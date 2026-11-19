"""Build the fixed, label-free trial feature table used by the autoresearch loop.

Runs once on the HABS backend, where the dataset is mounted. It reads the audited
trial grid (``data/aligned_trials.csv.gz`` from the annotation audit) and writes
one row per trial: 64-channel band powers, a short evoked response, a 20-channel
covariance, cardiac, skin and respiratory summaries, plus the ratings and the
metadata the harness needs to split the data. Eye tracking is excluded: the
organisers reported it misaligned with the other modalities.
"""

import json
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from habs.features import _ecg_features, _eda_features
from habs.temporal.data import physical_id

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
COV = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", "T7")
COV += ("C3", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2")
VIDEO_S, PRE_S, ERP_BINS, ERP_BIN_S = 10.0, 2.0, 5, 0.2
EEG_UNITS = {"V": 1e6, "mV": 1e3, "uV": 1.0}
META = (
    "cohort",
    "participant_id",
    "recorded_day",
    "event_order",
    "start_marker",
    "onset_s",
    "video",
    "video_theme",
    "default_emotion_label",
    "rating_valence",
    "rating_arousal",
    "rating_dominance",
    "recording",
    "age",
    "gender",
)


def _windows(x, fs, onsets, start, seconds):
    """Stack fixed-length windows; rows that leave the recording stay NaN."""
    n = round(seconds * fs)
    out = np.full((len(onsets), n), np.nan)
    for k, onset in enumerate(onsets):
        a = round((onset + start) * fs)
        if 0 <= a and a + n <= len(x):
            out[k] = x[a : a + n]
    return out


def _clean(w):
    """Blank windows with saturation, flat lines or non-finite samples."""
    bad = ~np.isfinite(w).all(axis=1)
    with np.errstate(invalid="ignore"):
        bad |= (np.ptp(w, axis=1) > 500) | (np.std(w, axis=1) < 0.01)
    w[bad] = np.nan
    return w


def _log_bandpower(w, fs, nperseg):
    f, p = welch(w, fs=fs, nperseg=nperseg, axis=-1)
    powers = []
    for low, high in BANDS.values():
        band = (f >= low) & (f < high)
        powers.append(trapezoid(p[:, band], f[band], axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.stack(powers, axis=1))


def _erp(x, fs, onsets):
    """Mean amplitude in five 200 ms bins after onset, minus a 200 ms baseline."""
    n = round(ERP_BIN_S * fs)  # native rates are not always integers
    w = _windows(x, fs, onsets, -ERP_BIN_S, n * (ERP_BINS + 1) / fs)
    w = w[:, : n * (ERP_BINS + 1)].reshape(len(onsets), ERP_BINS + 1, n).mean(axis=2)
    return w[:, 1:] - w[:, :1]


def _resp(x, fs, onsets):
    result = {"resp_amp": np.full(len(onsets), np.nan)}
    result["resp_rate"] = np.full(len(onsets), np.nan)
    if x is None or not np.isfinite(x).all():
        return result
    y = sosfiltfilt(butter(2, (0.1, 1.0), btype="bandpass", fs=fs, output="sos"), x)
    scale = np.std(y)
    if not scale > 0:
        return result
    w = _windows(y, fs, onsets, 0, VIDEO_S)
    result["resp_amp"] = np.std(w, axis=1) / scale
    for k, row in enumerate(w):
        if np.isfinite(row).all():
            peaks, _ = find_peaks(row, distance=round(1.5 * fs), prominence=0.5 * scale)
            result["resp_rate"][k] = len(peaks) * 60 / VIDEO_S
    return result


def _eeg_source(raw_dir, trial):
    """Prefer the released ICA-cleaned file when it shares the segment clock."""
    curated = Path(raw_dir) / trial["recording"]
    meta = Path(
        str(curated).replace("_task-emotion-video_multimodal.bdf", "_metadata.json")
    )
    alternative = Path(str(curated).replace("/CURATED/", "/PREPROCESSED/", 1))
    alt_meta = Path(str(meta).replace("/CURATED/", "/PREPROCESSED/", 1))
    if alternative == curated or not (alternative.is_file() and alt_meta.is_file()):
        return curated, 0.0
    origins = []
    for path in (meta, alt_meta):
        segments = json.loads(path.read_text())["dataset_export"]["segments"]
        origins.append(segments.get("emotion-video", {}).get("metadata_time_origin_s"))
    if None in origins:
        return curated, 0.0
    return alternative, float(origins[0]) - float(origins[1])


def _channel(reader, name):
    names = reader.getSignalLabels()
    if name not in names:
        return None, 0.0, ""
    i = names.index(name)
    unit = reader.getPhysicalDimension(i).replace("µ", "u").replace("μ", "u")
    return reader.readSignal(i), float(reader.getSampleFrequency(i)), unit


def _eeg_columns(path, onsets):
    """Band powers, evoked bins and covariance for every trial of one file."""
    columns, filtered = {}, {}
    with pyedflib.EdfReader(str(path)) as reader:
        names = reader.getSignalLabels()
        # EEG channels precede the peripheral block; PREPROCESSED drops some.
        names = names[
            : min(
                [names.index(n) for n in ("ECG", "Resp", "GSR") if n in names] or [64]
            )
        ]
        for name in names:
            x, fs, unit = _channel(reader, name)
            if unit not in EEG_UNITS or fs < 100:
                continue
            x = x * EEG_UNITS[unit]
            video = _clean(_windows(x, fs, onsets, 0, VIDEO_S))
            pre = _clean(_windows(x, fs, onsets, -PRE_S, PRE_S))
            for prefix, w, seg in (("bp_vid", video, 2.0), ("bp_pre", pre, 1.0)):
                values = _log_bandpower(w, fs, round(seg * fs))
                for b, band in enumerate(BANDS):
                    columns[f"{prefix}_{name}_{band}"] = values[:, b]
            erp = _erp(x, fs, onsets)
            erp[np.isnan(video).all(axis=1)] = np.nan
            for b in range(ERP_BINS):
                columns[f"erp_{name}_{b}"] = erp[:, b]
            if name in COV:
                sos = butter(4, (1, 45), btype="bandpass", fs=fs, output="sos")
                filtered[name] = _windows(sosfiltfilt(sos, x), fs, onsets, 0, VIDEO_S)
                filtered[name][np.isnan(video).all(axis=1)] = np.nan
    if len(filtered) == len(COV):
        stack = np.stack([filtered[n] for n in COV], axis=1)
        stack = stack - stack.mean(axis=2, keepdims=True)
        for i, j in zip(*np.triu_indices(len(COV)), strict=True):
            columns[f"cov_{COV[i]}_{COV[j]}"] = np.mean(
                stack[:, i] * stack[:, j], axis=1
            )
    return columns


def _recording_features(arguments):
    raw_dir, trials = arguments
    curated = Path(raw_dir) / trials[0]["recording"]
    onsets = np.array([float(t["onset_s"]) for t in trials])
    columns = {"trial_id": [physical_id(t) for t in trials]}
    eeg_path, shift = _eeg_source(raw_dir, trials[0])
    candidates = [(eeg_path, shift)] + ([(curated, 0.0)] if eeg_path != curated else [])
    source = "error"
    for path, offset in candidates:
        try:
            columns.update(_eeg_columns(path, onsets + offset))
            source = path.parts[-3]
            break
        except (OSError, ValueError) as error:
            print(f"EEG read failed for {path.name}: {error}", flush=True)
    columns["eeg_source"] = [source] * len(trials)
    with pyedflib.EdfReader(str(Path(raw_dir) / trials[0]["recording"])) as reader:
        ecg, ecg_fs, _ = _channel(reader, "ECG")
        resp, resp_fs, _ = _channel(reader, "Resp")
        names = reader.getSignalLabels()
        eda, eda_fs, eda_unit, bounds = None, 0.0, "", None
        if "GSR" in names:
            i = names.index("GSR")
            eda, eda_fs = reader.readSignal(i), float(reader.getSampleFrequency(i))
            eda_unit = reader.getPhysicalDimension(i)
            low, high = reader.getPhysicalMinimum(i), reader.getPhysicalMaximum(i)
            step = (high - low) / (
                reader.getDigitalMaximum(i) - reader.getDigitalMinimum(i)
            )
            bounds = (low, high, step)
    peripheral = []
    for onset in onsets:
        row = _ecg_features(ecg, ecg_fs, onset, VIDEO_S)
        row.update(_eda_features(eda, eda_fs, eda_unit, bounds, onset, VIDEO_S))
        pre = _eda_features(eda, eda_fs, eda_unit, bounds, onset - PRE_S, PRE_S)
        row["eda_pre_log_level"] = pre["eda_log_level"]
        peripheral.append(row)
    peripheral = pd.DataFrame(peripheral).drop(columns=["eda_reason"])
    columns.update({k: peripheral[k].to_numpy(dtype=float) for k in peripheral})
    columns.update(_resp(resp, resp_fs, onsets))
    return pd.DataFrame(columns)


def extract(trials, raw_dir, *, workers):
    """Return one feature row per audited trial, in the input order."""
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    if trials["trial_id"].duplicated().any():
        raise ValueError("Feature extraction requires unique physical trials.")
    jobs = [
        (str(raw_dir), group.to_dict("records"))
        for _, group in trials.groupby("recording", sort=True)
    ]
    parts = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for index, part in enumerate(pool.map(_recording_features, jobs), 1):
            parts.append(part)
            if index % 25 == 0:
                print(f"Recordings: {index}/{len(jobs)}", flush=True)
    features = pd.concat(parts, ignore_index=True)
    table = trials[["trial_id", *META]].merge(features, on="trial_id", validate="1:1")
    if len(table) != len(trials):
        raise ValueError("Feature extraction lost trials.")
    return table


def main():
    """Write the private feature table and a public aggregate manifest."""
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out = Path(os.environ["HABS_ANALYSIS_OUT"])
    trials = pd.read_csv(
        out / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    people = pd.read_csv(raw_dir / "participants_train.csv", na_values=["TESTSUBJECT"])
    people = people.assign(participant_id=people["participant_id"].astype(str))
    trials = trials.merge(people[["participant_id", "age"]], how="left", validate="m:1")
    workers = min(16, os.cpu_count() or 4)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    table = extract(trials, raw_dir, workers=workers)
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].astype("float32")
    (out / "autoresearch").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "autoresearch" / "trials.parquet", compression="zstd")
    groups = {}
    for column in table.columns:
        group = column.split("_")[0]
        groups[group] = groups.get(group, 0) + 1
    manifest = {
        "rows": len(table),
        "participants": int(table["participant_id"].nunique()),
        "themes": int(table["video_theme"].nunique()),
        "eeg_source": table["eeg_source"].value_counts().to_dict(),
        "column_groups": groups,
        "missing_fraction": {
            g: float(
                table[[c for c in table if c.startswith(g + "_")]].isna().mean().mean()
            )
            for g in ("bp", "erp", "cov", "ecg", "eda", "resp")
        },
        "workers": workers,
    }
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "autoresearch_features.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
