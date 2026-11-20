"""Heartbeat-evoked EEG responses around each video (study 06, part B).

Runs once on the HABS backend. For every audited trial it averages the EEG
locked to each R peak (-200 to +600 ms) over the beats of two windows, the 3 s
before video onset and the 10 s of the video, per scalp region, in 50 ms bins.
It stores the ECG averaged the same way, to check that an EEG difference is not
the heart's own electrical field, and the beat count and mean RR interval.
"""

import json
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, sosfiltfilt

from decoding.common import REGIONS
from decoding.extract import EEG_UNITS, _channel, _eeg_source, _r_peaks, _read_body
from habs.temporal.data import physical_id

WINDOWS = {"pre": (-3.0, 0.0), "video": (0.0, 10.0)}
EPOCH, BIN_S, NBINS, BASELINE = (-0.2, 0.6), 0.05, 16, 3  # bins 0-2: -200 to -50 ms
MAX_UV = 100.0  # reject an EEG epoch whose baseline-corrected peak exceeds this


def _band(x, fs):
    """0.5-30 Hz band; non-finite samples become NaN again after filtering."""
    bad = ~np.isfinite(x)
    sos = butter(2, (0.5, 30), btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, np.where(bad, 0.0, x))
    y[bad] = np.nan
    return y


def _refine(beats, ecg, fs):
    """Move each detected beat to the largest band-passed ECG deflection nearby."""
    sos = butter(2, (5, 25), btype="bandpass", fs=fs, output="sos")
    y = np.abs(sosfiltfilt(sos, ecg))
    half = round(0.06 * fs)
    out = []
    for b in np.round(beats * fs).astype(int):
        a, z = max(b - half, 0), min(b + half + 1, len(y))
        out.append((a + int(np.argmax(y[a:z]))) / fs)
    return np.array(out)


def _bin_index(fs):
    n = round((EPOCH[1] - EPOCH[0]) * fs)
    return n, np.minimum((np.arange(n) / fs / BIN_S).astype(int), NBINS - 1)


def _locked(x, fs, beats, start, stop, reject=None):
    """Mean R-locked epoch over beats in [start, stop), in 50 ms bins.

    Each epoch is baseline-corrected on -200 to -50 ms. With ``reject``, epochs
    whose absolute corrected value exceeds it are dropped. Returns the binned
    mean (NaN if no epoch is left) and the number of epochs used.
    """
    n, index = _bin_index(fs)
    chosen = beats[(beats >= start) & (beats < stop)]
    first = np.round((chosen + EPOCH[0]) * fs).astype(int)
    first = first[(first >= 0) & (first + n <= len(x))]
    if not len(first):
        return np.full(NBINS, np.nan), 0
    epochs = x[first[:, None] + np.arange(n)]
    epochs = epochs[np.isfinite(epochs).all(axis=1)]
    if not len(epochs):
        return np.full(NBINS, np.nan), 0
    binned = np.stack([epochs[:, index == k].mean(axis=1) for k in range(NBINS)], 1)
    binned -= binned[:, :BASELINE].mean(axis=1, keepdims=True)
    if reject is not None:
        epochs = epochs - epochs[:, index < BASELINE].mean(axis=1, keepdims=True)
        binned = binned[np.abs(epochs).max(axis=1) <= reject]
    if not len(binned):
        return np.full(NBINS, np.nan), 0
    return binned.mean(axis=0), len(binned)


def _regions(path, beats, onsets):
    """Region-averaged heartbeat-evoked EEG for every trial and window."""
    wanted = {ch: r for r, chs in REGIONS.items() for ch in chs}
    sums = {}
    with pyedflib.EdfReader(str(path)) as reader:
        for name in reader.getSignalLabels():
            if name not in wanted:
                continue
            x, fs, unit = _channel(reader, name)
            if unit not in EEG_UNITS or fs < 100:
                continue
            x = _band(x * EEG_UNITS[unit], fs)
            for window, (a, b) in WINDOWS.items():
                mean = np.array(
                    [_locked(x, fs, beats, t + a, t + b, MAX_UV)[0] for t in onsets]
                )
                sums.setdefault((wanted[name], window), []).append(mean)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for region in REGIONS:
            for window in WINDOWS:
                parts = sums.get((region, window))
                value = (
                    np.nanmean(parts, axis=0)
                    if parts
                    else np.full((len(onsets), NBINS), np.nan)
                )
                for k in range(NBINS):
                    out[f"hep_{window}_{region}_{k}"] = value[:, k]
    return out


def _recording(arguments):
    """Heartbeat-evoked EEG and ECG features for one recording."""
    raw_dir, trials = arguments
    onsets = np.array([float(t["onset_s"]) for t in trials])
    ecg, fs, *_ = _read_body(Path(raw_dir) / trials[0]["recording"])
    columns = {"trial_id": [physical_id(t) for t in trials]}
    if ecg is None or fs < 100 or not np.isfinite(ecg).all():
        return pd.DataFrame(columns)
    beats = _refine(_r_peaks(ecg, fs), ecg, fs)
    rr = np.diff(beats)
    filtered = _band(ecg, fs)
    for window, (a, b) in WINDOWS.items():
        locked = [_locked(filtered, fs, beats, t + a, t + b) for t in onsets]
        for k in range(NBINS):
            columns[f"ecg_{window}_{k}"] = [m[k] for m, _ in locked]
        columns[f"beats_{window}"] = [count for _, count in locked]
        columns[f"rr_{window}"] = [
            np.mean(rr[(beats[1:] >= t + a) & (beats[1:] < t + b)])
            if ((beats[1:] >= t + a) & (beats[1:] < t + b)).any()
            else np.nan
            for t in onsets
        ]
    path, shift = _eeg_source(raw_dir, trials[0])
    try:
        columns.update(_regions(path, beats + shift, onsets + shift))
    except (OSError, ValueError) as error:
        print(f"EEG read failed for {path.name}: {error}", flush=True)
    frame = pd.DataFrame(columns)
    numeric = frame.columns.drop("trial_id")
    frame[numeric] = frame[numeric].astype("float32")
    return frame


def run(trials, raw_dir, *, workers):
    """Apply ``_recording`` to every recording; report progress as they finish."""
    jobs = [
        (str(raw_dir), group.to_dict("records"))
        for _, group in trials.groupby("recording", sort=True)
    ]
    parts = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_recording, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            parts.append(future.result())
            if index % 25 == 0:
                print(f"Recordings: {index}/{len(jobs)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    """Write the private heartbeat-evoked table and a public coverage file."""
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out = Path(os.environ["HABS_ANALYSIS_OUT"])
    trials = pd.read_csv(
        out / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    table = run(trials, raw_dir, workers=min(16, os.cpu_count() or 4))
    (out / "features").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "features" / "hep.parquet", compression="zstd")
    coverage = {
        "rows": len(table),
        "coverage": {
            c: float(table[c].notna().mean()) if c in table else 0.0
            for c in ("hep_video_central_10", "hep_pre_central_10", "ecg_video_10")
        },
        "beats_video_median": float(table["beats_video"].median())
        if "beats_video" in table
        else None,
    }
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "hep.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
