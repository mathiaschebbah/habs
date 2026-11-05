"""Extract complementary peripheral and EEG measurements from reconciled trials."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from alignment import reconcile
from starting_notebooks.bdf_loader import uV_to_uS

FEATURES = ["eda_level", "eda_change", "eda_spread"]
RAW_FEATURES = ["voltage_level", "voltage_change", "voltage_spread"]
EEG_CHANNELS = ["F3", "F4", "Fz", "Cz", "P3", "P4", "O1", "O2"]
EEG_FEATURES = [
    f"eeg_{region}_{band}"
    for region in ("front", "back")
    for band in ("theta", "alpha", "beta")
]
ECG_FEATURES = ["ecg_rate", "ecg_change"]


def describe_window(values, rate):
    """Return level, end-minus-start change and interquartile spread."""
    early, late = values[: round(3 * rate)], values[round(7 * rate) :]
    return [
        float(np.median(values)),
        float(np.median(late) - np.median(early)),
        float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    ]


def read_recording(path, include_eeg=True):
    """Read only selected channels and the event annotations."""
    channels = {}
    with pyedflib.EdfReader(str(path)) as reader:
        names = reader.getSignalLabels()
        for name in ["GSR", "ECG", *(EEG_CHANNELS if include_eeg else [])]:
            if name in names:
                index = names.index(name)
                channels[name] = {
                    "values": reader.readSignal(index),
                    "rate": reader.getSampleFrequency(index),
                    "unit": reader.getPhysicalDimension(index),
                }
        onsets, _, descriptions = reader.readAnnotations()
    markers = defaultdict(list)
    for onset, description in zip(onsets, descriptions, strict=True):
        try:
            markers[int(float(description))].append(float(onset))
        except ValueError:
            continue
    if "ECG" in channels:
        values, rate = channels["ECG"]["values"], channels["ECG"]["rate"]
        finite = np.isfinite(values)
        if finite.any() and finite.mean() >= 0.99 and np.std(values[finite]) > 0:
            signal = np.interp(
                np.arange(len(values)), np.flatnonzero(finite), values[finite]
            )
            filtered = sosfiltfilt(
                butter(3, [5, 25], btype="bandpass", fs=rate, output="sos"), signal
            )
            scale = np.median(np.abs(filtered - np.median(filtered)))
            polarity = (
                1
                if np.quantile(filtered, 0.995) >= -np.quantile(filtered, 0.005)
                else -1
            )
            peaks, _ = find_peaks(
                polarity * filtered,
                distance=round(0.30 * rate),
                height=max(3 * scale, 1e-12),
                prominence=max(3 * scale, 1e-12),
            )
            channels["ECG"]["peaks"] = peaks / rate
    return channels, markers


def window(channel, onset):
    """Return the observed ten-second slice, allowing one sample of rounding."""
    rate, values = channel["rate"], channel["values"]
    start, stop = round(onset * rate), round((onset + 10) * rate)
    if start < 0 or stop > len(values) + 1:
        return None
    result = values[start : min(stop, len(values))]
    return result if len(result) >= round(10 * rate) - 1 else None


def eda_features(channel, onset):
    """Use the supplied conversion, while recording its clipping explicitly."""
    values = window(channel, onset)
    if values is None or channel["unit"] not in ("uV", "µV"):
        return {"eda_valid": False}
    finite = np.isfinite(values)
    if finite.mean() < 0.99:
        return {"eda_valid": False, "finite_fraction": float(finite.mean())}
    values = np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite])
    conductance = uV_to_uS(values)
    voltage = values / 1e6
    clipping = float(((conductance <= 1) | (conductance >= 40)).mean())
    flat = bool(np.max(conductance) == np.min(conductance))
    result = {
        "finite_fraction": float(finite.mean()),
        "clamp_fraction": clipping,
        "flat": flat,
        "raw_flat": bool(np.max(voltage) == np.min(voltage)),
        "raw_min": float(voltage.min()),
        "raw_max": float(voltage.max()),
        "eda_valid": not flat and clipping < 0.05,
    }
    result.update(
        dict(zip(FEATURES, describe_window(conductance, channel["rate"]), strict=True))
    )
    result.update(
        dict(zip(RAW_FEATURES, describe_window(voltage, channel["rate"]), strict=True))
    )
    return result


def eeg_features(channels, onset):
    """Summarize relative band powers in two prespecified electrode groups."""
    result = {"eeg_valid": False}
    for region, names in (("front", EEG_CHANNELS[:4]), ("back", EEG_CHANNELS[4:])):
        relative = []
        for name in names:
            if name not in channels:
                continue
            values = window(channels[name], onset)
            if values is None or not np.isfinite(values).all() or np.std(values) == 0:
                continue
            rate = channels[name]["rate"]
            frequencies, power = welch(
                values, fs=rate, nperseg=round(2 * rate), noverlap=round(rate)
            )
            total = power[(frequencies >= 1) & (frequencies <= 40)].sum()
            if not total > 0:
                continue
            relative.append(
                [
                    power[(frequencies >= low) & (frequencies < high)].sum() / total
                    for low, high in ((4, 8), (8, 13), (13, 30))
                ]
            )
        if len(relative) < 2:
            return result
        result.update(
            dict(
                zip(
                    [f"eeg_{region}_{band}" for band in ("theta", "alpha", "beta")],
                    np.mean(relative, axis=0).tolist(),
                    strict=True,
                )
            )
        )
    result["eeg_valid"] = True
    return result


def ecg_features(channel, onset):
    """Estimate heart-rate level and change without relying on amplitude units."""
    values = window(channel, onset)
    peaks = channel.get("peaks", np.array([]))
    if values is None or not np.isfinite(values).all():
        return {"ecg_valid": False}
    rr = np.diff(peaks[(peaks >= onset) & (peaks < onset + 10)])
    plausible = (rr >= 0.33) & (rr <= 1.5)
    if len(rr) < 4 or plausible.mean() < 0.8:
        return {"ecg_valid": False}
    rates = 60 / rr[plausible]
    half = len(rates) // 2
    return {
        "ecg_valid": True,
        "ecg_rate": float(np.median(rates)),
        "ecg_change": float(np.median(rates[half:]) - np.median(rates[:half])),
    }


def extract_session(task):
    """Compare raw and cleaned representations without consulting the ratings."""
    data_dir, rows = task
    paths = {
        info["path"]
        for row in rows
        for info in json.loads(row["variants"]).values()
        if info["bounded"]
    }
    has_cleaned = any(path.startswith("PREPROCESSED/") for path in paths)
    recordings = {
        path: read_recording(
            data_dir / path,
            include_eeg=path.startswith("PREPROCESSED/") or not has_cleaned,
        )
        for path in sorted(paths)
    }
    output = []
    for row in rows:
        row = dict(row)
        candidates = []
        for source, info in json.loads(row["variants"]).items():
            if not info["bounded"]:
                continue
            channels, markers = recordings[info["path"]]
            times = markers.get(int(row["start_marker"]), [])
            error = min((abs(t - info["onset"]) for t in times), default=float("inf"))
            verified = error <= 0.010
            measurement = {
                "source": source,
                "verified": verified,
                "timing_error": error,
            }
            measurement["eda"] = (
                eda_features(channels["GSR"], info["onset"])
                if "GSR" in channels
                else {"eda_valid": False}
            )
            measurement["eeg"] = eeg_features(channels, info["onset"])
            measurement["ecg"] = (
                ecg_features(channels["ECG"], info["onset"])
                if "ECG" in channels
                else {"ecg_valid": False}
            )
            candidates.append(measurement)
        row["annotation_verified"] = any(c["verified"] for c in candidates)
        row["timing_error"] = min(c["timing_error"] for c in candidates)
        row["raw_eda_valid"] = any(
            c["source"] == "CURATED" and c["verified"] and c["eda"]["eda_valid"]
            for c in candidates
        )
        for modality in ("eda", "eeg", "ecg"):
            preferred = "PREPROCESSED" if modality == "eeg" else "CURATED"
            chosen = min(
                candidates,
                key=lambda c: (
                    not (c["verified"] and c[modality][f"{modality}_valid"]),
                    c["source"] != preferred,
                ),
            )
            row.update(chosen[modality])
            row[f"{modality}_valid"] = bool(
                chosen["verified"] and chosen[modality][f"{modality}_valid"]
            )
            row[f"{modality}_source"] = chosen["source"]
        row["join_status"] = "matched"
        output.append(row)
    return output


def extract(data_dir, out_dir, workers=4):
    """Extract all aligned cohorts and report modality-specific usable samples.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only release root.
    out_dir : pathlib.Path
        Private output directory.
    workers : int
        Maximum number of concurrently loaded sessions.

    Returns
    -------
    pandas.DataFrame
        Aligned trial measurements with modality-specific validity flags.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    aligned = reconcile(data_dir, out_dir)
    accepted = aligned.loc[
        aligned.alignment.isin(["exact", "day_corrected", "day_recovered"])
    ]
    tasks = [
        (data_dir, group.to_dict("records"))
        for _, group in accepted.groupby(["participant_id", "day"])
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, batch in enumerate(pool.map(extract_session, tasks), 1):
            rows.extend(batch)
            if n % 25 == 0:
                print(f"Multimodal sessions processed: {n}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows)
    for name in [*FEATURES, *RAW_FEATURES, *EEG_FEATURES, *ECG_FEATURES]:
        if name not in frame:
            frame[name] = np.nan
    for name in [
        "eda_valid",
        "eeg_valid",
        "ecg_valid",
        "raw_eda_valid",
        "flat",
        "raw_flat",
    ]:
        frame[name] = (
            frame.get(name, pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
        )
    frame["any_valid"] = frame[["eda_valid", "eeg_valid", "ecg_valid"]].any(axis=1)
    frame.to_csv(
        out_dir / "trials.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    summary = {
        "rows": len(frame),
        "people": int(frame.participant_id.nunique()),
        "annotation_verified": int(frame.annotation_verified.sum()),
        "cohorts": {},
        "eda_rescued_by_preprocessing": int(
            (frame.eda_valid & ~frame.raw_eda_valid).sum()
        ),
        "sources": {
            m: dict(Counter(frame[f"{m}_source"])) for m in ("eda", "eeg", "ecg")
        },
    }
    for cohort, group in frame.groupby("cohort"):
        summary["cohorts"][str(int(cohort))] = {}
        for modality in ("raw_eda", "eda", "eeg", "ecg", "any"):
            valid = group.loc[group[f"{modality}_valid"]]
            sizes = valid.groupby(["participant_id", "day"]).size()
            days = sizes.loc[sizes >= 20].groupby("participant_id").size()
            summary["cohorts"][str(int(cohort))][modality] = {
                "trials": len(valid),
                "people": int(valid.participant_id.nunique()),
                "two_days_20_trials": int((days >= 2).sum()),
                "one_day_40_trials": int(
                    sizes.loc[sizes >= 40].index.get_level_values(0).nunique()
                ),
            }
    (out_dir / "quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("HABS_QUALITY=" + json.dumps(summary, separators=(",", ":")), flush=True)
    return frame
