"""Extract target-independent EEG, cardiac and skin features from native BDFs."""

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, sosfilt, welch

from habs.signals import eda_to_conductance

REGIONS = {"frontal": ("F3", "F4", "Fz"), "posterior": ("P3", "P4", "O1", "O2")}
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
EEG = tuple(f"eeg_{region}_{band}" for region in REGIONS for band in BANDS)
ECG = ("ecg_hr", "ecg_change")
EDA = ("eda_log_level", "eda_slope", "eda_log_sd")
PHYSIOLOGY = EEG + ECG + EDA
FEATURE_VERSION = "20261108-v1"


def _slice(values, rate, onset, seconds):
    start, count = max(0, round(onset * rate)), round(seconds * rate)
    return values[start : start + count]


def _eeg_features(signals, onset, seconds):
    result, regional_counts = {}, []
    for region, electrodes in REGIONS.items():
        band_values = []
        for name in electrodes:
            if name not in signals:
                continue
            values, rate = signals[name]
            segment = _slice(values, rate, onset, seconds)
            if (
                len(segment) != round(seconds * rate)
                or rate < 100
                or not np.isfinite(segment).all()
                or np.ptp(segment) > 500
                or not 0.001 < np.std(segment) < 150
            ):
                continue
            size = min(len(segment), round(2 * rate))
            frequencies, power = welch(
                segment, fs=rate, nperseg=size, noverlap=size // 2, detrend="constant"
            )
            reference = (frequencies >= 1) & (frequencies <= 40)
            total = trapezoid(power[reference], x=frequencies[reference])
            if not np.isfinite(total) or total <= 0:
                continue
            bands = []
            for low, high in BANDS.values():
                mask = (frequencies >= low) & (frequencies <= high)
                bands.append(trapezoid(power[mask], x=frequencies[mask]) / total)
            band_values.append(bands)
        regional_counts.append(len(band_values))
        for index, band in enumerate(BANDS):
            result[f"eeg_{region}_{band}"] = (
                float(np.median(np.asarray(band_values)[:, index]))
                if len(band_values) >= 2
                else np.nan
            )
    result["eeg_valid"] = min(regional_counts) >= 2
    return result


def _ecg_features(values, rate, onset, seconds):
    result = dict.fromkeys(ECG, np.nan)
    result["ecg_valid"] = False
    if values is None or rate < 100:
        return result
    earlier = max(0.0, onset - 2)
    segment = _slice(values, rate, earlier, onset - earlier + seconds)
    if len(segment) < round(seconds * rate) or not np.isfinite(segment).all():
        return result
    filtered = sosfilt(
        butter(2, (5, 25), btype="bandpass", fs=rate, output="sos"), segment
    )
    width = max(1, round(0.08 * rate))
    envelope = np.convolve(np.abs(filtered), np.ones(width) / width, mode="full")[
        : len(segment)
    ]
    peaks, _ = find_peaks(
        envelope, distance=round(0.3 * rate), prominence=np.std(envelope)
    )
    rr = np.diff(peaks) / rate
    times = peaks[1:] / rate - (onset - earlier)
    in_trial = (times >= 0) & (times < seconds)
    plausible = in_trial & (rr >= 0.35) & (rr <= 1.5)
    if plausible.sum() < (5 if seconds >= 10 else 3):
        return result
    if plausible.sum() < 0.8 * in_trial.sum():
        return result
    heart_rate = 60 / rr[plausible]
    late = times[plausible] >= seconds / 2
    if not late.any() or late.all():
        return result
    result.update(
        {
            "ecg_hr": float(np.median(heart_rate)),
            "ecg_change": float(np.mean(heart_rate[late]) - np.mean(heart_rate[~late])),
            "ecg_valid": True,
        }
    )
    return result


def _eda_features(values, rate, unit, bounds, onset, seconds):
    result = {**dict.fromkeys(EDA, np.nan), "eda_valid": False, "eda_reason": "missing"}
    if values is None:
        return result
    native = _slice(values, rate, onset, seconds)
    if len(native) != round(seconds * rate):
        return result
    conductance, quality = eda_to_conductance(native, unit)
    result["eda_reason"] = quality["status"]
    if quality["status"] != "ok":
        return result
    low, high, step = bounds
    usable_native = native[np.isfinite(conductance)]
    boundary_fraction = np.mean(
        (usable_native <= low + 2 * step) | (usable_native >= high - 2 * step)
    )
    if np.std(usable_native) <= 2 * step or np.mean(np.diff(usable_native) == 0) > 0.99:
        result["eda_reason"] = "flat"
        return result
    if boundary_fraction > 0.05:
        result["eda_reason"] = "export_boundary_plateau"
        return result
    valid = np.isfinite(conductance)
    times = np.arange(len(native))[valid] / rate
    values = conductance[valid]
    centered_time = times - np.mean(times)
    denominator = np.dot(centered_time, centered_time)
    result.update(
        {
            "eda_log_level": float(np.log(np.median(values))),
            "eda_slope": float(
                np.dot(centered_time, values - np.mean(values)) / denominator
            ),
            "eda_log_sd": float(np.log1p(np.std(values))),
            "eda_valid": True,
            "eda_reason": "ok",
        }
    )
    return result


def _recording_features(arguments):
    raw_dir, cache_dir, recording, trials = arguments
    path = Path(raw_dir) / recording
    signature = {
        "version": FEATURE_VERSION,
        "algorithm_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("signals.py").read_bytes()
        ).hexdigest(),
        "recording": recording,
        "size": path.stat().st_size,
        "mtime": path.stat().st_mtime_ns,
        "trials": [(r["trial_id"], r["onset_s"]) for r in trials],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"{digest}.csv.gz"
    if cache.is_file():
        return pd.read_csv(cache)
    rows = []
    with pyedflib.EdfReader(str(path)) as reader:
        names = reader.getSignalLabels()
        signals = {}
        for electrode in set(sum(REGIONS.values(), ())):
            if electrode not in names:
                continue
            index = names.index(electrode)
            unit = (
                reader.getPhysicalDimension(index).replace("µ", "u").replace("μ", "u")
            )
            factor = {"V": 1e6, "mV": 1e3, "uV": 1.0}.get(unit)
            if factor is not None:
                signals[electrode] = (
                    reader.readSignal(index) * factor,
                    float(reader.getSampleFrequency(index)),
                )
        ecg, ecg_rate, eda, eda_rate, eda_unit, bounds = None, 0, None, 0, "", None
        if "ECG" in names:
            index = names.index("ECG")
            ecg, ecg_rate = (
                reader.readSignal(index),
                float(reader.getSampleFrequency(index)),
            )
        if "GSR" in names:
            index = names.index("GSR")
            eda, eda_rate = (
                reader.readSignal(index),
                float(reader.getSampleFrequency(index)),
            )
            eda_unit = reader.getPhysicalDimension(index)
            low, high = (
                reader.getPhysicalMinimum(index),
                reader.getPhysicalMaximum(index),
            )
            digital_range = reader.getDigitalMaximum(index) - reader.getDigitalMinimum(
                index
            )
            bounds = (low, high, (high - low) / digital_range)
        for trial in trials:
            row = {"trial_id": trial["trial_id"]}
            for seconds, prefix in ((10, ""), (5, "early_")):
                onset = float(trial["onset_s"])
                features = {
                    **_eeg_features(signals, onset, seconds),
                    **_ecg_features(ecg, ecg_rate, onset, seconds),
                    **_eda_features(eda, eda_rate, eda_unit, bounds, onset, seconds),
                }
                row.update({prefix + key: value for key, value in features.items()})
            rows.append(row)
    result = pd.DataFrame(rows)
    temporary = cache.with_name(cache.name + ".tmp.gz")
    result.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(cache)
    return result


def extract_signals(trials, raw_dir, cache_dir, *, workers=4):
    """Materialize deterministic trial features with source-keyed checkpoints.

    Parameters
    ----------
    trials : pandas.DataFrame
        Reconciled trials with stable trial IDs, recording paths and onsets.
    raw_dir : str or pathlib.Path
        Read-only dataset root.
    cache_dir : str or pathlib.Path
        Private output directory for target-independent feature checkpoints.
    workers : int, default=4
        Independent recording readers. Each process opens a distinct BDF.

    Returns
    -------
    pandas.DataFrame
        Original rows joined one-to-one to full and early-window features.

    Raises
    ------
    ValueError
        If a trial identity is duplicated or feature extraction loses rows.
    """
    if trials["trial_id"].duplicated().any():
        raise ValueError("Signal extraction requires unique physical trial IDs.")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    arguments = [
        (str(raw_dir), str(cache_dir), name, group.to_dict("records"))
        for name, group in trials.groupby("recording", sort=True)
    ]
    parts = []
    if workers == 1:
        iterator = map(_recording_features, arguments)
        for part in iterator:
            parts.append(part)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for index, part in enumerate(pool.map(_recording_features, arguments), 1):
                parts.append(part)
                if index % 10 == 0:
                    print(f"Features: {index}/{len(arguments)} recordings", flush=True)
    features = pd.concat(parts, ignore_index=True)
    result = trials.merge(features, on="trial_id", how="left", validate="one_to_one")
    if len(result) != len(trials) or result["eeg_valid"].isna().any():
        raise ValueError("The feature table does not cover the requested trial grid.")
    return result
