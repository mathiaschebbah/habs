"""Second-by-second physiology around each video onset (study 05).

Runs once on the HABS backend. For every audited trial it writes 1-second bins
from 3 s before to 12 s after onset: instantaneous heart rate, skin
conductance, breathing amplitude and rate, and log EEG band power averaged
over six scalp regions. It also writes, for each recording, heart rate and
RMSSD on consecutive 60-second windows and on their first 5, 10, 20 and 30
seconds, to test ultra-short heart rate variability.
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
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch

from decoding.common import BANDS, REGIONS
from decoding.extract import (
    EEG_UNITS,
    _channel,
    _clean,
    _eeg_source,
    _r_peaks,
    _read_body,
    _windows,
)
from habs.signals import eda_to_conductance
from habs.temporal.data import physical_id

START, BINS = -3, 15  # bin k covers [k - 3, k - 2) seconds from onset
SHORT = (5, 10, 20, 30)
EXTRA = ("final_emotion_label", "self_reported_emotion_1")


def _binned(x, fs, onsets):
    """Mean of a continuous signal in each 1-second bin; NaN outside the file."""
    n = round(fs)
    w = _windows(x, fs, onsets, START, n * BINS / fs)[:, : n * BINS]
    return w.reshape(len(onsets), BINS, n).mean(axis=2)


def _interpolated(times, values, onsets, max_gap):
    """Bin means of an event series (one value per beat or breath) at 4 Hz."""
    out = np.full((len(onsets), BINS), np.nan)
    if len(times) < 2:
        return out
    offsets = START + np.arange(BINS * 4) / 4 + 0.125
    for k, onset in enumerate(onsets):
        grid = onset + offsets
        series = np.interp(grid, times, values, left=np.nan, right=np.nan)
        nearest = np.abs(
            times[np.clip(np.searchsorted(times, grid), 0, len(times) - 1)] - grid
        )
        series[nearest > max_gap] = np.nan
        out[k] = series.reshape(BINS, 4).mean(axis=1)
    return out


def _heart(ecg, fs, onsets):
    """Instantaneous heart rate per bin, and short-window HRV per recording."""
    empty = np.full((len(onsets), BINS), np.nan)
    if ecg is None or fs < 100 or not np.isfinite(ecg).all():
        return empty, pd.DataFrame()
    beats = _r_peaks(ecg, fs)
    rr = np.diff(beats)
    ok = (rr >= 0.35) & (rr <= 1.5)
    times, rate = beats[1:][ok], 60 / rr[ok]
    rows = []
    for index, start in enumerate(np.arange(0, beats[-1] - 60, 60)):
        row = {"window": index}
        for seconds in (*SHORT, 60):
            inside = (beats >= start) & (beats < start + seconds)
            intervals = np.diff(beats[inside])
            good = (intervals >= 0.35) & (intervals <= 1.5)
            if len(intervals) < 3 or good.mean() < 0.8:
                row[f"hr_{seconds}"] = row[f"rmssd_{seconds}"] = np.nan
                continue
            intervals = intervals[good]
            row[f"hr_{seconds}"] = 60 / intervals.mean()
            row[f"rmssd_{seconds}"] = np.sqrt(np.mean(np.diff(intervals) ** 2))
        rows.append(row)
    return _interpolated(times, rate, onsets, max_gap=2.0), pd.DataFrame(rows)


def _breathing(resp, fs, onsets):
    """Breathing amplitude envelope and instantaneous rate per bin."""
    empty = np.full((len(onsets), BINS), np.nan)
    if resp is None or fs < 10 or not np.isfinite(resp).all():
        return empty, empty
    y = sosfiltfilt(butter(2, (0.1, 1.0), btype="bandpass", fs=fs, output="sos"), resp)
    scale = np.std(y)
    if not scale > 0:
        return empty, empty
    envelope = np.abs(hilbert(y / scale))
    tops, _ = find_peaks(y, distance=round(1.5 * fs), prominence=0.3 * scale)
    times = tops / fs
    rate = _interpolated(times[1:], 60 / np.diff(times), onsets, max_gap=8.0)
    return _binned(envelope, fs, onsets), rate


def _eeg(path, onsets):
    """Log band power per region and bin, averaged over the region's channels."""
    sums = {(r, b): [] for r in REGIONS for b in BANDS}
    wanted = {ch: r for r, chs in REGIONS.items() for ch in chs}
    with pyedflib.EdfReader(str(path)) as reader:
        for name in reader.getSignalLabels():
            if name not in wanted:
                continue
            x, fs, unit = _channel(reader, name)
            if unit not in EEG_UNITS or fs < 100:
                continue
            n = round(fs)
            w = _windows(x * EEG_UNITS[unit], fs, onsets, START, n * BINS / fs)
            w = _clean(w[:, : n * BINS].reshape(len(onsets) * BINS, n))
            f, p = welch(w, fs=fs, nperseg=n, axis=-1)
            for band, (low, high) in zip(
                BANDS, ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)), strict=True
            ):
                mask = (f >= low) & (f < high)
                with np.errstate(divide="ignore", invalid="ignore"):
                    power = np.log10(trapezoid(p[:, mask], f[mask], axis=-1))
                sums[(wanted[name], band)].append(power.reshape(len(onsets), BINS))
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for (region, band), parts in sums.items():
            value = (
                np.nanmean(parts, axis=0)
                if parts
                else np.full((len(onsets), BINS), np.nan)
            )
            out[f"eeg_{region}_{band}"] = value
    return out


def _recording(arguments):
    """Time-resolved trial features and HRV windows for one recording."""
    raw_dir, trials = arguments
    onsets = np.array([float(t["onset_s"]) for t in trials])
    ecg, ecg_fs, resp, resp_fs, eda, eda_fs, eda_unit, _ = _read_body(
        Path(raw_dir) / trials[0]["recording"]
    )
    signals = {}
    signals["hr"], windows = _heart(ecg, ecg_fs, onsets)
    signals["resp_amp"], signals["resp_rate"] = _breathing(resp, resp_fs, onsets)
    signals["sc"] = np.full((len(onsets), BINS), np.nan)
    if eda is not None and eda_fs >= 10:
        conductance, _ = eda_to_conductance(eda, eda_unit, max_invalid_fraction=1.0)
        signals["sc"] = _binned(conductance, eda_fs, onsets)
    path, shift = _eeg_source(raw_dir, trials[0])
    try:
        signals.update(_eeg(path, onsets + shift))
    except (OSError, ValueError) as error:
        print(f"EEG read failed for {path.name}: {error}", flush=True)
    columns = {"trial_id": [physical_id(t) for t in trials]}
    for name, values in signals.items():
        for k in range(BINS):
            columns[f"tc_{name}_{k}"] = values[:, k].astype("float32")
    return pd.DataFrame(columns), windows.assign(recording=trials[0]["recording"])


def run(trials, raw_dir, *, workers):
    """Apply ``_recording`` to every recording; report progress as they finish."""
    jobs = [
        (str(raw_dir), group.to_dict("records"))
        for _, group in trials.groupby("recording", sort=True)
    ]
    parts, windows = [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_recording, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            part, window = future.result()
            parts.append(part)
            windows.append(window)
            if index % 25 == 0:
                print(f"Recordings: {index}/{len(jobs)}", flush=True)
    return pd.concat(parts, ignore_index=True), pd.concat(windows, ignore_index=True)


def main():
    """Write the private time-course and HRV tables and a public coverage file."""
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out = Path(os.environ["HABS_ANALYSIS_OUT"])
    trials = pd.read_csv(
        out / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    table, windows = run(trials, raw_dir, workers=min(16, os.cpu_count() or 4))
    table = trials[["trial_id", *EXTRA]].merge(table, on="trial_id", validate="1:1")
    (out / "features").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "features" / "timecourse.parquet", compression="zstd")
    windows.to_parquet(out / "features" / "hrv_windows.parquet", compression="zstd")
    coverage = {
        "rows": len(table),
        "hrv_windows": len(windows),
        "coverage": {
            s: float(table[f"tc_{s}_5"].notna().mean())
            for s in ("hr", "sc", "resp_amp", "resp_rate", "eeg_frontal_alpha")
        },
    }
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "timecourse.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
