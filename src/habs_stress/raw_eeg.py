"""Strictly prefix-limited signal processing for the raw EEG and body controls."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

FS = 200
LAGS = np.arange(0, 25)  # zero to 120 milliseconds of past ECG


def prefix(signal, fs, onsets, *, start=-5.0, stop=5.0, band=(0.5, 45.0)):
    """Crop before validation, filter forward and sample on a 200 Hz grid.

    The first untouched source sample is ceil((onset + stop) * fs).
    Downsampling selects the most recent filtered sample; no future padding.
    Out-of-bounds or nonfinite prefixes remain missing.
    """
    count = round((stop - start) * FS)
    result = np.full((len(onsets), count), np.nan, dtype=np.float32)
    if signal is None or fs < FS:
        return result
    native_count = round((stop - start) * fs)
    grid = np.floor(np.arange(count) * fs / FS).astype(int)
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos") if band else None
    zi = sosfilt_zi(sos) if sos is not None else None
    for row, onset in enumerate(onsets):
        end = int(np.ceil((onset + stop) * fs - 1e-8))
        begin = end - native_count
        if begin < 0 or end > len(signal):
            continue
        data = signal[begin:end]
        if not np.isfinite(data).all():
            continue
        if sos is not None:
            data, _ = sosfilt(sos, data, zi=zi * float(data[0]))
        result[row] = data[grid]
    return result


def reference_eeg(eeg):
    """Reference to available electrodes and retain a common, predeclared QC mask."""
    valid = np.isfinite(eeg).all(axis=-1)
    valid &= (np.ptp(eeg, axis=-1) <= 800) & (np.std(eeg, axis=-1) >= 0.01)
    n_good = valid.sum(axis=1)
    kept = np.where(valid[:, :, None], eeg, 0)
    mean = kept.sum(axis=1) / np.maximum(n_good[:, None], 1)
    return np.where(valid[:, :, None], kept - mean[:, None, :], 0).astype("float32"), valid


def ecg_design(ecg, *, warmup=360):
    """Lagged filtered ECG; scale from the warm-up preceding the EEG interval."""
    reference = ecg[:, :warmup]
    centered = (ecg - reference.mean(axis=1, keepdims=True)) / np.maximum(
        reference.std(axis=1, keepdims=True), 1e-6
    )
    return np.stack([centered[:, warmup - lag : ecg.shape[1] - lag] for lag in LAGS], axis=2)


def regress_ecg(eeg, valid, design, coefs):
    """Subtract the transfer coefficients frozen on independent calibration IDs."""
    predicted = np.einsum("ntl,lc->nct", design, coefs, optimize=True)
    return np.where(valid[:, :, None], eeg - predicted, 0.0).astype("float32")


def band_features(eeg):
    """Log powers in five bands over reference and early halves, including their change."""
    out = []
    for segment in (eeg[:, :, :640], eeg[:, :, 640:1140], eeg[:, :, 1140:]):
        taper = np.hanning(segment.shape[-1])
        spectrum = np.abs(np.fft.rfft(segment * taper, axis=-1)) ** 2 / np.sum(taper * taper)
        freq = np.fft.rfftfreq(segment.shape[-1], 1 / FS)
        out.append(
            np.stack(
                [
                    np.log10(spectrum[:, :, (freq >= lo) & (freq < hi)].mean(axis=-1) + 1e-8)
                    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
                ],
                axis=2,
            )
        )
    return np.concatenate((out[1] - out[0], out[2] - out[0]), axis=2).reshape(len(eeg), -1)
