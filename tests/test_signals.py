"""Tests for signal calibration, ECG polarity/delineation, and hard causal t = 5.000 s."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from habs_stress.signals import (
    causal_bandpass_windows,
    delineate_ecg,
    eda_to_conductance,
    extract_causal_ecg_eda_past,
    read_bdf_summary,
)


def _synthetic_ecg(fs: float = 250.0, duration_s: float = 25.0, hr_bpm: float = 70.0, sign: float = 1.0) -> np.ndarray:
    t = np.arange(round(fs * duration_s)) / fs
    sig = np.zeros_like(t)
    rr = 60.0 / hr_bpm
    for beat in np.arange(0.5, duration_s - 0.6, rr):
        dt = t - beat
        # P wave (-0.16 s), QRS (0 s), T wave (+0.26 s)
        sig += 0.15 * np.exp(-0.5 * ((dt + 0.16) / 0.025) ** 2)
        sig += 1.20 * np.exp(-0.5 * (dt / 0.012) ** 2)
        sig -= 0.25 * np.exp(-0.5 * ((dt - 0.022) / 0.010) ** 2)
        sig += 0.30 * np.exp(-0.5 * ((dt - 0.26) / 0.045) ** 2)
    return sign * sig


def test_eda_to_conductance_valid_and_mismatch():
    volts = np.array([-1.5, 0.0, 1.5])
    cond, q = eda_to_conductance(volts, "V")
    assert q["status"] == "ok" and not q["unit_mismatch"]
    assert np.all(np.isfinite(cond)) and np.all(cond > 0)

    # Microvolt-scale values mislabeled as uS trigger documented mismatch flag
    cond_m, q_m = eda_to_conductance(volts * 1e6, "uS")
    assert q_m["status"] == "ok" and q_m["unit_mismatch"]
    assert np.allclose(cond, cond_m)


def test_ecg_delineation_auto_polarity_fix():
    fs = 250.0
    up = delineate_ecg(_synthetic_ecg(fs, sign=+1.0), fs)
    inv = delineate_ecg(_synthetic_ecg(fs, sign=-1.0), fs)
    assert up["polarity"] == 1.0
    assert inv["polarity"] == -1.0
    assert up["good_qrs_frac"] > 0.95 and inv["good_qrs_frac"] > 0.95
    assert np.isclose(np.nanmedian(up["rr_s"]), np.nanmedian(inv["rr_s"]), atol=0.01)
    assert np.isclose(np.nanmedian(up["pr_pk"]), np.nanmedian(inv["pr_pk"]), atol=5.0)


def test_hard_causal_wall_bit_invariance():
    """Perturbing samples at t >= 5.000 s must have ZERO effect on past features."""
    fs = 200.0
    rng = np.random.default_rng(7)
    n = round(30.0 * fs)
    sig_a = rng.standard_normal(n)
    sig_b = sig_a.copy()
    onset = 10.0
    wall_idx = round((onset + 5.0) * fs)
    sig_b[wall_idx:] += 1e4  # massive future shock strictly at t >= 5.000 s

    wa, fa = causal_bandpass_windows(sig_a, fs, np.array([onset]), -3.0, 5.0, 1.0, 40.0)
    wb, fb = causal_bandpass_windows(sig_b, fs, np.array([onset]), -3.0, 5.0, 1.0, 40.0)
    assert fa[0] and fb[0]
    assert np.array_equal(wa, wb)

    ecg_a = _synthetic_ecg(fs, duration_s=30.0, sign=-1.0)
    ecg_b = ecg_a.copy()
    ecg_b[wall_idx:] = -999.0
    eda_a = np.full(n, 0.5)
    eda_b = eda_a.copy()
    eda_b[wall_idx:] = -2.0

    past_a = extract_causal_ecg_eda_past(ecg_a, fs, eda_a, fs, "V", np.array([onset]))
    past_b = extract_causal_ecg_eda_past(ecg_b, fs, eda_b, fs, "V", np.array([onset]))
    for k in past_a:
        assert np.array_equal(past_a[k], past_b[k], equal_nan=True)


def test_sample_bdf_audit_runs():
    sample_bdfs = sorted(Path("sample").rglob("*.bdf"))
    assert sample_bdfs, "Expected official sample/ BDF files."
    summary = read_bdf_summary(sample_bdfs[0])
    assert summary["n_channels"] > 0 and summary["duration_s"] > 0
