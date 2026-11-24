"""Causal-wall, oculometry alignment, and history-scaling tests for Study 09."""

from pathlib import Path

import numpy as np
import pandas as pd

from decoding import study09_extract


def test_hard_causal_wall_bit_for_bit_at_5s():
    """Mutating EEG/ECG/EDA/Resp at t >= onset + 5.0 s leaves past outputs identical."""
    rng = np.random.default_rng(42)
    fs = 1000.0
    dur_s = 40.0
    n = int(dur_s * fs)
    t = np.arange(n) / fs

    # Synthetic signals: ECG R-spikes every 0.85 s, EDA, 0.25 Hz resp, EEG
    ecg = rng.normal(0, 0.05, n)
    for bt in np.arange(0.4, dur_s - 0.5, 0.85):
        idx = int(bt * fs)
        ecg[idx - 10 : idx + 10] += 2.0 * np.exp(
            -0.5 * (np.arange(-10, 10) / 3.0) ** 2
        )
    # EDA raw voltage in uV inside calibration table [-2.14, 2.0] V -> 0.2 V = 2e5 uV
    eda = 2e5 + 3e4 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 1e3, n)
    resp = np.sin(2 * np.pi * 0.25 * t) + 0.1 * rng.normal(0, 1, n)
    eeg = 20.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 5.0, n)

    onsets = np.array([16.0])
    cut = round((onsets[0] + 5.0) * fs)

    # Mutated copies with catastrophic perturbation strictly at t >= onset + 5.0 s
    ecg_m, eda_m, resp_m, eeg_m = ecg.copy(), eda.copy(), resp.copy(), eeg.copy()
    ecg_m[cut:] += 1e6
    eda_m[cut:] += 1e6
    resp_m[cut:] += 1e6
    eeg_m[cut:] += 1e6

    # 1. Causal EEG window [-3.2, +5.0) s
    w0, f0 = study09_extract.causal_bandpass_windows(
        eeg, fs, onsets, -3.2, 5.0, 0.5, 45.0, order=4, warmup_s=0.5
    )
    w1, f1 = study09_extract.causal_bandpass_windows(
        eeg_m, fs, onsets, -3.2, 5.0, 0.5, 45.0, order=4, warmup_s=0.5
    )
    assert f0[0] and f1[0]
    assert np.array_equal(w0, w1)

    # 2. Causal Body past [-3, 5) s
    b0, r0 = study09_extract.extract_causal_body_past(
        ecg, fs, eda, fs, "uV", resp, fs, onsets
    )
    b1, r1 = study09_extract.extract_causal_body_past(
        ecg_m, fs, eda_m, fs, "uV", resp_m, fs, onsets
    )
    assert np.isfinite(b0["b05_ecg_hr_early"][0])
    assert b0["b05_sc_valid"][0] == 1.0
    assert b0["b05_resp_valid"][0] == 1.0
    for k in b0:
        assert np.array_equal(b0[k], b1[k], equal_nan=True), f"Leak detected in {k}"
    for k in r0:
        assert np.array_equal(r0[k], r1[k], equal_nan=True), f"Leak detected in ref {k}"

    # 3. Future targets finite on clean signals
    tgts = study09_extract.extract_future_body_targets(
        ecg, fs, eda, fs, "uV", resp, fs, onsets, r0
    )
    for k in (
        "tgt_ecg_hr_chg_5_10",
        "tgt_ecg_hr_chg_7_10",
        "tgt_eda_sc_chg_5_10",
        "tgt_eda_scr_new_5_10",
        "tgt_eda_sc_chg_6_9",
        "tgt_resp_amp_chg_5_10",
        "tgt_resp_amp_chg_7_10",
    ):
        assert np.isfinite(tgts[k][0]), f"Target {k} should be finite"


def test_sample_bdf_oculometry_alignment_and_zero_future_history():
    """Verify oculometry reader on real P001 sample BDF and causal history stats."""
    sample_bdf = Path(
        "sample/CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf"
    )
    if sample_bdf.is_file():
        onsets = np.array([20.0, 40.0, 60.0])
        feats, audit = study09_extract.extract_oculometry_recording(sample_bdf, onsets)
        assert audit["has_record_map"] is True
        assert audit["has_oculo_stream"] is True
        assert feats["oc_valid_early"].mean() == 1.0
        assert np.nanmedian(feats["oc_binoc_r"]) > 0.85
        assert 2.0 <= float(np.nanmedian(feats["oc_pupil_ref"])) <= 6.0

    # Causal history stats must be strictly invariant to future trials
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "participant_id": ["P1"] * 6 + ["P2"] * 6,
            "recorded_day": [1, 1, 1, 2, 2, 2] * 2,
            "event_order": [0, 1, 2, 0, 1, 2] * 2,
            "onset_s": [10.0, 30.0, 50.0, 10.0, 30.0, 50.0] * 2,
        }
    )
    m = rng.normal(0, 2, (12, 8)).astype(np.float32)
    v = rng.uniform(1, 4, (12, 8)).astype(np.float32)
    ok = np.ones(12, dtype=bool)
    mu0, sd0 = study09_extract.compute_causal_history_stats(df, m, v, ok, {})

    m_mut = m.copy()
    v_mut = v.copy()
    day2 = (df["recorded_day"] == 2).to_numpy()
    m_mut[day2] += 1000.0
    v_mut[day2] += 1000.0
    mu1, sd1 = study09_extract.compute_causal_history_stats(df, m_mut, v_mut, ok, {})
    assert np.array_equal(mu0[~day2], mu1[~day2])
    assert np.array_equal(sd0[~day2], sd1[~day2])
