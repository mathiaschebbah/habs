"""Leakage, invariance, and statistical engine tests for Study 09 protocol."""

import numpy as np
import pandas as pd

from decoding import common, ssl_eval, study09_protocol


def _make_dev_frame(
    n_people: int = 36, n_trials: int = 16, seed: int = 11
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_people):
        for t in range(n_trials):
            rows.append(
                {
                    "trial_id": f"P{p:03d}-{t:02d}",
                    "participant_id": f"P{p:03d}",
                    "recorded_day": 1 if t < n_trials // 2 else 2,
                    "event_order": t,
                    "onset_s": float(30 + t * 25),
                    "video": f"theme{t % 12}_clip{t}.mp4",
                    "video_theme": f"theme{t % 12}",
                    "rating_valence": float(rng.uniform(1.0, 9.0)),
                    "rating_arousal": float(rng.uniform(1.0, 9.0)),
                }
            )
    return common.assign_splits(pd.DataFrame(rows), study09_protocol.STUDY09_BASE_SALT)


def test_cold_start_single_trial_invariance():
    """Cold-start models give identical outputs for batch_size=1 vs batch_size=N."""
    rng = np.random.default_rng(0)
    frame = _make_dev_frame(n_people=24, n_trials=14)
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    pf, tf = study09_protocol.outer_blocked_folds(dev, k=3)
    tr_m = (pf != 0) & (tf != 0)
    ev_m = (pf == 0) & (tf == 0)

    x_body = rng.normal(0, 1, (len(dev), 12)).astype(np.float32)
    x_eeg = rng.normal(0, 1, (len(dev), 24)).astype(np.float32)
    y = 0.6 * x_body[:, 0] + 0.4 * x_eeg[:, 0] + rng.normal(0, 0.5, len(dev))

    # 1. ssl_eval._fit_linear_probes (fixed eval-batch SD leak!)
    _, sv_batch, pv_batch, pa_batch = ssl_eval._fit_linear_probes(
        x_body[tr_m], dev.iloc[tr_m], x_body[ev_m], max_pca=8
    )
    ev_idx = np.where(ev_m)[0]
    _, sv_one, pv_one, pa_one = ssl_eval._fit_linear_probes(
        x_body[tr_m], dev.iloc[tr_m], x_body[ev_idx[:1]], max_pca=8
    )
    assert np.allclose(sv_batch[0], sv_one[0], atol=1e-5)
    assert np.allclose(pv_batch[0], pv_one[0], atol=1e-5)
    assert np.allclose(pa_batch[0], pa_one[0], atol=1e-5)

    # 2. study09_protocol nested M0 / M2 residual & dual ridge
    m0_b, m2_b, _ = study09_protocol.fit_predict_m2_residual(
        x_body[tr_m],
        x_eeg[tr_m],
        y[tr_m],
        dev.iloc[tr_m],
        x_body[ev_m],
        x_eeg[ev_m],
        eeg_pca_grid=(8,),
    )
    m0_1, m2_1, _ = study09_protocol.fit_predict_m2_residual(
        x_body[tr_m],
        x_eeg[tr_m],
        y[tr_m],
        dev.iloc[tr_m],
        x_body[ev_idx[:1]],
        x_eeg[ev_idx[:1]],
        eeg_pca_grid=(8,),
    )
    assert np.allclose(m0_b[0], m0_1[0], atol=1e-5)
    assert np.allclose(m2_b[0], m2_1[0], atol=1e-5)

    # 3. study09_protocol nested affect/stress
    aff_b = study09_protocol.fit_predict_nested_affect_stress(
        x_body[tr_m], dev.iloc[tr_m], x_body[ev_m], max_pca=8
    )
    aff_1 = study09_protocol.fit_predict_nested_affect_stress(
        x_body[tr_m], dev.iloc[tr_m], x_body[ev_idx[:1]], max_pca=8
    )
    assert np.allclose(aff_b["stress_va"][0], aff_1["stress_va"][0], atol=1e-5)
    assert np.allclose(aff_b["stress_logreg"][0], aff_1["stress_logreg"][0], atol=1e-5)


def test_causal_online_zero_future_trial_leakage():
    """Mutating future trials j > i leaves causal_person_center(trial i) unchanged."""
    rng = np.random.default_rng(7)
    frame = _make_dev_frame(n_people=6, n_trials=12)
    x = rng.normal(0, 1, (len(frame), 8)).astype(np.float32)
    mu0 = np.zeros(8)

    c_orig = ssl_eval.causal_person_center(
        x, frame["participant_id"].to_numpy(), frame, prior_mean=mu0, shrinkage=4.0
    )

    # Corrupt the last 6 trials of every participant massively
    x_mut = x.copy()
    is_late = (frame["event_order"] >= 6).to_numpy()
    x_mut[is_late] += 9999.0
    c_mut = ssl_eval.causal_person_center(
        x_mut, frame["participant_id"].to_numpy(), frame, prior_mean=mu0, shrinkage=4.0
    )

    is_early = ~is_late
    assert np.allclose(c_orig[is_early], c_mut[is_early], atol=1e-6)


def test_fold_isolated_ssl_eval_blocks_cross_split_leak():
    """Mutating test_new_video rows must not change dev OOF predictions."""
    rng = np.random.default_rng(19)
    frame = _make_dev_frame(n_people=30, n_trials=16)
    x = rng.normal(0, 1, (len(frame), 10)).astype(np.float32)

    _, preds_orig = ssl_eval.probe_representation_all_regimes(
        frame,
        x,
        normalization_protocol="fold_isolated_session",
        eval_sealed=True,
        draws=20,
    )

    x_mut = x.copy()
    nv_mask = (frame["split"] == "test_new_video").to_numpy()
    assert nv_mask.any()
    x_mut[nv_mask] += 5000.0
    _, preds_mut = ssl_eval.probe_representation_all_regimes(
        frame,
        x_mut,
        normalization_protocol="fold_isolated_session",
        eval_sealed=True,
        draws=20,
    )
    assert np.allclose(
        preds_orig["dev_ridge_va"].to_numpy(),
        preds_mut["dev_ridge_va"].to_numpy(),
        atol=1e-5,
    )


def test_m2_minus_m0_detects_real_eeg_gain_and_rejects_null():
    """Nested M2 - M0 recovers real residual EEG signal and stays <=0 on noise."""
    rng = np.random.default_rng(101)
    frame = _make_dev_frame(n_people=36, n_trials=16)
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    pf, tf = study09_protocol.outer_blocked_folds(dev, k=3)
    n = len(dev)

    x_body = rng.normal(0, 1, (n, 8)).astype(np.float32)
    x_eeg_real = rng.normal(0, 1, (n, 16)).astype(np.float32)
    x_eeg_noise = rng.normal(0, 1, (n, 16)).astype(np.float32)
    y = (
        0.7 * x_body[:, 0]
        - 0.3 * x_body[:, 1]
        + 0.55 * x_eeg_real[:, 0]
        - 0.35 * x_eeg_real[:, 1]
        + rng.normal(0, 0.4, n)
    )

    oof_m0 = np.full(n, np.nan)
    oof_m2_real = np.full(n, np.nan)
    oof_m2_null = np.full(n, np.nan)
    for p in range(3):
        for t in range(3):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            p0, p2, _ = study09_protocol.fit_predict_m2_residual(
                x_body[tr_m],
                x_eeg_real[tr_m],
                y[tr_m],
                dev.iloc[tr_m],
                x_body[ev_m],
                x_eeg_real[ev_m],
                eeg_pca_grid=(None,),
            )
            _, p2_n, _ = study09_protocol.fit_predict_m2_residual(
                x_body[tr_m],
                x_eeg_noise[tr_m],
                y[tr_m],
                dev.iloc[tr_m],
                x_body[ev_m],
                x_eeg_noise[ev_m],
                eeg_pca_grid=(None,),
            )
            oof_m0[ev_m] = p0
            oof_m2_real[ev_m] = p2
            oof_m2_null[ev_m] = p2_n

    _, pp0 = study09_protocol.summarize_forecast(dev, y, oof_m0, draws=100)
    _, pp2 = study09_protocol.summarize_forecast(dev, y, oof_m2_real, draws=100)
    _, pp2n = study09_protocol.summarize_forecast(dev, y, oof_m2_null, draws=100)

    comp_real = study09_protocol.paired_forecast_comparison(pp2, pp0, draws=100)
    comp_null = study09_protocol.paired_forecast_comparison(pp2n, pp0, draws=100)
    assert comp_real["delta_r"]["ci95"][0] > 0.10
    assert comp_real["delta_r2_within"]["ci95"][0] > 0.10
    assert comp_real["delta_r"]["frac_improved"] > 0.80
    assert comp_null["delta_r"]["ci95"][0] <= 0.02


def test_trait_state_factorization_and_complementarity_gate():
    """Verify shrinkage denoising, trait preservation, and complementarity gate."""
    rng = np.random.default_rng(55)
    frame = _make_dev_frame(n_people=32, n_trials=14)
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    n, d = len(dev), 12
    pids = dev["participant_id"].astype("category").cat.codes.to_numpy()
    trait_vec = rng.normal(0, 1.5, (pids.max() + 1, d))[pids]
    state_vec = rng.normal(0, 1.0, (n, d))

    # Noisy 3-s baseline: same trait + heavy noise; stim = 0.5*trait + state
    x_base = trait_vec + rng.normal(0, 1.8, (n, d))
    x_stim = 0.5 * trait_vec + state_vec + rng.normal(0, 0.3, (n, d))

    pf, tf = study09_protocol.outer_blocked_folds(dev, k=3)
    tr_m = (pf != 0) & (tf != 0)
    ev_m = (pf == 0) & (tf == 0)

    _, s_blind, _ = study09_protocol.fit_transform_trait_state(
        x_stim[tr_m],
        x_base[tr_m],
        dev.iloc[tr_m],
        x_stim[ev_m],
        x_base[ev_m],
        mode="blind_ref_sub",
    )
    _, s_shrink, info_s = study09_protocol.fit_transform_trait_state(
        x_stim[tr_m],
        x_base[tr_m],
        dev.iloc[tr_m],
        x_stim[ev_m],
        x_base[ev_m],
        mode="state_only_shrinkage",
    )
    _, s_fact, info_f = study09_protocol.fit_transform_trait_state(
        x_stim[tr_m],
        x_base[tr_m],
        dev.iloc[tr_m],
        x_stim[ev_m],
        x_base[ev_m],
        mode="factored_trait_plus_state",
        n_trait_dims=4,
    )
    # Shrinkage weight is < 1 and yields lower noise variance than blind subtraction
    assert 0.0 < info_s["mean_shrink_w"] < 0.7
    assert np.var(s_shrink) < np.var(s_blind)
    assert s_fact.shape[1] == d + 4
    assert info_f["mean_bw_ratio"] > 0.1

    # Complementarity gate: redundant models -> False; complementary models -> True
    y_reg = state_vec[:, 0] + state_vec[:, 1] + rng.normal(0, 0.2, n)
    p_a = state_vec[:, 0] + rng.normal(0, 0.1, n)
    p_a_dup = p_a + rng.normal(0, 0.05, n)
    p_b = state_vec[:, 1] + rng.normal(0, 0.1, n)

    gate_red = study09_protocol.assess_complementarity(
        dev, y_reg, {"m1": p_a, "m2": p_a_dup}, task="regression", draws=100
    )
    assert gate_red["gate_passed"] is False

    gate_comp = study09_protocol.assess_complementarity(
        dev, y_reg, {"m1": p_a, "m2": p_b}, task="regression", draws=100
    )
    assert gate_comp["gate_passed"] is True
    st = study09_protocol.run_nested_oof_stacker(
        dev, y_reg, {"m1": p_a, "m2": p_b}, task="regression", k=3, draws=100
    )
    assert (
        st["metrics"]["r"]["mean"] > gate_comp["individual_scores"]["m1"]["mean"] + 0.10
    )
