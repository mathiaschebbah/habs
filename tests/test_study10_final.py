"""Unit and end-to-end anti-leakage tests for Study 10 (``decoding.study10_final``)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from decoding import common, study09_protocol, study10_final


def _make_synthetic_habs(
    n_people: int = 24, trials_per_person: int = 60, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate synthetic HABS multimodal trial table + REVE aux tables."""
    rng = np.random.default_rng(seed)
    rows = []
    tid = 0
    videos = [f"theme_{i % 15}_v{i}" for i in range(45)]

    for p_idx in range(n_people):
        pid = f"P{p_idx:03d}"
        cohort = (p_idx % 5) + 1
        trait_hr = rng.normal(70.0, 8.0)
        trait_sc = rng.normal(4.0, 1.5)
        for tr in range(trials_per_person):
            day = 1 if tr < trials_per_person // 2 else 2
            vid = videos[(p_idx * 7 + tr) % len(videos)]
            theme = vid.split("_v")[0]
            # Latent emotional state drives ratings + physiology
            latent_stress = rng.normal(0.0, 1.0)
            valence = float(
                np.clip(5.0 - 1.7 * latent_stress + rng.normal(0, 1.0), 1, 9)
            )
            arousal = float(
                np.clip(5.0 + 1.6 * latent_stress + rng.normal(0, 1.0), 1, 9)
            )
            if latent_stress > 0.8:
                emo = rng.choice(["fear", "disgust", "anger"])
            elif latent_stress < -0.7:
                emo = rng.choice(["joy", "relaxation"])
            else:
                emo = rng.choice(["neutral", "sadness"])
            row: dict = {
                "trial_id": tid,
                "participant_id": pid,
                "cohort": cohort,
                "recorded_day": day,
                "event_order": tr % (trials_per_person // 2),
                "onset_s": float((tr % (trials_per_person // 2)) * 20.0 + 10.0),
                "video": vid,
                "video_theme": theme,
                "rating_valence": valence,
                "rating_arousal": arousal,
                "self_reported_emotion_1": None if cohort == 1 else emo,
                # Core 4-phase body columns
                "ecg_hr_ref": trait_hr + rng.normal(0, 1.5),
                "ecg_hr_early": trait_hr - 1.2 * latent_stress + rng.normal(0, 1.5),
                "ecg_hr_late": trait_hr - 1.8 * latent_stress + rng.normal(0, 1.5),
                "ecg_hr_post": trait_hr - 0.5 * latent_stress + rng.normal(0, 1.5),
                "ecg_rr_ref": 0.85 + rng.normal(0, 0.03),
                "ecg_rr_early": 0.85 + 0.015 * latent_stress + rng.normal(0, 0.03),
                "ecg_rr_late": 0.85 + 0.020 * latent_stress + rng.normal(0, 0.03),
                "ecg_rr_post": 0.85 + 0.005 * latent_stress + rng.normal(0, 0.03),
                "sc_mean_ref": trait_sc + rng.normal(0, 0.2),
                "sc_mean_early": trait_sc + 0.1 * latent_stress + rng.normal(0, 0.2),
                "sc_mean_late": trait_sc + 0.35 * latent_stress + rng.normal(0, 0.2),
                "sc_mean_post": trait_sc + 0.25 * latent_stress + rng.normal(0, 0.2),
                "sc_rise_ref": rng.normal(0, 0.05),
                "sc_rise_early": 0.05 * latent_stress + rng.normal(0, 0.05),
                "sc_rise_late": 0.12 * latent_stress + rng.normal(0, 0.05),
                "sc_rise_post": 0.08 * latent_stress + rng.normal(0, 0.05),
                "sc_slope_ref": rng.normal(0, 0.05),
                "sc_slope_early": 0.03 * latent_stress + rng.normal(0, 0.05),
                "sc_slope_late": 0.06 * latent_stress + rng.normal(0, 0.05),
                "sc_slope_post": rng.normal(0, 0.05),
                "resp_amp_ref": 1.0 + rng.normal(0, 0.1),
                "resp_amp_early": 1.0 - 0.08 * latent_stress + rng.normal(0, 0.1),
                "resp_amp_late": 1.0 - 0.05 * latent_stress + rng.normal(0, 0.1),
                "resp_amp_post": 1.0 + rng.normal(0, 0.1),
                "resp_rate_ref": 15.0 + rng.normal(0, 1.0),
                "resp_rate_early": 15.0 + rng.normal(0, 1.0),
                "resp_rate_late": 15.0 + rng.normal(0, 1.0),
                "resp_rate_post": 15.0 + rng.normal(0, 1.0),
                "resp_slope_ref": rng.normal(0, 0.1),
                "resp_slope_early": rng.normal(0, 0.1),
                "resp_slope_late": rng.normal(0, 0.1),
                "resp_slope_post": rng.normal(0, 0.1),
                "scr_rise": max(0.0, 0.1 + 0.08 * latent_stress + rng.normal(0, 0.05)),
                # Study-09 causal forecasting columns
                "b05_hr_early": -1.0 * latent_stress + rng.normal(0, 1.0),
                "b05_hr_slope": -0.3 * latent_stress + rng.normal(0, 0.5),
                "b05_sc_early": 0.1 * latent_stress + rng.normal(0, 0.2),
                "tgt_ecg_hr_chg_5_10": -1.5 * latent_stress + rng.normal(0, 1.2),
                # Oculometry
                "oc_pupil_early_chg": 0.05 * latent_stress + rng.normal(0, 0.15),
                "oc_pupil_late_chg": 0.06 * latent_stress + rng.normal(0, 0.15),
            }
            # ECG morphology columns required by revision._build_phase_features
            for m in (
                "pr_pk",
                "pr_on",
                "pr_seg",
                "qrs",
                "qt",
                "qtc_baz",
                "qtc_frid",
                "rt_pk",
                "t_amp",
            ):
                base_v = rng.normal(0, 0.05)
                for ph in ("ref", "early", "late", "post", "vid"):
                    if m == "qtc_frid" and ph == "vid":
                        continue
                    row[f"ecg_{m}_{ph}"] = base_v + rng.normal(0, 0.02)

            # 4-phase EEG regional power
            for r in common.REGIONS:
                for b in common.BANDS:
                    b0 = rng.normal(0, 0.2)
                    sig_eff = (
                        -0.25 * latent_stress
                        if (r == "occipital" and b == "alpha")
                        else (
                            0.2 * latent_stress
                            if b == "delta"
                            else 0.05 * latent_stress
                        )
                    )
                    row[f"eeg_ep_ref_{r}_{b}"] = b0
                    row[f"eeg_ep_early_{r}_{b}"] = b0 + sig_eff + rng.normal(0, 0.2)
                    row[f"eeg_ep_late_{r}_{b}"] = (
                        b0 + 0.8 * sig_eff + rng.normal(0, 0.2)
                    )
                    row[f"eeg_ep_post_{r}_{b}"] = (
                        b0 + 0.2 * sig_eff + rng.normal(0, 0.2)
                    )

            # EEG spatial & temporal variation columns
            for fam in (
                "sp_std",
                "sp_iqr",
                "sp_ent",
                "reg_std",
                "ap_grad",
                "faa",
                "temp_asym",
            ):
                for b in common.BANDS:
                    b0 = rng.normal(0, 0.1)
                    for ph in ("ref", "early", "late", "post"):
                        row[f"eeg_{fam}_{ph}_{b}"] = b0 + rng.normal(0, 0.1)
            for b in common.BANDS:
                row[f"eeg_tvar_ref_{b}"] = rng.normal(0, 0.1)
                row[f"eeg_tvar_early_{b}"] = rng.normal(0, 0.1)
                row[f"eeg_tvar_late_{b}"] = rng.normal(0, 0.1)
                row[f"eeg_tslope_early_{b}"] = rng.normal(0, 0.1)
                row[f"eeg_tslope_late_{b}"] = rng.normal(0, 0.1)
                row[f"eeg_maxjump_early_{b}"] = rng.normal(0, 0.1)
                row[f"eeg_maxjump_late_{b}"] = rng.normal(0, 0.1)

            rows.append(row)
            tid += 1

    df = pd.DataFrame(rows)
    df = common.assign_splits(df, study09_protocol.STUDY09_BASE_SALT)
    # Ensure all rows belong to dev in small synthetic datasets when needed,
    # keeping realistic dev/test split structure if large enough
    if (df["split"] == "dev").sum() < 800:
        df["split"] = "dev"
        df["person_fold"] = study09_protocol.hash_buckets(
            df["participant_id"], "syn-p", 3
        )
        df["theme_fold"] = study09_protocol.hash_buckets(df["theme"], "syn-t", 3)

    # Synthetic REVE embeddings correlated with latent stress (valence - arousal)
    n = len(df)
    lat = (df["rating_arousal"] - df["rating_valence"]).to_numpy(float) / 4.0
    d_reve = 16
    w_lat = rng.normal(0, 0.4, size=(1, d_reve))
    x_vid10 = lat[:, None] * w_lat + rng.normal(0, 1.0, size=(n, d_reve))
    x_e5 = 0.8 * lat[:, None] * w_lat + rng.normal(0, 1.0, size=(n, d_reve))

    reve_pri = pd.DataFrame(
        {"trial_id": df["trial_id"].to_numpy()}
        | {
            f"reve_cs_ctx8_early_{i}": x_e5[:, i].astype(np.float32)
            for i in range(d_reve)
        }
        | {
            f"reve_cs_ctx8_pool_{i}": x_e5[:, i].astype(np.float32)
            for i in range(d_reve)
        }
        | {
            f"reve_cs_ctx8_diff_{i}": (0.5 * x_e5[:, i]).astype(np.float32)
            for i in range(d_reve)
        }
    )
    reve_ext = pd.DataFrame(
        {"trial_id": df["trial_id"].to_numpy()}
        | {
            f"reve_cs_vid10_{i}": x_vid10[:, i].astype(np.float32)
            for i in range(d_reve)
        }
    )
    return df, {"reve_primary": reve_pri, "reve_extra": reve_ext}


def test_coral_aligns_mean_and_covariance_and_is_row_independent():
    """CORAL matches source mean/cov and keeps eval rows strictly independent."""
    rng = np.random.default_rng(7)
    d = 6
    a_src = rng.normal(0, 1, (d, d))
    cov_src = a_src @ a_src.T + np.eye(d)
    z_src = rng.multivariate_normal(np.linspace(-1, 1, d), cov_src, size=500)

    a_tgt = rng.normal(0, 1.5, (d, d))
    cov_tgt = a_tgt @ a_tgt.T + 0.5 * np.eye(d)
    z_tgt_fit = rng.multivariate_normal(np.full(d, 3.0), cov_tgt, size=400)
    z_tgt_eval = rng.multivariate_normal(np.full(d, 3.0), cov_tgt, size=100)

    aligned_fit = study10_final.coral_align_target(
        z_src, z_tgt_fit, z_tgt_fit, reg=1e-4
    )
    assert np.allclose(aligned_fit.mean(axis=0), z_src.mean(axis=0), atol=1e-3)
    assert np.allclose(
        np.cov(aligned_fit, rowvar=False), np.cov(z_src, rowvar=False), atol=1e-2
    )

    # Perturbing eval row 1 must NEVER change aligned eval row 0!
    a1 = study10_final.coral_align_target(z_src, z_tgt_fit, z_tgt_eval, reg=1e-2)
    z_mod = z_tgt_eval.copy()
    z_mod[1] += 999.0
    a2 = study10_final.coral_align_target(z_src, z_tgt_fit, z_mod, reg=1e-2)
    assert np.allclose(a1[0], a2[0], atol=1e-12)


def test_fast_stress_solver_is_cold_start_single_trial_invariant():
    """Fast nested solver is strictly invariant to batch size = 1 evaluation."""
    df, _ = _make_synthetic_habs(n_people=12, trials_per_person=30, seed=1)
    x = study10_final._build_s07_feature_df(df).to_numpy(np.float32)
    tr = df.iloc[:240]
    res_batch = study10_final._fit_predict_stress_fast(
        x[:240], tr, x[240:270], max_pca=16, inner_k=3
    )
    res_single = study10_final._fit_predict_stress_fast(
        x[:240], tr, x[240:241], max_pca=16, inner_k=3
    )
    assert np.allclose(res_batch["stress_va"][0], res_single["stress_va"][0], atol=1e-5)
    assert np.allclose(
        res_batch["stress_logreg"][0], res_single["stress_logreg"][0], atol=1e-5
    )


def test_clinical_aggregation_auc_increases_monotonically_with_k():
    """Aggregating K=1..10 independent noisy episodes raises within-person AUC."""
    rng = np.random.default_rng(123)
    n_people = 30
    rows = []
    scores = []
    pos_list = []
    for p in range(n_people):
        for _ in range(20):
            rows.append({"participant_id": f"P{p}"})
            scores.append(rng.normal(0.5, 1.0))
            pos_list.append(True)
        for _ in range(20):
            rows.append({"participant_id": f"P{p}"})
            scores.append(rng.normal(0.0, 1.0))
            pos_list.append(False)
    df = pd.DataFrame(rows)
    pos = np.array(pos_list, dtype=bool)
    res = study10_final._simulate_clinical_aggregation(
        df, np.array(scores), pos, ~pos, n_rep=100, draws=100
    )
    aucs = [res["by_k"][str(k)]["auc"]["mean"] for k in study10_final.CLINICAL_K_GRID]
    assert aucs[0] > 0.60
    assert aucs[-1] > 0.80
    assert aucs[-1] > aucs[0] + 0.15
    assert all(aucs[i] <= aucs[i + 1] + 0.01 for i in range(len(aucs) - 1))
    assert res["by_k"]["3"]["disjoint_without_replacement_auc"]["mean"] > aucs[0]


def test_study10_end_to_end_smoke():
    """Fast end-to-end smoke test of all 6 Study-10 modules on synthetic HABS."""
    df, aux = _make_synthetic_habs(n_people=15, trials_per_person=58, seed=99)
    # Keep dev split large enough for fast CI execution (<10s)
    df["split"] = "dev"
    df["person_fold"] = study09_protocol.hash_buckets(
        df["participant_id"], "syn-p-fast", 3
    )
    df["theme_fold"] = study09_protocol.hash_buckets(df["theme"], "syn-t-fast", 3)
    res = study10_final.run_all(df, aux, draws=40)
    # Must serialize cleanly to valid JSON
    text = json.dumps(res)
    parsed = json.loads(text)

    for key in (
        "module10a_generalization_ablation",
        "module10b_longitudinal_habituation",
        "module10c_coral_kshot_loco",
        "module10d_coherence_bridge",
        "module10e_clinical_monitoring",
        "module10f_within_clip_emotions",
    ):
        assert key in parsed

    # Check key numerical fields are populated and reasonable on synthetic signal
    r4_auc = parsed["module10a_generalization_ablation"]["generalization_ladder"][
        "S07_enriched"
    ]["R4_double_unseen_cold_start"]["stress_ridge_va"]["stress"]["auc"]
    assert r4_auc is not None and r4_auc > 0.60

    k0 = parsed["module10c_coral_kshot_loco"]["kshot_chronological_dev"][
        "S07_enriched"
    ]["k_curve"]["0"]["cold_start"]["auc"]
    k40 = parsed["module10c_coral_kshot_loco"]["kshot_chronological_dev"][
        "S07_enriched"
    ]["k_curve"]["40"]["sup_eb_map_shrunk"]["auc"]
    assert k0 is not None and k40 is not None

    pca_var = parsed["module10d_coherence_bridge"]["coherence_all_314"][
        "pca_within_person_total"
    ]["explained_variance_ratio"]
    assert len(pca_var) >= 5 and abs(sum(pca_var) - 1.0) < 1e-2

    k1 = parsed["module10e_clinical_monitoring"]["stress_standard_vs_nonstress"][
        "S07_enriched"
    ]["by_k"]["1"]["auc"]["mean"]
    k10 = parsed["module10e_clinical_monitoring"]["stress_standard_vs_nonstress"][
        "S07_enriched"
    ]["by_k"]["10"]["auc"]["mean"]
    assert k10 > k1

    f_vs_d = parsed["module10f_within_clip_emotions"]["fear_vs_disgust_direct"]
    assert "eeg_global_gamma" in f_vs_d and "eda_sc" in f_vs_d
