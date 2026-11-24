"""End-to-end synthetic smoke test for Study-09 CPU analysis modules 1-7."""

import numpy as np
import pandas as pd

from decoding import study09_analysis, study09_protocol
from tests.test_decoding_revision import sim_folder  # noqa: F401


def test_study09_analysis_all_modules(sim_folder, monkeypatch):  # noqa: F811
    """Verify all 7 Study-09 CPU analysis modules run cleanly on synthetic parquets."""
    monkeypatch.setattr(study09_protocol, "DRAWS", 30)
    rng = np.random.default_rng(123)

    trials = pd.read_parquet(sim_folder / "trials.parquet")
    n = len(trials)
    tids = trials["trial_id"].to_numpy()

    # Create synthetic study09_causal.parquet
    c_cols: dict[str, object] = {"trial_id": tids}
    b0 = rng.normal(0, 1, n).astype(np.float32)
    e0 = rng.normal(0, 1, n).astype(np.float32)
    for k in range(6):
        c_cols[f"b05_feat_{k}"] = (
            b0 if k == 0 else rng.normal(0, 1, n).astype(np.float32)
        )
    c_cols["b05_ecg_qrs_valid_frac"] = np.full(n, 0.99, dtype=np.float32)
    c_cols["b05_sc_valid"] = np.ones(n, dtype=np.float32)
    for k in range(6):
        c_cols[f"e05_eachg_f_{k}"] = (
            e0 if k == 0 else rng.normal(0, 1, n).astype(np.float32)
        )
    for ak in (
        "art_cur_fp_ptp_early",
        "art_cur_rim_emg52_80_early",
        "art_ica_diff_rms_early",
    ):
        c_cols[ak] = rng.normal(0, 1, n).astype(np.float32)
    c_cols["art_bad_ch_count_early"] = np.zeros(n, dtype=np.float32)
    for pk in (
        "prev_ecg_hr_late",
        "prev_eda_sc_late",
        "prev_resp_amp_late",
        "trial_gap_s",
    ):
        c_cols[pk] = rng.normal(0, 1, n).astype(np.float32)

    for tgt in study09_analysis.EXTENDED_TARGETS:
        c_cols[tgt] = (0.6 * b0 + 0.35 * e0 + rng.normal(0, 0.5, n)).astype(np.float32)
    pd.DataFrame(c_cols).to_parquet(sim_folder / "study09_causal.parquet")

    # Synthetic study09_oculo.parquet (passing H4 gate)
    oc_cols: dict[str, object] = {
        "trial_id": tids,
        "oc_valid_early": np.ones(n, dtype=np.float32),
        "oc_valid_late": np.ones(n, dtype=np.float32),
        "oc_binoc_r": np.full(n, 0.92, dtype=np.float32),
        "oc_pupil_early_chg": rng.normal(-0.35, 0.15, n).astype(np.float32),
    }
    for k in (
        "oc_pupil_early_slope",
        "oc_pupil_early_std",
        "oc_dropout_early",
        "oc_gaze_disp_early",
        "oc_gaze_speed_early",
        "oc_pupil_late_chg",
        "oc_dropout_late",
        "oc_gaze_disp_late",
        "oc_gaze_speed_late",
    ):
        oc_cols[k] = rng.normal(0, 1, n).astype(np.float32)
    pd.DataFrame(oc_cols).to_parquet(sim_folder / "study09_oculo.parquet")

    # Synthetic REVE primary, extra, Study-08 fm_reve, and LoRA preds
    d = 12
    rp_cols: dict[str, object] = {
        "trial_id": tids,
        "reve_valid": np.ones(n, dtype=bool),
    }
    re_cols: dict[str, object] = {
        "trial_id": tids,
        "reve_valid": np.ones(n, dtype=bool),
    }
    fm_cols: dict[str, object] = {
        "trial_id": tids,
        "reve_valid": np.ones(n, dtype=bool),
    }
    for pref in (
        "reve_cs_pre",
        "reve_cs_early5",
        "reve_cs_ctx8_early",
        "reve_cs_ctx8_pool",
        "reve_cs_ctx8_diff",
    ):
        for j in range(d):
            rp_cols[f"{pref}_{j}"] = (e0 if j == 0 else rng.normal(0, 1, n)).astype(
                np.float16
            )
    for pref in (
        "reve_cs_interior_ctx8",
        "reve_cs_shuf_early5",
        "reve_hist_ctx8_early",
        "reve_cs_vid10",
        "reve_hist_vid10",
        "reve_rest_trait",
    ):
        for j in range(d):
            re_cols[f"{pref}_{j}"] = (e0 if j == 0 else rng.normal(0, 1, n)).astype(
                np.float16
            )
    for pref in ("reve_ref", "reve_early", "reve_late"):
        for j in range(d):
            fm_cols[f"{pref}_{j}"] = rng.normal(0, 1, n).astype(np.float32)
    pd.DataFrame(rp_cols).to_parquet(sim_folder / "study09_reve_primary.parquet")
    pd.DataFrame(re_cols).to_parquet(sim_folder / "study09_reve_extra.parquet")
    pd.DataFrame(fm_cols).to_parquet(sim_folder / "fm_reve.parquet")

    lora_cols: dict[str, object] = {"trial_id": tids}
    for tgt in study09_analysis.PRIMARY_TARGETS:
        lora_cols[f"reve_forecast_lora_ens_{tgt}"] = (0.6 * b0 + 0.4 * e0).astype(
            np.float32
        )
        lora_cols[f"reve_forecast_frozen_head_ens_{tgt}"] = (
            0.6 * b0 + 0.2 * e0
        ).astype(np.float32)
    for nm in (
        "reve_mt_frozen_rank",
        "reve_mt_lora_rank",
        "reve_mt_lora_norank",
        "reve_mt_lora_r4",
    ):
        for k in ("stress", "stress_va", "valence", "arousal"):
            lora_cols[f"{nm}_ens_{k}"] = rng.normal(0, 1, n).astype(np.float32)
    pd.DataFrame(lora_cols).to_parquet(sim_folder / "study09_lora_preds.parquet")

    frame, aux = study09_analysis.load_study09_tables(sim_folder)
    m1 = study09_analysis.module_audit_study07_08(frame, aux, draws=20)
    assert "normalization_audit_study07_enriched" in m1
    assert "bidirectional_and_stimulus_shortcut_audit" in m1

    m2, _ = study09_analysis.module_primary_forecasting(frame, aux, draws=20)
    assert "tgt_ecg_hr_chg_5_10" in m2["primary_cold_start_3x3"]
    assert "tgt_ecg_hr_chg_7_10" in m2["extended_delays_cold_start_3x3"]
    assert "B1_causal_online" in m2["protocol_comparison_primary"]
    assert "tgt_ecg_hr_chg_5_10" in m2["lora_forecast_integration"]

    m3 = study09_analysis.module_controls_and_negatives(frame, aux, draws=20)
    assert (
        "NC2_idiosyncratic_real_vs_vid_shuf"
        in m3["targets"]["tgt_ecg_hr_chg_5_10"]["paired_contrasts"]
    )

    m47 = study09_analysis.module_secondary_trait_oculo_stacking(frame, aux, draws=20)
    assert "reve_mt_lora_rank" in m47["module4_secondary_affect_and_lora"]
    assert (
        "factored_trait_plus_state"
        in m47["module5_trait_state_factorization"]["cold_start_reve_pre_base"]
    )
    assert (
        m47["module6_oculo_audit_and_baselines"]["dev_audit"]["h4_gate"]["gate_passed"]
        is True
    )
    assert "stress_complementarity" in m47["module7_complementarity_and_stacking"]
