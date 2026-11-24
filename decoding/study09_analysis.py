"""Study-09 CPU analysis suite: audit, M0/M1/M2 forecasting, controls & gates.

Runs on the ``decoding.yml`` CPU runner (``--study study09``) after GPU Job 1
(``study09_causal.parquet``, ``study09_oculo.parquet``,
``study09_reve_primary.parquet``, ``study09_reve_extra.parquet``) and GPU Job 2
(``study09_lora_preds.parquet``) complete.

All primary inference and model selection occur **exclusively inside ``dev``**
(``STUDY09_BASE_SALT = "habs-ssl-2026-11-24-b"``, 228 people x 66 themes) via
nested blocked ``person x theme`` cross-validation with fold-isolated
preprocessing. Study 08's ``test_new_both`` is explicitly marked EXPOSED and
used solely for a transparent retrospective leakage audit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decoding import common, revision, ssl_eval, study09_protocol
from decoding.stress import label

PRIMARY_TARGETS = (
    "tgt_ecg_hr_chg_5_10",
    "tgt_eda_sc_chg_5_10",
    "tgt_resp_amp_chg_5_10",
)
EXTENDED_TARGETS = (
    # Primary + secondary metrics at [5, 10) s
    "tgt_ecg_hr_chg_5_10",
    "tgt_ecg_rr_chg_5_10",
    "tgt_ecg_hr_inc_5_10",
    "tgt_eda_sc_chg_5_10",
    "tgt_eda_scr_new_5_10",
    "tgt_eda_sc_inc_5_10",
    "tgt_resp_amp_chg_5_10",
    "tgt_resp_rate_chg_5_10",
    "tgt_resp_amp_inc_5_10",
    # Prespecified physiological delay windows
    "tgt_ecg_hr_chg_5_8",
    "tgt_ecg_hr_chg_7_10",
    "tgt_ecg_hr_chg_10_12",
    "tgt_eda_sc_chg_6_9",
    "tgt_eda_sc_chg_7_10",
    "tgt_eda_sc_chg_10_12",
    "tgt_resp_amp_chg_5_8",
    "tgt_resp_amp_chg_7_10",
    "tgt_resp_amp_chg_10_12",
)
OCULO_EARLY_COLS = (
    "oc_pupil_early_chg",
    "oc_pupil_early_slope",
    "oc_pupil_early_std",
    "oc_dropout_early",
    "oc_gaze_disp_early",
    "oc_gaze_speed_early",
)
OCULO_FULL_COLS = (
    *OCULO_EARLY_COLS,
    "oc_pupil_late_chg",
    "oc_dropout_late",
    "oc_gaze_disp_late",
    "oc_gaze_speed_late",
)


def load_study09_tables(
    folder: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load base trial table plus all Study-07/08/09 private parquets aligned 1:1."""
    base = Path(folder or os.environ["TRIAL_DATA"])
    table = pd.read_parquet(base / "trials.parquet")
    if (base / "body.parquet").is_file():
        body = pd.read_parquet(base / "body.parquet")
        table = table.drop(columns=[c for c in body if c != "trial_id" and c in table])
        table = table.merge(body, on="trial_id", how="left", validate="1:1")
    for fname in (
        "timecourse.parquet",
        "rich.parquet",
        "study09_causal.parquet",
        "study09_oculo.parquet",
    ):
        p = base / fname
        if p.is_file():
            extra = pd.read_parquet(p)
            keep = ["trial_id"] + [c for c in extra.columns if c not in table.columns]
            table = table.merge(extra[keep], on="trial_id", how="left", validate="1:1")
    table = common.assign_splits(table, study09_protocol.STUDY09_BASE_SALT)

    aux: dict[str, pd.DataFrame] = {}
    for key, fname in (
        ("reve_primary", "study09_reve_primary.parquet"),
        ("reve_extra", "study09_reve_extra.parquet"),
        ("fm_reve_s08", "fm_reve.parquet"),
    ):
        p = base / fname
        if p.is_file():
            df_x = pd.read_parquet(p)
            aux[key] = table[["trial_id"]].merge(
                df_x, on="trial_id", how="left", validate="1:1"
            )

    # Merge split or combined LoRA prediction parquets seamlessly
    lora_parts = []
    for fname in (
        "study09_lora_preds.parquet",
        "study09_lora_forecast_preds.parquet",
        "study09_lora_mt_preds.parquet",
    ):
        p = base / fname
        if p.is_file():
            lora_parts.append(pd.read_parquet(p))
    if lora_parts:
        merged_l = table[["trial_id"]].copy()
        for df_l in lora_parts:
            keep = ["trial_id"] + [c for c in df_l.columns if c not in merged_l.columns]
            merged_l = merged_l.merge(
                df_l[keep], on="trial_id", how="left", validate="1:1"
            )
        aux["lora_preds"] = merged_l
    return table, aux


def _prefix_mat(df: pd.DataFrame | None, prefix: str) -> np.ndarray | None:
    if df is None:
        return None
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None
    return df[cols].to_numpy(dtype=np.float32)


def _body05_cols(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c.startswith("b05_"))


def _eeg05_hc_cols(frame: pd.DataFrame) -> list[str]:
    return sorted(
        c
        for c in frame.columns
        if c.startswith("e05_") and not c.startswith("e05_ref_")
    )


def _artefact_cols(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c.startswith("art_"))


def _tc1_cols(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in (
            "prev_ecg_hr_late",
            "prev_eda_sc_late",
            "prev_resp_amp_late",
            "trial_gap_s",
            "event_order",
            "onset_s",
        )
        if c in frame.columns
    ]


# ---------------------------------------------------------------------------
# MODULE 1: Quantitative Audit of Studies 07-08 Leaks (Req 1, 2, 3, 14)
# ---------------------------------------------------------------------------


def module_audit_study07_08(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = study09_protocol.DRAWS,
) -> dict:
    """Quantify transductive centering, bidirectional REVE, and seen-theme bias."""
    out: dict[str, Any] = {
        "exposed_split_notice": (
            "test_new_both of STUDY08_SALT='habs-ssl-2026-11-24-b' is EXPOSED "
            "from Study 08. Results on test_new_both/test_new_person/test_new_video "
            "below are strictly retrospective leakage audits, NOT independent "
            "validation."
        )
    }

    # 1A. Normalization audit on Study-07 enriched_full (global vs B2 vs B1 vs CS)
    if "ecg_hr_ref" in frame.columns and "eeg_ep_ref_frontal_alpha" in frame.columns:
        x_enr = revision._build_phase_features(
            frame,
            add_deltas=True,
            add_ecg_morph=True,
            add_eeg_spatial=True,
            add_eeg_tvar=True,
            person_center=False,
        ).to_numpy(np.float32)
        norm_audit = {}
        for proto in (
            "legacy_global",
            "fold_isolated_session",
            "causal_online",
            "cold_start",
        ):
            res, _ = ssl_eval.probe_representation_all_regimes(
                frame,
                x_enr,
                normalization_protocol=proto,
                eval_sealed=True,
                max_pca=256,
                draws=draws,
            )
            norm_audit[proto] = {
                "dev_ridge_va_auc": res["dev"]["stress_ridge_va"]["stress"],
                "dev_logreg_auc": res["dev"]["stress_logreg"]["stress"],
                "dev_affect_mean_rho": res["dev"]["affect"]["mean_rho"],
                "exposed_test_new_both_va": res["test_new_both"]["stress_ridge_va"][
                    "stress"
                ],
                "exposed_test_new_person_va": res["test_new_person"]["stress_ridge_va"][
                    "stress"
                ],
                "exposed_test_new_video_va": res["test_new_video"]["stress_ridge_va"][
                    "stress"
                ],
            }
        out["normalization_audit_study07_enriched"] = norm_audit

    # 1B. Bidirectional REVE (S08) vs Causal REVE (S09) & Seen-Theme Shortcut
    fm_s08 = aux.get("fm_reve_s08")
    reve_pri = aux.get("reve_primary")
    if fm_s08 is not None and reve_pri is not None:
        zr8 = _prefix_mat(fm_s08, "reve_ref_")
        ze8 = _prefix_mat(fm_s08, "reve_early_")
        x_s08_bidir = (ze8 - zr8) if (ze8 is not None and zr8 is not None) else None
        x_s09_causal = _prefix_mat(reve_pri, "reve_cs_ctx8_early_")
        x_s09_e5 = _prefix_mat(reve_pri, "reve_cs_early5_")

        bidir_audit: dict[str, Any] = {}
        dev_m = (frame["split"] == "dev").to_numpy()
        dev_df = frame[dev_m].reset_index(drop=True)
        pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
        t_both_m = (frame["split"] == "test_new_both").to_numpy()
        t_pers_m = (frame["split"] == "test_new_person").to_numpy()
        t_mix_m = t_both_m | t_pers_m

        target_pairs = (
            ("hr_5_10", "tgt_ecg_hr_chg_5_10"),
            ("sc_5_10", "tgt_eda_sc_chg_5_10"),
            ("resp_5_10", "tgt_resp_amp_chg_5_10"),
        )
        for label_x, x_mat in (
            ("s08_reve_bidirectional_13s", x_s08_bidir),
            ("s09_reve_causal_ctx8_strict", x_s09_causal),
            ("s09_reve_causal_early5_strict", x_s09_e5),
        ):
            if x_mat is None:
                continue
            bidir_audit[label_x] = {}
            for short_t, col_t in target_pairs:
                if col_t not in frame.columns:
                    continue
                y_all = frame[col_t].to_numpy(float)
                # (1) Strictly unseen people x unseen themes inside dev nested 3x3 CV
                oof_dev = np.full(len(dev_df), np.nan)
                x_dev = x_mat[dev_m]
                y_dev = y_all[dev_m]
                for p in range(3):
                    for t in range(3):
                        ev = (pf == p) & (tf == t)
                        tr = (pf != p) & (tf != t)
                        pr, _ = study09_protocol.fit_predict_nested_ridge(
                            x_dev[tr],
                            y_dev[tr],
                            dev_df.iloc[tr],
                            x_dev[ev],
                            pca_grid=(32, 64),
                            inner_k=3,
                        )
                        oof_dev[ev] = pr
                sm_dev, _ = study09_protocol.summarize_forecast(
                    dev_df, y_dev, oof_dev, draws=draws
                )

                # (2) Exposed audit: dev -> test_new_both (unseen) vs new_person (seen)
                pred_mix, _ = study09_protocol.fit_predict_nested_ridge(
                    x_dev, y_dev, dev_df, x_mat[t_mix_m], pca_grid=(32, 64), inner_k=3
                )
                df_mix = frame[t_mix_m].reset_index(drop=True)
                y_mix = y_all[t_mix_m]
                sm_mix, _ = study09_protocol.summarize_forecast(
                    df_mix, y_mix, pred_mix, draws=draws
                )
                m_nb = (df_mix["split"] == "test_new_both").to_numpy()
                m_np = (df_mix["split"] == "test_new_person").to_numpy()
                sm_nb, _ = study09_protocol.summarize_forecast(
                    df_mix[m_nb].reset_index(drop=True),
                    y_mix[m_nb],
                    pred_mix[m_nb],
                    draws=draws,
                )
                sm_np, _ = study09_protocol.summarize_forecast(
                    df_mix[m_np].reset_index(drop=True),
                    y_mix[m_np],
                    pred_mix[m_np],
                    draws=draws,
                )
                bidir_audit[label_x][short_t] = {
                    "dev_nested_unseen_both_r": sm_dev["r"],
                    "exposed_mixed_s08_style_r": sm_mix["r"],
                    "exposed_unseen_themes_new_both_r": sm_nb["r"],
                    "exposed_seen_themes_new_person_r": sm_np["r"],
                }
        out["bidirectional_and_stimulus_shortcut_audit"] = bidir_audit
    return out


# ---------------------------------------------------------------------------
# MODULE 2: Primary Question M0 vs M1 vs M2 & Prespecified Delays (Req 4-7, 15)
# ---------------------------------------------------------------------------


def _run_nested_forecast_suite(
    dev_df: pd.DataFrame,
    x_body: np.ndarray,
    x_eeg_reve: np.ndarray,
    x_eeg_hc: np.ndarray,
    y: np.ndarray,
    *,
    protocol: str = "cold_start",
    rest_eeg: np.ndarray | None = None,
    k_outer: int = 3,
    run_gbdt: bool = True,
    run_joint: bool = True,
    draws: int = study09_protocol.DRAWS,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run strictly nested outer k x k M0/M1/M2 comparison on one target."""
    n = len(dev_df)
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=k_outer)
    oofs: dict[str, np.ndarray] = {
        "M0_linear": np.full(n, np.nan),
        "M1_hc_eeg": np.full(n, np.nan),
        "M1_reve_frozen": np.full(n, np.nan),
        "M2_hc_resid": np.full(n, np.nan),
        "M2_reve_resid": np.full(n, np.nan),
    }
    if run_joint:
        oofs["M2_reve_joint"] = np.full(n, np.nan)
    if run_gbdt:
        oofs["M0_gbdt"] = np.full(n, np.nan)
        oofs["M2_gbdt_reve"] = np.full(n, np.nan)

    etas_reve, etas_hc = [], []
    for p in range(k_outer):
        for t in range(k_outer):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 40:
                continue
            df_tr, df_ev = dev_df.iloc[tr_m], dev_df.iloc[ev_m]
            y_tr = y[tr_m]

            # Apply protocol (A: cold_start vs B: rest_only/causal_online/session)
            xb_tr, xb_ev = study09_protocol.prepare_protocol_fold(
                x_body[tr_m],
                df_tr,
                x_body[ev_m],
                df_ev,
                protocol="cold_start" if protocol == "rest_only" else protocol,
            )
            xh_tr, xh_ev = study09_protocol.prepare_protocol_fold(
                x_eeg_hc[tr_m],
                df_tr,
                x_eeg_hc[ev_m],
                df_ev,
                protocol="cold_start" if protocol == "rest_only" else protocol,
            )
            xr_tr, xr_ev = study09_protocol.prepare_protocol_fold(
                x_eeg_reve[tr_m],
                df_tr,
                x_eeg_reve[ev_m],
                df_ev,
                protocol=protocol,
                rest_tr=rest_eeg[tr_m] if rest_eeg is not None else None,
                rest_ev=rest_eeg[ev_m] if rest_eeg is not None else None,
            )

            # M0 + M2_reve_resid
            p_m0, p_m2r, info_r = study09_protocol.fit_predict_m2_residual(
                xb_tr,
                xr_tr,
                y_tr,
                df_tr,
                xb_ev,
                xr_ev,
                eeg_pca_grid=(16, 32, 64),
                inner_k=3,
            )
            oofs["M0_linear"][ev_m] = p_m0
            oofs["M2_reve_resid"][ev_m] = p_m2r
            etas_reve.append(info_r["eta"])

            # M2_hc_resid
            _, p_m2h, info_h = study09_protocol.fit_predict_m2_residual(
                xb_tr,
                xh_tr,
                y_tr,
                df_tr,
                xb_ev,
                xh_ev,
                eeg_pca_grid=(16, 32, None),
                inner_k=3,
            )
            oofs["M2_hc_resid"][ev_m] = p_m2h
            etas_hc.append(info_h["eta"])

            # M1 (EEG alone)
            oofs["M1_reve_frozen"][ev_m], _ = study09_protocol.fit_predict_nested_ridge(
                xr_tr, y_tr, df_tr, xr_ev, pca_grid=(16, 32, 64), inner_k=3
            )
            oofs["M1_hc_eeg"][ev_m], _ = study09_protocol.fit_predict_nested_ridge(
                xh_tr, y_tr, df_tr, xh_ev, pca_grid=(16, 32, None), inner_k=3
            )

            # M2_reve_joint (dual-penalty Ridge)
            if run_joint:
                oofs["M2_reve_joint"][ev_m], _ = (
                    study09_protocol.fit_predict_m2_joint_dual_ridge(
                        xb_tr, xr_tr, y_tr, df_tr, xb_ev, xr_ev, eeg_pca=32, inner_k=3
                    )
                )

            # Nonlinear GBDT control: M0_gbdt vs M2_gbdt_reve
            if run_gbdt:
                oofs["M0_gbdt"][ev_m] = study09_protocol.fit_predict_gbdt_regressor(
                    xb_tr, y_tr, xb_ev, max_pca=None, seed=p * 10 + t
                )
                zr_tr, zr_ev = study09_protocol._preprocess_fit_transform(
                    xr_tr, xr_ev, max_pca=24
                )
                oofs["M2_gbdt_reve"][ev_m] = (
                    study09_protocol.fit_predict_gbdt_regressor(
                        np.hstack([xb_tr, zr_tr]),
                        y_tr,
                        np.hstack([xb_ev, zr_ev]),
                        max_pca=None,
                        seed=p * 10 + t,
                    )
                )

    summaries: dict[str, Any] = {
        "protocol": protocol,
        "k_outer": k_outer,
        "mean_eta_reve": round(float(np.mean(etas_reve)), 3) if etas_reve else None,
        "mean_eta_hc": round(float(np.mean(etas_hc)), 3) if etas_hc else None,
        "models": {},
        "paired_vs_m0_linear": {},
    }
    pp_map: dict[str, dict] = {}
    for m_name, pred in oofs.items():
        sm, pp = study09_protocol.summarize_forecast(dev_df, y, pred, draws=draws)
        summaries["models"][m_name] = sm
        pp_map[m_name] = pp

    for m_name in oofs:
        if m_name != "M0_linear":
            summaries["paired_vs_m0_linear"][m_name] = (
                study09_protocol.paired_forecast_comparison(
                    pp_map[m_name], pp_map["M0_linear"], draws=draws
                )
            )
    if run_gbdt:
        summaries["paired_m2_gbdt_vs_m0_gbdt"] = (
            study09_protocol.paired_forecast_comparison(
                pp_map["M2_gbdt_reve"], pp_map["M0_gbdt"], draws=draws
            )
        )
    return summaries, oofs


def module_primary_forecasting(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = study09_protocol.DRAWS,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Execute complete M0/M1/M2 forecasting benchmark strictly inside dev."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)

    bcols = _body05_cols(frame)
    ecols = _eeg05_hc_cols(frame)
    x_body = dev_df[bcols].to_numpy(np.float32)
    x_hc = dev_df[ecols].to_numpy(np.float32)

    reve_pri = aux["reve_primary"]
    reve_ext = aux["reve_extra"]
    x_reve_cs = np.hstack(
        [
            _prefix_mat(reve_pri, "reve_cs_ctx8_early_")[dev_m],
            _prefix_mat(reve_pri, "reve_cs_ctx8_pool_")[dev_m],
            _prefix_mat(reve_pri, "reve_cs_ctx8_diff_")[dev_m],
        ]
    )
    x_reve_hist = _prefix_mat(reve_ext, "reve_hist_ctx8_early_")[dev_m]
    x_reve_rest = _prefix_mat(reve_ext, "reve_rest_trait_")[dev_m]
    x_reve_e5 = _prefix_mat(reve_pri, "reve_cs_early5_")[dev_m]

    out: dict[str, Any] = {
        "n_dev_trials": int(len(dev_df)),
        "n_dev_people": int(dev_df["participant_id"].nunique()),
        "n_body05_features": len(bcols),
        "n_eeg05_hc_features": len(ecols),
        "primary_cold_start_3x3": {},
        "extended_delays_cold_start_3x3": {},
        "protocol_comparison_primary": {},
        "sensitivity_5x5_cold_start": {},
        "lora_forecast_integration": {},
    }
    saved_oofs: dict[str, dict[str, np.ndarray]] = {}

    # 1. Primary 3 targets under Cold-Start (3x3 outer + GBDT + Dual Joint + M2 Resid)
    pvals_primary: dict[str, float] = {}
    for tgt in PRIMARY_TARGETS:
        if tgt not in dev_df.columns:
            continue
        y = dev_df[tgt].to_numpy(float)
        res, oofs = _run_nested_forecast_suite(
            dev_df,
            x_body,
            x_reve_cs,
            x_hc,
            y,
            protocol="cold_start",
            k_outer=3,
            run_gbdt=True,
            run_joint=True,
            draws=draws,
        )
        out["primary_cold_start_3x3"][tgt] = res
        saved_oofs[tgt] = oofs
        pvals_primary[f"{tgt}|M2_reve_resid"] = res["paired_vs_m0_linear"][
            "M2_reve_resid"
        ]["delta_r"]["p_wilcoxon"]
        pvals_primary[f"{tgt}|M2_reve_joint"] = res["paired_vs_m0_linear"][
            "M2_reve_joint"
        ]["delta_r"]["p_wilcoxon"]
        pvals_primary[f"{tgt}|M2_hc_resid"] = res["paired_vs_m0_linear"]["M2_hc_resid"][
            "delta_r"
        ]["p_wilcoxon"]
        print(
            f"  Primary CS {tgt}: M0={res['models']['M0_linear']['r']['mean']}, "
            f"M1_reve={res['models']['M1_reve_frozen']['r']['mean']}, "
            f"M2_reve={res['models']['M2_reve_resid']['r']['mean']} "
            f"(d_r={res['paired_vs_m0_linear']['M2_reve_resid']['delta_r']['mean']}, "
            f"ci={res['paired_vs_m0_linear']['M2_reve_resid']['delta_r']['ci95']})",
            flush=True,
        )
    out["primary_bh_fdr_q"] = study09_protocol.bh_fdr(pvals_primary)

    # 2. All extended & prespecified physiological delay targets (Cold-Start 3x3)
    pvals_delays: dict[str, float] = {}
    for tgt in EXTENDED_TARGETS:
        if tgt not in dev_df.columns:
            continue
        if tgt in out["primary_cold_start_3x3"]:
            res = out["primary_cold_start_3x3"][tgt]
        else:
            y = dev_df[tgt].to_numpy(float)
            res, _ = _run_nested_forecast_suite(
                dev_df,
                x_body,
                x_reve_cs,
                x_hc,
                y,
                protocol="cold_start",
                k_outer=3,
                run_gbdt=False,
                run_joint=True,
                draws=draws,
            )
        out["extended_delays_cold_start_3x3"][tgt] = res
        pvals_delays[tgt] = res["paired_vs_m0_linear"]["M2_reve_resid"]["delta_r"][
            "p_wilcoxon"
        ]
    out["extended_delays_bh_fdr_q"] = study09_protocol.bh_fdr(pvals_delays)

    # 3. Protocol B comparisons on primary targets (B0 rest, B1 causal, B2 session)
    for proto_name, x_r_use, r_mat in (
        ("B0_rest_only", x_reve_e5, x_reve_rest),
        ("B1_causal_online", x_reve_hist, x_reve_rest),
        ("B2_session_isolated", x_reve_hist, None),
    ):
        proto_key = proto_name.split("_", 1)[1]
        out["protocol_comparison_primary"][proto_name] = {}
        for tgt in PRIMARY_TARGETS:
            if tgt not in dev_df.columns:
                continue
            y = dev_df[tgt].to_numpy(float)
            res_p, _ = _run_nested_forecast_suite(
                dev_df,
                x_body,
                x_r_use,
                x_hc,
                y,
                protocol=proto_key,
                rest_eeg=r_mat,
                k_outer=3,
                run_gbdt=False,
                run_joint=True,
                draws=draws,
            )
            out["protocol_comparison_primary"][proto_name][tgt] = res_p

    # 4. Outer 5x5 blocked CV sensitivity check on the 3 primary targets (Cold-Start)
    for tgt in PRIMARY_TARGETS:
        if tgt not in dev_df.columns:
            continue
        y = dev_df[tgt].to_numpy(float)
        res_5, _ = _run_nested_forecast_suite(
            dev_df,
            x_body,
            x_reve_cs,
            x_hc,
            y,
            protocol="cold_start",
            k_outer=5,
            run_gbdt=False,
            run_joint=True,
            draws=draws,
        )
        out["sensitivity_5x5_cold_start"][tgt] = res_5

    # 5. Join GPU Job 2 Forecasting LoRA preds (paired M2_LoRA vs M0 & M2_Frozen)
    lora_df = aux.get("lora_preds")
    if lora_df is not None:
        pfc = study09_protocol.paired_forecast_comparison
        lora_dev = lora_df[dev_m].reset_index(drop=True)
        for tgt in PRIMARY_TARGETS:
            col_lora = f"reve_forecast_lora_ens_{tgt}"
            col_fhead = f"reve_forecast_frozen_head_ens_{tgt}"
            if col_lora not in lora_dev.columns or tgt not in saved_oofs:
                continue
            y = dev_df[tgt].to_numpy(float)
            p_lora = lora_dev[col_lora].to_numpy(float)
            p_fhead = lora_dev[col_fhead].to_numpy(float)
            saved_oofs[tgt]["M2_reve_lora"] = p_lora
            saved_oofs[tgt]["M2_reve_frozen_head"] = p_fhead

            sm_lora, pp_lora = study09_protocol.summarize_forecast(
                dev_df, y, p_lora, draws=draws
            )
            sm_fh, pp_fh = study09_protocol.summarize_forecast(
                dev_df, y, p_fhead, draws=draws
            )
            _, pp_m0 = study09_protocol.summarize_forecast(
                dev_df, y, saved_oofs[tgt]["M0_linear"], draws=draws
            )
            _, pp_m2r = study09_protocol.summarize_forecast(
                dev_df, y, saved_oofs[tgt]["M2_reve_resid"], draws=draws
            )
            out["lora_forecast_integration"][tgt] = {
                "M2_reve_lora": sm_lora,
                "M2_reve_frozen_head": sm_fh,
                "paired_lora_vs_m0": pfc(pp_lora, pp_m0, draws=draws),
                "paired_lora_vs_frozen_head": pfc(pp_lora, pp_fh, draws=draws),
                "paired_lora_vs_m2_frozen_resid": pfc(pp_lora, pp_m2r, draws=draws),
            }
    return out, saved_oofs


# ---------------------------------------------------------------------------
# MODULE 3: Artefact, Temporal, Stimulus & Negative Controls (Req 8 & 16)
# ---------------------------------------------------------------------------


def _residualize_on_artefacts_fold(
    x_tr: np.ndarray, a_tr: np.ndarray, x_ev: np.ndarray, a_ev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly project out [0, 5) artefact covariates strictly fitted on train."""
    za_tr, za_ev = study09_protocol._preprocess_fit_transform(a_tr, a_ev, max_pca=None)
    # Ridge regression A -> X on train (stable when artefacts collinear)
    d_a = za_tr.shape[1]
    design_tr = np.column_stack([np.ones(len(za_tr)), za_tr])
    design_ev = np.column_stack([np.ones(len(za_ev)), za_ev])
    reg = np.eye(d_a + 1) * 10.0
    reg[0, 0] = 0.0
    x_tr_c = np.nan_to_num(x_tr.astype(np.float64), nan=0.0)
    x_ev_c = np.nan_to_num(x_ev.astype(np.float64), nan=0.0)
    beta = np.linalg.solve(design_tr.T @ design_tr + reg, design_tr.T @ x_tr_c)
    return (x_tr_c - design_tr @ beta).astype(np.float32), (
        x_ev_c - design_ev @ beta
    ).astype(np.float32)


def _permute_within_group(
    x: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    *,
    derange: bool = False,
) -> np.ndarray:
    """Permute rows of ``x`` strictly within each group label (e.g. person or video)."""
    out = x.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        perm = rng.permutation(idx)
        if derange and len(idx) >= 2:
            # Ensure no row keeps its own EEG (cyclic shift on fixed points)
            if np.any(perm == idx):
                perm = np.roll(idx, 1)
        out[idx] = x[perm]
    return out


def module_controls_and_negatives(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = study09_protocol.DRAWS,
) -> dict:
    """Run artefact (AC1-AC3), temporal (TC1, NC4), & negative (NC1-NC3) controls."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    n = len(dev_df)

    x_body = dev_df[_body05_cols(frame)].to_numpy(np.float32)
    acols = _artefact_cols(frame)
    x_art = (
        dev_df[acols].to_numpy(np.float32) if acols else np.zeros((n, 1), np.float32)
    )
    tcols = _tc1_cols(frame)
    x_tc1 = (
        dev_df[tcols].to_numpy(np.float32) if tcols else np.zeros((n, 1), np.float32)
    )

    reve_pri = aux["reve_primary"]
    reve_ext = aux["reve_extra"]
    x_reve_real = np.hstack(
        [
            _prefix_mat(reve_pri, "reve_cs_ctx8_early_")[dev_m],
            _prefix_mat(reve_pri, "reve_cs_ctx8_pool_")[dev_m],
            _prefix_mat(reve_pri, "reve_cs_ctx8_diff_")[dev_m],
        ]
    )
    x_reve_e5 = _prefix_mat(reve_pri, "reve_cs_early5_")[dev_m]
    x_reve_pre = _prefix_mat(reve_pri, "reve_cs_pre_")[dev_m]
    x_reve_int46 = _prefix_mat(reve_ext, "reve_cs_interior_ctx8_")[dev_m]
    x_reve_shuf5 = _prefix_mat(reve_ext, "reve_cs_shuf_early5_")[dev_m]

    rng = np.random.default_rng(20261124)
    x_reve_nc1_person = _permute_within_group(
        x_reve_real, dev_df["participant_id"].astype(str).to_numpy(), rng, derange=True
    )
    x_reve_nc2_video = _permute_within_group(
        x_reve_real, dev_df["video"].astype(str).to_numpy(), rng, derange=True
    )

    # AC3 clean-trial mask
    clean_m = np.ones(n, dtype=bool)
    if "art_bad_ch_count_early" in dev_df:
        clean_m &= dev_df["art_bad_ch_count_early"].fillna(99).to_numpy(float) == 0.0
    if "b05_ecg_qrs_valid_frac" in dev_df:
        clean_m &= dev_df["b05_ecg_qrs_valid_frac"].fillna(0.0).to_numpy(float) >= 0.95

    out: dict[str, Any] = {
        "n_artefact_covariates": len(acols),
        "n_temporal_covariates": len(tcols),
        "clean_subset_fraction": round(float(clean_m.mean()), 4),
        "targets": {},
    }

    for tgt in PRIMARY_TARGETS:
        if tgt not in dev_df.columns:
            continue
        y = dev_df[tgt].to_numpy(float)
        keys = (
            "M0",
            "M2_real",
            "M0_plus_art",
            "M2_plus_art",
            "M2_interior46",
            "M2_art_residualized",
            "M0_plus_tc1",
            "M2_plus_tc1",
            "M2_early5_real",
            "M2_NC4_time_shuf5",
            "M2_NC1_within_person_shuf",
            "M2_NC2_within_video_shuf",
            "M2_NC3_pre_stim_only",
        )
        oofs = {k: np.full(n, np.nan) for k in keys}

        for p in range(3):
            for t in range(3):
                ev_m = (pf == p) & (tf == t)
                tr_m = (pf != p) & (tf != t)
                if ev_m.sum() == 0 or tr_m.sum() < 40:
                    continue
                df_tr = dev_df.iloc[tr_m]
                y_tr = y[tr_m]

                # Reference M0 & M2_real
                p0, p2, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_real[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_real[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M0"][ev_m] = p0
                oofs["M2_real"][ev_m] = p2

                # AC1: M0 + artefacts vs M2 + artefacts
                xb_a_tr = np.hstack([x_body[tr_m], x_art[tr_m]])
                xb_a_ev = np.hstack([x_body[ev_m], x_art[ev_m]])
                p0a, p2a, _ = study09_protocol.fit_predict_m2_residual(
                    xb_a_tr,
                    x_reve_real[tr_m],
                    y_tr,
                    df_tr,
                    xb_a_ev,
                    x_reve_real[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M0_plus_art"][ev_m] = p0a
                oofs["M2_plus_art"][ev_m] = p2a

                # AC2a: 46 interior non-rim channels ONLY
                _, p2_i46, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_int46[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_int46[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M2_interior46"][ev_m] = p2_i46

                # AC2b: REVE linearly residualized on artefact proxies inside train fold
                xr_res_tr, xr_res_ev = _residualize_on_artefacts_fold(
                    x_reve_real[tr_m], x_art[tr_m], x_reve_real[ev_m], x_art[ev_m]
                )
                _, p2_res, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    xr_res_tr,
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    xr_res_ev,
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M2_art_residualized"][ev_m] = p2_res

                # TC1: M0 + previous trial + order vs M2 + previous trial + order
                xb_t_tr = np.hstack([x_body[tr_m], x_tc1[tr_m]])
                xb_t_ev = np.hstack([x_body[ev_m], x_tc1[ev_m]])
                p0t, p2t, _ = study09_protocol.fit_predict_m2_residual(
                    xb_t_tr,
                    x_reve_real[tr_m],
                    y_tr,
                    df_tr,
                    xb_t_ev,
                    x_reve_real[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M0_plus_tc1"][ev_m] = p0t
                oofs["M2_plus_tc1"][ev_m] = p2t

                # NC4: Early-5 real vs 1-s time-shuffled Early-5
                _, p2_e5, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_e5[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_e5[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                _, p2_sh5, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_shuf5[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_shuf5[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M2_early5_real"][ev_m] = p2_e5
                oofs["M2_NC4_time_shuf5"][ev_m] = p2_sh5

                # NC1: Within-person EEG shuffle (sanity detector: MUST be <= 0)
                _, p2_nc1, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_nc1_person[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_nc1_person[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M2_NC1_within_person_shuf"][ev_m] = p2_nc1

                # NC2: Within-video inter-subject EEG shuffle (stimulus shortcut!)
                # Train on REAL data; eval with another person's EEG on SAME video!
                _, p2_nc2, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_real[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_nc2_video[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M2_NC2_within_video_shuf"][ev_m] = p2_nc2

                # NC3: Pre-stimulus [-3, 0) s EEG only
                _, p2_nc3, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_pre[tr_m],
                    y_tr,
                    df_tr,
                    x_body[ev_m],
                    x_reve_pre[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                oofs["M2_NC3_pre_stim_only"][ev_m] = p2_nc3

        summaries, pps = {}, {}
        for k, pr in oofs.items():
            sm, pp = study09_protocol.summarize_forecast(dev_df, y, pr, draws=draws)
            summaries[k] = sm
            pps[k] = pp

        # AC3: Evaluate M0 vs M2_real restricted to high-quality clean trials
        df_cl = dev_df[clean_m].reset_index(drop=True)
        _, pp0_cl = study09_protocol.summarize_forecast(
            df_cl, y[clean_m], oofs["M0"][clean_m], draws=draws
        )
        _, pp2_cl = study09_protocol.summarize_forecast(
            df_cl, y[clean_m], oofs["M2_real"][clean_m], draws=draws
        )

        # Within-video paired gain (M2_real - M0 on exact same videos!)
        vids = dev_df["video"].astype(str).to_numpy()
        vid_diffs = []
        for vid in np.unique(vids):
            m = (
                (vids == vid)
                & np.isfinite(y)
                & np.isfinite(oofs["M0"])
                & np.isfinite(oofs["M2_real"])
            )
            if m.sum() >= study09_protocol.MIN_VIDEO_VIEWERS and np.std(y[m]) > 1e-12:
                r0 = (
                    float(np.corrcoef(y[m], oofs["M0"][m])[0, 1])
                    if np.std(oofs["M0"][m]) > 1e-12
                    else 0.0
                )
                r2 = (
                    float(np.corrcoef(y[m], oofs["M2_real"][m])[0, 1])
                    if np.std(oofs["M2_real"][m]) > 1e-12
                    else 0.0
                )
                vid_diffs.append(r2 - r0)
        wv_gain = ssl_eval.boot_ci(vid_diffs, draws=draws)

        pfc = study09_protocol.paired_forecast_comparison
        pc = {
            "M2_real_vs_M0": pfc(pps["M2_real"], pps["M0"], draws=draws),
            "AC1_M2_art_vs_M0_art": pfc(
                pps["M2_plus_art"], pps["M0_plus_art"], draws=draws
            ),
            "AC2a_interior46_vs_M0": pfc(pps["M2_interior46"], pps["M0"], draws=draws),
            "AC2b_art_resid_vs_M0": pfc(
                pps["M2_art_residualized"], pps["M0"], draws=draws
            ),
            "AC3_clean_subset_M2_vs_M0": pfc(pp2_cl, pp0_cl, draws=draws),
            "TC1_M2_tc1_vs_M0_tc1": pfc(
                pps["M2_plus_tc1"], pps["M0_plus_tc1"], draws=draws
            ),
            "NC1_within_person_shuf_vs_M0": pfc(
                pps["M2_NC1_within_person_shuf"], pps["M0"], draws=draws
            ),
            "NC2_stimulus_mediated_vs_M0": pfc(
                pps["M2_NC2_within_video_shuf"], pps["M0"], draws=draws
            ),
            "NC2_idiosyncratic_real_vs_vid_shuf": pfc(
                pps["M2_real"], pps["M2_NC2_within_video_shuf"], draws=draws
            ),
            "NC3_pre_stim_only_vs_M0": pfc(
                pps["M2_NC3_pre_stim_only"], pps["M0"], draws=draws
            ),
            "NC4_early5_real_vs_time_shuf5": pfc(
                pps["M2_early5_real"], pps["M2_NC4_time_shuf5"], draws=draws
            ),
            "within_video_idiosyncratic_M2_minus_M0": wv_gain,
        }
        out["targets"][tgt] = {"summaries": summaries, "paired_contrasts": pc}
        print(
            f"  Controls {tgt}: AC1={pc['AC1_M2_art_vs_M0_art']['delta_r']['mean']}, "
            f"AC2a={pc['AC2a_interior46_vs_M0']['delta_r']['mean']}, "
            f"NC1={pc['NC1_within_person_shuf_vs_M0']['delta_r']['mean']}, "
            f"NC2_stim={pc['NC2_stimulus_mediated_vs_M0']['delta_r']['mean']}, "
            f"NC2_idio={pc['NC2_idiosyncratic_real_vs_vid_shuf']['delta_r']['mean']}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# MODULES 4, 5, 6, 7: Secondary LoRA, Trait/State, Oculo Gate, Stacking Gate
# ---------------------------------------------------------------------------


def _nested_dev_stress_eval(
    dev_df: pd.DataFrame,
    x: np.ndarray,
    *,
    protocol: str = "cold_start",
    rest_arr: np.ndarray | None = None,
    max_pca: int = 64,
    draws: int = study09_protocol.DRAWS,
) -> tuple[dict, dict[str, np.ndarray], dict[str, float]]:
    """Run nested 3x3 dev CV for stress/affect under Cold-Start or Protocol B."""
    n = len(dev_df)
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    oofs = {
        k: np.full(n, np.nan, dtype=np.float32)
        for k in ("stress_logreg", "stress_prob", "stress_va", "valence", "arousal")
    }
    for p in range(3):
        for t in range(3):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 40:
                continue
            x_tr, x_ev = study09_protocol.prepare_protocol_fold(
                x[tr_m],
                dev_df.iloc[tr_m],
                x[ev_m],
                dev_df.iloc[ev_m],
                protocol=protocol,
                rest_tr=rest_arr[tr_m] if rest_arr is not None else None,
                rest_ev=rest_arr[ev_m] if rest_arr is not None else None,
            )
            pr = study09_protocol.fit_predict_nested_affect_stress(
                x_tr, dev_df.iloc[tr_m], x_ev, max_pca=max_pca, inner_k=3
            )
            for k in oofs:
                oofs[k][ev_m] = pr[k]

    res_log = ssl_eval.evaluate_stress(
        dev_df,
        pd.Series(oofs["stress_logreg"], index=dev_df.index),
        draws=draws,
        return_per_person=True,
    )
    res_log.pop("_per_person")
    res_va = ssl_eval.evaluate_stress(
        dev_df,
        pd.Series(oofs["stress_va"], index=dev_df.index),
        draws=draws,
        return_per_person=True,
    )
    pp_va = res_va.pop("_per_person")
    res_aff = ssl_eval.evaluate_valence_arousal(
        dev_df,
        pd.Series(oofs["valence"], index=dev_df.index),
        pd.Series(oofs["arousal"], index=dev_df.index),
        draws=draws,
    )
    cal = study09_protocol.evaluate_brier_and_pooled(
        dev_df, oofs["stress_prob"], oofs["stress_logreg"]
    )
    return (
        {
            "protocol": protocol,
            "stress_logreg": res_log,
            "stress_ridge_va": res_va,
            "affect": res_aff,
            "calibration_and_trait": cal,
        },
        oofs,
        pp_va,
    )


def module_secondary_trait_oculo_stacking(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = study09_protocol.DRAWS,
) -> dict:
    """Run Modules 4 (LoRA), 5 (Trait/State), 6 (Oculo), 7 (Stacking) on dev."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    n = len(dev_df)

    reve_pri = aux["reve_primary"]
    reve_ext = aux["reve_extra"]
    x_reve_vid_cs = _prefix_mat(reve_ext, "reve_cs_vid10_")[dev_m]
    x_reve_vid_hist = _prefix_mat(reve_ext, "reve_hist_vid10_")[dev_m]
    x_reve_pre = _prefix_mat(reve_pri, "reve_cs_pre_")[dev_m]
    x_reve_rest = _prefix_mat(reve_ext, "reve_rest_trait_")[dev_m]

    out: dict[str, Any] = {
        "module4_secondary_affect_and_lora": {},
        "module5_trait_state_factorization": {},
        "module6_oculo_audit_and_baselines": {},
        "module7_complementarity_and_stacking": {},
    }

    # --- MODULE 4: Secondary Affect & Stress (Cold-Start & Protocol B1 + LoRA) ---
    m4: dict[str, Any] = {}
    oof_stress_pool: dict[str, np.ndarray] = {}
    pp_store: dict[str, dict[str, float]] = {}

    # Study-07 enriched causal features (strictly uncentered before fold protocol!)
    if "ecg_hr_ref" in dev_df.columns and "eeg_ep_ref_frontal_alpha" in dev_df.columns:
        x_s07_full = revision._build_phase_features(
            dev_df,
            add_deltas=True,
            add_ecg_morph=True,
            add_eeg_spatial=True,
            add_eeg_tvar=True,
            person_center=False,
        ).to_numpy(np.float32)
        eeg_only_df = revision._build_phase_features(
            dev_df,
            add_deltas=True,
            add_ecg_morph=False,
            add_eeg_spatial=True,
            add_eeg_tvar=True,
            person_center=False,
        )
        x_s07_eeg = eeg_only_df[
            [c for c in eeg_only_df if c.startswith("eeg_")]
        ].to_numpy(np.float32)

        for tag, x_m, proto in (
            ("s07_enriched_cold_start", x_s07_full, "cold_start"),
            ("s07_enriched_causal_online", x_s07_full, "causal_online"),
            ("s07_eeg_cold_start", x_s07_eeg, "cold_start"),
            ("s07_eeg_causal_online", x_s07_eeg, "causal_online"),
        ):
            res, oofs, pp = _nested_dev_stress_eval(
                dev_df, x_m, protocol=proto, max_pca=128, draws=draws
            )
            m4[tag] = res
            pp_store[tag] = pp
            if tag == "s07_enriched_cold_start":
                oof_stress_pool["S07_enriched_CS"] = oofs["stress_va"]

    for tag, x_m, proto, r_arr in (
        ("reve_frozen_linear_cold_start", x_reve_vid_cs, "cold_start", None),
        (
            "reve_frozen_linear_causal_online",
            x_reve_vid_hist,
            "causal_online",
            x_reve_rest,
        ),
        ("reve_frozen_linear_rest_only", x_reve_vid_cs, "rest_only", x_reve_rest),
    ):
        res, oofs, pp = _nested_dev_stress_eval(
            dev_df, x_m, protocol=proto, rest_arr=r_arr, max_pca=64, draws=draws
        )
        m4[tag] = res
        pp_store[tag] = pp
        if tag == "reve_frozen_linear_cold_start":
            oof_stress_pool["REVE_frozen_linear_CS"] = oofs["stress_va"]

    # Join GPU Job 2 Multitask LoRA predictions on dev
    lora_df = aux.get("lora_preds")
    if lora_df is not None:
        ldev = lora_df[dev_m].reset_index(drop=True)
        for m_name in (
            "reve_mt_frozen_rank",
            "reve_mt_lora_rank",
            "reve_mt_lora_norank",
            "reve_mt_lora_r4",
        ):
            c_va = f"{m_name}_ens_stress_va"
            c_dir = f"{m_name}_ens_stress"
            if c_va not in ldev.columns:
                continue
            s_va = ldev[c_va].to_numpy(float)
            s_dir = ldev[c_dir].to_numpy(float)
            p_v = ldev[f"{m_name}_ens_valence"].to_numpy(float)
            p_a = ldev[f"{m_name}_ens_arousal"].to_numpy(float)
            r_va = ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(s_va, index=dev_df.index),
                draws=draws,
                return_per_person=True,
            )
            pp_va = r_va.pop("_per_person")
            r_dir = ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(s_dir, index=dev_df.index),
                draws=draws,
                return_per_person=True,
            )
            r_dir.pop("_per_person")
            r_aff = ssl_eval.evaluate_valence_arousal(
                dev_df,
                pd.Series(p_v, index=dev_df.index),
                pd.Series(p_a, index=dev_df.index),
                draws=draws,
            )
            prob_dir = 1.0 / (1.0 + np.exp(-np.clip(s_dir, -20.0, 20.0)))
            m4[m_name] = {
                "protocol": "cold_start",
                "stress_direct": r_dir,
                "stress_va": r_va,
                "affect": r_aff,
                "calibration_and_trait": study09_protocol.evaluate_brier_and_pooled(
                    dev_df, prob_dir, s_dir
                ),
            }
            pp_store[m_name] = pp_va
            if m_name in ("reve_mt_frozen_rank", "reve_mt_lora_rank"):
                oof_stress_pool[m_name] = s_va

        # Paired LoRA comparisons
        paired_m4 = {}
        for a, b in (
            ("reve_mt_lora_rank", "reve_mt_frozen_rank"),
            ("reve_mt_lora_rank", "reve_frozen_linear_cold_start"),
            ("reve_mt_lora_rank", "reve_mt_lora_norank"),
            ("reve_mt_lora_rank", "s07_eeg_cold_start"),
            ("reve_mt_lora_rank", "s07_enriched_cold_start"),
        ):
            if a in pp_store and b in pp_store:
                paired_m4[f"{a}_minus_{b}"] = ssl_eval.paired_gain_ci(
                    pp_store[a], pp_store[b], draws=draws
                )
        m4["paired_comparisons"] = paired_m4
    out["module4_secondary_affect_and_lora"] = m4

    # --- MODULE 5: Trait / State Factorization (Req 10) ---
    m5: dict[str, Any] = {"cold_start_reve_pre_base": {}, "unlabeled_rest_base": {}}
    for fam_key, x_base_use in (
        ("cold_start_reve_pre_base", x_reve_pre),
        ("unlabeled_rest_base", x_reve_rest),
    ):
        for mode in (
            "raw_unfactored",
            "blind_ref_sub",
            "state_only_shrinkage",
            "trait_only",
            "factored_trait_plus_state",
        ):
            oofs = {
                k: np.full(n, np.nan, dtype=np.float32)
                for k in (
                    "stress_logreg",
                    "stress_prob",
                    "stress_va",
                    "valence",
                    "arousal",
                )
            }
            infos = []
            for p in range(3):
                for t in range(3):
                    ev_m = (pf == p) & (tf == t)
                    tr_m = (pf != p) & (tf != t)
                    x_tr_f, x_ev_f, info_f = study09_protocol.fit_transform_trait_state(
                        x_reve_vid_cs[tr_m],
                        x_base_use[tr_m],
                        dev_df.iloc[tr_m],
                        x_reve_vid_cs[ev_m],
                        x_base_use[ev_m],
                        mode=mode,
                        n_trait_dims=16,
                    )
                    infos.append(info_f)
                    pr = study09_protocol.fit_predict_nested_affect_stress(
                        x_tr_f, dev_df.iloc[tr_m], x_ev_f, max_pca=64, inner_k=3
                    )
                    for k in oofs:
                        oofs[k][ev_m] = pr[k]
            r_va = ssl_eval.evaluate_stress(
                dev_df, pd.Series(oofs["stress_va"], index=dev_df.index), draws=draws
            )
            r_log = ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(oofs["stress_logreg"], index=dev_df.index),
                draws=draws,
            )
            r_aff = ssl_eval.evaluate_valence_arousal(
                dev_df,
                pd.Series(oofs["valence"], index=dev_df.index),
                pd.Series(oofs["arousal"], index=dev_df.index),
                draws=draws,
            )
            cal = study09_protocol.evaluate_brier_and_pooled(
                dev_df, oofs["stress_prob"], oofs["stress_logreg"]
            )
            m5[fam_key][mode] = {
                "stress_ridge_va": r_va["stress"],
                "stress_logreg": r_log["stress"],
                "affect_mean_rho": r_aff["mean_rho"],
                "calibration_and_trait": cal,
                "factorization_info_fold0": infos[0] if infos else {},
            }

        # Also add blind_person_center (Protocol B1 causal_online) for direct comparison
        res_pc, _, _ = _nested_dev_stress_eval(
            dev_df,
            x_reve_vid_cs,
            protocol="causal_online",
            rest_arr=x_reve_rest if fam_key == "unlabeled_rest_base" else None,
            max_pca=64,
            draws=draws,
        )
        m5[fam_key]["blind_person_center_causal_b1"] = {
            "stress_ridge_va": res_pc["stress_ridge_va"]["stress"],
            "stress_logreg": res_pc["stress_logreg"]["stress"],
            "affect_mean_rho": res_pc["affect"]["mean_rho"],
            "calibration_and_trait": res_pc["calibration_and_trait"],
        }

    # Geometry of extracted trait vs state subspaces (subject R^2 & Day1->Day2)
    z_tr_s, _, _ = study09_protocol.fit_transform_trait_state(
        x_reve_vid_cs,
        x_reve_pre,
        dev_df,
        x_reve_vid_cs,
        x_reve_pre,
        mode="state_only_shrinkage",
    )
    z_tr_t, _, _ = study09_protocol.fit_transform_trait_state(
        x_reve_vid_cs,
        x_reve_pre,
        dev_df,
        x_reve_vid_cs,
        x_reve_pre,
        mode="trait_only",
        n_trait_dims=16,
    )
    m5["geometry_dev"] = {
        "raw_vid10_identity": ssl_eval.probe_subject_identity(dev_df, x_reve_vid_cs),
        "shrinkage_state_identity": ssl_eval.probe_subject_identity(dev_df, z_tr_s),
        "anova_trait_identity": ssl_eval.probe_subject_identity(
            dev_df, z_tr_t, max_pca=16
        ),
    }
    out["module5_trait_state_factorization"] = m5

    # --- MODULE 6: Pupil / Gaze Audit & Conditional Baselines (Req 11) ---
    m6: dict[str, Any] = {}
    if "oc_valid_early" in dev_df.columns:
        cov_ea = float(dev_df["oc_valid_early"].fillna(0).mean())
        cov_la = float(dev_df["oc_valid_late"].fillna(0).mean())
        br = dev_df["oc_binoc_r"].dropna().to_numpy(float)
        med_br = float(np.median(br)) if len(br) else None
        # Check evoked pupil onset change on dev (person-averaged)
        pe = dev_df.groupby("participant_id")["oc_pupil_early_chg"].mean().dropna()
        from scipy.stats import ttest_1samp

        t_ev, p_ev = ttest_1samp(pe, 0.0) if len(pe) >= 15 else (np.nan, np.nan)
        max_coh_cov = 0.0
        by_coh = {}
        if "cohort" in dev_df.columns:
            for c, g in dev_df.groupby("cohort"):
                vc = float(g["oc_valid_early"].fillna(0).mean())
                max_coh_cov = max(max_coh_cov, vc)
                bc = g["oc_binoc_r"].dropna()
                by_coh[str(int(c))] = {
                    "n_trials": int(len(g)),
                    "valid_early_frac": round(vc, 4),
                    "median_binoc_r": round(float(bc.median()), 4) if len(bc) else None,
                }
        pass_cov = bool(cov_ea >= 0.50 or max_coh_cov >= 0.60)
        pass_br = bool(med_br is not None and med_br >= 0.50)
        pass_pupil_align = bool(np.isfinite(t_ev) and abs(t_ev) > 4.0 and p_ev < 1e-4)
        # Check gaze-speed availability as secondary alignment cue (t=0.00 s)
        gaze_valid_frac = (
            float(dev_df["oc_gaze_speed_early"].notna().mean())
            if "oc_gaze_speed_early" in dev_df.columns
            else 0.0
        )
        gate_h4_strict = bool(pass_cov and pass_br and pass_pupil_align)
        run_oculo_baselines = bool(
            pass_cov and pass_br and (pass_pupil_align or gaze_valid_frac >= 0.80)
        )
        m6["dev_audit"] = {
            "valid_early_frac": round(cov_ea, 4),
            "valid_late_frac": round(cov_la, 4),
            "gaze_valid_early_frac": round(gaze_valid_frac, 4),
            "median_binoc_r": round(med_br, 4) if med_br is not None else None,
            "pupil_early_chg_mean_mm": round(float(pe.mean()), 4) if len(pe) else None,
            "pupil_early_chg_t": round(float(t_ev), 3) if np.isfinite(t_ev) else None,
            "pupil_early_chg_p": float(p_ev) if np.isfinite(p_ev) else None,
            "by_cohort": by_coh,
            "h4_gate": {
                "pass_coverage": pass_cov,
                "pass_binoc_r": pass_br,
                "pass_pupil_mean_alignment": pass_pupil_align,
                "gate_passed": gate_h4_strict,
                "run_baselines_status": (
                    "strict_h4_passed"
                    if gate_h4_strict
                    else (
                        "gaze_onset_verified_exploratory"
                        if run_oculo_baselines
                        else "abstained_h4_gate_failed"
                    )
                ),
            },
        }
        if run_oculo_baselines:
            # Run simple causal oculometry baselines on valid subset / full dev!
            oc_cols_use = [c for c in OCULO_FULL_COLS if c in dev_df.columns]
            x_oc = dev_df[oc_cols_use].to_numpy(np.float32)
            res_oc_cs, oofs_oc, _ = _nested_dev_stress_eval(
                dev_df, x_oc, protocol="cold_start", max_pca=16, draws=draws
            )
            res_oc_b1, _, _ = _nested_dev_stress_eval(
                dev_df, x_oc, protocol="causal_online", max_pca=16, draws=draws
            )
            # Also valid-only stress AUC
            val_m = (dev_df["oc_valid_early"].fillna(0) > 0.5).to_numpy()
            res_oc_val = ssl_eval.evaluate_stress(
                dev_df[val_m],
                pd.Series(oofs_oc["stress_va"][val_m], index=dev_df.index[val_m]),
                draws=draws,
            )
            oof_stress_pool["Oculo_simple_CS"] = oofs_oc["stress_va"]

            # Forecasting control: does adding causal pupil/gaze[0:5] alter M2-M0?
            oc_ea_cols = [c for c in OCULO_EARLY_COLS if c in dev_df.columns]
            x_oc_ea = dev_df[oc_ea_cols].to_numpy(np.float32)
            x_body = dev_df[_body05_cols(frame)].to_numpy(np.float32)
            x_reve_cs = np.hstack(
                [
                    _prefix_mat(reve_pri, "reve_cs_ctx8_early_")[dev_m],
                    _prefix_mat(reve_pri, "reve_cs_ctx8_pool_")[dev_m],
                    _prefix_mat(reve_pri, "reve_cs_ctx8_diff_")[dev_m],
                ]
            )
            oc_fcast = {}
            for tgt in PRIMARY_TARGETS:
                if tgt not in dev_df.columns:
                    continue
                y = dev_df[tgt].to_numpy(float)
                p0_oc = np.full(n, np.nan)
                p2_oc = np.full(n, np.nan)
                for p in range(3):
                    for t in range(3):
                        ev_m = (pf == p) & (tf == t)
                        tr_m = (pf != p) & (tf != t)
                        xb_oc_tr = np.hstack([x_body[tr_m], x_oc_ea[tr_m]])
                        xb_oc_ev = np.hstack([x_body[ev_m], x_oc_ea[ev_m]])
                        a0, a2, _ = study09_protocol.fit_predict_m2_residual(
                            xb_oc_tr,
                            x_reve_cs[tr_m],
                            y[tr_m],
                            dev_df.iloc[tr_m],
                            xb_oc_ev,
                            x_reve_cs[ev_m],
                            eeg_pca_grid=(16, 32, 64),
                        )
                        p0_oc[ev_m] = a0
                        p2_oc[ev_m] = a2
                s0, pp0 = study09_protocol.summarize_forecast(
                    dev_df, y, p0_oc, draws=draws
                )
                s2, pp2 = study09_protocol.summarize_forecast(
                    dev_df, y, p2_oc, draws=draws
                )
                pfc_oc = study09_protocol.paired_forecast_comparison
                oc_fcast[tgt] = {
                    "M0_plus_oculo": s0["r"],
                    "M2_plus_oculo": s2["r"],
                    "paired_M2_vs_M0_ctrl_oculo": pfc_oc(pp2, pp0, draws=draws),
                }
            m6["baselines"] = {
                "status": m6["dev_audit"]["h4_gate"]["run_baselines_status"],
                "stress_cold_start": res_oc_cs,
                "stress_causal_online": res_oc_b1,
                "stress_valid_subset_cs": res_oc_val["stress"],
                "forecast_control_with_oculo": oc_fcast,
            }
        else:
            m6["baselines"] = {
                "status": "abstained_h4_gate_failed",
                "reason": (
                    "Pupil/gaze failed pre-registered coverage/binocular/"
                    "alignment gate."
                ),
            }
    else:
        m6["status"] = "oculo_table_missing"
    out["module6_oculo_audit_and_baselines"] = m6

    # --- MODULE 7: Complementarity Gate & Conditional Nested OOF Stacking ---
    m7: dict[str, Any] = {}
    y_stress = label(dev_df).to_numpy(int)
    if len(oof_stress_pool) >= 2:
        comp_st = study09_protocol.assess_complementarity(
            dev_df, y_stress, oof_stress_pool, task="stress", draws=draws
        )
        m7["stress_complementarity"] = comp_st
        if comp_st["gate_passed"]:
            best_nm = comp_st["best_single_model"]
            sel_models = [best_nm, *comp_st["qualifying_secondary_models"]]
            st_res = study09_protocol.run_nested_oof_stacker(
                dev_df,
                y_stress,
                {k: oof_stress_pool[k] for k in sel_models},
                task="stress",
                k=3,
                draws=draws,
            )
            pp_st = st_res.pop("_per_person")
            st_res.pop("_oof_pred")
            # Paired gain vs best single model
            r_best = ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(oof_stress_pool[best_nm], index=dev_df.index),
                draws=draws,
                return_per_person=True,
            )
            pp_best = r_best.pop("_per_person")
            st_res["paired_gain_vs_best_single"] = ssl_eval.paired_gain_ci(
                pp_st, pp_best, draws=draws
            )
            m7["stress_nested_stacking"] = st_res
        else:
            m7["stress_nested_stacking"] = {
                "status": "abstained_h5_complementarity_gate_not_met"
            }

    # Forecasting complementarity & conditional stacking on primary targets
    x_body = dev_df[_body05_cols(frame)].to_numpy(np.float32)
    x_hc = dev_df[_eeg05_hc_cols(frame)].to_numpy(np.float32)
    x_reve_cs = np.hstack(
        [
            _prefix_mat(reve_pri, "reve_cs_ctx8_early_")[dev_m],
            _prefix_mat(reve_pri, "reve_cs_ctx8_pool_")[dev_m],
            _prefix_mat(reve_pri, "reve_cs_ctx8_diff_")[dev_m],
        ]
    )
    m7["forecast_complementarity"] = {}
    for tgt in PRIMARY_TARGETS:
        if tgt not in dev_df.columns:
            continue
        y = dev_df[tgt].to_numpy(float)
        _, f_oofs = _run_nested_forecast_suite(
            dev_df,
            x_body,
            x_reve_cs,
            x_hc,
            y,
            protocol="cold_start",
            k_outer=3,
            run_gbdt=True,
            run_joint=False,
            draws=draws,
        )
        cand = {
            "M0_linear": f_oofs["M0_linear"],
            "M0_gbdt": f_oofs["M0_gbdt"],
            "M1_reve_frozen": f_oofs["M1_reve_frozen"],
            "M1_hc_eeg": f_oofs["M1_hc_eeg"],
        }
        if lora_df is not None and f"reve_forecast_lora_ens_{tgt}" in lora_df.columns:
            cand["M2_reve_lora"] = lora_df[dev_m][
                f"reve_forecast_lora_ens_{tgt}"
            ].to_numpy(float)
        comp_f = study09_protocol.assess_complementarity(
            dev_df, y, cand, task="regression", draws=draws
        )
        entry_f: dict[str, Any] = {"complementarity": comp_f}
        if comp_f["gate_passed"]:
            best_f = comp_f["best_single_model"]
            sel_f = [best_f, *comp_f["qualifying_secondary_models"]]
            st_f = study09_protocol.run_nested_oof_stacker(
                dev_df,
                y,
                {k: cand[k] for k in sel_f},
                task="regression",
                k=3,
                draws=draws,
            )
            pp_stf = st_f.pop("_per_person")
            st_f.pop("_oof_pred")
            _, pp_bf = study09_protocol.summarize_forecast(
                dev_df, y, cand[best_f], draws=draws
            )
            st_f["paired_gain_vs_best_single"] = (
                study09_protocol.paired_forecast_comparison(pp_stf, pp_bf, draws=draws)
            )
            entry_f["nested_stacking"] = st_f
        else:
            entry_f["nested_stacking"] = {
                "status": "abstained_h5_complementarity_gate_not_met"
            }
        m7["forecast_complementarity"][tgt] = entry_f

    out["module7_complementarity_and_stacking"] = m7
    return out


def main() -> None:
    """CLI entrypoint compatible with ``decoding.yml`` CPU workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("all", "audit_forecast", "controls", "secondary"),
        default="all",
    )
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    t0 = time.perf_counter()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    frame, aux = load_study09_tables()
    result: dict[str, Any] = {
        "study": "09_eeg_predictive_physiology",
        "split": args.split,
        "salt": study09_protocol.STUDY09_BASE_SALT,
        "splits": common.summary(frame),
        "available_aux_tables": sorted(aux.keys()),
    }

    if args.split in ("all", "audit_forecast"):
        print("=== Study 09 Module 1: Studies 07-08 Audit ===", flush=True)
        result["module1_audit"] = module_audit_study07_08(frame, aux)
        print("=== Study 09 Module 2: Primary M0/M1/M2 & Delays ===", flush=True)
        m2_res, _ = module_primary_forecasting(frame, aux)
        result["module2_primary_forecasting"] = m2_res

    if args.split in ("all", "controls"):
        print(
            "=== Study 09 Module 3: Artefact, Temporal & Negative Controls ===",
            flush=True,
        )
        result["module3_controls"] = module_controls_and_negatives(frame, aux)

    if args.split in ("all", "secondary"):
        print(
            "=== Study 09 Modules 4-7: LoRA, Trait/State, Oculo, Stacking ===",
            flush=True,
        )
        result["modules4_to_7"] = module_secondary_trait_oculo_stacking(frame, aux)

    result["elapsed_cpu_seconds"] = round(time.perf_counter() - t0, 2)
    text = json.dumps(result, indent=1, default=lambda x: None)
    print(text[:5000], flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(text)


if __name__ == "__main__":
    main()
