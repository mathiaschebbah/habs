"""Study 10 (Final Experiment): Generalization, Stability, Coherence & Monitoring.

Runs 100% on CPU (``decoding.yml --study study10_final --split all``) using the
already-extracted Study 04-09 private parquets (zero new GPU jobs).

Pre-registered protocol: ``journal/10_final_generalization_coherence_clinical.md``
Hypotheses addressed:
- H10.1 (Module 10A): Generalization ladder R1 -> R4 & sensor modality ablation.
- H10.2 (Module 10B): Day-1 -> Day-2 physiological stability, cross-day transfer,
  and within-session habituation (Sakimura et al. 2025; Chen et al. 2014).
- H10.3 (Module 10C): Unsupervised CORAL, chronological k-shot personalization
  (k = 0..40) + J1 -> J2 calibration, and Leave-One-Cohort-Out (LOCO-CV)
  (Mihirette et al. 2025; Khan & Ishikawa 2026; Engemann et al. 2018).
- H10.4 (Module 10D): Autonomic-subjective coherence & 2D divergence space +
  bridge to Study 09 EEG->Heart predictive gain (Cuve et al. 2023; Brown 2019).
- H10.5 (Module 10E): Operational multi-episode clinical passive monitoring
  curve (K = 1..10 episodes = 10-100 s) across 3 clinical contrasts
  (Danel 2025; Roue 2025; Adama 2025; Paniagua-Gomez 2025).
- H10.6 (Module 10F): Within-clip early [0, 5) s vs late [5, 10) s EEG-band
  and peripheral dynamics by self-reported emotion (Si et al. 2026; Qin 2023).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from decoding import (
    common,
    person,
    revision,
    ssl_eval,
    study09_analysis,
    study09_protocol,
)
from decoding.stress import contrasts, label

DRAWS = study09_protocol.DRAWS
MIN_EACH = 3
K_SHOT_GRID = (0, 5, 10, 15, 20, 30, 40)
K_SHOT_GRID_30 = (0, 5, 10, 15, 20, 30)
CLINICAL_K_GRID = (1, 2, 3, 5, 8, 10)
EMOTIONS_TARGET = ("fear", "disgust", "anger", "sadness", "joy", "relaxation")


# ---------------------------------------------------------------------------
# Shared fast leak-free primitives & CORAL domain alignment
# ---------------------------------------------------------------------------


def _build_s07_feature_df(frame: pd.DataFrame) -> pd.DataFrame:
    """Build uncentered Study-07 4-phase enriched multimodal feature table."""
    return revision._build_phase_features(
        frame,
        add_deltas=True,
        add_ecg_morph=True,
        add_eeg_spatial=True,
        add_eeg_tvar=True,
        person_center=False,
    )


def _modality_columns(feat_df: pd.DataFrame) -> dict[str, list[str]]:
    """Partition Study-07 enriched columns cleanly by physiological modality."""
    cols = list(feat_df.columns)
    ecg = [c for c in cols if c.startswith("ecg_")]
    eda = [c for c in cols if c.startswith("sc_")]
    resp = [c for c in cols if c.startswith("resp_")]
    eeg = [c for c in cols if c.startswith("eeg_")]
    return {
        "All": cols,
        "minus_ECG": [c for c in cols if not c.startswith("ecg_")],
        "minus_EDA": [c for c in cols if not c.startswith("sc_")],
        "minus_Resp": [c for c in cols if not c.startswith("resp_")],
        "minus_EEG": [c for c in cols if not c.startswith("eeg_")],
        "ECG_only": ecg,
        "EDA_only": eda,
        "Resp_only": resp,
        "EEG_hc_only": eeg,
    }


def _get_reve_vid10(aux: dict[str, pd.DataFrame]) -> np.ndarray | None:
    """Return frozen Cold-Start 10-s REVE embeddings (fallback to causal ctx8)."""
    x = study09_analysis._prefix_mat(aux.get("reve_extra"), "reve_cs_vid10_")
    if x is not None:
        return x
    return study09_analysis._prefix_mat(aux.get("reve_primary"), "reve_cs_ctx8_early_")


def _get_reve_causal_forecast(aux: dict[str, pd.DataFrame]) -> np.ndarray | None:
    """Return Study-09 canonical causal REVE forecasting stack [early, pool, diff]."""
    pri = aux.get("reve_primary")
    parts = [
        study09_analysis._prefix_mat(pri, pfx)
        for pfx in (
            "reve_cs_ctx8_early_",
            "reve_cs_ctx8_pool_",
            "reve_cs_ctx8_diff_",
        )
    ]
    valid = [p for p in parts if p is not None]
    return np.hstack(valid) if valid else None


def _sym_matrix_power(mat: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
    """Symmetric positive-definite matrix power via eigendecomposition."""
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, eps)
    return (vecs * (vals**power)[None, :]) @ vecs.T


def coral_align_target(
    z_src: np.ndarray,
    z_tgt_fit: np.ndarray,
    z_tgt_eval: np.ndarray,
    *,
    reg: float = 1e-2,
    mean_shrink: float = 0.0,
    cov_shrink: float = 0.0,
) -> np.ndarray:
    """Unsupervised CORAL domain alignment (Sun et al. 2016; Mihirette 2025).

    Aligns unlabeled target evaluation rows ``z_tgt_eval`` to source distribution
    ``z_src`` using target statistics estimated STRICTLY from ``z_tgt_fit``
    (zero target labels, zero leakage from ``z_tgt_eval`` when separated).
    Optional Empirical-Bayes shrinkage ``(mean_shrink, cov_shrink)`` stabilizes
    covariance whitening when ``len(z_tgt_fit)`` is small (few-shot regime).
    """
    zs = np.asarray(z_src, dtype=np.float64)
    zt_f = np.asarray(z_tgt_fit, dtype=np.float64)
    zt_e = np.asarray(z_tgt_eval, dtype=np.float64)
    d = zs.shape[1]
    n_t = len(zt_f)
    if n_t == 0:
        return zt_e.copy()

    mu_s = zs.mean(axis=0)
    cs_raw = np.cov(zs, rowvar=False) if len(zs) > 1 else np.eye(d)
    cs_raw = np.atleast_2d(cs_raw)
    cs = cs_raw + reg * np.eye(d)

    mu_t_raw = zt_f.mean(axis=0)
    mu_t = (n_t * mu_t_raw + mean_shrink * mu_s) / max(n_t + mean_shrink, 1e-8)

    if n_t >= 2:
        ct_raw = np.atleast_2d(np.cov(zt_f, rowvar=False))
    else:
        ct_raw = cs_raw.copy()
    ct_shrunk = (n_t * ct_raw + cov_shrink * cs_raw) / max(n_t + cov_shrink, 1e-8)
    ct = ct_shrunk + reg * np.eye(d)

    whitening = _sym_matrix_power(ct, -0.5)
    coloring = _sym_matrix_power(cs, 0.5)
    transform = whitening @ coloring
    return (zt_e - mu_t[None, :]) @ transform + mu_s[None, :]


def _svd_ridge_coef(
    z_tr: np.ndarray, y_tr: np.ndarray, alpha: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form SVD Ridge returning ``(y_mean, x_mean, coef)``."""
    y_mean = float(np.mean(y_tr))
    yc = y_tr - y_mean
    x_mean = np.mean(z_tr, axis=0)
    xc = z_tr - x_mean[None, :]
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    d = s / (s * s + alpha)
    coef = vt.T @ (d * (u.T @ yc))
    return y_mean, x_mean, coef


def _fit_predict_stress_fast(
    x_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    *,
    max_pca: int = 64,
    cs: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    alphas: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0),
    inner_k: int = 3,
    coral_eval: bool = False,
    coral_reg: float = 1e-2,
    return_params: bool = False,
) -> dict[str, Any]:
    """Fast fold-isolated nested LogReg + Ridge V&A (mathematically matches S09).

    Reuses a single fold-isolated ``_preprocess_fit_transform`` per inner fold
    across LogReg, Valence Ridge, and Arousal Ridge (~3x faster than calling
    them separately). Optionally applies strictly unsupervised CORAL alignment
    in standardized/PCA space to ``z_ev``.
    """
    ipf, itf = study09_protocol.inner_blocked_folds(df_tr, k=inner_k)
    y_s = label(df_tr).to_numpy(int)
    ppl = df_tr["participant_id"]
    ppl_arr = ppl.to_numpy()
    yv = (
        df_tr["rating_valence"] - df_tr["rating_valence"].groupby(ppl).transform("mean")
    ).to_numpy(float)
    ya = (
        df_tr["rating_arousal"] - df_tr["rating_arousal"].groupby(ppl).transform("mean")
    ).to_numpy(float)

    oof_c = {c: np.full(len(df_tr), np.nan, dtype=float) for c in cs}
    oof_v = {a: np.full(len(df_tr), np.nan, dtype=float) for a in alphas}
    oof_a = {a: np.full(len(df_tr), np.nan, dtype=float) for a in alphas}

    for p in range(inner_k):
        for t in range(inner_k):
            ev_m = (ipf == p) & (itf == t)
            tr_m = (ipf != p) & (itf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 20 or len(np.unique(y_s[tr_m])) < 2:
                continue
            z_tr_i, z_ev_i = study09_protocol._preprocess_fit_transform(
                x_tr[tr_m], x_tr[ev_m], max_pca=max_pca
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for c_val in cs:
                    clf = LogisticRegression(
                        C=c_val, class_weight="balanced", max_iter=500
                    ).fit(z_tr_i, y_s[tr_m])
                    oof_c[c_val][ev_m] = clf.decision_function(z_ev_i)

            # Shared SVD across all Ridge alphas and both targets (V and A)
            xm = z_tr_i.mean(axis=0, keepdims=True)
            xc = z_tr_i - xm
            xec = z_ev_i - xm
            u, s, vt = np.linalg.svd(xc, full_matrices=False)
            ym_v = float(np.mean(yv[tr_m]))
            ym_a = float(np.mean(ya[tr_m]))
            ut_v = u.T @ (yv[tr_m] - ym_v)
            ut_a = u.T @ (ya[tr_m] - ym_a)
            for a in alphas:
                dv = s / (s * s + a)
                oof_v[a][ev_m] = ym_v + xec @ (vt.T @ (dv * ut_v))
                oof_a[a][ev_m] = ym_a + xec @ (vt.T @ (dv * ut_a))

    # Select C by inner within-person stress AUC
    best_c, best_auc = cs[len(cs) // 2], -1.0
    for c_val in cs:
        pmap = ssl_eval.person_stress_aucs(
            df_tr,
            pd.Series(oof_c[c_val], index=df_tr.index),
            pd.Series(y_s.astype(bool), index=df_tr.index),
            pd.Series(~y_s.astype(bool), index=df_tr.index),
        )
        sc = float(np.mean(list(pmap.values()))) if pmap else 0.5
        if sc > best_auc:
            best_auc, best_c = sc, c_val

    best_av, best_lv = alphas[len(alphas) // 2], float("inf")
    best_aa, best_la = alphas[len(alphas) // 2], float("inf")
    for a in alphas:
        lv = study09_protocol._within_person_mse(yv, oof_v[a], ppl_arr)
        la = study09_protocol._within_person_mse(ya, oof_a[a], ppl_arr)
        if lv < best_lv:
            best_lv, best_av = lv, a
        if la < best_la:
            best_la, best_aa = la, a

    z_tr_f, z_ev_f = study09_protocol._preprocess_fit_transform(
        x_tr, x_ev, max_pca=max_pca
    )
    z_ev_use = (
        coral_align_target(z_tr_f, z_ev_f, z_ev_f, reg=coral_reg)
        if coral_eval
        else z_ev_f
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_f = LogisticRegression(C=best_c, class_weight="balanced", max_iter=500).fit(
            z_tr_f, y_s
        )
        sc_log = clf_f.decision_function(z_ev_use)
        prob_log = clf_f.predict_proba(z_ev_use)[:, 1]

    ym_v, xm_f, w_v = _svd_ridge_coef(z_tr_f, yv, best_av)
    ym_a, _, w_a = _svd_ridge_coef(z_tr_f, ya, best_aa)
    pv = ym_v + (z_ev_use - xm_f[None, :]) @ w_v
    pa = ym_a + (z_ev_use - xm_f[None, :]) @ w_a

    ok_v = np.isfinite(oof_v[best_av])
    ok_a = np.isfinite(oof_a[best_aa])
    sd_v = max(float(np.std(oof_v[best_av][ok_v])) if ok_v.sum() > 5 else 1.0, 1e-6)
    sd_a = max(float(np.std(oof_a[best_aa][ok_a])) if ok_a.sum() > 5 else 1.0, 1e-6)
    sc_va = pa / sd_a - pv / sd_v

    out: dict[str, Any] = {
        "stress_logreg": sc_log.astype(np.float32),
        "stress_prob": prob_log.astype(np.float32),
        "stress_va": sc_va.astype(np.float32),
        "valence": pv.astype(np.float32),
        "arousal": pa.astype(np.float32),
    }
    if return_params:
        out["params"] = {
            "z_tr": z_tr_f,
            "z_ev": z_ev_f,
            "xm": xm_f,
            "ym_v": ym_v,
            "ym_a": ym_a,
            "w_v": w_v,
            "w_a": w_a,
            "sd_v": sd_v,
            "sd_a": sd_a,
            "best_c": best_c,
            "best_av": best_av,
            "best_aa": best_aa,
        }
    return out


def _eval_stress_bundle(
    df: pd.DataFrame, oofs: dict[str, np.ndarray], *, draws: int = DRAWS
) -> tuple[dict, dict[str, float]]:
    """Summarize stress (Ridge V&A + LogReg) and affect metrics cleanly."""
    st_va = ssl_eval.evaluate_stress(
        df,
        pd.Series(oofs["stress_va"], index=df.index),
        draws=draws,
        return_per_person=True,
    )
    pp_va = st_va.pop("_per_person")
    st_lr = ssl_eval.evaluate_stress(
        df,
        pd.Series(oofs["stress_logreg"], index=df.index),
        draws=draws,
        return_per_person=False,
    )
    aff = ssl_eval.evaluate_valence_arousal(
        df,
        pd.Series(oofs["valence"], index=df.index),
        pd.Series(oofs["arousal"], index=df.index),
        draws=draws,
    )
    return {
        "stress_ridge_va": st_va,
        "stress_logreg": st_lr,
        "affect": aff,
    }, pp_va


def _run_cv_regime(
    dev_df: pd.DataFrame,
    x_mat: np.ndarray,
    regime: str,
    *,
    max_pca: int = 64,
    draws: int = DRAWS,
) -> tuple[dict, dict[str, np.ndarray], dict[str, float]]:
    """Evaluate a representation under R1, R2, R3, or R4 CV inside ``dev``."""
    n = len(dev_df)
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    oofs = {
        k: np.full(n, np.nan, dtype=np.float32)
        for k in ("stress_logreg", "stress_prob", "stress_va", "valence", "arousal")
    }

    if regime == "R1_random_5fold":
        rng = np.random.default_rng(20261125)
        rf = rng.integers(0, 5, size=n)
        splits = [(rf != k, rf == k) for k in range(5)]
    elif regime == "R2_within_person_new_themes":
        splits = [(tf != t, tf == t) for t in range(3)]
    elif regime == "R3_loso_seen_themes":
        splits = [(pf != p, pf == p) for p in range(3)]
    elif regime == "R4_double_unseen_cold_start":
        splits = [
            ((pf != p) & (tf != t), (pf == p) & (tf == t))
            for p in range(3)
            for t in range(3)
        ]
    else:
        raise ValueError(f"Unknown regime: {regime}")

    for tr_m, ev_m in splits:
        if ev_m.sum() == 0 or tr_m.sum() < 40:
            continue
        preds = _fit_predict_stress_fast(
            x_mat[tr_m], dev_df.iloc[tr_m], x_mat[ev_m], max_pca=max_pca
        )
        for k in oofs:
            oofs[k][ev_m] = preds[k]

    summary, pp_va = _eval_stress_bundle(dev_df, oofs, draws=draws)
    return summary, oofs, pp_va


# ---------------------------------------------------------------------------
# MODULE 10A: Generalization Ladder (R1 -> R4) & Sensor Modality Ablation
# ---------------------------------------------------------------------------


def module_10a_generalization_and_ablation(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = DRAWS,
) -> tuple[dict, dict[str, np.ndarray]]:
    """H10.1: Quantify R1->R4 generalization gap and R4 modality ablation."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    feat_df = _build_s07_feature_df(dev_df)
    mod_cols = _modality_columns(feat_df)
    x_all = feat_df[mod_cols["All"]].to_numpy(np.float32)

    x_reve_full = _get_reve_vid10(aux)
    x_reve = x_reve_full[dev_m] if x_reve_full is not None else None

    regimes = (
        "R1_random_5fold",
        "R2_within_person_new_themes",
        "R3_loso_seen_themes",
        "R4_double_unseen_cold_start",
    )
    ladder: dict[str, dict] = {"S07_enriched": {}, "REVE_frozen": {}}
    saved_r4_oofs: dict[str, np.ndarray] = {}
    pp_ref_r4: dict[str, dict[str, float]] = {}

    for reg in regimes:
        s_enr, o_enr, pp_enr = _run_cv_regime(
            dev_df, x_all, reg, max_pca=64, draws=draws
        )
        ladder["S07_enriched"][reg] = s_enr
        if reg == "R4_double_unseen_cold_start":
            saved_r4_oofs["S07_enriched"] = o_enr["stress_va"]
            pp_ref_r4["S07_enriched"] = pp_enr
        if x_reve is not None:
            s_rev, o_rev, pp_rev = _run_cv_regime(
                dev_df, x_reve, reg, max_pca=64, draws=draws
            )
            ladder["REVE_frozen"][reg] = s_rev
            if reg == "R4_double_unseen_cold_start":
                saved_r4_oofs["REVE_frozen"] = o_rev["stress_va"]
                pp_ref_r4["REVE_frozen"] = pp_rev

    # Paired R1/R2/R3 - R4 optimism gaps for S07_enriched & REVE_frozen
    optimism_gaps: dict[str, dict] = {}
    for mname in ("S07_enriched", "REVE_frozen"):
        if not ladder[mname]:
            continue
        r4_wp = ladder[mname]["R4_double_unseen_cold_start"]["stress_ridge_va"][
            "stress"
        ]["auc"]
        r4_pool = ladder[mname]["R4_double_unseen_cold_start"]["stress_ridge_va"][
            "pooled_auc"
        ]
        r4_lr_wp = ladder[mname]["R4_double_unseen_cold_start"]["stress_logreg"][
            "stress"
        ]["auc"]
        r4_lr_pool = ladder[mname]["R4_double_unseen_cold_start"]["stress_logreg"][
            "pooled_auc"
        ]
        optimism_gaps[mname] = {}
        for reg in regimes[:3]:
            r_va = ladder[mname][reg]["stress_ridge_va"]
            r_lr = ladder[mname][reg]["stress_logreg"]
            optimism_gaps[mname][f"{reg}_minus_R4"] = {
                "va_within_person_delta": (
                    round(r_va["stress"]["auc"] - r4_wp, 4)
                    if r_va["stress"]["auc"] is not None and r4_wp is not None
                    else None
                ),
                "va_pooled_delta": (
                    round(r_va["pooled_auc"] - r4_pool, 4)
                    if r_va["pooled_auc"] is not None and r4_pool is not None
                    else None
                ),
                "logreg_within_person_delta": (
                    round(r_lr["stress"]["auc"] - r4_lr_wp, 4)
                    if r_lr["stress"]["auc"] is not None and r4_lr_wp is not None
                    else None
                ),
                "logreg_pooled_delta": (
                    round(r_lr["pooled_auc"] - r4_lr_pool, 4)
                    if r_lr["pooled_auc"] is not None and r4_lr_pool is not None
                    else None
                ),
            }

    # Sensor modality ablation strictly on R4 Cold-Start (same 3x3 folds)
    ablation: dict[str, Any] = {}
    pp_all = pp_ref_r4["S07_enriched"]
    ablation["All"] = {
        "n_features": len(mod_cols["All"]),
        **ladder["S07_enriched"]["R4_double_unseen_cold_start"],
        "paired_vs_All": {"gain": 0.0, "ci95": [0.0, 0.0], "n": len(pp_all)},
    }
    for key in (
        "minus_ECG",
        "minus_EDA",
        "minus_Resp",
        "minus_EEG",
        "ECG_only",
        "EDA_only",
        "Resp_only",
        "EEG_hc_only",
    ):
        cols = mod_cols[key]
        if not cols:
            continue
        xm = feat_df[cols].to_numpy(np.float32)
        s_m, o_m, pp_m = _run_cv_regime(
            dev_df,
            xm,
            "R4_double_unseen_cold_start",
            max_pca=min(64, xm.shape[1]),
            draws=draws,
        )
        ablation[key] = {
            "n_features": len(cols),
            **s_m,
            "paired_vs_All": ssl_eval.paired_gain_ci(pp_m, pp_all, draws=draws),
        }
        if key in ("ECG_only", "EDA_only", "EEG_hc_only"):
            saved_r4_oofs[key] = o_m["stress_va"]

    if x_reve is not None:
        ablation["REVE_only"] = {
            "n_features": int(x_reve.shape[1]),
            **ladder["REVE_frozen"]["R4_double_unseen_cold_start"],
            "paired_vs_All": ssl_eval.paired_gain_ci(
                pp_ref_r4["REVE_frozen"], pp_all, draws=draws
            ),
        }
        # Also test S07_enriched + REVE_frozen multimodal fusion on R4
        x_comb = np.hstack([x_all, x_reve])
        s_cb, _, pp_cb = _run_cv_regime(
            dev_df, x_comb, "R4_double_unseen_cold_start", max_pca=64, draws=draws
        )
        ablation["All_plus_REVE"] = {
            "n_features": int(x_comb.shape[1]),
            **s_cb,
            "paired_vs_All": ssl_eval.paired_gain_ci(pp_cb, pp_all, draws=draws),
        }

    out = {
        "n_dev_trials": int(len(dev_df)),
        "n_dev_people": int(dev_df["participant_id"].nunique()),
        "generalization_ladder": ladder,
        "optimism_gaps": optimism_gaps,
        "r4_modality_ablation": ablation,
    }
    print(
        "  10A Ladder S07_VA (WP/Pool): "
        + ", ".join(
            f"{r[:2]}={ladder['S07_enriched'][r]['stress_ridge_va']['stress']['auc']}/"
            f"{ladder['S07_enriched'][r]['stress_ridge_va']['pooled_auc']}"
            for r in regimes
        ),
        flush=True,
    )
    return out, saved_r4_oofs


# ---------------------------------------------------------------------------
# MODULE 10B: Longitudinal Stability (J1 -> J2) & Within-Session Habituation
# ---------------------------------------------------------------------------


def _extract_interpretable_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute interpretable baseline-corrected trial responses for 10B/10D."""
    sigs: dict[str, pd.Series] = {}
    if "tc_hr_0" in frame.columns:
        sigs["ecg_hr"] = person.response(frame, "heart")
    else:
        sigs["ecg_hr"] = (
            0.5 * (frame["ecg_hr_early"] + frame["ecg_hr_late"]) - frame["ecg_hr_ref"]
        )
    if {"ecg_rr_early", "ecg_rr_late", "ecg_rr_ref"}.issubset(frame.columns):
        sigs["ecg_rr"] = (
            0.5 * (frame["ecg_rr_early"] + frame["ecg_rr_late"]) - frame["ecg_rr_ref"]
        )
    if "tc_sc_0" in frame.columns:
        sigs["eda_sc"] = person.response(frame, "skin")
    else:
        sigs["eda_sc"] = (
            0.5 * (frame["sc_mean_early"] + frame["sc_mean_late"])
            - frame["sc_mean_ref"]
        )
    if "scr_rise" in frame.columns:
        sigs["eda_scr"] = frame["scr_rise"].astype(float)
    elif {"sc_rise_early", "sc_rise_late", "sc_rise_ref"}.issubset(frame.columns):
        sigs["eda_scr"] = (
            0.5 * (frame["sc_rise_early"] + frame["sc_rise_late"])
            - frame["sc_rise_ref"]
        )
    if "tc_resp_amp_0" in frame.columns:
        sigs["resp_amp"] = person.response(frame, "breathing")
    else:
        sigs["resp_amp"] = (
            0.5 * (frame["resp_amp_early"] + frame["resp_amp_late"])
            - frame["resp_amp_ref"]
        )
    if {
        "eeg_ep_early_occipital_alpha",
        "eeg_ep_late_occipital_alpha",
        "eeg_ep_ref_occipital_alpha",
    }.issubset(frame.columns):
        sigs["eeg_alpha_occ"] = (
            0.5
            * (
                frame["eeg_ep_early_occipital_alpha"]
                + frame["eeg_ep_late_occipital_alpha"]
            )
            - frame["eeg_ep_ref_occipital_alpha"]
        )
    else:
        sigs["eeg_alpha_occ"] = person.response(frame, "alpha")

    if {
        "eeg_ep_early_central_delta",
        "eeg_ep_early_parietal_delta",
        "eeg_ep_ref_central_delta",
        "eeg_ep_ref_parietal_delta",
    }.issubset(frame.columns):
        sigs["eeg_delta_cp"] = 0.5 * (
            (frame["eeg_ep_early_central_delta"] - frame["eeg_ep_ref_central_delta"])
            + (
                frame["eeg_ep_early_parietal_delta"]
                - frame["eeg_ep_ref_parietal_delta"]
            )
        )
    else:
        sigs["eeg_delta_cp"] = person.response(frame, "delta")

    if {
        "eeg_ep_early_frontal_gamma",
        "eeg_ep_late_frontal_gamma",
        "eeg_ep_ref_frontal_gamma",
    }.issubset(frame.columns):
        sigs["eeg_gamma_fr"] = (
            0.5
            * (frame["eeg_ep_early_frontal_gamma"] + frame["eeg_ep_late_frontal_gamma"])
            - frame["eeg_ep_ref_frontal_gamma"]
        )
    if {"oc_pupil_early_chg", "oc_pupil_late_chg"}.issubset(frame.columns):
        sigs["oc_pupil"] = 0.5 * (
            frame["oc_pupil_early_chg"] + frame["oc_pupil_late_chg"]
        )
    return sigs


def _boot_corr_ci(
    a: np.ndarray, b: np.ndarray, *, method: str = "pearson", draws: int = DRAWS
) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    x, y = a[ok], b[ok]
    if len(x) < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"r": None, "ci95": [None, None], "p": None, "n": int(len(x))}
    if method == "spearman":
        r_obs, p_obs = spearmanr(x, y)
    else:
        r_obs, p_obs = pearsonr(x, y)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), (draws, len(x)))
    xb, yb = x[idx], y[idx]
    if method == "spearman":
        rs = np.array(
            [
                spearmanr(xb[i], yb[i]).statistic
                for i in range(min(draws, 600))
                if np.std(xb[i]) > 0 and np.std(yb[i]) > 0
            ],
            dtype=float,
        )
    else:
        xc = xb - xb.mean(axis=1, keepdims=True)
        yc = yb - yb.mean(axis=1, keepdims=True)
        denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
        rs = np.where(denom > 1e-12, (xc * yc).sum(axis=1) / denom, np.nan)
        rs = rs[np.isfinite(rs)]
    lo, hi = np.percentile(rs, [2.5, 97.5]) if len(rs) else (np.nan, np.nan)
    return {
        "r": round(float(r_obs), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p": round(float(p_obs), 6),
        "n": int(len(x)),
    }


def _cross_day_repeatability_subset(
    sub_df: pd.DataFrame, *, draws: int = DRAWS
) -> dict:
    """Compute J1->J2 repeatability of stress reactivity, distress slope, & baseline."""
    if "recorded_day" not in sub_df.columns:
        return {"error": "recorded_day_missing"}

    sigs = _extract_interpretable_signals(sub_df)
    stressed = label(sub_df).to_numpy(bool)
    distress = sub_df["rating_arousal"].to_numpy(float) - sub_df[
        "rating_valence"
    ].to_numpy(float)
    arousal = sub_df["rating_arousal"].to_numpy(float)
    days = sub_df["recorded_day"].to_numpy()
    pids = sub_df["participant_id"].astype(str).to_numpy()

    # Participants present on both Day 1 and Day 2
    d1_pids = set(pids[days == 1])
    d2_pids = set(pids[days == 2])
    both_pids = sorted(d1_pids & d2_pids)

    reactivity: dict[str, Any] = {}
    per_sig_pairs: dict[str, dict[str, tuple[float, float]]] = {}

    for sname, svals_s in sigs.items():
        vals = svals_s.to_numpy(float)
        d_pairs, rho_dist_pairs, rho_aro_pairs = [], [], []
        pid_map: dict[str, tuple[float, float]] = {}
        for pid in both_pids:
            m1 = (pids == pid) & (days == 1) & np.isfinite(vals)
            m2 = (pids == pid) & (days == 2) & np.isfinite(vals)
            # Stress Cohen's d per day
            a1, b1 = vals[m1 & stressed], vals[m1 & ~stressed]
            a2, b2 = vals[m2 & stressed], vals[m2 & ~stressed]
            if (
                len(a1) >= MIN_EACH
                and len(b1) >= MIN_EACH
                and len(a2) >= MIN_EACH
                and len(b2) >= MIN_EACH
            ):
                s1 = np.sqrt(
                    (
                        np.var(a1, ddof=1) * (len(a1) - 1)
                        + np.var(b1, ddof=1) * (len(b1) - 1)
                    )
                    / (len(a1) + len(b1) - 2)
                )
                s2 = np.sqrt(
                    (
                        np.var(a2, ddof=1) * (len(a2) - 1)
                        + np.var(b2, ddof=1) * (len(b2) - 1)
                    )
                    / (len(a2) + len(b2) - 2)
                )
                if s1 > 1e-12 and s2 > 1e-12:
                    d_pairs.append(
                        ((a1.mean() - b1.mean()) / s1, (a2.mean() - b2.mean()) / s2)
                    )

            # Within-day Spearman rho with distress (A - V) and arousal
            if m1.sum() >= 8 and m2.sum() >= 8:
                if (
                    np.std(vals[m1]) > 1e-12
                    and np.std(vals[m2]) > 1e-12
                    and np.std(distress[m1]) > 1e-12
                    and np.std(distress[m2]) > 1e-12
                ):
                    r1 = float(spearmanr(vals[m1], distress[m1]).statistic)
                    r2 = float(spearmanr(vals[m2], distress[m2]).statistic)
                    if np.isfinite(r1) and np.isfinite(r2):
                        rho_dist_pairs.append((r1, r2))
                        pid_map[pid] = (r1, r2)
                if (
                    np.std(vals[m1]) > 1e-12
                    and np.std(vals[m2]) > 1e-12
                    and np.std(arousal[m1]) > 1e-12
                    and np.std(arousal[m2]) > 1e-12
                ):
                    ra1 = float(spearmanr(vals[m1], arousal[m1]).statistic)
                    ra2 = float(spearmanr(vals[m2], arousal[m2]).statistic)
                    if np.isfinite(ra1) and np.isfinite(ra2):
                        rho_aro_pairs.append((ra1, ra2))

        per_sig_pairs[sname] = pid_map
        dp = np.array(d_pairs, dtype=float) if d_pairs else np.empty((0, 2))
        rp = (
            np.array(rho_dist_pairs, dtype=float)
            if rho_dist_pairs
            else np.empty((0, 2))
        )
        ap = np.array(rho_aro_pairs, dtype=float) if rho_aro_pairs else np.empty((0, 2))
        reactivity[sname] = {
            "stress_d_j1_mean": round(float(dp[:, 0].mean()), 4) if len(dp) else None,
            "stress_d_j2_mean": round(float(dp[:, 1].mean()), 4) if len(dp) else None,
            "stress_d_repeatability": _boot_corr_ci(dp[:, 0], dp[:, 1], draws=draws)
            if len(dp)
            else {"r": None, "ci95": [None, None], "n": 0},
            "distress_rho_j1_mean": round(float(rp[:, 0].mean()), 4)
            if len(rp)
            else None,
            "distress_rho_j2_mean": round(float(rp[:, 1].mean()), 4)
            if len(rp)
            else None,
            "distress_rho_repeatability": _boot_corr_ci(rp[:, 0], rp[:, 1], draws=draws)
            if len(rp)
            else {"r": None, "ci95": [None, None], "n": 0},
            "arousal_rho_repeatability": _boot_corr_ci(ap[:, 0], ap[:, 1], draws=draws)
            if len(ap)
            else {"r": None, "ci95": [None, None], "n": 0},
        }

    # Direct bootstrap comparison r(ECG_HR) - r(EDA_SC) on matched multi-day people
    paired_stability: dict[str, Any] = {}
    for s_a, s_b in (
        ("ecg_hr", "eda_sc"),
        ("eeg_alpha_occ", "eda_sc"),
        ("ecg_hr", "resp_amp"),
    ):
        ma, mb = per_sig_pairs.get(s_a, {}), per_sig_pairs.get(s_b, {})
        common_p = sorted(set(ma) & set(mb))
        if len(common_p) >= 15:
            a1 = np.array([ma[p][0] for p in common_p])
            a2 = np.array([ma[p][1] for p in common_p])
            b1 = np.array([mb[p][0] for p in common_p])
            b2 = np.array([mb[p][1] for p in common_p])
            obs = float(pearsonr(a1, a2)[0] - pearsonr(b1, b2)[0])
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(common_p), (draws, len(common_p)))
            diffs = []
            for row in idx:
                if (
                    np.std(a1[row]) > 0
                    and np.std(a2[row]) > 0
                    and np.std(b1[row]) > 0
                    and np.std(b2[row]) > 0
                ):
                    diffs.append(
                        float(
                            np.corrcoef(a1[row], a2[row])[0, 1]
                            - np.corrcoef(b1[row], b2[row])[0, 1]
                        )
                    )
            lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
            paired_stability[f"{s_a}_minus_{s_b}_distress_r"] = {
                "diff": round(obs, 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "n_people": len(common_p),
            }

    # Tonic baseline stability J1 -> J2 (resting/pre-stimulus levels)
    baseline_cols = {
        "ecg_hr_ref": "ecg_hr_ref",
        "eda_sc_ref": "sc_mean_ref",
        "resp_amp_ref": "resp_amp_ref",
        "eeg_alpha_occ_ref": "eeg_ep_ref_occipital_alpha",
        "eeg_gamma_fr_ref": "eeg_ep_ref_frontal_gamma",
        "subjective_stress_rate": None,
    }
    baseline_stab: dict[str, Any] = {}
    for bname, col in baseline_cols.items():
        arr = (
            stressed.astype(float)
            if col is None
            else (sub_df[col].to_numpy(float) if col in sub_df.columns else None)
        )
        if arr is None:
            continue
        p1, p2 = [], []
        for pid in both_pids:
            m1 = (pids == pid) & (days == 1) & np.isfinite(arr)
            m2 = (pids == pid) & (days == 2) & np.isfinite(arr)
            if m1.sum() >= 5 and m2.sum() >= 5:
                p1.append(float(arr[m1].mean()))
                p2.append(float(arr[m2].mean()))
        if p1:
            baseline_stab[bname] = _boot_corr_ci(
                np.array(p1), np.array(p2), draws=draws
            )

    return {
        "n_multi_day_people": len(both_pids),
        "reactivity_repeatability": reactivity,
        "paired_repeatability_contrasts": paired_stability,
        "tonic_baseline_stability": baseline_stab,
    }


def _within_session_habituation(sub_df: pd.DataFrame, *, draws: int = DRAWS) -> dict:
    """Test chronological habituation across session terciles T1, T2, T3."""
    sigs = _extract_interpretable_signals(sub_df)
    sigs["subjective_distress"] = sub_df["rating_arousal"] - sub_df["rating_valence"]
    sort_cols = [c for c in ("event_order", "onset_s") if c in sub_df.columns]
    if not sort_cols:
        return {"error": "no_order_columns"}

    # Assign within-session terciles (0=T1, 1=T2, 2=T3) & normalized rank [0, 1]
    sess_keys = (
        ["participant_id", "recorded_day"]
        if "recorded_day" in sub_df.columns
        else ["participant_id"]
    )
    order_df = sub_df[[*sess_keys, *sort_cols]].copy()
    order_df["_idx"] = np.arange(len(sub_df))
    order_df = order_df.sort_values([*sess_keys, *sort_cols, "_idx"], kind="mergesort")
    tercile = np.full(len(sub_df), -1, dtype=int)
    norm_rank = np.full(len(sub_df), np.nan, dtype=float)
    for _, grp in order_df.groupby(sess_keys, sort=False):
        m = len(grp)
        if m < 15:
            continue
        pos = grp["_idx"].to_numpy(int)
        ranks = np.arange(m, dtype=float) / max(m - 1, 1)
        norm_rank[pos] = ranks
        tercile[pos] = np.minimum((np.arange(m) * 3) // m, 2)

    stressed = label(sub_df).to_numpy(bool)
    pids = sub_df["participant_id"].astype(str).to_numpy()
    out: dict[str, Any] = {}

    for sname, svals in sigs.items():
        vals = svals.to_numpy(float)
        # Video-consensus adjusted signal (removes any video-order confound!)
        _, _, vals_idio_s = person.components(sub_df, vals)
        vals_idio = vals_idio_s.to_numpy(float)

        # Tercile mean response on stressed trials (and stress Cohen's d per tercile)
        t_means: dict[int, dict[str, float]] = {0: {}, 1: {}, 2: {}}
        t_idio_stress: dict[int, dict[str, float]] = {0: {}, 1: {}, 2: {}}
        rank_slopes_all, rank_slopes_stress = [], []

        for pid in np.unique(pids):
            pm = (pids == pid) & (tercile >= 0) & np.isfinite(vals)
            if pm.sum() < 15:
                continue
            for t_i in (0, 1, 2):
                mt = pm & (tercile == t_i) & stressed
                if mt.sum() >= 2:
                    t_means[t_i][pid] = float(vals[mt].mean())
                    if np.isfinite(vals_idio[mt]).sum() >= 2:
                        t_idio_stress[t_i][pid] = float(np.nanmean(vals_idio[mt]))
            # Within-person rank correlation with video-adjusted response
            ok_i = pm & np.isfinite(vals_idio) & np.isfinite(norm_rank)
            if ok_i.sum() >= 15 and np.std(vals_idio[ok_i]) > 1e-12:
                r_all = float(spearmanr(norm_rank[ok_i], vals_idio[ok_i]).statistic)
                if np.isfinite(r_all):
                    rank_slopes_all.append(r_all)
            ok_s = ok_i & stressed
            if ok_s.sum() >= 6 and np.std(vals_idio[ok_s]) > 1e-12:
                r_st = float(spearmanr(norm_rank[ok_s], vals_idio[ok_s]).statistic)
                if np.isfinite(r_st):
                    rank_slopes_stress.append(r_st)

        out[sname] = {
            "stressed_mean_T1": ssl_eval.boot_ci(
                list(t_means[0].values()), draws=draws
            ),
            "stressed_mean_T2": ssl_eval.boot_ci(
                list(t_means[1].values()), draws=draws
            ),
            "stressed_mean_T3": ssl_eval.boot_ci(
                list(t_means[2].values()), draws=draws
            ),
            "paired_T3_minus_T1_raw": ssl_eval.paired_gain_ci(
                t_means[2], t_means[0], draws=draws
            ),
            "paired_T3_minus_T1_video_adj": ssl_eval.paired_gain_ci(
                t_idio_stress[2], t_idio_stress[0], draws=draws
            ),
            "within_person_rank_rho_all_video_adj": ssl_eval.boot_ci(
                rank_slopes_all, draws=draws
            ),
            "within_person_rank_rho_stressed_video_adj": ssl_eval.boot_ci(
                rank_slopes_stress, draws=draws
            ),
        }
    return out


def module_10b_longitudinal_and_habituation(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = DRAWS,
) -> dict:
    """H10.2: J1->J2 repeatability, cross-day decoder transfer, & habituation."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    feat_dev = _build_s07_feature_df(dev_df)
    mod_cols = _modality_columns(feat_dev)
    x_reve_full = _get_reve_vid10(aux)
    x_reve_dev = x_reve_full[dev_m] if x_reve_full is not None else None

    # 1. Cross-day physiological repeatability (dev + full 314 descriptive)
    rep_dev = _cross_day_repeatability_subset(dev_df, draws=draws)
    rep_all = _cross_day_repeatability_subset(frame.reset_index(drop=True), draws=draws)

    # 2. Cross-day decoder transfer (Train Day 1 -> Test Day 2 unseen themes in dev)
    transfer: dict[str, Any] = {}
    if "recorded_day" in dev_df.columns:
        pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
        d1 = (dev_df["recorded_day"] == 1).to_numpy()
        d2 = (dev_df["recorded_day"] == 2).to_numpy()
        d1_pids = set(dev_df.loc[d1, "participant_id"].astype(str))
        in_d1_mask = dev_df["participant_id"].astype(str).isin(d1_pids).to_numpy()

        modalities: dict[str, np.ndarray] = {
            "S07_enriched": feat_dev[mod_cols["All"]].to_numpy(np.float32),
            "ECG_only": feat_dev[mod_cols["ECG_only"]].to_numpy(np.float32),
            "EDA_only": feat_dev[mod_cols["EDA_only"]].to_numpy(np.float32),
            "Resp_only": feat_dev[mod_cols["Resp_only"]].to_numpy(np.float32),
            "EEG_hc_only": feat_dev[mod_cols["EEG_hc_only"]].to_numpy(np.float32),
        }
        if x_reve_dev is not None:
            modalities["REVE_frozen"] = x_reve_dev

        for mname, xm in modalities.items():
            # (a) Within-Day-1 reference (3x3 blocked on Day 1 only)
            sc_d1 = np.full(len(dev_df), np.nan, dtype=np.float32)
            # (b) Train Day 1 -> Test Day 2 (unseen people AND unseen themes)
            sc_d1_to_d2_newp = np.full(len(dev_df), np.nan, dtype=np.float32)
            # (c) Train Day 1 -> Test Day 2 (SAME people, unseen themes)
            sc_d1_to_d2_samep = np.full(len(dev_df), np.nan, dtype=np.float32)

            mpca = min(64, xm.shape[1])
            for p in range(3):
                for t in range(3):
                    tr_d1 = d1 & (pf != p) & (tf != t)
                    ev_d1 = d1 & (pf == p) & (tf == t)
                    ev_d2_np = d2 & (pf == p) & (tf == t)
                    if tr_d1.sum() < 40:
                        continue
                    ev_union = ev_d1 | ev_d2_np
                    if ev_union.sum() == 0:
                        continue
                    pr = _fit_predict_stress_fast(
                        xm[tr_d1], dev_df.iloc[tr_d1], xm[ev_union], max_pca=mpca
                    )["stress_va"]
                    u_idx = np.where(ev_union)[0]
                    sc_d1[u_idx[ev_d1[u_idx]]] = pr[ev_d1[u_idx]]
                    sc_d1_to_d2_newp[u_idx[ev_d2_np[u_idx]]] = pr[ev_d2_np[u_idx]]

            for t in range(3):
                tr_d1_all = d1 & (tf != t)
                ev_d2_sp = d2 & in_d1_mask & (tf == t)
                if tr_d1_all.sum() < 40 or ev_d2_sp.sum() == 0:
                    continue
                sc_d1_to_d2_samep[ev_d2_sp] = _fit_predict_stress_fast(
                    xm[tr_d1_all], dev_df.iloc[tr_d1_all], xm[ev_d2_sp], max_pca=mpca
                )["stress_va"]

            # Evaluate on Day 1 and Day 2 subsets + paired J2 - J1 change
            res_d1 = ssl_eval.evaluate_stress(
                dev_df[d1],
                pd.Series(sc_d1[d1], index=dev_df.index[d1]),
                draws=draws,
                return_per_person=True,
            )
            pp_d1 = res_d1.pop("_per_person")
            res_d2_np = ssl_eval.evaluate_stress(
                dev_df[d2],
                pd.Series(sc_d1_to_d2_newp[d2], index=dev_df.index[d2]),
                draws=draws,
                return_per_person=True,
            )
            pp_d2_np = res_d2_np.pop("_per_person")
            res_d2_sp = ssl_eval.evaluate_stress(
                dev_df[d2 & in_d1_mask],
                pd.Series(
                    sc_d1_to_d2_samep[d2 & in_d1_mask],
                    index=dev_df.index[d2 & in_d1_mask],
                ),
                draws=draws,
                return_per_person=True,
            )
            pp_d2_sp = res_d2_sp.pop("_per_person")

            transfer[mname] = {
                "within_day1_3x3": res_d1["stress"],
                "j1_to_j2_new_people_unseen_themes": res_d2_np["stress"],
                "j1_to_j2_same_people_unseen_themes": res_d2_sp["stress"],
                "paired_j2_newp_minus_j1": ssl_eval.paired_gain_ci(
                    pp_d2_np, pp_d1, draws=draws
                ),
                "paired_j2_samep_minus_j2_newp": ssl_eval.paired_gain_ci(
                    pp_d2_sp, pp_d2_np, draws=draws
                ),
            }

    # 3. Within-session habituation (full 314 + dev)
    hab_all = _within_session_habituation(frame.reset_index(drop=True), draws=draws)
    hab_dev = _within_session_habituation(dev_df, draws=draws)

    rr_all = rep_all.get("reactivity_repeatability", {})
    hr_r = rr_all.get("ecg_hr", {}).get("distress_rho_repeatability", {}).get("r")
    eda_r = rr_all.get("eda_sc", {}).get("distress_rho_repeatability", {}).get("r")
    print(
        f"  10B Multi-day people (dev/all): {rep_dev.get('n_multi_day_people')}/"
        f"{rep_all.get('n_multi_day_people')}; "
        f"HR r(J1,J2)={hr_r}, EDA r(J1,J2)={eda_r}",
        flush=True,
    )
    return {
        "repeatability_dev": rep_dev,
        "repeatability_all_314": rep_all,
        "cross_day_decoder_transfer_dev": transfer,
        "within_session_habituation_all_314": hab_all,
        "within_session_habituation_dev": hab_dev,
    }


# ---------------------------------------------------------------------------
# MODULE 10C: Unsupervised CORAL, k-shot Personalization & LOCO-CV
# ---------------------------------------------------------------------------


def _eb_map_ridge_predict(
    z_cal: np.ndarray,
    y_cal: np.ndarray,
    z_eval: np.ndarray,
    w_pop: np.ndarray,
    xm_pop: np.ndarray,
    prior_lambda: float,
) -> np.ndarray:
    """Bayesian MAP Ridge adaptation with Gaussian prior centered at ``w_pop``."""
    zc = z_cal - xm_pop[None, :]
    ze = z_eval - xm_pop[None, :]
    yc = y_cal - float(np.mean(y_cal))
    resid = yc - zc @ w_pop
    d = zc.shape[1]
    a_mat = zc.T @ zc + prior_lambda * np.eye(d)
    delta_w = np.linalg.solve(a_mat, zc.T @ resid)
    return ze @ (w_pop + delta_w)


def _naive_personal_ridge_predict(
    z_cal: np.ndarray,
    y_cal: np.ndarray,
    z_eval: np.ndarray,
    alpha: float = 10.0,
) -> np.ndarray:
    """Naive personal Ridge fitted exclusively on k labeled trials (Khan 2026)."""
    if len(z_cal) < 3 or np.std(y_cal) < 1e-12:
        return np.zeros(len(z_eval), dtype=float)
    ym, xm, w = _svd_ridge_coef(z_cal, y_cal, alpha=alpha)
    return ym + (z_eval - xm[None, :]) @ w


def _run_kshot_personalization(
    dev_df: pd.DataFrame,
    x_mat: np.ndarray,
    *,
    max_pca: int = 64,
    min_trials: int = 55,
    eval_start: int = 40,
    k_grid: tuple[int, ...] = K_SHOT_GRID,
    eb_lambda: float = 1000.0,
    draws: int = DRAWS,
) -> dict:
    """Strictly chronological k-shot personalization curve on common future tail."""
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    n = len(dev_df)
    sort_cols = [
        c for c in ("recorded_day", "event_order", "onset_s") if c in dev_df.columns
    ]

    methods = (
        "cold_start",
        "unsup_b1_center",
        "unsup_coral",
        "sup_naive_personal",
        "sup_eb_map_shrunk",
    )
    scores: dict[int, dict[str, np.ndarray]] = {
        k: {m: np.full(n, np.nan, dtype=np.float32) for m in methods} for k in k_grid
    }

    yv_all = dev_df["rating_valence"].to_numpy(float)
    ya_all = dev_df["rating_arousal"].to_numpy(float)
    order_df = dev_df[["participant_id", *sort_cols]].copy()
    order_df["_pos"] = np.arange(n)
    order_df = order_df.sort_values(
        ["participant_id", *sort_cols, "_pos"], kind="mergesort"
    )
    person_chron: dict[str, np.ndarray] = {
        str(pid): grp["_pos"].to_numpy(int)
        for pid, grp in order_df.groupby("participant_id", sort=False)
    }

    eligible_people = [p for p, idxs in person_chron.items() if len(idxs) >= min_trials]
    eval_mask_global = np.zeros(n, dtype=bool)
    for pid in eligible_people:
        eval_mask_global[person_chron[pid][eval_start:]] = True

    # Fit each (p, t) outer population model ONCE and project all rows of pf==p
    for p in range(3):
        p_people = [pid for pid in eligible_people if pf[person_chron[pid][0]] == p]
        if not p_people:
            continue
        p_rows_mask = pf == p
        p_rows_idx = np.where(p_rows_mask)[0]
        pos_to_local = {int(g): i for i, g in enumerate(p_rows_idx)}

        for t in range(3):
            tr_m = (pf != p) & (tf != t)
            if tr_m.sum() < 40:
                continue
            res = _fit_predict_stress_fast(
                x_mat[tr_m],
                dev_df.iloc[tr_m],
                x_mat[p_rows_mask],
                max_pca=max_pca,
                return_params=True,
            )
            par = res["params"]
            z_tr_f, z_p_all = par["z_tr"], par["z_ev"]
            xm, w_v, w_a = par["xm"], par["w_v"], par["w_a"]
            sd_v, sd_a = par["sd_v"], par["sd_a"]

            for pid in p_people:
                chron = person_chron[pid]
                tail = chron[eval_start:]
                # Strictly evaluate only tail trials on unseen theme fold == t
                tail_t = tail[tf[tail] == t]
                if len(tail_t) == 0:
                    continue
                loc_tail = np.array([pos_to_local[int(g)] for g in tail_t], dtype=int)
                z_tail = z_p_all[loc_tail]

                pv_cs = (z_tail - xm[None, :]) @ w_v
                pa_cs = (z_tail - xm[None, :]) @ w_a
                sc_cs = pa_cs / sd_a - pv_cs / sd_v

                for k in k_grid:
                    scores[k]["cold_start"][tail_t] = sc_cs
                    if k == 0:
                        for m in methods[1:]:
                            scores[k][m][tail_t] = sc_cs
                        continue

                    cal_idx = chron[:k]
                    loc_cal = np.array(
                        [pos_to_local[int(g)] for g in cal_idx], dtype=int
                    )
                    z_cal = z_p_all[loc_cal]

                    # 1. Unsupervised B1 shrunk centering on k unlabeled past trials
                    mu_hat = (k * z_cal.mean(axis=0) + 5.0 * xm) / (k + 5.0)
                    z_tail_b1 = z_tail - (mu_hat - xm)[None, :]
                    scores[k]["unsup_b1_center"][tail_t] = (
                        (z_tail_b1 - xm[None, :]) @ w_a
                    ) / sd_a - ((z_tail_b1 - xm[None, :]) @ w_v) / sd_v

                    # 2. Unsupervised shrunk CORAL on k unlabeled past trials
                    z_tail_cor = coral_align_target(
                        z_tr_f,
                        z_cal,
                        z_tail,
                        reg=1e-1,
                        mean_shrink=5.0,
                        cov_shrink=25.0,
                    )
                    scores[k]["unsup_coral"][tail_t] = (
                        (z_tail_cor - xm[None, :]) @ w_a
                    ) / sd_a - ((z_tail_cor - xm[None, :]) @ w_v) / sd_v

                    # 3. Supervised Naive Personal Ridge on k labeled past trials
                    # Zero eval-batch stats: uses training-fold (sd_v, sd_a) only!
                    yv_cal = yv_all[cal_idx]
                    ya_cal = ya_all[cal_idx]
                    pv_nv = _naive_personal_ridge_predict(
                        z_cal, yv_cal, z_tail, alpha=10.0
                    )
                    pa_nv = _naive_personal_ridge_predict(
                        z_cal, ya_cal, z_tail, alpha=10.0
                    )
                    scores[k]["sup_naive_personal"][tail_t] = (
                        pa_nv / sd_a - pv_nv / sd_v
                    )

                    # 4. Supervised Empirical-Bayes MAP shrunk update around w_pop
                    pv_eb = _eb_map_ridge_predict(
                        z_cal, yv_cal, z_tail, w_v, xm, prior_lambda=eb_lambda
                    )
                    pa_eb = _eb_map_ridge_predict(
                        z_cal, ya_cal, z_tail, w_a, xm, prior_lambda=eb_lambda
                    )
                    scores[k]["sup_eb_map_shrunk"][tail_t] = pa_eb / sd_a - pv_eb / sd_v

    tail_df = dev_df[eval_mask_global].reset_index(drop=True)
    pos_m = label(tail_df)
    neg_m = ~pos_m
    curve: dict[str, Any] = {}
    pp_cs_ref = ssl_eval.person_stress_aucs(
        tail_df,
        pd.Series(scores[0]["cold_start"][eval_mask_global], index=tail_df.index),
        pos_m,
        neg_m,
    )
    for k in k_grid:
        entry: dict[str, Any] = {}
        for m in methods:
            sc_sub = pd.Series(scores[k][m][eval_mask_global], index=tail_df.index)
            pmap = ssl_eval.person_stress_aucs(tail_df, sc_sub, pos_m, neg_m)
            ci = ssl_eval.boot_ci(list(pmap.values()), draws=draws)
            gain = ssl_eval.paired_gain_ci(pmap, pp_cs_ref, draws=draws)
            entry[m] = {
                "auc": ci["mean"],
                "ci95": ci["ci95"],
                "people": ci["n"],
                "gain_vs_cold_start": gain,
            }
        curve[str(k)] = entry

    return {
        "n_eligible_people": len(eligible_people),
        "eval_start_trial": eval_start,
        "n_common_tail_trials": int(eval_mask_global.sum()),
        "eb_prior_lambda": eb_lambda,
        "k_curve": curve,
    }


def _run_j1_to_j2_calibration(
    dev_df: pd.DataFrame,
    x_mat: np.ndarray,
    *,
    max_pca: int = 64,
    eb_lambda: float = 1000.0,
    draws: int = DRAWS,
) -> dict:
    """Full Day-1 calibration -> Day-2 unseen-theme evaluation on multi-day people."""
    if "recorded_day" not in dev_df.columns:
        return {"error": "recorded_day_missing"}
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    d1 = (dev_df["recorded_day"] == 1).to_numpy()
    d2 = (dev_df["recorded_day"] == 2).to_numpy()
    pids = dev_df["participant_id"].astype(str).to_numpy()
    both_pids = sorted(set(pids[d1]) & set(pids[d2]))
    if len(both_pids) < 10:
        return {"error": "too_few_multiday_people", "n": len(both_pids)}

    methods = (
        "cold_start",
        "j1_unsup_b1_center",
        "j1_unsup_coral",
        "j1_sup_naive_personal",
        "j1_sup_eb_map_shrunk",
    )
    n = len(dev_df)
    scores = {m: np.full(n, np.nan, dtype=np.float32) for m in methods}
    yv_all = dev_df["rating_valence"].to_numpy(float)
    ya_all = dev_df["rating_arousal"].to_numpy(float)

    for p in range(3):
        p_people = [pid for pid in both_pids if pf[np.where(pids == pid)[0][0]] == p]
        if not p_people:
            continue
        p_mask = pf == p
        p_idx = np.where(p_mask)[0]
        pos_map = {int(g): i for i, g in enumerate(p_idx)}

        for t in range(3):
            tr_m = (pf != p) & (tf != t)
            if tr_m.sum() < 40:
                continue
            res = _fit_predict_stress_fast(
                x_mat[tr_m],
                dev_df.iloc[tr_m],
                x_mat[p_mask],
                max_pca=max_pca,
                return_params=True,
            )
            par = res["params"]
            z_tr_f, z_p_all = par["z_tr"], par["z_ev"]
            xm, w_v, w_a = par["xm"], par["w_v"], par["w_a"]
            sd_v, sd_a = par["sd_v"], par["sd_a"]

            for pid in p_people:
                cal_idx = np.where((pids == pid) & d1)[0]
                ev_idx = np.where((pids == pid) & d2 & (tf == t))[0]
                if len(cal_idx) < 5 or len(ev_idx) == 0:
                    continue
                z_cal = z_p_all[[pos_map[int(g)] for g in cal_idx]]
                z_ev = z_p_all[[pos_map[int(g)] for g in ev_idx]]
                k = len(cal_idx)

                sc_cs = ((z_ev - xm[None, :]) @ w_a) / sd_a - (
                    (z_ev - xm[None, :]) @ w_v
                ) / sd_v
                scores["cold_start"][ev_idx] = sc_cs

                mu_hat = (k * z_cal.mean(axis=0) + 5.0 * xm) / (k + 5.0)
                z_ev_b1 = z_ev - (mu_hat - xm)[None, :]
                scores["j1_unsup_b1_center"][ev_idx] = (
                    (z_ev_b1 - xm[None, :]) @ w_a
                ) / sd_a - ((z_ev_b1 - xm[None, :]) @ w_v) / sd_v

                z_ev_cor = coral_align_target(
                    z_tr_f, z_cal, z_ev, reg=1e-1, mean_shrink=5.0, cov_shrink=25.0
                )
                scores["j1_unsup_coral"][ev_idx] = (
                    (z_ev_cor - xm[None, :]) @ w_a
                ) / sd_a - ((z_ev_cor - xm[None, :]) @ w_v) / sd_v

                yv_c, ya_c = yv_all[cal_idx], ya_all[cal_idx]
                pv_n = _naive_personal_ridge_predict(z_cal, yv_c, z_ev, alpha=10.0)
                pa_n = _naive_personal_ridge_predict(z_cal, ya_c, z_ev, alpha=10.0)
                scores["j1_sup_naive_personal"][ev_idx] = pa_n / sd_a - pv_n / sd_v

                pv_e = _eb_map_ridge_predict(z_cal, yv_c, z_ev, w_v, xm, eb_lambda)
                pa_e = _eb_map_ridge_predict(z_cal, ya_c, z_ev, w_a, xm, eb_lambda)
                scores["j1_sup_eb_map_shrunk"][ev_idx] = pa_e / sd_a - pv_e / sd_v

    eval_m = d2 & np.isfinite(scores["cold_start"])
    d2_df = dev_df[eval_m].reset_index(drop=True)
    pos_m = label(d2_df)
    neg_m = ~pos_m
    pp_cs = ssl_eval.person_stress_aucs(
        d2_df, pd.Series(scores["cold_start"][eval_m], index=d2_df.index), pos_m, neg_m
    )
    out: dict[str, Any] = {
        "n_multiday_people_eval": len(pp_cs),
        "n_day2_trials": int(eval_m.sum()),
        "methods": {},
    }
    for m in methods:
        sc_s = pd.Series(scores[m][eval_m], index=d2_df.index)
        pmap = ssl_eval.person_stress_aucs(d2_df, sc_s, pos_m, neg_m)
        ci = ssl_eval.boot_ci(list(pmap.values()), draws=draws)
        out["methods"][m] = {
            "auc": ci["mean"],
            "ci95": ci["ci95"],
            "people": ci["n"],
            "gain_vs_cold_start": ssl_eval.paired_gain_ci(pmap, pp_cs, draws=draws),
        }
    return out


def _run_loco_cv(
    dev_df: pd.DataFrame,
    x_s07: np.ndarray,
    x_reve: np.ndarray | None,
    x_body05: np.ndarray | None,
    x_reve_fcast: np.ndarray | None,
    *,
    draws: int = DRAWS,
) -> dict:
    """Leave-One-Cohort-Out (LOCO-CV) across Cohorts 1-5: Raw vs Unsupervised CORAL."""
    if "cohort" not in dev_df.columns:
        return {"error": "cohort_missing"}
    cohorts = sorted(int(c) for c in dev_df["cohort"].dropna().unique())
    coh_arr = dev_df["cohort"].to_numpy()
    n = len(dev_df)

    # 1. Stress detection LOCO (Raw vs Unsupervised CORAL)
    stress_loco: dict[str, Any] = {}
    for mname, xm in (("S07_enriched", x_s07), ("REVE_frozen", x_reve)):
        if xm is None:
            continue
        oof_raw = np.full(n, np.nan, dtype=np.float32)
        oof_cor = np.full(n, np.nan, dtype=np.float32)
        oof_lr_raw = np.full(n, np.nan, dtype=np.float32)
        oof_lr_cor = np.full(n, np.nan, dtype=np.float32)
        by_coh: dict[str, Any] = {}

        for c in cohorts:
            ev_m = coh_arr == c
            tr_m = coh_arr != c
            if ev_m.sum() < 30 or tr_m.sum() < 50:
                continue
            pr_raw = _fit_predict_stress_fast(
                xm[tr_m], dev_df.iloc[tr_m], xm[ev_m], max_pca=64, coral_eval=False
            )
            pr_cor = _fit_predict_stress_fast(
                xm[tr_m], dev_df.iloc[tr_m], xm[ev_m], max_pca=64, coral_eval=True
            )
            oof_raw[ev_m] = pr_raw["stress_va"]
            oof_cor[ev_m] = pr_cor["stress_va"]
            oof_lr_raw[ev_m] = pr_raw["stress_logreg"]
            oof_lr_cor[ev_m] = pr_cor["stress_logreg"]

            df_c = dev_df[ev_m].reset_index(drop=True)
            s_raw = ssl_eval.evaluate_stress(
                df_c, pd.Series(pr_raw["stress_va"], index=df_c.index), draws=draws
            )
            s_cor = ssl_eval.evaluate_stress(
                df_c, pd.Series(pr_cor["stress_va"], index=df_c.index), draws=draws
            )
            by_coh[f"cohort_{c}"] = {
                "n_trials": int(ev_m.sum()),
                "n_people": int(df_c["participant_id"].nunique()),
                "raw_va_wp_auc": s_raw["stress"]["auc"],
                "raw_va_pooled_auc": s_raw["pooled_auc"],
                "coral_va_wp_auc": s_cor["stress"]["auc"],
                "coral_va_pooled_auc": s_cor["pooled_auc"],
            }

        res_raw = ssl_eval.evaluate_stress(
            dev_df,
            pd.Series(oof_raw, index=dev_df.index),
            draws=draws,
            return_per_person=True,
        )
        pp_raw = res_raw.pop("_per_person")
        res_cor = ssl_eval.evaluate_stress(
            dev_df,
            pd.Series(oof_cor, index=dev_df.index),
            draws=draws,
            return_per_person=True,
        )
        pp_cor = res_cor.pop("_per_person")
        res_lr_raw = ssl_eval.evaluate_stress(
            dev_df, pd.Series(oof_lr_raw, index=dev_df.index), draws=draws
        )
        res_lr_cor = ssl_eval.evaluate_stress(
            dev_df, pd.Series(oof_lr_cor, index=dev_df.index), draws=draws
        )
        stress_loco[mname] = {
            "overall_raw_va": res_raw,
            "overall_coral_va": res_cor,
            "paired_coral_minus_raw_va": ssl_eval.paired_gain_ci(
                pp_cor, pp_raw, draws=draws
            ),
            "overall_raw_logreg": res_lr_raw["stress"],
            "overall_coral_logreg": res_lr_cor["stress"],
            "by_cohort": by_coh,
        }

    # 2. Study-09 Causal Cardiac Forecasting LOCO (M0 vs M2_reve_resid, Raw vs CORAL)
    fcast_loco: dict[str, Any] = {}
    tgt = "tgt_ecg_hr_chg_5_10"
    if x_body05 is not None and x_reve_fcast is not None and tgt in dev_df.columns:
        y = dev_df[tgt].to_numpy(float)
        p0_raw = np.full(n, np.nan)
        p2_raw = np.full(n, np.nan)
        p0_cor = np.full(n, np.nan)
        p2_cor = np.full(n, np.nan)
        by_coh_f: dict[str, Any] = {}

        for c in cohorts:
            ev_m = coh_arr == c
            tr_m = coh_arr != c
            if ev_m.sum() < 30 or tr_m.sum() < 50:
                continue
            a0, a2, _ = study09_protocol.fit_predict_m2_residual(
                x_body05[tr_m],
                x_reve_fcast[tr_m],
                y[tr_m],
                dev_df.iloc[tr_m],
                x_body05[ev_m],
                x_reve_fcast[ev_m],
                eeg_pca_grid=(16, 32, 64),
            )
            p0_raw[ev_m], p2_raw[ev_m] = a0, a2

            # CORAL-aligned target body and EEG (unsupervised target cohort alignment!)
            zb_tr, zb_ev = study09_protocol._preprocess_fit_transform(
                x_body05[tr_m], x_body05[ev_m], max_pca=None
            )
            zb_ev_cor = coral_align_target(zb_tr, zb_ev, zb_ev, reg=1e-2)
            ze_tr, ze_ev = study09_protocol._preprocess_fit_transform(
                x_reve_fcast[tr_m], x_reve_fcast[ev_m], max_pca=64
            )
            ze_ev_cor = coral_align_target(ze_tr, ze_ev, ze_ev, reg=1e-2)
            c0, c2, _ = study09_protocol.fit_predict_m2_residual(
                zb_tr,
                ze_tr,
                y[tr_m],
                dev_df.iloc[tr_m],
                zb_ev_cor,
                ze_ev_cor,
                eeg_pca_grid=(16, 32, None),
            )
            p0_cor[ev_m], p2_cor[ev_m] = c0, c2

            df_c = dev_df[ev_m].reset_index(drop=True)
            s0c, pp0c = study09_protocol.summarize_forecast(
                df_c, y[ev_m], a0, draws=draws
            )
            s2c, pp2c = study09_protocol.summarize_forecast(
                df_c, y[ev_m], a2, draws=draws
            )
            s2cc, pp2cc = study09_protocol.summarize_forecast(
                df_c, y[ev_m], c2, draws=draws
            )
            by_coh_f[f"cohort_{c}"] = {
                "n_people": s0c["n_people"],
                "M0_raw_r": s0c["r"]["mean"],
                "M2_raw_r": s2c["r"]["mean"],
                "delta_M2_minus_M0_raw": study09_protocol.paired_forecast_comparison(
                    pp2c, pp0c, draws=draws
                ).get("delta_r"),
                "M2_coral_r": s2cc["r"]["mean"],
            }

        s0_all, pp0_all = study09_protocol.summarize_forecast(
            dev_df, y, p0_raw, draws=draws
        )
        s2_all, pp2_all = study09_protocol.summarize_forecast(
            dev_df, y, p2_raw, draws=draws
        )
        s0_cor, pp0_cor = study09_protocol.summarize_forecast(
            dev_df, y, p0_cor, draws=draws
        )
        s2_cor, pp2_cor = study09_protocol.summarize_forecast(
            dev_df, y, p2_cor, draws=draws
        )
        pfc = study09_protocol.paired_forecast_comparison
        fcast_loco = {
            "target": tgt,
            "M0_raw": s0_all["r"],
            "M2_raw": s2_all["r"],
            "paired_M2_minus_M0_raw": pfc(pp2_all, pp0_all, draws=draws),
            "M0_coral": s0_cor["r"],
            "M2_coral": s2_cor["r"],
            "paired_M2_minus_M0_coral": pfc(pp2_cor, pp0_cor, draws=draws),
            "paired_M2_coral_minus_raw": pfc(pp2_cor, pp2_all, draws=draws),
            "by_cohort": by_coh_f,
        }

    return {"stress_loco": stress_loco, "cardiac_forecast_loco": fcast_loco}


def module_10c_coral_kshot_loco(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = DRAWS,
) -> dict:
    """H10.3: Chronological k-shot, J1->J2 calibration, and LOCO-CV + CORAL."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    feat_dev = _build_s07_feature_df(dev_df)
    x_s07 = feat_dev.to_numpy(np.float32)

    x_reve_full = _get_reve_vid10(aux)
    x_reve = x_reve_full[dev_m] if x_reve_full is not None else None
    bcols = study09_analysis._body05_cols(frame)
    x_body05 = dev_df[bcols].to_numpy(np.float32) if bcols else None
    x_reve_fc_full = _get_reve_causal_forecast(aux)
    x_reve_fc = x_reve_fc_full[dev_m] if x_reve_fc_full is not None else None

    kshot: dict[str, Any] = {
        "S07_enriched": _run_kshot_personalization(
            dev_df, x_s07, min_trials=55, eval_start=40, k_grid=K_SHOT_GRID, draws=draws
        ),
        "S07_enriched_k30_highN": _run_kshot_personalization(
            dev_df,
            x_s07,
            min_trials=45,
            eval_start=30,
            k_grid=K_SHOT_GRID_30,
            draws=draws,
        ),
    }
    j1_j2_cal: dict[str, Any] = {
        "S07_enriched": _run_j1_to_j2_calibration(dev_df, x_s07, draws=draws)
    }
    if x_reve is not None:
        kshot["REVE_frozen"] = _run_kshot_personalization(
            dev_df,
            x_reve,
            min_trials=55,
            eval_start=40,
            k_grid=K_SHOT_GRID,
            draws=draws,
        )
        j1_j2_cal["REVE_frozen"] = _run_j1_to_j2_calibration(
            dev_df, x_reve, draws=draws
        )

    loco = _run_loco_cv(dev_df, x_s07, x_reve, x_body05, x_reve_fc, draws=draws)

    kc = kshot["S07_enriched"]["k_curve"]
    print(
        f"  10C k-shot S07 (k=0/10/20/40): "
        f"CS={kc['0']['cold_start']['auc']}, "
        f"Naive10={kc['10']['sup_naive_personal']['auc']}, "
        f"EB10={kc['10']['sup_eb_map_shrunk']['auc']}, "
        f"EB40={kc['40']['sup_eb_map_shrunk']['auc']}, "
        f"CORAL40={kc['40']['unsup_coral']['auc']}",
        flush=True,
    )
    return {
        "kshot_chronological_dev": kshot,
        "j1_to_j2_calibration_dev": j1_j2_cal,
        "loco_cross_cohort_dev": loco,
    }


# ---------------------------------------------------------------------------
# MODULE 10D: Autonomic-Subjective Coherence & Bridge to Brain->Heart Gain
# ---------------------------------------------------------------------------


def _varimax(
    loadings: np.ndarray, *, max_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """Orthogonal Varimax rotation of factor loadings matrix (p x k)."""
    p, k = loadings.shape
    rot = np.eye(k)
    d_old = 0.0
    for _ in range(max_iter):
        lam = loadings @ rot
        u, s, vt = np.linalg.svd(
            loadings.T @ (lam**3 - (1.0 / p) * lam @ np.diag(np.sum(lam**2, axis=0)))
        )
        rot = u @ vt
        d_new = float(np.sum(s))
        if d_old > 0 and abs(d_new - d_old) / d_old < tol:
            break
        d_old = d_new
    return loadings @ rot


def _within_person_zscore_df(df: pd.DataFrame, pids: np.ndarray) -> pd.DataFrame:
    """Z-score every column strictly within participant."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    tmp = df.copy()
    tmp["_pid"] = pids
    for col in df.columns:
        g = tmp.groupby("_pid", sort=False)[col]
        mu = g.transform("mean")
        sd = g.transform("std")
        out[col] = np.where(sd > 1e-12, (df[col] - mu) / sd, np.nan)
    return out


def _pca_coherence_space(z_df: pd.DataFrame, cols: list[str]) -> dict:
    """Run SVD/PCA + Varimax 2-factor & 3-factor loadings on within-person z-matrix."""
    sub = z_df[cols].dropna()
    if len(sub) < 50:
        return {"n_complete_trials": int(len(sub)), "error": "too_few_complete"}
    x = sub.to_numpy(float)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(
        x.std(axis=0, keepdims=True), 1e-12
    )
    corr = np.corrcoef(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    var_exp = eigvals / eigvals.sum()
    loadings_raw = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))[None, :]

    # Canonical sign orientation: PC1 loads positively on Arousal (index 1)
    if loadings_raw[1, 0] < 0:
        loadings_raw[:, 0] *= -1.0

    rot2 = _varimax(loadings_raw[:, :2])
    if rot2[1, 0] < 0:
        rot2[:, 0] *= -1.0
    rot3 = _varimax(loadings_raw[:, :3]) if len(cols) >= 3 else rot2

    return {
        "n_complete_trials": int(len(sub)),
        "variables": cols,
        "corr_matrix": {
            cols[i]: {cols[j]: round(float(corr[i, j]), 4) for j in range(len(cols))}
            for i in range(len(cols))
        },
        "eigenvalues": [round(float(v), 4) for v in eigvals],
        "explained_variance_ratio": [round(float(v), 4) for v in var_exp],
        "pc_loadings": {
            f"PC{k + 1}": {
                cols[i]: round(float(loadings_raw[i, k]), 4) for i in range(len(cols))
            }
            for k in range(min(4, len(cols)))
        },
        "varimax_2f_loadings": {
            f"F{k + 1}": {
                cols[i]: round(float(rot2[i, k]), 4) for i in range(len(cols))
            }
            for k in range(2)
        },
        "varimax_3f_loadings": {
            f"F{k + 1}": {
                cols[i]: round(float(rot3[i, k]), 4) for i in range(len(cols))
            }
            for k in range(min(3, len(cols)))
        },
    }


def _compute_coherence_summary(
    sub_df: pd.DataFrame, *, draws: int = DRAWS
) -> tuple[dict, dict[str, float]]:
    """Cuve et al. (2023) 2D space + Brown et al. (2019) individual coherence."""
    sigs = _extract_interpretable_signals(sub_df)
    pids = sub_df["participant_id"].astype(str).to_numpy()

    raw_tbl = pd.DataFrame(
        {
            "valence": sub_df["rating_valence"].to_numpy(float),
            "arousal": sub_df["rating_arousal"].to_numpy(float),
            "d_hr": sigs["ecg_hr"].to_numpy(float),
            "d_sc": sigs["eda_sc"].to_numpy(float),
            "d_scr": sigs.get(
                "eda_scr", pd.Series(np.nan, index=sub_df.index)
            ).to_numpy(float),
            "d_resp": sigs["resp_amp"].to_numpy(float),
            "d_alpha_occ": sigs["eeg_alpha_occ"].to_numpy(float),
            "d_delta_cp": sigs["eeg_delta_cp"].to_numpy(float),
            "d_gamma_fr": sigs.get(
                "eeg_gamma_fr", pd.Series(np.nan, index=sub_df.index)
            ).to_numpy(float),
        },
        index=sub_df.index,
    )
    core_cols = [c for c in raw_tbl.columns if raw_tbl[c].notna().mean() > 0.3]
    z_within = _within_person_zscore_df(raw_tbl[core_cols], pids)

    # Also video-residualized idiosyncratic space
    idio_cols = {}
    for col in core_cols:
        _, _, rest = person.components(sub_df, raw_tbl[col])
        idio_cols[col] = rest.to_numpy(float)
    z_idio = _within_person_zscore_df(pd.DataFrame(idio_cols, index=sub_df.index), pids)

    pca_total = _pca_coherence_space(z_within, core_cols)
    pca_idio = _pca_coherence_space(z_idio, core_cols)

    # Optional 10-var version including pupil
    pca_with_pupil = None
    if "oc_pupil" in sigs and sigs["oc_pupil"].notna().mean() > 0.3:
        tbl_p = raw_tbl[core_cols].copy()
        tbl_p["d_pupil"] = sigs["oc_pupil"].to_numpy(float)
        z_p = _within_person_zscore_df(tbl_p, pids)
        pca_with_pupil = _pca_coherence_space(z_p, [*core_cols, "d_pupil"])

    # Brown et al. (2019) per-person coherence distribution
    distress_z = z_within["arousal"] - z_within["valence"]
    stress_oriented = {
        "heart_decel": -z_within["d_hr"],
        "eda_sc_rise": z_within["d_sc"],
        "resp_shallow": -z_within["d_resp"],
        "alpha_occ_supp": -z_within["d_alpha_occ"],
        "delta_cp_inc": z_within["d_delta_cp"],
    }
    if "d_scr" in z_within:
        stress_oriented["eda_scr_rise"] = z_within["d_scr"]
    stress_oriented["autonomic_composite"] = pd.concat(
        [
            stress_oriented["heart_decel"],
            stress_oriented["eda_sc_rise"],
            stress_oriented["resp_shallow"],
        ],
        axis=1,
    ).mean(axis=1)
    stress_oriented["neuro_autonomic_composite"] = pd.concat(
        [
            stress_oriented["heart_decel"],
            stress_oriented["eda_sc_rise"],
            stress_oriented["alpha_occ_supp"],
            stress_oriented["delta_cp_inc"],
        ],
        axis=1,
    ).mean(axis=1)

    indiv_coherence: dict[str, Any] = {}
    person_auto_coh: dict[str, float] = {}
    days = (
        sub_df["recorded_day"].to_numpy() if "recorded_day" in sub_df.columns else None
    )

    for ch_name, ch_s in stress_oriented.items():
        ch_arr = ch_s.to_numpy(float)
        d_arr = distress_z.to_numpy(float)
        rhos, hi_rhos, lo_rhos = [], [], []
        j1_map, j2_map = {}, {}

        for pid in np.unique(pids):
            m = (pids == pid) & np.isfinite(d_arr) & np.isfinite(ch_arr)
            if m.sum() < 15:
                continue
            dv, cv = d_arr[m], ch_arr[m]
            if np.std(dv) < 1e-12 or np.std(cv) < 1e-12:
                continue
            r_i = float(spearmanr(dv, cv).statistic)
            if not np.isfinite(r_i):
                continue
            rhos.append(r_i)
            if ch_name == "neuro_autonomic_composite":
                person_auto_coh[pid] = r_i

            # Intensity effect (Brown 2019): top 50% vs bottom 50% |distress|
            med_abs = float(np.median(np.abs(dv)))
            hi_m = np.abs(dv) >= med_abs
            lo_m = ~hi_m
            if (
                hi_m.sum() >= 6
                and lo_m.sum() >= 6
                and np.std(dv[hi_m]) > 1e-12
                and np.std(cv[hi_m]) > 1e-12
                and np.std(dv[lo_m]) > 1e-12
                and np.std(cv[lo_m]) > 1e-12
            ):
                r_hi = float(spearmanr(dv[hi_m], cv[hi_m]).statistic)
                r_lo = float(spearmanr(dv[lo_m], cv[lo_m]).statistic)
                if np.isfinite(r_hi) and np.isfinite(r_lo):
                    hi_rhos.append(r_hi)
                    lo_rhos.append(r_lo)

            # Day 1 vs Day 2 trait stability of coherence
            if days is not None:
                m1 = m & (days == 1)
                m2 = m & (days == 2)
                if (
                    m1.sum() >= 8
                    and m2.sum() >= 8
                    and np.std(d_arr[m1]) > 1e-12
                    and np.std(ch_arr[m1]) > 1e-12
                    and np.std(d_arr[m2]) > 1e-12
                    and np.std(ch_arr[m2]) > 1e-12
                ):
                    r1 = float(spearmanr(d_arr[m1], ch_arr[m1]).statistic)
                    r2 = float(spearmanr(d_arr[m2], ch_arr[m2]).statistic)
                    if np.isfinite(r1) and np.isfinite(r2):
                        j1_map[pid] = r1
                        j2_map[pid] = r2

        arr = np.array(rhos, dtype=float)
        ci = ssl_eval.boot_ci(arr, draws=draws)
        dist_stats = {}
        if len(arr):
            q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
            dist_stats = {
                "median": round(float(q50), 4),
                "iqr": [round(float(q25), 4), round(float(q75), 4)],
                "p10_p90": [round(float(q10), 4), round(float(q90), 4)],
                "frac_positive": round(float(np.mean(arr > 0.0)), 4),
                "frac_strong_gt_020": round(float(np.mean(arr > 0.20)), 4),
                "frac_discordant_lt_0": round(float(np.mean(arr < 0.0)), 4),
            }
        intensity_diff = None
        if len(hi_rhos) >= 10:
            hi_a, lo_a = np.array(hi_rhos), np.array(lo_rhos)
            intensity_diff = {
                "high_intensity_mean": ssl_eval.boot_ci(hi_a, draws=draws),
                "moderate_intensity_mean": ssl_eval.boot_ci(lo_a, draws=draws),
                "paired_high_minus_mod": ssl_eval.boot_ci(hi_a - lo_a, draws=draws),
            }
        j_common = sorted(set(j1_map) & set(j2_map))
        cross_day_coh = (
            _boot_corr_ci(
                np.array([j1_map[p] for p in j_common]),
                np.array([j2_map[p] for p in j_common]),
                draws=draws,
            )
            if len(j_common) >= 15
            else {"r": None, "ci95": [None, None], "n": len(j_common)}
        )
        indiv_coherence[ch_name] = {
            **ci,
            **dist_stats,
            "intensity_effect": intensity_diff,
            "cross_day_trait_stability": cross_day_coh,
        }

    return {
        "n_people": int(sub_df["participant_id"].nunique()),
        "n_trials": int(len(sub_df)),
        "pca_within_person_total": pca_total,
        "pca_within_person_idiosyncratic": pca_idio,
        "pca_with_pupil": pca_with_pupil,
        "brown_individual_coherence": indiv_coherence,
    }, person_auto_coh


def module_10d_coherence_and_bridge(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = DRAWS,
) -> dict:
    """H10.4: 2D coherence space (all 314 & dev) + bridge to S09 ECG M2-M0 gain."""
    coh_all, _ = _compute_coherence_summary(frame.reset_index(drop=True), draws=draws)
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    coh_dev, dev_coh_map = _compute_coherence_summary(dev_df, draws=draws)

    # Bridge: correlate per-person Study-09 M2-M0 gain with coherence & traits
    bridge: dict[str, Any] = {}
    tgt = "tgt_ecg_hr_chg_5_10"
    bcols = study09_analysis._body05_cols(frame)
    x_reve_fc_full = _get_reve_causal_forecast(aux)
    if tgt in dev_df.columns and bcols and x_reve_fc_full is not None:
        x_body = dev_df[bcols].to_numpy(np.float32)
        x_reve_fc = x_reve_fc_full[dev_m]
        y = dev_df[tgt].to_numpy(float)
        pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
        p0 = np.full(len(dev_df), np.nan)
        p2 = np.full(len(dev_df), np.nan)
        for p in range(3):
            for t in range(3):
                ev_m = (pf == p) & (tf == t)
                tr_m = (pf != p) & (tf != t)
                if ev_m.sum() == 0 or tr_m.sum() < 40:
                    continue
                a0, a2, _ = study09_protocol.fit_predict_m2_residual(
                    x_body[tr_m],
                    x_reve_fc[tr_m],
                    y[tr_m],
                    dev_df.iloc[tr_m],
                    x_body[ev_m],
                    x_reve_fc[ev_m],
                    eeg_pca_grid=(16, 32, 64),
                )
                p0[ev_m], p2[ev_m] = a0, a2
        _, pp0 = study09_protocol.summarize_forecast(dev_df, y, p0, draws=draws)
        _, pp2 = study09_protocol.summarize_forecast(dev_df, y, p2, draws=draws)
        delta_r = {k: pp2[k]["r"] - pp0[k]["r"] for k in set(pp0) & set(pp2)}

        # Per-person traits in dev
        stressed = label(dev_df).to_numpy(float)
        hr_resp = _extract_interpretable_signals(dev_df)["ecg_hr"].to_numpy(float)
        pids = dev_df["participant_id"].astype(str).to_numpy()
        trait_maps: dict[str, dict[str, float]] = {
            "neuro_autonomic_coherence": dev_coh_map,
            "trait_stress_rate": {},
            "trait_baseline_hr_ref": {},
            "individual_hr_decel_reactivity": {},
        }
        for pid in np.unique(pids):
            m = pids == pid
            trait_maps["trait_stress_rate"][pid] = float(np.nanmean(stressed[m]))
            if "ecg_hr_ref" in dev_df.columns:
                v_ref = dev_df.loc[m, "ecg_hr_ref"].to_numpy(float)
                if np.isfinite(v_ref).any():
                    trait_maps["trait_baseline_hr_ref"][pid] = float(np.nanmean(v_ref))
            m_pos = m & (stressed > 0.5) & np.isfinite(hr_resp)
            m_neg = m & (stressed < 0.5) & np.isfinite(hr_resp)
            if m_pos.sum() >= 3 and m_neg.sum() >= 3:
                trait_maps["individual_hr_decel_reactivity"][pid] = float(
                    -(hr_resp[m_pos].mean() - hr_resp[m_neg].mean())
                )

        bridge["m2_minus_m0_mean_delta_r"] = ssl_eval.boot_ci(
            list(delta_r.values()), draws=draws
        )
        bridge["correlates_of_brain_to_heart_gain"] = {}
        for tname, tmap in trait_maps.items():
            ck = sorted(set(delta_r) & set(tmap))
            if len(ck) >= 15:
                bridge["correlates_of_brain_to_heart_gain"][tname] = _boot_corr_ci(
                    np.array([delta_r[k] for k in ck]),
                    np.array([tmap[k] for k in ck]),
                    method="spearman",
                    draws=draws,
                )

    c_na = coh_all["brown_individual_coherence"]["neuro_autonomic_composite"]
    evr = coh_all["pca_within_person_total"].get("explained_variance_ratio", [])[:3]
    print(
        f"  10D Coherence (N=314): neuro-auto median={c_na.get('median')}, "
        f"mean={c_na.get('mean')}, >0={c_na.get('frac_positive')}, "
        f"PCA var={evr}",
        flush=True,
    )
    return {
        "coherence_all_314": coh_all,
        "coherence_dev": coh_dev,
        "bridge_to_study09_brain_heart_gain": bridge,
    }


# ---------------------------------------------------------------------------
# MODULE 10E: Clinical Passive Multi-Episode Monitoring Aggregation
# ---------------------------------------------------------------------------


def _sens_at_spec(pos_scores: np.ndarray, neg_scores: np.ndarray, spec: float) -> float:
    """Empirical sensitivity at target specificity ``spec`` (e.g. 0.80 or 0.90)."""
    thr = float(np.quantile(neg_scores, spec))
    return float(np.mean(pos_scores > thr))


def _simulate_clinical_aggregation(
    df: pd.DataFrame,
    scores: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    n_rep: int = 200,
    draws: int = DRAWS,
) -> dict:
    """Simulate K-episode passive aggregation (K=1..10) + disjoint sensitivity check."""
    pids = df["participant_id"].astype(str).to_numpy()
    sc = np.asarray(scores, dtype=float)
    ok = np.isfinite(sc)
    rng = np.random.default_rng(20261125)

    # Pre-slice per-person positive and negative score arrays
    person_pools: list[tuple[np.ndarray, np.ndarray]] = []
    for pid in np.unique(pids):
        pm = (pids == pid) & ok & pos_mask
        nm = (pids == pid) & ok & neg_mask
        if pm.sum() >= MIN_EACH and nm.sum() >= MIN_EACH:
            person_pools.append((sc[pm], sc[nm]))

    by_k: dict[str, Any] = {}
    for k in CLINICAL_K_GRID:
        aucs, sens80, sens90 = [], [], []
        disj_aucs = []
        for a, b in person_pools:
            if k == 1:
                y_i = np.concatenate([np.ones(len(a), int), np.zeros(len(b), int)])
                s_i = np.concatenate([a, b])
                aucs.append(float(roc_auc_score(y_i, s_i)))
                sens80.append(_sens_at_spec(a, b, 0.80))
                sens90.append(_sens_at_spec(a, b, 0.90))
                disj_aucs.append(aucs[-1])
            else:
                pa = a[rng.integers(0, len(a), (n_rep, k))].mean(axis=1)
                pb = b[rng.integers(0, len(b), (n_rep, k))].mean(axis=1)
                y_i = np.concatenate([np.ones(n_rep, int), np.zeros(n_rep, int)])
                s_i = np.concatenate([pa, pb])
                aucs.append(float(roc_auc_score(y_i, s_i)))
                sens80.append(_sens_at_spec(pa, pb, 0.80))
                sens90.append(_sens_at_spec(pa, pb, 0.90))

                # Strict WITHOUT-replacement check when person has >= 2*k episodes
                if k <= 5 and len(a) >= 2 * k and len(b) >= 2 * k:
                    n_blocks_a = len(a) // k
                    n_blocks_b = len(b) // k
                    pa_d = a[: n_blocks_a * k].reshape(n_blocks_a, k).mean(axis=1)
                    pb_d = b[: n_blocks_b * k].reshape(n_blocks_b, k).mean(axis=1)
                    yd = np.concatenate(
                        [np.ones(n_blocks_a, int), np.zeros(n_blocks_b, int)]
                    )
                    sd = np.concatenate([pa_d, pb_d])
                    disj_aucs.append(float(roc_auc_score(yd, sd)))

        by_k[str(k)] = {
            "window_seconds": 10 * k,
            "auc": ssl_eval.boot_ci(aucs, draws=draws),
            "sens_at_spec80": ssl_eval.boot_ci(sens80, draws=draws),
            "sens_at_spec90": ssl_eval.boot_ci(sens90, draws=draws),
            "disjoint_without_replacement_auc": (
                ssl_eval.boot_ci(disj_aucs, draws=draws)
                if len(disj_aucs) >= 10
                else None
            ),
        }
    return {"n_eligible_people": len(person_pools), "by_k": by_k}


def module_10e_clinical_monitoring(
    frame: pd.DataFrame,
    r4_oofs: dict[str, np.ndarray],
    *,
    draws: int = DRAWS,
) -> dict:
    """H10.5: Operational K-episode passive monitoring on R4 Cold-Start OOF."""
    dev_m = (frame["split"] == "dev").to_numpy()
    dev_df = frame[dev_m].reset_index(drop=True)
    v = dev_df["rating_valence"].to_numpy(float)
    a = dev_df["rating_arousal"].to_numpy(float)
    ctrs = contrasts(dev_df)

    contrast_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "stress_standard_vs_nonstress": (
            ctrs["stress"][0].to_numpy(bool),
            ctrs["stress"][1].to_numpy(bool),
        ),
        "stress_strict37_vs_relaxed64": (
            (v <= 3.0) & (a >= 7.0),
            (v >= 6.0) & (a <= 4.0),
        ),
    }
    if "self_reported_emotion_1" in dev_df.columns:
        emo = revision._clean_str(dev_df["self_reported_emotion_1"])
        contrast_masks["acute_threat_vs_calm_neutral"] = (
            emo.isin(["fear", "anger", "disgust"]).to_numpy(bool),
            emo.isin(["neutral", "relaxation"]).to_numpy(bool),
        )

    out: dict[str, Any] = {}
    for cname, (pm, nm) in contrast_masks.items():
        out[cname] = {}
        for mname, sc in r4_oofs.items():
            out[cname][mname] = _simulate_clinical_aggregation(
                dev_df, sc, pm, nm, draws=draws
            )

    s07_std = out["stress_standard_vs_nonstress"]["S07_enriched"]["by_k"]
    s07_str = out["stress_strict37_vs_relaxed64"]["S07_enriched"]["by_k"]
    print(
        f"  10E Clinical S07 (K=1/3/5/10): "
        f"Std={s07_std['1']['auc']['mean']}/{s07_std['3']['auc']['mean']}/"
        f"{s07_std['5']['auc']['mean']}/{s07_std['10']['auc']['mean']}; "
        f"StrictVsRel={s07_str['1']['auc']['mean']}/{s07_str['3']['auc']['mean']}/"
        f"{s07_str['5']['auc']['mean']}/{s07_str['10']['auc']['mean']}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# MODULE 10F: Within-Clip Early [0, 5) vs Late [5, 10) Emotion Dynamics
# ---------------------------------------------------------------------------


def module_10f_within_clip_emotion_dynamics(
    frame: pd.DataFrame, *, draws: int = DRAWS
) -> dict:
    """H10.6: Early vs late within-clip EEG-band & peripheral emotion profiles."""
    if "self_reported_emotion_1" not in frame.columns:
        return {"error": "self_reported_emotion_1_missing"}

    emo = revision._clean_str(frame["self_reported_emotion_1"])
    known = emo.notna() & (emo != "") & (emo != "nan") & (emo != "none")
    sub = frame[known].reset_index(drop=True)
    emo_s = emo[known].reset_index(drop=True)
    people = sub["participant_id"].astype(str).to_numpy()
    neutral_m = (emo_s == "neutral").to_numpy(bool)

    # Assemble early [0,5), late [5,10), and delta (late - early) for all signals
    channels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for band in common.BANDS:
        # Global 6-region average
        r_cols = [
            f"eeg_ep_ref_{r}_{band}"
            for r in common.REGIONS
            if f"eeg_ep_ref_{r}_{band}" in sub
        ]
        e_cols = [
            f"eeg_ep_early_{r}_{band}"
            for r in common.REGIONS
            if f"eeg_ep_early_{r}_{band}" in sub
        ]
        l_cols = [
            f"eeg_ep_late_{r}_{band}"
            for r in common.REGIONS
            if f"eeg_ep_late_{r}_{band}" in sub
        ]
        if r_cols and len(r_cols) == len(e_cols) == len(l_cols):
            ref = sub[r_cols].mean(axis=1).to_numpy(float)
            ea = sub[e_cols].mean(axis=1).to_numpy(float)
            la = sub[l_cols].mean(axis=1).to_numpy(float)
            channels[f"eeg_global_{band}"] = (ea - ref, la - ref, la - ea)

    # Key focal cortical regions
    for name, regs, band in (
        ("eeg_occ_alpha", ("occipital",), "alpha"),
        ("eeg_cp_delta", ("central", "parietal"), "delta"),
        ("eeg_fr_gamma", ("frontal",), "gamma"),
        ("eeg_fr_beta", ("frontal",), "beta"),
    ):
        rc = [f"eeg_ep_ref_{r}_{band}" for r in regs if f"eeg_ep_ref_{r}_{band}" in sub]
        ec = [
            f"eeg_ep_early_{r}_{band}"
            for r in regs
            if f"eeg_ep_early_{r}_{band}" in sub
        ]
        lc = [
            f"eeg_ep_late_{r}_{band}" for r in regs if f"eeg_ep_late_{r}_{band}" in sub
        ]
        if rc and len(rc) == len(ec) == len(lc):
            ref = sub[rc].mean(axis=1).to_numpy(float)
            ea = sub[ec].mean(axis=1).to_numpy(float)
            la = sub[lc].mean(axis=1).to_numpy(float)
            channels[name] = (ea - ref, la - ref, la - ea)

    # Peripheral channels
    for cname, pfx in (
        ("ecg_hr", "ecg_hr"),
        ("eda_sc", "sc_mean"),
        ("resp_amp", "resp_amp"),
    ):
        if {f"{pfx}_ref", f"{pfx}_early", f"{pfx}_late"}.issubset(sub.columns):
            ref = sub[f"{pfx}_ref"].to_numpy(float)
            ea = sub[f"{pfx}_early"].to_numpy(float)
            la = sub[f"{pfx}_late"].to_numpy(float)
            channels[cname] = (ea - ref, la - ref, la - ea)

    by_emotion: dict[str, Any] = {}
    for em in EMOTIONS_TARGET:
        em_m = (emo_s == em).to_numpy(bool)
        by_emotion[em] = {
            "n_trials": int(em_m.sum()),
            "channels": {},
        }
        for ch, (ea, la, dl) in channels.items():
            by_emotion[em]["channels"][ch] = {
                "early_d_vs_neutral": revision._person_d(
                    ea, em_m, neutral_m, people, min_each=MIN_EACH
                ),
                "late_d_vs_neutral": revision._person_d(
                    la, em_m, neutral_m, people, min_each=MIN_EACH
                ),
                "late_minus_early_d_vs_neutral": revision._person_d(
                    dl, em_m, neutral_m, people, min_each=MIN_EACH
                ),
            }

    # Direct Fear vs Disgust paired contrast (testing Si et al. 2026 directly!)
    fear_m = (emo_s == "fear").to_numpy(bool)
    disg_m = (emo_s == "disgust").to_numpy(bool)
    fear_vs_disgust: dict[str, Any] = {}
    for ch, (ea, la, dl) in channels.items():
        fear_vs_disgust[ch] = {
            "early_d": revision._person_d(
                ea, fear_m, disg_m, people, min_each=MIN_EACH
            ),
            "late_d": revision._person_d(la, fear_m, disg_m, people, min_each=MIN_EACH),
            "late_minus_early_d": revision._person_d(
                dl, fear_m, disg_m, people, min_each=MIN_EACH
            ),
        }

    g_ea = fear_vs_disgust.get("eeg_global_gamma", {}).get("early_d")
    d_la = fear_vs_disgust.get("eeg_global_delta", {}).get("late_d")
    sc_la = fear_vs_disgust.get("eda_sc", {}).get("late_d")
    print(
        f"  10F Within-clip (N={sub['participant_id'].nunique()}, n={len(sub)}): "
        f"Fear-Disg gamma early={g_ea}, delta late={d_la}, EDA late={sc_la}",
        flush=True,
    )
    return {
        "n_trials": int(len(sub)),
        "n_people": int(sub["participant_id"].nunique()),
        "n_neutral_trials": int(neutral_m.sum()),
        "by_emotion_vs_neutral": by_emotion,
        "fear_vs_disgust_direct": fear_vs_disgust,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint compatible with decoding.yml
# ---------------------------------------------------------------------------


def run_all(
    frame: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Run all 6 Study-10 final modules and return JSON-serializable report."""
    t0 = time.perf_counter()
    print("=== Study 10A: Generalization Ladder (R1->R4) & Ablation ===", flush=True)
    m10a, r4_oofs = module_10a_generalization_and_ablation(frame, aux, draws=draws)

    print(
        "=== Study 10B: Longitudinal Stability (J1->J2) & Habituation ===", flush=True
    )
    m10b = module_10b_longitudinal_and_habituation(frame, aux, draws=draws)

    print("=== Study 10C: CORAL, k-shot Personalization & LOCO-CV ===", flush=True)
    m10c = module_10c_coral_kshot_loco(frame, aux, draws=draws)

    print("=== Study 10D: Autonomic-Subjective Coherence & S09 Bridge ===", flush=True)
    m10d = module_10d_coherence_and_bridge(frame, aux, draws=draws)

    print("=== Study 10E: Clinical Passive Multi-Episode Monitoring ===", flush=True)
    m10e = module_10e_clinical_monitoring(frame, r4_oofs, draws=draws)

    print("=== Study 10F: Within-Clip Early vs Late Emotion Dynamics ===", flush=True)
    m10f = module_10f_within_clip_emotion_dynamics(frame, draws=draws)

    return {
        "study": "10_final_generalization_coherence_clinical",
        "salt": study09_protocol.STUDY09_BASE_SALT,
        "splits": common.summary(frame),
        "available_aux_tables": sorted(aux.keys()),
        "module10a_generalization_ablation": m10a,
        "module10b_longitudinal_habituation": m10b,
        "module10c_coral_kshot_loco": m10c,
        "module10d_coherence_bridge": m10d,
        "module10e_clinical_monitoring": m10e,
        "module10f_within_clip_emotions": m10f,
        "elapsed_cpu_seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    """CLI entrypoint compatible with ``decoding.yml`` CPU workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("all",), default="all")
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    frame, aux = study09_analysis.load_study09_tables()
    result = run_all(frame, aux, draws=DRAWS)
    text = json.dumps(result, indent=1, default=lambda x: None)
    print(text[:4000], flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(text)


if __name__ == "__main__":
    main()
