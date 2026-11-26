"""Causal early-EEG to future autonomic forecasting (M0 vs. M1 vs. M2) and controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from habs_stress.protocol import (
    MIN_FORECAST_TRIALS,
    _within_person_mse,
    inner_blocked_folds,
    paired_person_bootstrap,
    preprocess_fit_transform,
    svd_ridge_fit_predict,
)


def fit_predict_nested_ridge(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    *,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    pca_grid: tuple[int | None, ...] = (None,),
    inner_k: int = 3,
    return_inner_oof: bool = False,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Ridge with nested inner person x theme CV minimizing within-person MSE."""
    ok = np.isfinite(y_tr)
    if ok.sum() < 20:
        pred = np.full(len(x_ev), float(np.nanmean(y_tr)) if ok.any() else 0.0)
        info = {"alpha": None, "pca": None}
        if return_inner_oof:
            return pred, np.full(len(x_tr), np.nan), info
        return pred, info

    sub_x, sub_y = x_tr[ok], y_tr[ok].astype(np.float64)
    sub_df = df_tr.iloc[np.where(ok)[0]]
    ipf, itf = inner_blocked_folds(sub_df, k=inner_k)
    ppl = sub_df["participant_id"].to_numpy()
    valid_pcas = [
        nc
        for nc in pca_grid
        if nc is None or (nc < sub_x.shape[1] and nc < int(0.4 * len(sub_x)))
    ] or [None]

    best_loss, best_cfg, best_oof = float("inf"), (alphas[len(alphas) // 2], valid_pcas[0]), None
    for nc in valid_pcas:
        oofs = {a: np.full(len(sub_x), np.nan) for a in alphas}
        for p in range(inner_k):
            for t in range(inner_k):
                ev_m = (ipf == p) & (itf == t)
                tr_m = (ipf != p) & (itf != t)
                if ev_m.sum() == 0 or tr_m.sum() < 15:
                    continue
                zt, ze = preprocess_fit_transform(sub_x[tr_m], sub_x[ev_m], max_pca=nc)
                for a in alphas:
                    oofs[a][ev_m] = svd_ridge_fit_predict(zt, sub_y[tr_m], ze, a)
        for a in alphas:
            loss = _within_person_mse(sub_y, oofs[a], ppl)
            if loss < best_loss:
                best_loss, best_cfg, best_oof = loss, (a, nc), oofs[a]

    best_a, best_nc = best_cfg
    zt_f, ze_f = preprocess_fit_transform(sub_x, x_ev, max_pca=best_nc)
    pred_ev = svd_ridge_fit_predict(zt_f, sub_y, ze_f, best_a)
    info = {"alpha": best_a, "pca": best_nc}
    if return_inner_oof:
        full_oof = np.full(len(x_tr), np.nan)
        full_oof[ok] = best_oof
        return pred_ev, full_oof, info
    return pred_ev, info


def within_person_pearson(
    y_true: np.ndarray, y_pred: np.ndarray, people: np.ndarray
) -> dict[str, float]:
    """Compute per-participant Pearson r across participants with >= 5 valid trials."""
    out: dict[str, float] = {}
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    for pid in np.unique(people[ok]):
        m = ok & (people == pid)
        if m.sum() < MIN_FORECAST_TRIALS:
            continue
        yt, yp = y_true[m], y_pred[m]
        if np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
            out[str(pid)] = float(np.corrcoef(yt, yp)[0, 1])
    return out


def residualize_artefacts(
    x_eeg_tr: np.ndarray,
    art_tr: np.ndarray,
    x_eeg_ev: np.ndarray,
    art_ev: np.ndarray,
    *,
    alpha: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-isolated OLS/Ridge regression removing nuisance/artefact covariates from EEG."""
    za_tr, za_ev = preprocess_fit_transform(art_tr, art_ev)
    x_tr_f, x_ev_f = preprocess_fit_transform(x_eeg_tr, x_eeg_ev)
    u, s, vt = np.linalg.svd(za_tr, full_matrices=False)
    coef = vt.T @ ((s / (s * s + alpha))[:, None] * (u.T @ x_tr_f))
    return x_tr_f - za_tr @ coef, x_ev_f - za_ev @ coef


def evaluate_causal_forecast(
    df: pd.DataFrame,
    x_body: np.ndarray,
    x_eeg: np.ndarray,
    y_future: np.ndarray,
    *,
    eeg_pca_grid: tuple[int | None, ...] = (None, 32, 64),
    etas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Run blocked outer 3x3 person x theme causal forecasting (M0 vs. M1 vs. M2_resid).

    Tests whether early EEG [-3.2, 5.0) s improves prediction of future autonomic
    physiology [5.0, 10.0) s beyond the full past autonomic trajectory M0.
    """
    n = len(df)
    pf = df["person_fold"].to_numpy(int)
    tf = df["theme_fold"].to_numpy(int)
    pred_m0 = np.full(n, np.nan)
    pred_m1 = np.full(n, np.nan)
    pred_m2 = np.full(n, np.nan)

    for p in range(3):
        for t in range(3):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 25:
                continue
            df_tr = df.iloc[np.where(tr_m)[0]]
            p0_ev, p0_oof, _ = fit_predict_nested_ridge(
                x_body[tr_m], y_future[tr_m], df_tr, x_body[ev_m], return_inner_oof=True
            )
            p1_ev, _ = fit_predict_nested_ridge(
                x_eeg[tr_m], y_future[tr_m], df_tr, x_eeg[ev_m], pca_grid=eeg_pca_grid
            )
            resid_tr = y_future[tr_m] - p0_oof
            pr_ev, pr_oof, _ = fit_predict_nested_ridge(
                x_eeg[tr_m], resid_tr, df_tr, x_eeg[ev_m],
                pca_grid=eeg_pca_grid, return_inner_oof=True,
            )
            ppl_tr = df_tr["participant_id"].to_numpy()
            best_eta = min(
                etas,
                key=lambda e: _within_person_mse(
                    y_future[tr_m], p0_oof + e * pr_oof, ppl_tr
                ),
            )
            pred_m0[ev_m] = p0_ev
            pred_m1[ev_m] = p1_ev
            pred_m2[ev_m] = p0_ev + best_eta * pr_ev

    ppl = df["participant_id"].to_numpy()
    r0 = within_person_pearson(y_future, pred_m0, ppl)
    r1 = within_person_pearson(y_future, pred_m1, ppl)
    r2 = within_person_pearson(y_future, pred_m2, ppl)
    common = sorted(set(r0) & set(r2))
    a0 = np.array([r0[k] for k in common])
    a2 = np.array([r2[k] for k in common])
    return {
        "M0_linear": paired_person_bootstrap(a0),
        "M1_eeg_only": paired_person_bootstrap(np.array([r1[k] for k in common if k in r1])),
        "M2_resid": paired_person_bootstrap(a2),
        "M2_minus_M0": paired_person_bootstrap(a2, a0),
        "predictions": {"M0": pred_m0, "M1": pred_m1, "M2_resid": pred_m2},
    }


def idiosyncratic_same_video_test(
    df: pd.DataFrame,
    y_future: np.ndarray,
    pred_m0: np.ndarray,
    pred_m2_own: np.ndarray,
    pred_m2_other: np.ndarray,
) -> dict[str, Any]:
    """Compare own-EEG M2 vs. same-video other-viewer EEG M2 (idiosyncratic test)."""
    ppl = df["participant_id"].to_numpy()
    r_own = within_person_pearson(y_future, pred_m2_own, ppl)
    r_oth = within_person_pearson(y_future, pred_m2_other, ppl)
    r_m0 = within_person_pearson(y_future, pred_m0, ppl)
    common = sorted(set(r_own) & set(r_oth) & set(r_m0))
    return {
        "own_vs_other_viewer": paired_person_bootstrap(
            np.array([r_own[k] for k in common]),
            np.array([r_oth[k] for k in common]),
        ),
        "other_viewer_vs_m0": paired_person_bootstrap(
            np.array([r_oth[k] for k in common]),
            np.array([r_m0[k] for k in common]),
        ),
    }
