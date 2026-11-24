"""Core Study-09 zero-leakage protocol, nested blocked CV, and statistical engine.

Implements all pre-registered methodological guarantees of Study 09
(``journal/09_eeg_predictive_physiology.md``):

1. Strict separation of **Protocol A (Cold-Start)** vs **Protocol B (Unlabeled
   History: B0 resting-state, B1 strictly causal online Empirical Bayes, B2
   fold-isolated session)**.
2. All model selection, scaling, imputation, PCA, trait/state factorization,
   and optional OOF stacking occur strictly inside ``dev`` via **nested blocked
   person x theme cross-validation** (outer 3x3 primary + outer 5x5 sensitivity,
   with inner 3x3 blocked person x theme CV inside each outer training block).
3. Fast closed-form SVD Ridge with nested grouped selection for M0, M1,
   residual M2, and dual-penalty joint M2 forecasting, plus nonlinear GBDT
   controls.
4. Trait / State factorization (optimal shrinkage baseline residualization +
   ANOVA between/within trait subspace) preserving useful trait variance.
5. Paired per-subject inference: within-person Pearson r, within-person R^2,
   raw R^2, RMSE reduction, within-video idiosyncratic r, 2,000-draw paired
   person bootstrap 95% CIs, Wilcoxon signed-rank p-values, BH-FDR q-values,
   and full per-subject distributions (median, IQR, deciles, % improved).
6. Pre-registered Complementarity Gate and strictly nested conditional OOF
   stacker.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from decoding import ssl_eval
from decoding.stress import label

STUDY09_BASE_SALT = "habs-ssl-2026-11-24-b"
DRAWS = 2000
MIN_FORECAST_TRIALS = 5
MIN_VIDEO_VIEWERS = 5


# ---------------------------------------------------------------------------
# 1. Nested blocked person x theme split helpers (strictly inside dev)
# ---------------------------------------------------------------------------


def hash_buckets(keys: pd.Series | np.ndarray, salt: str, k: int) -> np.ndarray:
    """Deterministic hash assignment of group keys into ``0..k-1``."""
    return np.array(
        [
            int(hashlib.sha256(f"{salt}|{str(key)}".encode()).hexdigest(), 16) % k
            for key in keys
        ],
        dtype=int,
    )


def outer_blocked_folds(
    dev_df: pd.DataFrame, *, k: int = 3, salt: str = STUDY09_BASE_SALT
) -> tuple[np.ndarray, np.ndarray]:
    """Return outer ``(person_fold, theme_fold)`` arrays strictly inside ``dev``.

    For ``k == 3`` reuses the exact canonical Study-08 ``dev`` folds when present
    so CPU linear/GBDT models align 1-to-1 with GPU REVE-LoRA folds. For
    ``k != 3`` (e.g. ``k = 5`` sensitivity check) hashes with a dedicated salt.
    """
    if k == 3 and "person_fold" in dev_df.columns and "theme_fold" in dev_df.columns:
        return (
            dev_df["person_fold"].to_numpy(dtype=int),
            dev_df["theme_fold"].to_numpy(dtype=int),
        )
    pf = hash_buckets(dev_df["participant_id"], f"{salt}|study09-outer-p-{k}", k)
    tf = hash_buckets(dev_df["theme"], f"{salt}|study09-outer-t-{k}", k)
    return pf, tf


def inner_blocked_folds(
    tr_df: pd.DataFrame, *, k: int = 3, salt: str = "study09-inner-v1"
) -> tuple[np.ndarray, np.ndarray]:
    """Independent blocked ``(inner_p, inner_t)`` folds strictly inside ``tr_df``."""
    pf = hash_buckets(tr_df["participant_id"], f"{salt}|p-{k}", k)
    tf = hash_buckets(tr_df["theme"], f"{salt}|t-{k}", k)
    return pf, tf


# ---------------------------------------------------------------------------
# 2. Protocol A (Cold-Start) vs Protocol B (Unlabeled History) preparation
# ---------------------------------------------------------------------------


def prepare_protocol_fold(
    x_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    df_ev: pd.DataFrame,
    *,
    protocol: str = "cold_start",
    rest_tr: np.ndarray | None = None,
    rest_ev: np.ndarray | None = None,
    shrinkage: float = 5.0,
    rest_weight: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare train/eval matrices under strictly separated normalization protocols.

    Protocols
    ---------
    - ``"cold_start"`` (Protocol A): zero subject history at inference time.
      Strictly invariant to predicting one trial at a time (``batch_size = 1``).
    - ``"rest_only"`` (Protocol B0): subtracts each participant's unlabeled
      resting-state vector (shrunk toward training population mean if missing).
    - ``"causal_online"`` (Protocol B1): subtracts each participant's strictly
      past unlabeled trials ``< i`` (plus optional resting-state prior), shrunk
      via Empirical Bayes toward the training population mean. Zero future leak.
    - ``"session_isolated"`` (Protocol B2): centers ``x_tr`` strictly within
      ``x_tr`` and ``x_ev`` strictly within ``x_ev`` (explicit transductive
      unlabeled-session evaluation, strictly fold-isolated).
    """
    a_tr = np.asarray(x_tr, dtype=np.float32)
    a_ev = np.asarray(x_ev, dtype=np.float32)
    if protocol == "cold_start":
        return a_tr.copy(), a_ev.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_tr = np.nanmean(a_tr, axis=0)
    mu_tr = np.where(np.isfinite(mu_tr), mu_tr, 0.0)

    if protocol == "rest_only":
        if rest_tr is None or rest_ev is None:
            raise ValueError("rest_only protocol requires rest_tr and rest_ev arrays.")
        r_tr = np.asarray(rest_tr, dtype=np.float64)
        r_ev = np.asarray(rest_ev, dtype=np.float64)
        ok_tr = np.isfinite(r_tr)
        ok_ev = np.isfinite(r_ev)
        base_tr = np.where(
            ok_tr,
            (rest_weight * np.where(ok_tr, r_tr, 0.0) + shrinkage * mu_tr)
            / (rest_weight + shrinkage),
            mu_tr,
        )
        base_ev = np.where(
            ok_ev,
            (rest_weight * np.where(ok_ev, r_ev, 0.0) + shrinkage * mu_tr)
            / (rest_weight + shrinkage),
            mu_tr,
        )
        return (a_tr - base_tr).astype(np.float32), (a_ev - base_ev).astype(np.float32)

    if protocol == "causal_online":
        rest_map_tr = _build_rest_map(df_tr, rest_tr)
        rest_map_ev = _build_rest_map(df_ev, rest_ev)
        c_tr = ssl_eval.causal_person_center(
            a_tr,
            df_tr["participant_id"].to_numpy(),
            df_tr,
            prior_mean=mu_tr,
            shrinkage=shrinkage,
            rest_baseline=rest_map_tr,
            rest_weight=rest_weight,
        )
        c_ev = ssl_eval.causal_person_center(
            a_ev,
            df_ev["participant_id"].to_numpy(),
            df_ev,
            prior_mean=mu_tr,
            shrinkage=shrinkage,
            rest_baseline=rest_map_ev,
            rest_weight=rest_weight,
        )
        return c_tr, c_ev

    if protocol == "session_isolated":
        c_tr = ssl_eval.person_center_matrix(
            a_tr,
            df_tr["participant_id"].to_numpy(),
            prior_mean=mu_tr,
            shrinkage=shrinkage,
        )
        c_ev = ssl_eval.person_center_matrix(
            a_ev,
            df_ev["participant_id"].to_numpy(),
            prior_mean=mu_tr,
            shrinkage=shrinkage,
        )
        return c_tr, c_ev

    raise ValueError(f"Unknown Study-09 protocol: {protocol}")


def _build_rest_map(
    df: pd.DataFrame, rest_arr: np.ndarray | None
) -> dict[str, np.ndarray] | None:
    if rest_arr is None:
        return None
    arr = np.asarray(rest_arr, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    pids = df["participant_id"].astype(str).to_numpy()
    for pid in np.unique(pids):
        rows = arr[pids == pid]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(rows, axis=0)
        if np.isfinite(m).any():
            out[str(pid)] = m
    return out


# ---------------------------------------------------------------------------
# 3. Fast closed-form SVD Ridge & Nested Grouped Model Fitting
# ---------------------------------------------------------------------------


def _svd_ridge_fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form Ridge regression ``(X^T X + alpha I)^-1 X^T y`` via SVD."""
    y_mean = float(np.mean(y_tr))
    yc = y_tr - y_mean
    x_mean = np.mean(x_tr, axis=0, keepdims=True)
    xc = x_tr - x_mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    d = s / (s * s + alpha)
    coef = vt.T @ (d * (u.T @ yc))
    return y_mean + (x_ev - x_mean) @ coef


def _preprocess_fit_transform(
    x_tr: np.ndarray, x_ev: np.ndarray, *, max_pca: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-isolated median imputation, zero-variance guard, standardization, PCA."""
    x_tr_f = np.asarray(x_tr, dtype=np.float64)
    x_ev_f = np.asarray(x_ev, dtype=np.float64)
    # Drop all-NaN columns in training fold safely
    valid_cols = np.isfinite(x_tr_f).any(axis=0)
    if not valid_cols.any():
        return np.zeros((len(x_tr_f), 1)), np.zeros((len(x_ev_f), 1))
    x_tr_f = x_tr_f[:, valid_cols]
    x_ev_f = x_ev_f[:, valid_cols]

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    z_tr = scl.fit_transform(imp.fit_transform(x_tr_f))
    z_ev = scl.transform(imp.transform(x_ev_f))
    if max_pca is not None and z_tr.shape[1] > max_pca and z_tr.shape[0] > max_pca + 5:
        pca = PCA(n_components=max_pca, random_state=0)
        z_tr = pca.fit_transform(z_tr)
        z_ev = pca.transform(z_ev)
    return z_tr, z_ev


def _within_person_mse(
    y_true: np.ndarray, y_pred: np.ndarray, people: np.ndarray
) -> float:
    """Mean within-person centered squared error (lower is better)."""
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 10:
        return float("inf")
    df = pd.DataFrame({"p": people[ok], "y": y_true[ok], "pr": y_pred[ok]})
    mses = []
    for _, g in df.groupby("p", sort=False):
        if len(g) < MIN_FORECAST_TRIALS:
            continue
        yc = g["y"].to_numpy(float)
        pc = g["pr"].to_numpy(float)
        yc = yc - yc.mean()
        pc = pc - pc.mean()
        mses.append(float(np.mean((yc - pc) ** 2)))
    if not mses:
        return float(np.mean((y_true[ok] - y_pred[ok]) ** 2))
    return float(np.mean(mses))


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
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, np.ndarray, dict]:
    """Fit Ridge with nested inner ``person x theme`` CV hyperparameter selection."""
    ok_tr = np.isfinite(y_tr)
    if ok_tr.sum() < 20:
        pred_ev = np.full(len(x_ev), float(np.nanmean(y_tr)) if ok_tr.any() else 0.0)
        info = {"alpha": None, "pca": None}
        if return_inner_oof:
            return pred_ev, np.full(len(x_tr), np.nan), info
        return pred_ev, info

    sub_x = x_tr[ok_tr]
    sub_y = y_tr[ok_tr].astype(np.float64)
    sub_df = df_tr.iloc[np.where(ok_tr)[0]]
    ipf, itf = inner_blocked_folds(sub_df, k=inner_k)
    ppl = sub_df["participant_id"].to_numpy()

    # Pre-filter candidate PCAs so we never exceed feature or sample count
    d_raw = sub_x.shape[1]
    clean_pca_grid: list[int | None] = []
    for nc in pca_grid:
        if nc is None:
            clean_pca_grid.append(None)
        elif nc < d_raw and nc < int(0.4 * len(sub_x)):
            clean_pca_grid.append(int(nc))
    if not clean_pca_grid:
        clean_pca_grid = [None]

    # Fit Imputer + Scaler + max_pca ONCE per inner fold (PCA components are
    # nested/ordered, so slicing [:, :nc] is mathematically exact and 3x faster!)
    non_none_pcas = [c for c in clean_pca_grid if c is not None]
    need_full = None in clean_pca_grid
    max_inner_pca = max(non_none_pcas) if non_none_pcas else None

    inner_cache: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ] = []
    for p in range(inner_k):
        for t in range(inner_k):
            ev_m = (ipf == p) & (itf == t)
            tr_m = (ipf != p) & (itf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 15:
                continue
            z_tr_full_i = z_ev_full_i = z_tr_pca_i = z_ev_pca_i = None
            if need_full:
                z_tr_full_i, z_ev_full_i = _preprocess_fit_transform(
                    sub_x[tr_m], sub_x[ev_m], max_pca=None
                )
            if max_inner_pca is not None:
                if z_tr_full_i is not None and z_tr_full_i.shape[1] > max_inner_pca:
                    pca_i = PCA(n_components=max_inner_pca, random_state=0)
                    z_tr_pca_i = pca_i.fit_transform(z_tr_full_i)
                    z_ev_pca_i = pca_i.transform(z_ev_full_i)
                else:
                    z_tr_pca_i, z_ev_pca_i = _preprocess_fit_transform(
                        sub_x[tr_m], sub_x[ev_m], max_pca=max_inner_pca
                    )
            inner_cache.append(
                (ev_m, sub_y[tr_m], z_tr_full_i, z_ev_full_i, z_tr_pca_i, z_ev_pca_i)
            )

    best_score = float("inf")
    best_cfg: tuple[int | None, float] = (clean_pca_grid[0], alphas[len(alphas) // 2])
    best_oof_sub = np.full(len(sub_x), np.nan, dtype=float)

    for n_pca in clean_pca_grid:
        oof_by_alpha = {a: np.full(len(sub_x), np.nan, dtype=float) for a in alphas}
        for ev_m, y_tr_i, ztf, zef, ztp, zep in inner_cache:
            if n_pca is None:
                z_tr_i, z_ev_i = ztf, zef
            else:
                z_tr_i, z_ev_i = ztp[:, :n_pca], zep[:, :n_pca]
            y_mean = float(np.mean(y_tr_i))
            yc = y_tr_i - y_mean
            x_mean = np.mean(z_tr_i, axis=0, keepdims=True)
            xc = z_tr_i - x_mean
            u, s, vt = np.linalg.svd(xc, full_matrices=False)
            uty = u.T @ yc
            xe_c = z_ev_i - x_mean
            for a in alphas:
                d_vec = s / (s * s + a)
                coef = vt.T @ (d_vec * uty)
                oof_by_alpha[a][ev_m] = y_mean + xe_c @ coef

        for a in alphas:
            loss = _within_person_mse(sub_y, oof_by_alpha[a], ppl)
            if loss < best_score:
                best_score = loss
                best_cfg = (n_pca, a)
                best_oof_sub = oof_by_alpha[a]

    best_pca, best_alpha = best_cfg
    z_tr_full, z_ev_full = _preprocess_fit_transform(sub_x, x_ev, max_pca=best_pca)
    pred_ev = _svd_ridge_fit_predict(z_tr_full, sub_y, z_ev_full, best_alpha)
    info = {
        "alpha": float(best_alpha),
        "pca": int(best_pca) if best_pca is not None else None,
        "inner_wp_mse": round(float(best_score), 6),
    }
    if return_inner_oof:
        oof_full = np.full(len(x_tr), np.nan, dtype=float)
        oof_full[ok_tr] = best_oof_sub
        return pred_ev, oof_full, info
    return pred_ev, info


def fit_predict_m2_residual(
    x_body_tr: np.ndarray,
    x_eeg_tr: np.ndarray,
    y_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_body_ev: np.ndarray,
    x_eeg_ev: np.ndarray,
    *,
    body_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    eeg_alphas: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0, 50000.0),
    eeg_pca_grid: tuple[int | None, ...] = (16, 32, 64),
    eta_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    inner_k: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nested two-stage residual M2: ``M0(Body) + eta * g(EEG -> y - M0_oof)``.

    Returns ``(pred_m0_ev, pred_m2_ev, info)``. Every parameter (``M0`` alpha,
    EEG PCA/alpha, and mixing weight ``eta``) is selected strictly inside
    ``df_tr`` by inner blocked ``person x theme`` CV.
    """
    pred_m0_ev, oof_m0, info_m0 = fit_predict_nested_ridge(
        x_body_tr,
        y_tr,
        df_tr,
        x_body_ev,
        alphas=body_alphas,
        pca_grid=(None,),
        inner_k=inner_k,
        return_inner_oof=True,
    )
    resid_tr = y_tr - oof_m0
    pred_e_ev, oof_e, info_e = fit_predict_nested_ridge(
        x_eeg_tr,
        resid_tr,
        df_tr,
        x_eeg_ev,
        alphas=eeg_alphas,
        pca_grid=eeg_pca_grid,
        inner_k=inner_k,
        return_inner_oof=True,
    )
    ppl = df_tr["participant_id"].to_numpy()
    best_eta = 0.0
    best_loss = float("inf")
    for eta in eta_grid:
        comb = oof_m0 + eta * np.nan_to_num(oof_e, nan=0.0)
        loss = _within_person_mse(y_tr, comb, ppl)
        if loss < best_loss:
            best_loss = loss
            best_eta = float(eta)

    pred_m2_ev = pred_m0_ev + best_eta * pred_e_ev
    return (
        pred_m0_ev,
        pred_m2_ev,
        {
            "m0": info_m0,
            "eeg_resid": info_e,
            "eta": best_eta,
            "inner_comb_wp_mse": round(float(best_loss), 6),
        },
    )


def fit_predict_m2_joint_dual_ridge(
    x_body_tr: np.ndarray,
    x_eeg_tr: np.ndarray,
    y_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_body_ev: np.ndarray,
    x_eeg_ev: np.ndarray,
    *,
    body_alphas: tuple[float, ...] = (10.0, 100.0, 1000.0),
    eeg_alphas: tuple[float, ...] = (100.0, 1000.0, 10000.0, 50000.0),
    eeg_pca: int = 32,
    inner_k: int = 3,
) -> tuple[np.ndarray, dict]:
    """Joint dual-penalty Ridge allowing separate regularization ``(lam_b, lam_e)``.

    Solves ``min ||y - X_b w_b - X_e w_e||^2 + lam_b ||w_b||^2 + lam_e ||w_e||^2``
    by scaling ``[X_b / sqrt(lam_b), X_e / sqrt(lam_e)]`` with unit Ridge penalty.
    """
    ok_tr = np.isfinite(y_tr)
    if ok_tr.sum() < 20:
        return np.full(
            len(x_body_ev), float(np.nanmean(y_tr)) if ok_tr.any() else 0.0
        ), {}

    xb = x_body_tr[ok_tr]
    xe = x_eeg_tr[ok_tr]
    sub_y = y_tr[ok_tr].astype(np.float64)
    sub_df = df_tr.iloc[np.where(ok_tr)[0]]
    ipf, itf = inner_blocked_folds(sub_df, k=inner_k)
    ppl = sub_df["participant_id"].to_numpy()

    # Precompute fold-isolated standardized body & PCA-reduced EEG blocks once
    inner_blocks: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for p in range(inner_k):
        for t in range(inner_k):
            ev_m = (ipf == p) & (itf == t)
            tr_m = (ipf != p) & (itf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 15:
                continue
            zb_tr, zb_ev = _preprocess_fit_transform(xb[tr_m], xb[ev_m], max_pca=None)
            ze_tr, ze_ev = _preprocess_fit_transform(
                xe[tr_m], xe[ev_m], max_pca=eeg_pca
            )
            inner_blocks.append((ev_m, sub_y[tr_m], zb_tr, zb_ev, ze_tr, ze_ev))

    best_loss = float("inf")
    best_pair = (body_alphas[1], eeg_alphas[1])
    for lb in body_alphas:
        sb = 1.0 / np.sqrt(lb)
        for le in eeg_alphas:
            se = 1.0 / np.sqrt(le)
            oof = np.full(len(sub_y), np.nan, dtype=float)
            for ev_m, y_i, zb_tr, zb_ev, ze_tr, ze_ev in inner_blocks:
                z_tr = np.hstack([zb_tr * sb, ze_tr * se])
                z_ev = np.hstack([zb_ev * sb, ze_ev * se])
                oof[ev_m] = _svd_ridge_fit_predict(z_tr, y_i, z_ev, alpha=1.0)
            loss = _within_person_mse(sub_y, oof, ppl)
            if loss < best_loss:
                best_loss = loss
                best_pair = (lb, le)

    lb_opt, le_opt = best_pair
    zb_tr_f, zb_ev_f = _preprocess_fit_transform(xb, x_body_ev, max_pca=None)
    ze_tr_f, ze_ev_f = _preprocess_fit_transform(xe, x_eeg_ev, max_pca=eeg_pca)
    z_tr_f = np.hstack([zb_tr_f / np.sqrt(lb_opt), ze_tr_f / np.sqrt(le_opt)])
    z_ev_f = np.hstack([zb_ev_f / np.sqrt(lb_opt), ze_ev_f / np.sqrt(le_opt)])
    pred_ev = _svd_ridge_fit_predict(z_tr_f, sub_y, z_ev_f, alpha=1.0)
    return pred_ev, {
        "lam_body": float(lb_opt),
        "lam_eeg": float(le_opt),
        "eeg_pca": int(eeg_pca),
        "inner_wp_mse": round(float(best_loss), 6),
    }


def fit_predict_gbdt_regressor(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    *,
    max_pca: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Regularized nonlinear HistGradientBoostingRegressor control (fold-isolated)."""
    ok_tr = np.isfinite(y_tr)
    if ok_tr.sum() < 30:
        return np.full(len(x_ev), float(np.nanmean(y_tr)) if ok_tr.any() else 0.0)
    z_tr, z_ev = _preprocess_fit_transform(x_tr[ok_tr], x_ev, max_pca=max_pca)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = HistGradientBoostingRegressor(
            max_iter=120,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=40,
            l2_regularization=10.0,
            random_state=seed,
        ).fit(z_tr, y_tr[ok_tr].astype(np.float64))
        return reg.predict(z_ev)


# ---------------------------------------------------------------------------
# 4. Trait / State Factorization (Requirement 10)
# ---------------------------------------------------------------------------


def fit_transform_trait_state(
    x_stim_tr: np.ndarray,
    x_base_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_stim_ev: np.ndarray,
    x_base_ev: np.ndarray,
    *,
    mode: str = "factored_trait_plus_state",
    n_trait_dims: int = 16,
    shrinkage_ridge: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fold-isolated Trait / State factorization without blindly discarding trait info.

    Modes
    -----
    - ``"raw_unfactored"``: ``x_stim`` unchanged.
    - ``"blind_ref_sub"``: naive ``x_stim - x_base`` (fixes ``w = 1``, amplifies
      3-s baseline noise when ``corr < 1``).
    - ``"state_only_shrinkage"``: learns diagonal shrinkage weights ``w_j`` on
      ``train`` minimizing ``Var(x_stim_j - w_j * x_base_j)``, returning denoised
      transient state ``z_state`` only.
    - ``"trait_only"``: extracts regularized ANOVA between-person trait subspace
      ``z_trait`` from ``x_base`` (maximizing between- vs within-person variance
      ratio strictly inside ``train``).
    - ``"factored_trait_plus_state"``: returns ``[z_state, z_trait]`` preserving
      useful trait predictors alongside denoised transient state.
    """
    s_tr = np.asarray(x_stim_tr, dtype=np.float64)
    b_tr = np.asarray(x_base_tr, dtype=np.float64)
    s_ev = np.asarray(x_stim_ev, dtype=np.float64)
    b_ev = np.asarray(x_base_ev, dtype=np.float64)

    imp_s = SimpleImputer(strategy="median").fit(s_tr)
    imp_b = SimpleImputer(strategy="median").fit(b_tr)
    s_tr = imp_s.transform(s_tr)
    s_ev = imp_s.transform(s_ev)
    b_tr = imp_b.transform(b_tr)
    b_ev = imp_b.transform(b_ev)

    if mode == "raw_unfactored":
        return s_tr.astype(np.float32), s_ev.astype(np.float32), {"mode": mode}

    if mode == "blind_ref_sub":
        return (
            (s_tr - b_tr).astype(np.float32),
            (s_ev - b_ev).astype(np.float32),
            {"mode": mode},
        )

    # Optimal per-feature shrinkage regression of x_stim on x_base (train only!)
    mu_s = s_tr.mean(axis=0, keepdims=True)
    mu_b = b_tr.mean(axis=0, keepdims=True)
    sc_tr = s_tr - mu_s
    bc_tr = b_tr - mu_b
    cov = np.mean(sc_tr * bc_tr, axis=0)
    var_b = np.mean(bc_tr * bc_tr, axis=0)
    w_shrink = np.clip(cov / (var_b + shrinkage_ridge), 0.0, 1.0)

    state_tr = (s_tr - mu_s) - (b_tr - mu_b) * w_shrink[None, :]
    state_ev = (s_ev - mu_s) - (b_ev - mu_b) * w_shrink[None, :]

    if mode == "state_only_shrinkage":
        return (
            state_tr.astype(np.float32),
            state_ev.astype(np.float32),
            {
                "mode": mode,
                "mean_shrink_w": round(float(np.mean(w_shrink)), 4),
                "median_shrink_w": round(float(np.median(w_shrink)), 4),
            },
        )

    # Regularized ANOVA / Between-Person Trait Subspace on x_base_tr
    scl_b = StandardScaler().fit(b_tr)
    bz_tr = scl_b.transform(b_tr)
    bz_ev = scl_b.transform(b_ev)
    # Reduce dimension with PCA first if high-dim for stable covariance
    d_work = min(64, bz_tr.shape[1], bz_tr.shape[0] - 5)
    pca_b = PCA(n_components=d_work, random_state=0).fit(bz_tr)
    bp_tr = pca_b.transform(bz_tr)
    bp_ev = pca_b.transform(bz_ev)

    pids_tr = df_tr["participant_id"].to_numpy()
    df_bp = pd.DataFrame(bp_tr)
    p_means = df_bp.groupby(pids_tr, sort=False).transform("mean").to_numpy()
    within = bp_tr - p_means
    cov_b = (p_means.T @ p_means) / max(len(bp_tr) - 1, 1)
    cov_w = (within.T @ within) / max(len(bp_tr) - 1, 1)
    # Regularized generalized eigenproblem: cov_b v = lam (cov_w + 0.1 I) v
    cov_w_reg = cov_w + 0.1 * np.eye(d_work)
    eig_w, vec_w = np.linalg.eigh(cov_w_reg)
    eig_w = np.maximum(eig_w, 1e-6)
    inv_sqrt = vec_w @ np.diag(1.0 / np.sqrt(eig_w)) @ vec_w.T
    mat = inv_sqrt @ cov_b @ inv_sqrt
    eig_vals, eig_vecs = np.linalg.eigh(0.5 * (mat + mat.T))
    order = np.argsort(eig_vals)[::-1]
    k_t = min(n_trait_dims, d_work)
    proj = inv_sqrt @ eig_vecs[:, order[:k_t]]
    trait_tr = bp_tr @ proj
    trait_ev = bp_ev @ proj
    top_ratio = float(np.mean(eig_vals[order[:k_t]]))

    if mode == "trait_only":
        return (
            trait_tr.astype(np.float32),
            trait_ev.astype(np.float32),
            {
                "mode": mode,
                "n_trait_dims": int(k_t),
                "mean_bw_ratio": round(top_ratio, 3),
            },
        )

    if mode == "factored_trait_plus_state":
        return (
            np.hstack([state_tr, trait_tr]).astype(np.float32),
            np.hstack([state_ev, trait_ev]).astype(np.float32),
            {
                "mode": mode,
                "n_state_dims": int(state_tr.shape[1]),
                "n_trait_dims": int(k_t),
                "mean_shrink_w": round(float(np.mean(w_shrink)), 4),
                "mean_bw_ratio": round(top_ratio, 3),
            },
        )
    raise ValueError(f"Unknown trait/state mode: {mode}")


# ---------------------------------------------------------------------------
# 5. Nested Stress & Affect Linear Probing (Cold-Start & Unlabeled History)
# ---------------------------------------------------------------------------


def fit_predict_nested_affect_stress(
    x_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    *,
    max_pca: int = 64,
    cs: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    alphas: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0),
    inner_k: int = 3,
) -> dict[str, np.ndarray]:
    """Fit balanced LogReg + Ridge V&A with nested inner person x theme CV.

    Guarantees zero eval-batch leakage: ``sd_pv`` and ``sd_pa`` are computed
    exclusively from inner-OOF training predictions.
    """
    ipf, itf = inner_blocked_folds(df_tr, k=inner_k)
    y_s = label(df_tr).to_numpy(int)
    ppl = df_tr["participant_id"]
    yv = (
        df_tr["rating_valence"] - df_tr["rating_valence"].groupby(ppl).transform("mean")
    ).to_numpy(float)
    ya = (
        df_tr["rating_arousal"] - df_tr["rating_arousal"].groupby(ppl).transform("mean")
    ).to_numpy(float)

    # Select C by inner grouped person x theme OOF within-person AUC
    best_c = cs[len(cs) // 2]
    best_auc = -1.0
    oof_by_c = {c: np.full(len(df_tr), np.nan, dtype=float) for c in cs}
    for p in range(inner_k):
        for t in range(inner_k):
            ev_m = (ipf == p) & (itf == t)
            tr_m = (ipf != p) & (itf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 20 or len(np.unique(y_s[tr_m])) < 2:
                continue
            z_tr_i, z_ev_i = _preprocess_fit_transform(
                x_tr[tr_m], x_tr[ev_m], max_pca=max_pca
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for c_val in cs:
                    clf = LogisticRegression(
                        C=c_val, class_weight="balanced", max_iter=600
                    ).fit(z_tr_i, y_s[tr_m])
                    oof_by_c[c_val][ev_m] = clf.decision_function(z_ev_i)

    for c_val in cs:
        pmap = ssl_eval.person_stress_aucs(
            df_tr,
            pd.Series(oof_by_c[c_val], index=df_tr.index),
            pd.Series(y_s.astype(bool), index=df_tr.index),
            pd.Series(~y_s.astype(bool), index=df_tr.index),
        )
        score = float(np.mean(list(pmap.values()))) if pmap else 0.5
        if score > best_auc:
            best_auc = score
            best_c = c_val

    z_tr_f, z_ev_f = _preprocess_fit_transform(x_tr, x_ev, max_pca=max_pca)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_f = LogisticRegression(C=best_c, class_weight="balanced", max_iter=600).fit(
            z_tr_f, y_s
        )
        sc_log = clf_f.decision_function(z_ev_f)
        prob_log = clf_f.predict_proba(z_ev_f)[:, 1]

    # Nested Ridge for valence & arousal (train inner-OOF SD scaling!)
    pv, oof_v, _ = fit_predict_nested_ridge(
        x_tr,
        yv,
        df_tr,
        x_ev,
        alphas=alphas,
        pca_grid=(max_pca,),
        inner_k=inner_k,
        return_inner_oof=True,
    )
    pa, oof_a, _ = fit_predict_nested_ridge(
        x_tr,
        ya,
        df_tr,
        x_ev,
        alphas=alphas,
        pca_grid=(max_pca,),
        inner_k=inner_k,
        return_inner_oof=True,
    )
    ok_v = np.isfinite(oof_v)
    ok_a = np.isfinite(oof_a)
    sd_v = max(float(np.std(oof_v[ok_v])) if ok_v.sum() > 5 else 1.0, 1e-6)
    sd_a = max(float(np.std(oof_a[ok_a])) if ok_a.sum() > 5 else 1.0, 1e-6)
    sc_va = pa / sd_a - pv / sd_v
    return {
        "stress_logreg": sc_log.astype(np.float32),
        "stress_prob": prob_log.astype(np.float32),
        "stress_va": sc_va.astype(np.float32),
        "valence": pv.astype(np.float32),
        "arousal": pa.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# 6. Per-subject Forecasting Metrics, Paired CIs, Wilcoxon, & BH-FDR
# ---------------------------------------------------------------------------


def evaluate_forecast_per_person(
    frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    """Compute per-participant forecasting metrics (r, R^2_within, R^2_raw, RMSE)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    out: dict[str, dict[str, float]] = {}
    pids = frame["participant_id"].astype(str).to_numpy()
    for pid in np.unique(pids):
        m = (pids == pid) & np.isfinite(y) & np.isfinite(p)
        if m.sum() < MIN_FORECAST_TRIALS:
            continue
        yi, pi = y[m], p[m]
        sy, sp = float(np.std(yi)), float(np.std(pi))
        if sy <= 1e-12:
            continue
        r = float(np.corrcoef(yi, pi)[0, 1]) if sp > 1e-12 else 0.0
        yc = yi - yi.mean()
        pc = pi - pi.mean()
        ss_tot = float(np.sum(yc * yc))
        r2_within = float(1.0 - np.sum((yc - pc) ** 2) / ss_tot)
        r2_raw = float(1.0 - np.sum((yi - pi) ** 2) / ss_tot)
        rmse = float(np.sqrt(np.mean((yi - pi) ** 2)))
        rmse_centered = float(np.sqrt(np.mean((yc - pc) ** 2)))
        out[str(pid)] = {
            "r": r,
            "r2_within": r2_within,
            "r2_raw": r2_raw,
            "rmse": rmse,
            "rmse_centered": rmse_centered,
            "n_trials": int(m.sum()),
        }
    return out


def summarize_forecast(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> tuple[dict, dict[str, dict[str, float]]]:
    """Aggregate per-person and pooled forecasting metrics with 95% bootstrap CIs."""
    pp = evaluate_forecast_per_person(frame, y_true, y_pred)
    summary: dict[str, Any] = {"n_people": len(pp)}
    for key in ("r", "r2_within", "r2_raw", "rmse", "rmse_centered"):
        vals = np.array([v[key] for v in pp.values()], dtype=float)
        ci = ssl_eval.boot_ci(vals, draws=draws, seed=seed)
        if len(vals) > 0:
            q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90])
            summary[key] = {
                **ci,
                "median": round(float(q50), 4),
                "iqr": [round(float(q25), 4), round(float(q75), 4)],
                "p10_p90": [round(float(q10), 4), round(float(q90), 4)],
            }
        else:
            summary[key] = ci

    # Pooled R^2 & Pearson r (explicitly separated from within-person!)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() > 20 and np.std(y[ok]) > 1e-12 and np.std(p[ok]) > 1e-12:
        pr = float(np.corrcoef(y[ok], p[ok])[0, 1])
        ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
        r2_pool = float(1.0 - np.sum((y[ok] - p[ok]) ** 2) / ss_tot)
        summary["pooled"] = {
            "r": round(pr, 4),
            "r2": round(r2_pool, 4),
            "rmse": round(float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))), 4),
            "n_trials": int(ok.sum()),
        }
    else:
        summary["pooled"] = {
            "r": None,
            "r2": None,
            "rmse": None,
            "n_trials": int(ok.sum()),
        }

    # Within-video idiosyncratic r (across viewers watching the exact same video!)
    summary["within_video"] = evaluate_forecast_within_video(
        frame, y_true, y_pred, draws=draws, seed=seed
    )
    return summary, pp


def evaluate_forecast_within_video(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict:
    """Within-video Pearson r across viewers of the exact same clip."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if "video" not in frame.columns:
        return {"mean": None, "ci95": [None, None], "n_videos": 0}
    vids = frame["video"].astype(str).to_numpy()
    rs = []
    for vid in np.unique(vids):
        m = (vids == vid) & np.isfinite(y) & np.isfinite(p)
        if m.sum() < MIN_VIDEO_VIEWERS:
            continue
        yi, pi = y[m], p[m]
        if np.std(yi) > 1e-12 and np.std(pi) > 1e-12:
            rs.append(float(np.corrcoef(yi, pi)[0, 1]))
    ci = ssl_eval.boot_ci(rs, draws=draws, seed=seed)
    return {"mean": ci["mean"], "ci95": ci["ci95"], "n_videos": ci["n"]}


def paired_forecast_comparison(
    pp_target: dict[str, dict[str, float]],
    pp_ref: dict[str, dict[str, float]],
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict:
    """Paired ``target - ref`` across matched participants with CIs & Wilcoxon p."""
    keys = sorted(set(pp_target) & set(pp_ref))
    if len(keys) < 5:
        return {"n_people": len(keys), "error": "insufficient_matched_people"}
    out: dict[str, Any] = {"n_people": len(keys)}
    for key, higher_is_better in (
        ("r", True),
        ("r2_within", True),
        ("r2_raw", True),
        ("rmse", False),
        ("rmse_centered", False),
    ):
        a = np.array([pp_target[k][key] for k in keys], dtype=float)
        b = np.array([pp_ref[k][key] for k in keys], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        diff = (a[ok] - b[ok]) if higher_is_better else (b[ok] - a[ok])
        ci = ssl_eval.boot_ci(diff, draws=draws, seed=seed)
        if len(diff) >= 5 and np.any(diff != 0):
            try:
                pval = float(wilcoxon(diff, alternative="two-sided").pvalue)
            except ValueError:
                pval = 1.0
        else:
            pval = 1.0
        q10, q25, q50, q75, q90 = np.percentile(diff, [10, 25, 50, 75, 90])
        label_key = f"delta_{key}" if higher_is_better else f"reduction_{key}"
        out[label_key] = {
            "mean": ci["mean"],
            "ci95": ci["ci95"],
            "n": ci["n"],
            "p_wilcoxon": round(pval, 6),
            "median": round(float(q50), 4),
            "iqr": [round(float(q25), 4), round(float(q75), 4)],
            "p10_p90": [round(float(q10), 4), round(float(q90), 4)],
            "frac_improved": round(float(np.mean(diff > 0)), 4),
        }
    return out


def bh_fdr(pvals: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg FDR q-values for a mapping ``{name: p_value}``."""
    keys = [k for k, v in pvals.items() if v is not None and np.isfinite(v)]
    if not keys:
        return {}
    ps = np.array([pvals[k] for k in keys], dtype=float)
    order = np.argsort(ps)
    n = len(ps)
    q_ord = np.empty(n, dtype=float)
    running = 1.0
    for rank_rev in range(n - 1, -1, -1):
        idx = order[rank_rev]
        running = min(running, ps[idx] * n / (rank_rev + 1))
        q_ord[rank_rev] = running
    return {
        keys[idx]: round(float(min(1.0, q_ord[pos])), 6)
        for pos, idx in enumerate(order)
    }


# ---------------------------------------------------------------------------
# 7. Complementarity Gate & Conditional Nested OOF Stacking (Requirement 12)
# ---------------------------------------------------------------------------


def assess_complementarity(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    oof_preds: dict[str, np.ndarray],
    *,
    task: str = "stress",
    draws: int = DRAWS,
    max_corr_threshold: float = 0.85,
) -> dict:
    """Quantify pairwise redundancy & conditional contribution BEFORE any stacking.

    Applies the pre-registered Study-09 Complementarity Gate:
    1. At least two models reliably above chance (95% CI > 0.50 for stress AUC,
       or > 0 for regression r);
    2. Pairwise within-person prediction correlation < ``max_corr_threshold``;
    3. At least one secondary model exhibits strictly positive within-person
       partial correlation ``r(y, pred_b | pred_best)`` with 95% CI > 0.
    """
    names = list(oof_preds.keys())
    y = np.asarray(y_true, dtype=float)
    pids = frame["participant_id"].astype(str).to_numpy()

    indiv: dict[str, dict] = {}
    for nm in names:
        pr = np.asarray(oof_preds[nm], dtype=float)
        if task == "stress":
            pmap = ssl_eval.person_stress_aucs(
                frame,
                pd.Series(pr, index=frame.index),
                pd.Series(y.astype(bool), index=frame.index),
                pd.Series(~y.astype(bool), index=frame.index),
            )
            ci = ssl_eval.boot_ci(list(pmap.values()), draws=draws)
            chance = 0.5
        else:
            pp = evaluate_forecast_per_person(frame, y, pr)
            ci = ssl_eval.boot_ci([v["r"] for v in pp.values()], draws=draws)
            chance = 0.0
        above_chance = bool(ci["ci95"][0] is not None and ci["ci95"][0] > chance)
        indiv[nm] = {**ci, "above_chance": above_chance}

    # Best individual model
    ranked = sorted(
        names,
        key=lambda k: indiv[k]["mean"] if indiv[k]["mean"] is not None else -1e9,
        reverse=True,
    )
    best_name = ranked[0] if ranked else None

    # Pairwise within-person prediction correlation matrix
    pair_corr: dict[str, dict[str, float | None]] = {a: {} for a in names}
    for i, a in enumerate(names):
        pa = np.asarray(oof_preds[a], dtype=float)
        for j, b in enumerate(names):
            if i == j:
                pair_corr[a][b] = 1.0
                continue
            pb = np.asarray(oof_preds[b], dtype=float)
            rs = []
            for pid in np.unique(pids):
                m = (pids == pid) & np.isfinite(pa) & np.isfinite(pb)
                if (
                    m.sum() >= MIN_FORECAST_TRIALS
                    and np.std(pa[m]) > 1e-12
                    and np.std(pb[m]) > 1e-12
                ):
                    rs.append(float(np.corrcoef(pa[m], pb[m])[0, 1]))
            pair_corr[a][b] = round(float(np.mean(rs)), 4) if rs else None

    # Within-person partial correlation r(y, pred_b | pred_best)
    partial_vs_best: dict[str, dict] = {}
    qualifying_secondary: list[str] = []
    if best_name is not None:
        p_best = np.asarray(oof_preds[best_name], dtype=float)
        for nm in names:
            if nm == best_name:
                continue
            p_sec = np.asarray(oof_preds[nm], dtype=float)
            part_rs = []
            for pid in np.unique(pids):
                m = (
                    (pids == pid)
                    & np.isfinite(y)
                    & np.isfinite(p_best)
                    & np.isfinite(p_sec)
                )
                if m.sum() < MIN_FORECAST_TRIALS + 2:
                    continue
                yi, bi, si = y[m], p_best[m], p_sec[m]
                if np.std(yi) < 1e-12 or np.std(bi) < 1e-12 or np.std(si) < 1e-12:
                    continue
                # Residualize both y and p_sec on p_best within person
                bc = bi - bi.mean()
                denom = float(np.dot(bc, bc))
                if denom <= 1e-12:
                    continue
                ry = (yi - yi.mean()) - (float(np.dot(yi - yi.mean(), bc)) / denom) * bc
                rs_vec = (si - si.mean()) - (
                    float(np.dot(si - si.mean(), bc)) / denom
                ) * bc
                if np.std(ry) > 1e-12 and np.std(rs_vec) > 1e-12:
                    part_rs.append(float(np.corrcoef(ry, rs_vec)[0, 1]))
            ci_p = ssl_eval.boot_ci(part_rs, draws=draws)
            corr_with_best = pair_corr[best_name][nm]
            passes = bool(
                indiv[best_name]["above_chance"]
                and indiv[nm]["above_chance"]
                and (corr_with_best is not None and corr_with_best < max_corr_threshold)
                and (ci_p["ci95"][0] is not None and ci_p["ci95"][0] > 0.0)
            )
            if passes:
                qualifying_secondary.append(nm)
            partial_vs_best[nm] = {
                "corr_with_best": corr_with_best,
                "partial_r_given_best": ci_p["mean"],
                "partial_r_ci95": ci_p["ci95"],
                "n_people": ci_p["n"],
                "passes_complementarity": passes,
            }

    gate_passed = len(qualifying_secondary) >= 1
    return {
        "task": task,
        "best_single_model": best_name,
        "individual_scores": indiv,
        "pairwise_within_person_corr": pair_corr,
        "partial_contribution_vs_best": partial_vs_best,
        "qualifying_secondary_models": qualifying_secondary,
        "gate_passed": gate_passed,
    }


def run_nested_oof_stacker(
    dev_df: pd.DataFrame,
    y_true: np.ndarray,
    oof_preds: dict[str, np.ndarray],
    *,
    task: str = "stress",
    k: int = 3,
    draws: int = DRAWS,
) -> dict:
    """Fold-isolated nested OOF stacking (run ONLY if complementarity gate passed)."""
    names = list(oof_preds.keys())
    x_meta = np.column_stack([np.asarray(oof_preds[nm], dtype=float) for nm in names])
    pf, tf = outer_blocked_folds(dev_df, k=k)
    y = np.asarray(y_true, dtype=float)
    stacked = np.full(len(dev_df), np.nan, dtype=float)
    fold_weights = []

    for p in range(k):
        for t in range(k):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 20:
                continue
            z_tr, z_ev = _preprocess_fit_transform(x_meta[tr_m], x_meta[ev_m])
            if task == "stress":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(
                        C=0.1, class_weight="balanced", max_iter=500
                    ).fit(z_tr, y[tr_m].astype(int))
                    stacked[ev_m] = clf.decision_function(z_ev)
                    fold_weights.append(clf.coef_[0].tolist())
            else:
                pred_ev, _ = fit_predict_nested_ridge(
                    z_tr,
                    y[tr_m],
                    dev_df.iloc[tr_m],
                    z_ev,
                    alphas=(1.0, 10.0, 100.0, 1000.0),
                    pca_grid=(None,),
                    inner_k=3,
                )
                stacked[ev_m] = pred_ev
                # Approximate weight direction for interpretability
                ok = np.isfinite(y[tr_m])
                w = np.linalg.lstsq(
                    z_tr[ok].T @ z_tr[ok] + 100.0 * np.eye(z_tr.shape[1]),
                    z_tr[ok].T @ (y[tr_m][ok] - np.mean(y[tr_m][ok])),
                    rcond=None,
                )[0]
                fold_weights.append(w.tolist())

    mean_w = (
        {
            nm: round(float(v), 4)
            for nm, v in zip(names, np.mean(fold_weights, axis=0), strict=True)
        }
        if fold_weights
        else {}
    )
    if task == "stress":
        res_st = ssl_eval.evaluate_stress(
            dev_df,
            pd.Series(stacked, index=dev_df.index),
            draws=draws,
            return_per_person=True,
        )
        pp_st = res_st.pop("_per_person")
        return {
            "models": names,
            "mean_meta_weights": mean_w,
            "metrics": res_st,
            "_per_person": pp_st,
            "_oof_pred": stacked,
        }
    summ, pp = summarize_forecast(dev_df, y, stacked, draws=draws)
    return {
        "models": names,
        "mean_meta_weights": mean_w,
        "metrics": summ,
        "_per_person": pp,
        "_oof_pred": stacked,
    }


def evaluate_brier_and_pooled(
    frame: pd.DataFrame, prob: np.ndarray, score: np.ndarray
) -> dict:
    """Compute pooled AUC, Brier score, and within-person Spearman rho on stress."""
    y = label(frame).to_numpy(int)
    ok_s = np.isfinite(score)
    ok_p = np.isfinite(prob)
    pooled_auc = (
        round(float(roc_auc_score(y[ok_s], score[ok_s])), 4)
        if ok_s.sum() > 20 and len(np.unique(y[ok_s])) == 2
        else None
    )
    brier = (
        round(float(brier_score_loss(y[ok_p], np.clip(prob[ok_p], 1e-6, 1 - 1e-6))), 4)
        if ok_p.sum() > 20
        else None
    )
    # Trait-level stress propensity correlation across participants
    df_t = pd.DataFrame(
        {"p": frame["participant_id"].to_numpy(), "y": y.astype(float), "s": score}
    ).dropna()
    agg = df_t.groupby("p").agg(rate=("y", "mean"), pred=("s", "mean"), n=("y", "size"))
    agg = agg[agg["n"] >= 5]
    trait_r = (
        round(float(spearmanr(agg["rate"], agg["pred"]).statistic), 4)
        if len(agg) >= 15 and agg["rate"].std() > 0 and agg["pred"].std() > 0
        else None
    )
    return {
        "pooled_auc": pooled_auc,
        "brier": brier,
        "between_person_trait_rho": trait_r,
        "n_trait_people": int(len(agg)),
    }
