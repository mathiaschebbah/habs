"""Domain adaptation (CORAL), k-shot EB-MAP, Varimax coherence, and passive monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from habs_stress.protocol import MIN_EACH, paired_person_bootstrap


def _sym_matrix_power(mat: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
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
    """Unsupervised CORAL domain alignment with optional Empirical-Bayes shrinkage.

    Aligns target evaluation rows ``z_tgt_eval`` to source distribution ``z_src``
    using statistics estimated strictly on unlabeled ``z_tgt_fit``.
    """
    zs = np.asarray(z_src, dtype=np.float64)
    zt_f = np.asarray(z_tgt_fit, dtype=np.float64)
    zt_e = np.asarray(z_tgt_eval, dtype=np.float64)
    n_t, d = len(zt_f), zs.shape[1]
    if n_t == 0:
        return zt_e.copy()

    mu_s = zs.mean(axis=0)
    cs_raw = np.atleast_2d(np.cov(zs, rowvar=False)) if len(zs) > 1 else np.eye(d)
    cs = cs_raw + reg * np.eye(d)

    mu_t = (n_t * zt_f.mean(axis=0) + mean_shrink * mu_s) / max(n_t + mean_shrink, 1e-8)
    ct_raw = np.atleast_2d(np.cov(zt_f, rowvar=False)) if n_t >= 2 else cs_raw.copy()
    ct = (n_t * ct_raw + cov_shrink * cs_raw) / max(n_t + cov_shrink, 1e-8) + reg * np.eye(d)

    transform = _sym_matrix_power(ct, -0.5) @ _sym_matrix_power(cs, 0.5)
    return (zt_e - mu_t[None, :]) @ transform + mu_s[None, :]


def eb_map_adapt_ridge(
    z_cal: np.ndarray,
    y_cal: np.ndarray,
    w_prior: np.ndarray,
    z_eval: np.ndarray,
    *,
    prior_lambda: float = 1000.0,
) -> np.ndarray:
    """Empirical-Bayes MAP Ridge update shrunk toward population weights ``w_prior``.

    Solves ``min_w ||z_cal @ w - y_cal||^2 + prior_lambda ||w - w_prior||^2``,
    which smoothly interpolates from Cold-Start (k = 0) toward personal weights
    without small-sample overfitting.
    """
    zc = np.asarray(z_cal, dtype=np.float64)
    yc = np.asarray(y_cal, dtype=np.float64)
    wp = np.asarray(w_prior, dtype=np.float64)
    ze = np.asarray(z_eval, dtype=np.float64)
    if len(zc) == 0:
        return ze @ wp
    zc_c = zc - zc.mean(axis=0, keepdims=True)
    yc_c = yc - yc.mean()
    resid = yc_c - zc_c @ wp
    d = zc.shape[1]
    delta_w = np.linalg.solve(zc_c.T @ zc_c + prior_lambda * np.eye(d), zc_c.T @ resid)
    return ze @ (wp + delta_w)


def varimax_rotation(loadings: np.ndarray, *, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Orthogonal Varimax rotation for interpretable factor loadings."""
    phi = np.asarray(loadings, dtype=np.float64).copy()
    p, k = phi.shape
    rot = np.eye(k)
    d_prev = 0.0
    for _ in range(max_iter):
        lam = phi @ rot
        u, s, vt = np.linalg.svd(
            phi.T @ (lam**3 - (1.0 / p) * lam * np.sum(lam**2, axis=0, keepdims=True))
        )
        rot = u @ vt
        d_new = float(np.sum(s))
        if abs(d_new - d_prev) < tol:
            break
        d_prev = d_new
    return phi @ rot


def subjective_autonomic_coherence(
    df: pd.DataFrame,
    variables: list[str],
    *,
    n_factors: int = 3,
) -> dict[str, Any]:
    """Within-person Varimax factor structure and per-subject subjective-autonomic coherence."""
    sub = df[["participant_id", *variables]].dropna().copy()
    centered = sub[variables] - sub.groupby("participant_id")[variables].transform("mean")
    z = (centered - centered.mean()) / centered.std(ddof=0).replace(0, 1.0)
    corr = np.corrcoef(z.to_numpy(float), rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    raw_load = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0.0))[None, :]
    rot_load = varimax_rotation(raw_load)

    return {
        "n_complete_trials": int(len(sub)),
        "variables": variables,
        "explained_variance_ratio": [float(v / len(variables)) for v in eigvals[:n_factors]],
        "varimax_loadings": {
            var: [round(float(rot_load[i, f]), 4) for f in range(n_factors)]
            for i, var in enumerate(variables)
        },
    }


def individual_coherence_distribution(
    df: pd.DataFrame,
    subjective_col: str,
    physiological_col: str,
    *,
    min_trials: int = 10,
) -> dict[str, Any]:
    """Distribution of within-person Spearman correlations (coherence vs. discordance)."""
    rhos = []
    for _, g in df.groupby("participant_id"):
        sub = g[[subjective_col, physiological_col]].dropna()
        if len(sub) < min_trials:
            continue
        r = spearmanr(sub[subjective_col], sub[physiological_col]).statistic
        if np.isfinite(r):
            rhos.append(float(r))
    arr = np.array(rhos, dtype=float)
    base = paired_person_bootstrap(arr)
    base["frac_positive"] = float((arr > 0).mean()) if len(arr) else np.nan
    base["frac_discordant"] = float((arr <= 0).mean()) if len(arr) else np.nan
    return base


def passive_monitoring_curve(
    df: pd.DataFrame,
    scores: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    k_grid: tuple[int, ...] = (1, 2, 3, 5, 8, 10),
    n_draws: int = 200,
    seed: int = 0,
) -> dict[int, dict[str, Any]]:
    """Aggregate K episodes (10 s to 100 s) within person for clinical monitoring curves.

    Reports within-person ROC-AUC and sensitivity at 80% and 90% specificity as
    temporal integration increases from a single 10-s episode (K = 1) to 100 s (K = 10).
    """
    rng = np.random.default_rng(seed)
    sc = np.asarray(scores, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    neg = np.asarray(negative, dtype=bool)
    ppl = df["participant_id"].astype(str).to_numpy()

    out: dict[int, dict[str, Any]] = {}
    for k in k_grid:
        aucs, s80s, s90s = [], [], []
        for pid in np.unique(ppl):
            m = (ppl == pid) & np.isfinite(sc)
            p_vals = sc[m & pos]
            n_vals = sc[m & neg]
            if len(p_vals) < MIN_EACH or len(n_vals) < MIN_EACH:
                continue
            if k == 1:
                y = np.r_[np.ones(len(p_vals)), np.zeros(len(n_vals))]
                s = np.r_[p_vals, n_vals]
            else:
                p_agg = p_vals[rng.integers(0, len(p_vals), size=(n_draws, k))].mean(axis=1)
                n_agg = n_vals[rng.integers(0, len(n_vals), size=(n_draws, k))].mean(axis=1)
                y = np.r_[np.ones(n_draws), np.zeros(n_draws)]
                s = np.r_[p_agg, n_agg]
            aucs.append(float(roc_auc_score(y, s)))
            fpr, tpr, _ = roc_curve(y, s)
            s80s.append(float(np.interp(0.20, fpr, tpr)))
            s90s.append(float(np.interp(0.10, fpr, tpr)))
        out[int(k)] = {
            "window_seconds": int(10 * k),
            "auc": paired_person_bootstrap(np.array(aucs)),
            "sens_at_spec80": paired_person_bootstrap(np.array(s80s)),
            "sens_at_spec90": paired_person_bootstrap(np.array(s90s)),
        }
    return out
