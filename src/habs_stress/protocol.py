"""Zero-leakage person x theme splits, nested CV, normalization, and inference."""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

STUDY_SALT = "habs-ssl-2026-11-24-b"
MIN_EACH = 3
MIN_FORECAST_TRIALS = 5


def hash_buckets(keys: pd.Series | np.ndarray, salt: str, k: int) -> np.ndarray:
    """Deterministically hash group identifiers into k buckets (0..k-1)."""
    return np.array(
        [
            int(hashlib.sha256(f"{salt}|{str(key)}".encode()).hexdigest(), 16) % k
            for key in keys
        ],
        dtype=int,
    )


def assign_splits(table: pd.DataFrame, salt: str = STUDY_SALT) -> pd.DataFrame:
    """Assign double-unseen person x theme holdout and 3 x 3 dev folds."""
    theme = table["video_theme"].astype("string")
    stem = table["video"].astype(str).str.replace(r"_\d+(\.\w+)?$", "", regex=True)
    df = table.assign(theme=theme.fillna(stem).astype(str))
    people = df["participant_id"].astype(str)
    p5 = hash_buckets(people, f"{salt}|people", 5)
    t5 = hash_buckets(df["theme"], f"{salt}|themes", 5)
    split = np.where(
        p5 == 0, np.where(t5 == 0, "test_new_both", "test_new_person"), "dev"
    )
    split = np.where((p5 != 0) & (t5 == 0), "test_new_video", split)
    return df.assign(
        split=split,
        person_fold=hash_buckets(people, f"{salt}|people-cv", 3),
        theme_fold=hash_buckets(df["theme"], f"{salt}|themes-cv", 3),
    )


def inner_blocked_folds(
    df: pd.DataFrame, *, k: int = 3, salt: str = "inner-v1"
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent inner (person_fold, theme_fold) arrays inside df."""
    return (
        hash_buckets(df["participant_id"], f"{salt}|p-{k}", k),
        hash_buckets(df["theme"], f"{salt}|t-{k}", k),
    )


def causal_person_center(
    x: np.ndarray,
    people: np.ndarray,
    order_keys: np.ndarray,
    *,
    prior_mean: np.ndarray,
    shrinkage: float = 5.0,
) -> np.ndarray:
    """Protocol B1: strictly causal online Empirical-Bayes centering.

    Trial i for a participant uses only strictly earlier trials (< i) shrunk
    toward ``prior_mean`` estimated on the training fold. Zero future leakage.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float32)
    prior = np.asarray(prior_mean, dtype=np.float64)
    pids = np.asarray(people, dtype=str)
    order = np.asarray(order_keys)

    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        seq = idx[np.argsort(order[idx], kind="mergesort")]
        sum_vec = np.zeros(arr.shape[1], dtype=np.float64)
        cnt_vec = np.zeros(arr.shape[1], dtype=np.float64)
        for row_i in seq:
            base = (sum_vec + shrinkage * prior) / (cnt_vec + shrinkage)
            out[row_i] = (arr[row_i] - base).astype(np.float32)
            fin = np.isfinite(arr[row_i])
            sum_vec[fin] += arr[row_i, fin]
            cnt_vec[fin] += 1.0
    return out


def prepare_protocol_fold(
    x_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    df_ev: pd.DataFrame,
    *,
    protocol: str = "cold_start",
    shrinkage: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate Cold-Start (Protocol A) from Causal Online (B1) and Session (B2)."""
    a_tr = np.asarray(x_tr, dtype=np.float32)
    a_ev = np.asarray(x_ev, dtype=np.float32)
    if protocol == "cold_start":
        return a_tr.copy(), a_ev.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_tr = np.nanmean(a_tr, axis=0)
    mu_tr = np.where(np.isfinite(mu_tr), mu_tr, 0.0)

    if protocol == "causal_online":
        ord_tr = df_tr["onset_s"].to_numpy() if "onset_s" in df_tr else np.arange(len(df_tr))
        ord_ev = df_ev["onset_s"].to_numpy() if "onset_s" in df_ev else np.arange(len(df_ev))
        return (
            causal_person_center(
                a_tr, df_tr["participant_id"].to_numpy(), ord_tr,
                prior_mean=mu_tr, shrinkage=shrinkage,
            ),
            causal_person_center(
                a_ev, df_ev["participant_id"].to_numpy(), ord_ev,
                prior_mean=mu_tr, shrinkage=shrinkage,
            ),
        )

    if protocol == "session_isolated":
        def _center_fold(mat: np.ndarray, pids: np.ndarray) -> np.ndarray:
            res = np.empty_like(mat, dtype=np.float32)
            for pid in np.unique(pids):
                m = pids == pid
                sub = mat[m]
                cnt = np.isfinite(sub).sum(axis=0)
                s = np.nansum(sub, axis=0)
                base = (s + shrinkage * mu_tr) / (cnt + shrinkage)
                res[m] = (sub - base).astype(np.float32)
            return res

        return (
            _center_fold(a_tr, df_tr["participant_id"].astype(str).to_numpy()),
            _center_fold(a_ev, df_ev["participant_id"].astype(str).to_numpy()),
        )
    raise ValueError(f"Unknown protocol: {protocol}")


def preprocess_fit_transform(
    x_tr: np.ndarray, x_ev: np.ndarray, *, max_pca: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-isolated median imputation, standardization, and optional PCA."""
    tr = np.asarray(x_tr, dtype=np.float64).copy()
    ev = np.asarray(x_ev, dtype=np.float64).copy()
    valid_cols = np.isfinite(tr).any(axis=0)
    if not valid_cols.any():
        return np.zeros((len(tr), 1)), np.zeros((len(ev), 1))
    tr, ev = tr[:, valid_cols], ev[:, valid_cols]
    meds = np.nanmedian(tr, axis=0)
    meds = np.where(np.isfinite(meds), meds, 0.0)
    tr = np.where(np.isfinite(tr), tr, meds[None, :])
    ev = np.where(np.isfinite(ev), ev, meds[None, :])

    mu = tr.mean(axis=0, keepdims=True)
    sd = tr.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z_tr = (tr - mu) / sd
    z_ev = (ev - mu) / sd
    if max_pca is not None and z_tr.shape[1] > max_pca and z_tr.shape[0] > max_pca + 5:
        pca = PCA(n_components=max_pca, svd_solver="full", random_state=0)
        z_tr = pca.fit_transform(z_tr)
        z_ev = pca.transform(z_ev)
    return z_tr, z_ev


def svd_ridge_fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form Ridge regression via SVD."""
    y_mean = float(np.mean(y_tr))
    x_mean = np.mean(x_tr, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_tr - x_mean, full_matrices=False)
    coef = vt.T @ ((s / (s * s + alpha)) * (u.T @ (y_tr - y_mean)))
    return y_mean + (x_ev - x_mean) @ coef


def fit_predict_nested_stress(
    x_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    *,
    max_pca: int = 64,
    cs: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
    alphas: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0),
    inner_k: int = 3,
) -> dict[str, np.ndarray]:
    """Fold-isolated nested LogisticRegression + Ridge Valence/Arousal scorer."""
    y_s = ((df_tr["rating_valence"] <= 4) & (df_tr["rating_arousal"] >= 6)).to_numpy(int)
    ppl = df_tr["participant_id"]
    yv = (df_tr["rating_valence"] - df_tr["rating_valence"].groupby(ppl).transform("mean")).to_numpy(float)
    ya = (df_tr["rating_arousal"] - df_tr["rating_arousal"].groupby(ppl).transform("mean")).to_numpy(float)
    ipf, itf = inner_blocked_folds(df_tr, k=inner_k)

    oof_c = {c: np.full(len(df_tr), np.nan) for c in cs}
    oof_v = {a: np.full(len(df_tr), np.nan) for a in alphas}
    oof_a = {a: np.full(len(df_tr), np.nan) for a in alphas}

    for p in range(inner_k):
        for t in range(inner_k):
            ev_m = (ipf == p) & (itf == t)
            tr_m = (ipf != p) & (itf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 15 or len(np.unique(y_s[tr_m])) < 2:
                continue
            zt, ze = preprocess_fit_transform(x_tr[tr_m], x_tr[ev_m], max_pca=max_pca)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for c_val in cs:
                    clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=500).fit(zt, y_s[tr_m])
                    oof_c[c_val][ev_m] = clf.decision_function(ze)
            for a in alphas:
                oof_v[a][ev_m] = svd_ridge_fit_predict(zt, yv[tr_m], ze, a)
                oof_a[a][ev_m] = svd_ridge_fit_predict(zt, ya[tr_m], ze, a)

    best_c = max(
        cs,
        key=lambda c: _mean_person_auc(df_tr["participant_id"].to_numpy(), y_s, oof_c[c]),
    )
    best_av = min(alphas, key=lambda a: _within_person_mse(yv, oof_v[a], ppl.to_numpy()))
    best_aa = min(alphas, key=lambda a: _within_person_mse(ya, oof_a[a], ppl.to_numpy()))

    zt_f, ze_f = preprocess_fit_transform(x_tr, x_ev, max_pca=max_pca)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_f = LogisticRegression(C=best_c, class_weight="balanced", max_iter=500).fit(zt_f, y_s)
        sc_log = clf_f.decision_function(ze_f)
    pv = svd_ridge_fit_predict(zt_f, yv, ze_f, best_av)
    pa = svd_ridge_fit_predict(zt_f, ya, ze_f, best_aa)
    sd_v = max(float(np.nanstd(oof_v[best_av])), 1e-6)
    sd_a = max(float(np.nanstd(oof_a[best_aa])), 1e-6)
    return {
        "stress_logreg": sc_log.astype(np.float32),
        "stress_va": (pa / sd_a - pv / sd_v).astype(np.float32),
        "valence": pv.astype(np.float32),
        "arousal": pa.astype(np.float32),
    }


def _mean_person_auc(people: np.ndarray, y: np.ndarray, scores: np.ndarray) -> float:
    aucs = []
    for pid in np.unique(people):
        m = (people == pid) & np.isfinite(scores)
        if y[m].sum() >= MIN_EACH and (1 - y[m]).sum() >= MIN_EACH:
            aucs.append(float(roc_auc_score(y[m], scores[m])))
    return float(np.mean(aucs)) if aucs else 0.5


def _within_person_mse(
    y_true: np.ndarray, y_pred: np.ndarray, people: np.ndarray
) -> float:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 10:
        return float("inf")
    mses = []
    for pid in np.unique(people[ok]):
        m = ok & (people == pid)
        if m.sum() < MIN_FORECAST_TRIALS:
            continue
        yc = y_true[m] - y_true[m].mean()
        pc = y_pred[m] - y_pred[m].mean()
        mses.append(float(np.mean((yc - pc) ** 2)))
    return float(np.mean(mses)) if mses else float(np.mean((y_true[ok] - y_pred[ok]) ** 2))


def paired_person_bootstrap(
    values: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap 95% CI and Wilcoxon test over per-participant metrics."""
    a = np.asarray(values, dtype=float)
    if reference is not None:
        b = np.asarray(reference, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        diff = a[ok] - b[ok]
    else:
        diff = a[np.isfinite(a)]
    if len(diff) == 0:
        return {"mean": np.nan, "ci95": [np.nan, np.nan], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    boot = diff[idx].mean(axis=1)
    res: dict[str, Any] = {
        "mean": float(diff.mean()),
        "median": float(np.median(diff)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": int(len(diff)),
    }
    if reference is not None:
        res["frac_improved"] = float((diff > 0).mean())
        if len(diff) >= 5 and np.any(diff != 0):
            _, pval = wilcoxon(diff)
            res["p_wilcoxon"] = float(pval)
        else:
            res["p_wilcoxon"] = 1.0
    return res


def benjamini_hochberg(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q
