"""Publication-revision CPU experiments (Priorities 1, 2, 3 + Figure 1 coverage).

Runs strictly inside the audited development split (``habs-ssl-2026-11-24-b``,
12,693 trials, 228 participant IDs, 66 themes) using the private Parquet tables
cached in HABS Azure Storage. Outputs publication-safe aggregate metrics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

STUDY09_BASE_SALT = "habs-ssl-2026-11-24-b"
TIMECOURSE_SALT = "habs-timecourse-2026-11-20"
DRAWS = 2000
MIN_EACH = 3
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
REGIONS = ("frontal", "left_temporal", "right_temporal", "central", "parietal", "occipital")
CS_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ALPHAS_GRID = (10.0, 100.0, 1000.0, 10000.0)


# ---------------------------------------------------------------------------
# 1. Deterministic splitting, feature assembly, and post-column auditing
# ---------------------------------------------------------------------------


def hash_buckets(keys: pd.Series | np.ndarray, salt: str, k: int) -> np.ndarray:
    """Deterministic SHA-256 bucket assignment into ``0..k-1``."""
    return np.array(
        [int(hashlib.sha256(f"{salt}|{str(key)}".encode()).hexdigest(), 16) % k for key in keys],
        dtype=int,
    )


def assign_splits(table: pd.DataFrame, salt: str = STUDY09_BASE_SALT) -> pd.DataFrame:
    """Reproduce exact canonical Study-08/09/10 ID x theme split allocation."""
    theme = table["video_theme"].astype("string")
    stem = table["video"].astype(str).str.replace(r"_\d+(\.\w+)?$", "", regex=True)
    out = table.assign(theme=theme.fillna(stem).astype(str))
    people = out["participant_id"].astype(str)
    p5 = hash_buckets(people, f"{salt}|people", 5)
    t5 = hash_buckets(out["theme"], f"{salt}|themes", 5)
    split = np.where(p5 == 0, np.where(t5 == 0, "test_new_both", "test_new_person"), "dev")
    split = np.where((p5 != 0) & (t5 == 0), "test_new_video", split)
    return out.assign(
        split=split,
        person_fold=hash_buckets(people, f"{salt}|people-cv", 3),
        theme_fold=hash_buckets(out["theme"], f"{salt}|themes-cv", 3),
    )


def stress_label(frame: pd.DataFrame) -> pd.Series:
    """Primary acute distress target: valence <= 4 and arousal >= 6."""
    return (frame["rating_valence"] <= 4) & (frame["rating_arousal"] >= 6)


def stress_contrasts(frame: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Primary stress target plus four prespecified specificity contrasts."""
    v, a = frame["rating_valence"], frame["rating_arousal"]
    s = (v <= 4) & (a >= 6)
    return {
        "stress": (s, ~s),
        "strict": ((v <= 3) & (a >= 7), ~((v <= 3) & (a >= 7))),
        "vs_excitement": (s, (v >= 6) & (a >= 6)),
        "vs_calm_negative": (s, (v <= 4) & (a <= 4)),
        "vs_relaxed": (s, (v >= 6) & (a <= 4)),
    }


def build_s07_feature_df(frame: pd.DataFrame) -> pd.DataFrame:
    """Build exact uncentered Study-07 4-phase enriched feature table (448 cols)."""
    cols: dict[str, pd.Series] = {}

    def _add_phase_group(prefix: str, names: tuple[str, ...]) -> None:
        for nm in names:
            ref = frame[f"{prefix}_{nm}_ref"]
            early = frame[f"{prefix}_{nm}_early"]
            late = frame[f"{prefix}_{nm}_late"]
            post = frame[f"{prefix}_{nm}_post"]
            cols[f"{prefix}_{nm}_e_chg"] = early - ref
            cols[f"{prefix}_{nm}_l_chg"] = late - ref
            cols[f"{prefix}_{nm}_p_chg"] = post - ref
            cols[f"{prefix}_{nm}_d_l_e"] = late - early
            cols[f"{prefix}_{nm}_d_p_l"] = post - late

    _add_phase_group("ecg", ("hr", "rr"))
    _add_phase_group("sc", ("mean", "rise", "slope"))
    _add_phase_group("resp", ("amp", "rate", "slope"))

    for r in REGIONS:
        for b in BANDS:
            ref = frame[f"eeg_ep_ref_{r}_{b}"]
            e = frame[f"eeg_ep_early_{r}_{b}"]
            l_ = frame[f"eeg_ep_late_{r}_{b}"]
            p_ = frame[f"eeg_ep_post_{r}_{b}"]
            cols[f"eeg_{r}_{b}_e_chg"] = e - ref
            cols[f"eeg_{r}_{b}_l_chg"] = l_ - ref
            cols[f"eeg_{r}_{b}_p_chg"] = p_ - ref
            cols[f"eeg_{r}_{b}_d_l_e"] = l_ - e
            cols[f"eeg_{r}_{b}_d_p_l"] = p_ - l_

    _add_phase_group(
        "ecg",
        ("pr_pk", "pr_on", "pr_seg", "qrs", "qt", "qtc_baz", "qtc_frid", "rt_pk", "t_amp"),
    )
    for m in ("pr_pk", "pr_on", "pr_seg", "qrs", "qt", "qtc_baz", "rt_pk", "t_amp"):
        cols[f"ecg_{m}_vid_chg"] = frame[f"ecg_{m}_vid"] - frame[f"ecg_{m}_ref"]

    for fam in ("sp_std", "sp_iqr", "sp_ent", "reg_std", "ap_grad", "faa", "temp_asym"):
        for b in BANDS:
            ref = frame[f"eeg_{fam}_ref_{b}"]
            e = frame[f"eeg_{fam}_early_{b}"]
            l_ = frame[f"eeg_{fam}_late_{b}"]
            p_ = frame[f"eeg_{fam}_post_{b}"]
            cols[f"eeg_{fam}_{b}_e_chg"] = e - ref
            cols[f"eeg_{fam}_{b}_l_chg"] = l_ - ref
            cols[f"eeg_{fam}_{b}_p_chg"] = p_ - ref
            cols[f"eeg_{fam}_{b}_d_l_e"] = l_ - e
            cols[f"eeg_{fam}_{b}_d_p_l"] = p_ - l_

    for b in BANDS:
        tref = frame[f"eeg_tvar_ref_{b}"]
        cols[f"eeg_tvar_e_{b}"] = frame[f"eeg_tvar_early_{b}"] - tref
        cols[f"eeg_tvar_l_{b}"] = frame[f"eeg_tvar_late_{b}"] - tref
        cols[f"eeg_tslope_e_{b}"] = frame[f"eeg_tslope_early_{b}"]
        cols[f"eeg_tslope_l_{b}"] = frame[f"eeg_tslope_late_{b}"]
        cols[f"eeg_maxjump_e_{b}"] = frame[f"eeg_maxjump_early_{b}"]
        cols[f"eeg_maxjump_l_{b}"] = frame[f"eeg_maxjump_late_{b}"]

    return pd.DataFrame(cols, index=frame.index)


def split_post_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Strictly separate features referencing ``post [10, 12) s`` from pre-10s features."""
    post_cols = [
        c
        for c in columns
        if ("post" in c.lower()) or c.endswith("_p_chg") or c.endswith("_d_p_l")
    ]
    no_post_cols = [c for c in columns if c not in set(post_cols)]
    return no_post_cols, post_cols


def build_session_order_and_prev_trial(
    dev_df: pd.DataFrame,
    x_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build within-session trial indices and strictly previous-trial unlabeled features.

    Resets cleanly at every ``(participant_id, recorded_day)`` boundary.
    Returns ``(idx_raw, idx_norm, is_first, x_prev)`` aligned to ``dev_df``.
    """
    n = len(dev_df)
    sess_keys = (
        ["participant_id", "recorded_day"]
        if "recorded_day" in dev_df.columns
        else ["participant_id"]
    )
    sort_cols = [c for c in ("event_order", "onset_s") if c in dev_df.columns]
    order_df = dev_df[[*sess_keys, *sort_cols]].copy()
    order_df["_pos"] = np.arange(n, dtype=int)
    order_df = order_df.sort_values([*sess_keys, *sort_cols, "_pos"], kind="mergesort")

    idx_raw = np.zeros(n, dtype=np.float32)
    idx_norm = np.zeros(n, dtype=np.float32)
    is_first = np.zeros(n, dtype=bool)
    x_prev = np.full(x_source.shape, np.nan, dtype=np.float32)

    for _, grp in order_df.groupby(sess_keys, sort=False):
        pos = grp["_pos"].to_numpy(int)
        m = len(pos)
        ranks = np.arange(m, dtype=np.float32)
        idx_raw[pos] = ranks
        idx_norm[pos] = ranks / max(1.0, float(m - 1))
        is_first[pos[0]] = True
        if m > 1:
            x_prev[pos[1:]] = x_source[pos[:-1]]

    return idx_raw, idx_norm, is_first, x_prev


# ---------------------------------------------------------------------------
# 2. Fold-isolated preprocessing, fast nested fitting, and paired metrics
# ---------------------------------------------------------------------------


def _preprocess_fit_transform(
    x_tr: np.ndarray, x_ev: np.ndarray, *, max_pca: int | None = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-isolated median imputation, standardization, and optional PCA."""
    x_tr_f = np.asarray(x_tr, dtype=np.float32)
    x_ev_f = np.asarray(x_ev, dtype=np.float32)
    valid_cols = np.isfinite(x_tr_f).any(axis=0)
    if not valid_cols.any():
        return np.zeros((len(x_tr_f), 1), dtype=np.float32), np.zeros(
            (len(x_ev_f), 1), dtype=np.float32
        )
    x_tr_f = x_tr_f[:, valid_cols].copy()
    x_ev_f = x_ev_f[:, valid_cols].copy()

    nan_tr = ~np.isfinite(x_tr_f)
    nan_ev = ~np.isfinite(x_ev_f)
    if nan_tr.any() or nan_ev.any():
        need = nan_tr.any(axis=0) | nan_ev.any(axis=0)
        cidx = np.where(need)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            meds = np.nanmedian(x_tr_f[:, cidx], axis=0)
        meds = np.where(np.isfinite(meds), meds, 0.0).astype(np.float32)
        rt, ct = np.where(nan_tr[:, cidx])
        if len(rt):
            x_tr_f[rt, cidx[ct]] = meds[ct]
        re, ce = np.where(nan_ev[:, cidx])
        if len(re):
            x_ev_f[re, cidx[ce]] = meds[ce]

    mu = x_tr_f.mean(axis=0, keepdims=True)
    sd = x_tr_f.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z_tr = (x_tr_f - mu) / sd
    z_ev = (x_ev_f - mu) / sd

    if max_pca is not None and z_tr.shape[1] > max_pca and z_tr.shape[0] > max_pca + 5:
        if z_tr.shape[1] >= 128:
            pca = PCA(
                n_components=max_pca,
                svd_solver="randomized",
                iterated_power=4,
                random_state=0,
            )
        else:
            pca = PCA(n_components=max_pca, svd_solver="full")
        z_tr = pca.fit_transform(z_tr).astype(np.float32)
        z_ev = pca.transform(z_ev).astype(np.float32)
    return z_tr, z_ev


def _svd_ridge_coef(
    z_tr: np.ndarray, y_tr: np.ndarray, alpha: float
) -> tuple[float, np.ndarray, np.ndarray]:
    ym = float(np.mean(y_tr))
    xm = z_tr.mean(axis=0)
    xc = z_tr - xm[None, :]
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    d = s / (s * s + alpha)
    w = vt.T @ (d * (u.T @ (y_tr - ym)))
    return ym, xm, w


def _within_person_mse(y: np.ndarray, pred: np.ndarray, ppl: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 10:
        return float("inf")
    df = pd.DataFrame({"p": ppl[ok], "e": (y[ok] - pred[ok]) ** 2})
    return float(df.groupby("p")["e"].mean().mean())


def person_stress_aucs(
    frame: pd.DataFrame,
    scores: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    min_each: int = MIN_EACH,
) -> dict[str, float]:
    """Within-ID ROC-AUC for every ID with >= ``min_each`` positive & negative trials."""
    sc = np.asarray(scores, dtype=float)
    ok = np.isfinite(sc)
    pids = frame["participant_id"].astype(str).to_numpy()
    out: dict[str, float] = {}
    for pid in np.unique(pids[ok & (pos_mask | neg_mask)]):
        m = pids == pid
        p_idx = m & ok & pos_mask
        n_idx = m & ok & neg_mask
        if p_idx.sum() < min_each or n_idx.sum() < min_each:
            continue
        y = np.concatenate([np.ones(p_idx.sum(), int), np.zeros(n_idx.sum(), int)])
        s = np.concatenate([sc[p_idx], sc[n_idx]])
        out[str(pid)] = float(roc_auc_score(y, s))
    return out


def bootstrap_mean(vals: np.ndarray, *, draws: int = DRAWS, seed: int = 0) -> dict[str, Any]:
    """Mean + 95% percentile bootstrap CI."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (draws, len(arr)))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": round(float(arr.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n": int(len(arr)),
    }


def paired_auc_diff(
    pmap_a: dict[str, float],
    pmap_b: dict[str, float],
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired ID-bootstrap difference ``A - B`` across shared eligible IDs."""
    shared = sorted(set(pmap_a) & set(pmap_b))
    if not shared:
        return {"diff": None, "ci95": [None, None], "n": 0}
    diffs = np.array([pmap_a[p] - pmap_b[p] for p in shared], dtype=float)
    res = bootstrap_mean(diffs, draws=draws, seed=seed)
    return {"diff": res["mean"], "ci95": res["ci95"], "n": res["n"]}


def same_video_auc(
    frame: pd.DataFrame,
    scores: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """Within-video across-viewer ROC-AUC with video-level bootstrap (weighted by n+*n-)."""
    sc = np.asarray(scores, dtype=float)
    ok = np.isfinite(sc)
    vids = frame["video"].astype(str).to_numpy()
    v_aucs, v_wts = [], []
    for vid in np.unique(vids[ok & (pos_mask | neg_mask)]):
        m = vids == vid
        p = m & ok & pos_mask
        n = m & ok & neg_mask
        if p.sum() < 3 or n.sum() < 3:
            continue
        y = np.concatenate([np.ones(p.sum(), int), np.zeros(n.sum(), int)])
        s = np.concatenate([sc[p], sc[n]])
        v_aucs.append(float(roc_auc_score(y, s)))
        v_wts.append(float(p.sum() * n.sum()))
    if not v_aucs:
        return {"auc_weighted": None, "ci95_weighted": [None, None], "videos": 0}
    aucs = np.array(v_aucs, dtype=float)
    wts = np.array(v_wts, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(aucs), (draws, len(aucs)))
    bw = (aucs[idx] * wts[idx]).sum(axis=1) / wts[idx].sum(axis=1)
    bu = aucs[idx].mean(axis=1)
    lo_w, hi_w = np.percentile(bw, [2.5, 97.5])
    lo_u, hi_u = np.percentile(bu, [2.5, 97.5])
    return {
        "auc_weighted": round(float(np.average(aucs, weights=wts)), 4),
        "ci95_weighted": [round(float(lo_w), 4), round(float(hi_w), 4)],
        "auc_unweighted": round(float(aucs.mean()), 4),
        "ci95_unweighted": [round(float(lo_u), 4), round(float(hi_u), 4)],
        "videos": int(len(aucs)),
        "pairs": int(wts.sum()),
    }


def evaluate_score_vector(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    draws: int = DRAWS,
    include_contrasts: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute within-ID stress AUC, pooled AUC, and optional specificity/fixed-video contrasts."""
    y_all = stress_label(frame).to_numpy(bool)
    ok = np.isfinite(scores)
    pmap = person_stress_aucs(frame, scores, y_all, ~y_all)
    st = bootstrap_mean(np.array(list(pmap.values()), dtype=float), draws=draws)
    pooled = (
        round(float(roc_auc_score(y_all[ok].astype(int), scores[ok])), 4)
        if ok.sum() > 20 and len(np.unique(y_all[ok])) == 2
        else None
    )
    out: dict[str, Any] = {
        "within_id_auc": st["mean"],
        "ci95": st["ci95"],
        "n_ids": st["n"],
        "pooled_auc": pooled,
        "n_trials_scored": int(ok.sum()),
    }
    if include_contrasts:
        cdict = stress_contrasts(frame)
        for cname in ("strict", "vs_excitement", "vs_calm_negative", "vs_relaxed"):
            pm, nm = cdict[cname]
            cmap = person_stress_aucs(frame, scores, pm.to_numpy(bool), nm.to_numpy(bool))
            cs = bootstrap_mean(np.array(list(cmap.values()), dtype=float), draws=draws)
            out[cname] = {"auc": cs["mean"], "ci95": cs["ci95"], "n_ids": cs["n"]}
        out["same_video"] = same_video_auc(frame, scores, y_all, ~y_all, draws=draws)
        calm = (frame["rating_valence"] >= 5) & (frame["rating_arousal"] <= 5)
        out["same_video_vs_calm"] = same_video_auc(
            frame, scores, y_all, calm.to_numpy(bool), draws=draws
        )
    return out, pmap


def fit_predict_nested_linear(
    x_tr: np.ndarray,
    df_tr: pd.DataFrame,
    x_ev: np.ndarray,
    *,
    max_pca: int | None = 64,
    inner_k: int = 3,
    fit_logreg: bool = True,
) -> dict[str, np.ndarray]:
    """Nested fold-isolated Ridge VA + optional LogisticRegression matching Study 10."""
    ipf = hash_buckets(df_tr["participant_id"], f"study09-inner-v1|p-{inner_k}", inner_k)
    itf = hash_buckets(df_tr["theme"], f"study09-inner-v1|t-{inner_k}", inner_k)
    y_s = stress_label(df_tr).to_numpy(int)
    ppl = df_tr["participant_id"]
    ppl_arr = ppl.to_numpy()
    yv = (df_tr["rating_valence"] - df_tr["rating_valence"].groupby(ppl).transform("mean")).to_numpy(
        float
    )
    ya = (df_tr["rating_arousal"] - df_tr["rating_arousal"].groupby(ppl).transform("mean")).to_numpy(
        float
    )

    oof_c = {c: np.full(len(df_tr), np.nan, dtype=float) for c in CS_GRID}
    oof_v = {a: np.full(len(df_tr), np.nan, dtype=float) for a in ALPHAS_GRID}
    oof_a = {a: np.full(len(df_tr), np.nan, dtype=float) for a in ALPHAS_GRID}

    for p in range(inner_k):
        for t in range(inner_k):
            ev_m = (ipf == p) & (itf == t)
            tr_m = (ipf != p) & (itf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 20 or len(np.unique(y_s[tr_m])) < 2:
                continue
            z_tr_i, z_ev_i = _preprocess_fit_transform(x_tr[tr_m], x_tr[ev_m], max_pca=max_pca)
            if fit_logreg:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for c_val in CS_GRID:
                        clf = LogisticRegression(
                            C=c_val, class_weight="balanced", max_iter=500
                        ).fit(z_tr_i, y_s[tr_m])
                        oof_c[c_val][ev_m] = clf.decision_function(z_ev_i)

            xm = z_tr_i.mean(axis=0, keepdims=True)
            xc = z_tr_i - xm
            xec = z_ev_i - xm
            u, s, vt = np.linalg.svd(xc, full_matrices=False)
            ym_v, ym_a = float(np.mean(yv[tr_m])), float(np.mean(ya[tr_m]))
            ut_v = u.T @ (yv[tr_m] - ym_v)
            ut_a = u.T @ (ya[tr_m] - ym_a)
            for a in ALPHAS_GRID:
                dv = s / (s * s + a)
                oof_v[a][ev_m] = ym_v + xec @ (vt.T @ (dv * ut_v))
                oof_a[a][ev_m] = ym_a + xec @ (vt.T @ (dv * ut_a))

    best_av, best_lv = ALPHAS_GRID[len(ALPHAS_GRID) // 2], float("inf")
    best_aa, best_la = ALPHAS_GRID[len(ALPHAS_GRID) // 2], float("inf")
    for a in ALPHAS_GRID:
        lv = _within_person_mse(yv, oof_v[a], ppl_arr)
        la = _within_person_mse(ya, oof_a[a], ppl_arr)
        if lv < best_lv:
            best_lv, best_av = lv, a
        if la < best_la:
            best_la, best_aa = la, a

    z_tr_f, z_ev_f = _preprocess_fit_transform(x_tr, x_ev, max_pca=max_pca)
    ym_v, xm_f, w_v = _svd_ridge_coef(z_tr_f, yv, best_av)
    ym_a, _, w_a = _svd_ridge_coef(z_tr_f, ya, best_aa)
    pv = ym_v + (z_ev_f - xm_f[None, :]) @ w_v
    pa = ym_a + (z_ev_f - xm_f[None, :]) @ w_a

    ok_v = np.isfinite(oof_v[best_av])
    ok_a = np.isfinite(oof_a[best_aa])
    sd_v = max(float(np.std(oof_v[best_av][ok_v])) if ok_v.sum() > 5 else 1.0, 1e-6)
    sd_a = max(float(np.std(oof_a[best_aa][ok_a])) if ok_a.sum() > 5 else 1.0, 1e-6)
    sc_va = (pa / sd_a - pv / sd_v).astype(np.float32)

    out: dict[str, np.ndarray] = {"stress_va": sc_va}
    if fit_logreg:
        best_c, best_auc = CS_GRID[len(CS_GRID) // 2], -1.0
        for c_val in CS_GRID:
            pmap = person_stress_aucs(df_tr, oof_c[c_val], y_s.astype(bool), ~y_s.astype(bool))
            sc = float(np.mean(list(pmap.values()))) if pmap else 0.5
            if sc > best_auc:
                best_auc, best_c = sc, c_val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_f = LogisticRegression(C=best_c, class_weight="balanced", max_iter=500).fit(
                z_tr_f, y_s
            )
            out["stress_logreg"] = clf_f.decision_function(z_ev_f).astype(np.float32)
    return out


def run_r4_3x3(
    dev_df: pd.DataFrame,
    x_mat: np.ndarray,
    *,
    max_pca: int | None = 64,
    fit_logreg: bool = True,
) -> dict[str, np.ndarray]:
    """Standard audited Cold-Start 3x3 blocked CV on ``dev_df``."""
    pf = dev_df["person_fold"].to_numpy(int)
    tf = dev_df["theme_fold"].to_numpy(int)
    n = len(dev_df)
    oof_va = np.full(n, np.nan, dtype=np.float32)
    oof_lr = np.full(n, np.nan, dtype=np.float32)
    for p in range(3):
        for t in range(3):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 40:
                continue
            preds = fit_predict_nested_linear(
                x_mat[tr_m], dev_df.iloc[tr_m], x_mat[ev_m], max_pca=max_pca, fit_logreg=fit_logreg
            )
            oof_va[ev_m] = preds["stress_va"]
            if fit_logreg:
                oof_lr[ev_m] = preds["stress_logreg"]
    out = {"stress_va": oof_va}
    if fit_logreg:
        out["stress_logreg"] = oof_lr
    return out


# ---------------------------------------------------------------------------
# 3. Priority 1: Matched-N_train R1-R4 ladder on IDENTICAL 3x3 test cells
# ---------------------------------------------------------------------------


def build_matched_cell_pools(
    pf: np.ndarray,
    tf: np.ndarray,
    p: int,
    t: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return test cell mask ``E(p,t)`` and the four disjoint training pools."""
    ev_m = (pf == p) & (tf == t)
    pools = {
        "R4_double_unseen": (pf != p) & (tf != t),
        "R3_new_id_seen_themes": (pf != p),
        "R2_seen_id_new_themes": (tf != t),
        "R1_random_pool": ~ev_m,
    }
    return ev_m, pools


def run_priority1_matched_ladder(
    dev_df: pd.DataFrame,
    models: dict[str, np.ndarray],
    *,
    n_repeats: int = 10,
    max_pca: int = 64,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Evaluate R1-R4 on identical 3x3 test cells at matched N_train (= |R4|) AND full size."""
    pf = dev_df["person_fold"].to_numpy(int)
    tf = dev_df["theme_fold"].to_numpy(int)
    pids = dev_df["participant_id"].astype(str).to_numpy()
    themes = dev_df["theme"].astype(str).to_numpy()
    n = len(dev_df)
    regimes = (
        "R1_random_pool",
        "R2_seen_id_new_themes",
        "R3_new_id_seen_themes",
        "R4_double_unseen",
    )

    # Pre-generate deterministic matched subsamples per (cell, regime, seed) so all models
    # share the EXACT same training rows on every seed and test cell!
    cell_specs: list[dict[str, Any]] = []
    pool_audit: dict[str, list[dict[str, float]]] = {
        f"{mode}_{r}": [] for mode in ("matched", "full") for r in regimes
    }

    for p in range(3):
        for t in range(3):
            ev_m, pools = build_matched_cell_pools(pf, tf, p, t)
            ev_pids = set(pids[ev_m])
            ev_themes = set(themes[ev_m])
            n_match = int(pools["R4_double_unseen"].sum())
            spec: dict[str, Any] = {
                "p": p,
                "t": t,
                "ev_idx": np.where(ev_m)[0],
                "n_match": n_match,
                "full_idx": {},
                "matched_idx": {r: [] for r in regimes},
            }
            for r in regimes:
                full_i = np.where(pools[r])[0]
                spec["full_idx"][r] = full_i
                pool_audit[f"full_{r}"].append(
                    {
                        "n_rows": float(len(full_i)),
                        "n_ids": float(len(set(pids[full_i]))),
                        "n_themes": float(len(set(themes[full_i]))),
                        "frac_same_id": float(np.isin(pids[full_i], list(ev_pids)).mean()),
                        "frac_same_theme": float(np.isin(themes[full_i], list(ev_themes)).mean()),
                    }
                )
                for rep in range(n_repeats):
                    if r == "R4_double_unseen":
                        sub_i = full_i
                    else:
                        rng = np.random.default_rng(20261126 + 1000 * rep + 37 * (3 * p + t) + len(r))
                        sub_i = np.sort(rng.choice(full_i, size=n_match, replace=False))
                    spec["matched_idx"][r].append(sub_i)
                    if rep == 0:
                        pool_audit[f"matched_{r}"].append(
                            {
                                "n_rows": float(len(sub_i)),
                                "n_ids": float(len(set(pids[sub_i]))),
                                "n_themes": float(len(set(themes[sub_i]))),
                                "frac_same_id": float(np.isin(pids[sub_i], list(ev_pids)).mean()),
                                "frac_same_theme": float(
                                    np.isin(themes[sub_i], list(ev_themes)).mean()
                                ),
                            }
                        )
            cell_specs.append(spec)

    audit_summary = {
        k: {
            m: round(float(np.mean([row[m] for row in rows])), 4)
            for m in ("n_rows", "n_ids", "n_themes", "frac_same_id", "frac_same_theme")
        }
        for k, rows in pool_audit.items()
    }

    contrast_pairs = (
        ("R4_minus_R1", "R4_double_unseen", "R1_random_pool"),
        ("R1_minus_R4", "R1_random_pool", "R4_double_unseen"),
        ("R3_minus_R4", "R3_new_id_seen_themes", "R4_double_unseen"),
        ("R2_minus_R4", "R2_seen_id_new_themes", "R4_double_unseen"),
        ("R1_minus_R2", "R1_random_pool", "R2_seen_id_new_themes"),
        ("R1_minus_R3", "R1_random_pool", "R3_new_id_seen_themes"),
        ("R3_minus_R2", "R3_new_id_seen_themes", "R2_seen_id_new_themes"),
    )

    out_models: dict[str, Any] = {}
    for mname, x_mat in models.items():
        t0 = time.perf_counter()
        print(f"  [P1] Running matched-N ({n_repeats} repeats) & full ladder for {mname}...", flush=True)

        # 1. R4 once (identical across matched & full and all repeats)
        r4_va = np.full(n, np.nan, dtype=np.float32)
        r4_lr = np.full(n, np.nan, dtype=np.float32)
        for spec in cell_specs:
            ev_i = spec["ev_idx"]
            tr_i = spec["full_idx"]["R4_double_unseen"]
            preds = fit_predict_nested_linear(
                x_mat[tr_i], dev_df.iloc[tr_i], x_mat[ev_i], max_pca=max_pca, fit_logreg=True
            )
            r4_va[ev_i] = preds["stress_va"]
            r4_lr[ev_i] = preds["stress_logreg"]

        # 2. Full-pool R1/R2/R3 on the exact same 9 test cells
        full_scores: dict[str, dict[str, np.ndarray]] = {
            "R4_double_unseen": {"stress_va": r4_va, "stress_logreg": r4_lr}
        }
        for r in ("R1_random_pool", "R2_seen_id_new_themes", "R3_new_id_seen_themes"):
            va = np.full(n, np.nan, dtype=np.float32)
            lr = np.full(n, np.nan, dtype=np.float32)
            for spec in cell_specs:
                ev_i = spec["ev_idx"]
                tr_i = spec["full_idx"][r]
                preds = fit_predict_nested_linear(
                    x_mat[tr_i], dev_df.iloc[tr_i], x_mat[ev_i], max_pca=max_pca, fit_logreg=True
                )
                va[ev_i] = preds["stress_va"]
                lr[ev_i] = preds["stress_logreg"]
            full_scores[r] = {"stress_va": va, "stress_logreg": lr}

        # 3. Matched-N across n_repeats independent subsamples
        matched_pmaps: dict[str, dict[str, list[dict[str, float]]]] = {
            head: {r: [] for r in regimes} for head in ("stress_va", "stress_logreg")
        }
        matched_within_seeds: dict[str, dict[str, list[float]]] = {
            head: {r: [] for r in regimes} for head in ("stress_va", "stress_logreg")
        }
        matched_pooled_seeds: dict[str, dict[str, list[float]]] = {
            head: {r: [] for r in regimes} for head in ("stress_va", "stress_logreg")
        }

        for head in ("stress_va", "stress_logreg"):
            res_r4, pmap_r4 = evaluate_score_vector(
                dev_df, full_scores["R4_double_unseen"][head], draws=draws, include_contrasts=False
            )
            for _ in range(n_repeats):
                matched_pmaps[head]["R4_double_unseen"].append(pmap_r4)
                matched_within_seeds[head]["R4_double_unseen"].append(float(res_r4["within_id_auc"]))
                matched_pooled_seeds[head]["R4_double_unseen"].append(float(res_r4["pooled_auc"]))

        for rep in range(n_repeats):
            for r in ("R1_random_pool", "R2_seen_id_new_themes", "R3_new_id_seen_themes"):
                va = np.full(n, np.nan, dtype=np.float32)
                lr = np.full(n, np.nan, dtype=np.float32)
                for spec in cell_specs:
                    ev_i = spec["ev_idx"]
                    tr_i = spec["matched_idx"][r][rep]
                    preds = fit_predict_nested_linear(
                        x_mat[tr_i],
                        dev_df.iloc[tr_i],
                        x_mat[ev_i],
                        max_pca=max_pca,
                        fit_logreg=True,
                    )
                    va[ev_i] = preds["stress_va"]
                    lr[ev_i] = preds["stress_logreg"]
                for head, sc_arr in (("stress_va", va), ("stress_logreg", lr)):
                    res_k, pmap_k = evaluate_score_vector(
                        dev_df, sc_arr, draws=200, include_contrasts=False
                    )
                    matched_pmaps[head][r].append(pmap_k)
                    matched_within_seeds[head][r].append(float(res_k["within_id_auc"]))
                    matched_pooled_seeds[head][r].append(float(res_k["pooled_auc"]))

        # Summarize per head (mean-over-seeds per ID for paired ID bootstrap + seed SD!)
        model_res: dict[str, Any] = {"elapsed_s": round(time.perf_counter() - t0, 2)}
        for head in ("stress_va", "stress_logreg"):
            full_summary: dict[str, Any] = {}
            full_pmaps: dict[str, dict[str, float]] = {}
            for r in regimes:
                ev, pm = evaluate_score_vector(
                    dev_df, full_scores[r][head], draws=draws, include_contrasts=False
                )
                full_summary[r] = ev
                full_pmaps[r] = pm
            full_contrasts = {
                cname: {
                    **paired_auc_diff(full_pmaps[ra], full_pmaps[rb], draws=draws),
                    "pooled_diff": round(
                        float(full_summary[ra]["pooled_auc"] - full_summary[rb]["pooled_auc"]), 4
                    ),
                }
                for cname, ra, rb in contrast_pairs
            }

            matched_summary: dict[str, Any] = {}
            matched_avg_pmap: dict[str, dict[str, float]] = {}
            for r in regimes:
                all_ids = sorted(set.intersection(*(set(pm) for pm in matched_pmaps[head][r])))
                avg_pm = {
                    pid: float(np.mean([pm[pid] for pm in matched_pmaps[head][r]]))
                    for pid in all_ids
                }
                matched_avg_pmap[r] = avg_pm
                bm = bootstrap_mean(np.array(list(avg_pm.values()), dtype=float), draws=draws)
                w_seeds = np.array(matched_within_seeds[head][r], dtype=float)
                p_seeds = np.array(matched_pooled_seeds[head][r], dtype=float)
                matched_summary[r] = {
                    "within_id_auc": bm["mean"],
                    "ci95_id": bm["ci95"],
                    "n_ids": bm["n"],
                    "seed_sd_within": round(float(w_seeds.std(ddof=1)), 5) if n_repeats > 1 else 0.0,
                    "seed_min_max_within": [
                        round(float(w_seeds.min()), 4),
                        round(float(w_seeds.max()), 4),
                    ],
                    "pooled_auc_mean": round(float(p_seeds.mean()), 4),
                    "seed_sd_pooled": round(float(p_seeds.std(ddof=1)), 5) if n_repeats > 1 else 0.0,
                }

            matched_contrasts = {}
            for cname, ra, rb in contrast_pairs:
                pd_res = paired_auc_diff(
                    matched_avg_pmap[ra], matched_avg_pmap[rb], draws=draws
                )
                seed_diffs = (
                    np.array(matched_within_seeds[head][ra])
                    - np.array(matched_within_seeds[head][rb])
                )
                pooled_diffs = (
                    np.array(matched_pooled_seeds[head][ra])
                    - np.array(matched_pooled_seeds[head][rb])
                )
                matched_contrasts[cname] = {
                    **pd_res,
                    "seed_sd_within_diff": round(float(seed_diffs.std(ddof=1)), 5)
                    if n_repeats > 1
                    else 0.0,
                    "pooled_diff_mean": round(float(pooled_diffs.mean()), 4),
                    "seed_sd_pooled_diff": round(float(pooled_diffs.std(ddof=1)), 5)
                    if n_repeats > 1
                    else 0.0,
                }

            size_effect = {
                r: paired_auc_diff(full_pmaps[r], matched_avg_pmap[r], draws=draws)
                for r in ("R1_random_pool", "R2_seen_id_new_themes", "R3_new_id_seen_themes")
            }

            model_res[head] = {
                "matched_n_train": {
                    "regimes": matched_summary,
                    "contrasts": matched_contrasts,
                },
                "full_pool_same_cells": {
                    "regimes": full_summary,
                    "contrasts": full_contrasts,
                },
                "size_effect_full_minus_matched": size_effect,
            }
        out_models[mname] = model_res
        r4v = model_res["stress_va"]["matched_n_train"]["regimes"]["R4_double_unseen"][
            "within_id_auc"
        ]
        r1v = model_res["stress_va"]["matched_n_train"]["regimes"]["R1_random_pool"][
            "within_id_auc"
        ]
        r3v = model_res["stress_va"]["matched_n_train"]["regimes"]["R3_new_id_seen_themes"][
            "within_id_auc"
        ]
        print(
            f"  [P1] {mname} matched-N VA: R1={r1v} R3={r3v} R4={r4v} ({model_res['elapsed_s']}s)",
            flush=True,
        )

    return {
        "n_repeats": n_repeats,
        "max_pca": max_pca,
        "pool_audit_mean_across_9_cells": audit_summary,
        "models": out_models,
    }


# ---------------------------------------------------------------------------
# 4. Priority 2: Ridge WITH vs. WITHOUT post [10, 12) s on exact same folds
# ---------------------------------------------------------------------------


def run_priority2_post_ablation(
    dev_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    *,
    draws: int = DRAWS,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Compare Ridge VA & LogReg with vs. without all ``post [10, 12) s`` features."""
    all_cols = list(feat_df.columns)
    no_post_cols, post_cols = split_post_columns(all_cols)
    assert not any(
        ("post" in c.lower()) or c.endswith("_p_chg") or c.endswith("_d_p_l")
        for c in no_post_cols
    ), "Post-phase feature leaked into no_post_cols!"

    subsets = {
        "all_with_post": all_cols,
        "all_no_post": no_post_cols,
        "eeg_with_post": [c for c in all_cols if c.startswith("eeg_")],
        "eeg_no_post": [c for c in no_post_cols if c.startswith("eeg_")],
        "body_with_post": [c for c in all_cols if not c.startswith("eeg_")],
        "body_no_post": [c for c in no_post_cols if not c.startswith("eeg_")],
    }

    results_by_pca: dict[str, Any] = {}
    saved_oofs: dict[str, dict[str, np.ndarray]] = {}

    for max_pca in (64, 128):
        key_pca = f"pca_{max_pca}"
        sub_res: dict[str, Any] = {}
        pmaps: dict[str, dict[str, dict[str, float]]] = {"stress_va": {}, "stress_logreg": {}}
        for sname, cols in subsets.items():
            x_mat = feat_df[cols].to_numpy(dtype=np.float32)
            oofs = run_r4_3x3(dev_df, x_mat, max_pca=max_pca, fit_logreg=True)
            if max_pca == 64:
                saved_oofs[sname] = oofs
            ev_va, pm_va = evaluate_score_vector(dev_df, oofs["stress_va"], draws=draws)
            ev_lr, pm_lr = evaluate_score_vector(dev_df, oofs["stress_logreg"], draws=draws)
            pmaps["stress_va"][sname] = pm_va
            pmaps["stress_logreg"][sname] = pm_lr
            sub_res[sname] = {
                "n_features": len(cols),
                "stress_va": ev_va,
                "stress_logreg": ev_lr,
            }

        paired = {}
        for scope in ("all", "eeg", "body"):
            paired[f"{scope}_with_minus_no_post"] = {
                "stress_va": paired_auc_diff(
                    pmaps["stress_va"][f"{scope}_with_post"],
                    pmaps["stress_va"][f"{scope}_no_post"],
                    draws=draws,
                ),
                "stress_logreg": paired_auc_diff(
                    pmaps["stress_logreg"][f"{scope}_with_post"],
                    pmaps["stress_logreg"][f"{scope}_no_post"],
                    draws=draws,
                ),
            }
        results_by_pca[key_pca] = {"subsets": sub_res, "paired_with_minus_no_post": paired}
        print(
            f"  [P2] {key_pca}: all_with={sub_res['all_with_post']['stress_va']['within_id_auc']} "
            f"all_no={sub_res['all_no_post']['stress_va']['within_id_auc']} "
            f"diff={paired['all_with_minus_no_post']['stress_va']}",
            flush=True,
        )

    return (
        {
            "n_total_features": len(all_cols),
            "n_post_features_dropped": len(post_cols),
            "n_no_post_features_kept": len(no_post_cols),
            "dropped_post_columns_sample": post_cols[:20],
            "upstream_caveat": (
                "Both with_post and no_post draw from rich.parquet, where upstream "
                "recording-level sosfiltfilt (ECG/resp) and HABS PREPROCESSED EEG "
                "(zero-phase FIR + full-recording Picard ICA) are shared."
            ),
            "by_pca": results_by_pca,
        },
        saved_oofs,
    )


# ---------------------------------------------------------------------------
# 5. Priority 3: Trial-index, cohort x index, strict pre-0, and prev-trial
# ---------------------------------------------------------------------------


def _run_gbdt_r4_3x3(dev_df: pd.DataFrame, x_mat: np.ndarray) -> dict[str, np.ndarray]:
    """Fold-isolated nonlinear HistGradientBoosting baseline on exact 3x3 Cold-Start folds."""
    pf = dev_df["person_fold"].to_numpy(int)
    tf = dev_df["theme_fold"].to_numpy(int)
    y_s = stress_label(dev_df).to_numpy(int)
    ppl = dev_df["participant_id"]
    yv = (dev_df["rating_valence"] - dev_df["rating_valence"].groupby(ppl).transform("mean")).to_numpy(
        float
    )
    ya = (dev_df["rating_arousal"] - dev_df["rating_arousal"].groupby(ppl).transform("mean")).to_numpy(
        float
    )
    n = len(dev_df)
    oof_va = np.full(n, np.nan, dtype=np.float32)
    oof_clf = np.full(n, np.nan, dtype=np.float32)
    for p in range(3):
        for t in range(3):
            ev_m = (pf == p) & (tf == t)
            tr_m = (pf != p) & (tf != t)
            if ev_m.sum() == 0 or tr_m.sum() < 40:
                continue
            clf = HistGradientBoostingClassifier(
                max_depth=3, max_iter=80, min_samples_leaf=30, random_state=0
            ).fit(x_mat[tr_m], y_s[tr_m])
            oof_clf[ev_m] = clf.predict_proba(x_mat[ev_m])[:, 1]
            rv = HistGradientBoostingRegressor(
                max_depth=3, max_iter=80, min_samples_leaf=30, random_state=0
            ).fit(x_mat[tr_m], yv[tr_m])
            ra = HistGradientBoostingRegressor(
                max_depth=3, max_iter=80, min_samples_leaf=30, random_state=0
            ).fit(x_mat[tr_m], ya[tr_m])
            pv = rv.predict(x_mat[ev_m])
            pa = ra.predict(x_mat[ev_m])
            oof_va[ev_m] = (pa / max(float(np.std(ya[tr_m])), 1e-6)) - (
                pv / max(float(np.std(yv[tr_m])), 1e-6)
            )
    return {"stress_va": oof_va, "stress_clf": oof_clf}


def _evaluate_by_cohort(
    dev_df: pd.DataFrame,
    scores: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Evaluate within-ID AUC overall, per cohort 1..5, and grouped 1-2 vs 3-5."""
    use = np.ones(len(dev_df), dtype=bool) if mask is None else mask.copy()
    sub = dev_df.loc[use]
    sc = scores[use]
    res_all, pmap_all = evaluate_score_vector(sub, sc, draws=draws, include_contrasts=False)
    out: dict[str, Any] = {"overall": res_all, "by_cohort": {}, "by_order_protocol": {}}
    if "cohort" in sub.columns:
        coh = sub["cohort"].to_numpy()
        for c in sorted(int(v) for v in sub["cohort"].dropna().unique()):
            cm = coh == c
            ev_c, _ = evaluate_score_vector(
                sub.loc[cm], sc[cm], draws=draws, include_contrasts=False
            )
            out["by_cohort"][f"cohort_{c}"] = ev_c
        for gname, cset in (
            ("cohorts_1_2_fully_random", {1, 2}),
            ("cohorts_3_5_randomized_blocks", {3, 4, 5}),
        ):
            gm = np.isin(coh, list(cset))
            ev_g, _ = evaluate_score_vector(
                sub.loc[gm], sc[gm], draws=draws, include_contrasts=False
            )
            out["by_order_protocol"][gname] = ev_g
    return out, pmap_all


def run_priority3_confound_controls(
    dev_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    s07_no_post_oofs: dict[str, np.ndarray],
    s07_with_post_oofs: dict[str, np.ndarray],
    *,
    full_frame: pd.DataFrame,
    dev_mask: np.ndarray,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Evaluate trial-index, cohort x index, strict pre-0, and previous-trial baselines."""
    no_post_cols, _ = split_post_columns(list(feat_df.columns))
    # Determine the actual predecessor before masking out non-development themes.
    full_features = build_s07_feature_df(full_frame)[no_post_cols].to_numpy(dtype=np.float32)
    idx_raw, idx_norm, is_first, x_prev_s07 = (
        arr[dev_mask] for arr in build_session_order_and_prev_trial(full_frame, full_features)
    )
    non_first = ~is_first

    # Cohort x index design matrix (cohort constants alone do not affect within-ID AUC)
    coh_vals = (
        dev_df["cohort"].fillna(-1).to_numpy(int)
        if "cohort" in dev_df.columns
        else np.zeros(len(dev_df), dtype=int)
    )
    unique_cohs = sorted(int(c) for c in np.unique(coh_vals) if c >= 0)
    coh_x_idx_cols = [idx_norm, idx_norm**2]
    for c in unique_cohs:
        ind = (coh_vals == c).astype(np.float32)
        coh_x_idx_cols.extend([ind * idx_norm, ind * (idx_norm**2)])
    x_coh_x_idx = np.column_stack(coh_x_idx_cols).astype(np.float32)
    x_idx_lin = idx_norm[:, None].astype(np.float32)
    x_idx_poly3 = np.column_stack([idx_norm, idx_norm**2, idx_norm**3]).astype(np.float32)
    x_idx_gbdt = np.column_stack([idx_raw, idx_norm]).astype(np.float32)
    x_coh_idx_gbdt = np.column_stack([coh_vals.astype(np.float32), idx_raw, idx_norm]).astype(
        np.float32
    )

    # Strictly pre-stimulus (< 0 s) columns from study09_causal.parquet
    strict_pre0_body_cols = [c for c in ("b05_sc_ref", "b05_resp_amp_ref") if c in dev_df.columns]
    pre0_body_ecg_cols = [
        c
        for c in ("b05_sc_ref", "b05_resp_amp_ref", "b05_ecg_hr_ref", "b05_ecg_rr_ref")
        if c in dev_df.columns
    ]
    pre0_eeg_cols = sorted(c for c in dev_df.columns if c.startswith("e05_ref_"))

    # Strictly causal pre-5s physiology of the preceding trial (i - 1)
    causal_pre5_cols = sorted(
        c
        for c in dev_df.columns
        if (c.startswith("b05_") or c.startswith("e05_")) and c != "b05_sc_valid"
    )
    x_causal_pre5 = (
        full_frame[causal_pre5_cols].to_numpy(dtype=np.float32) if causal_pre5_cols else None
    )
    x_prev_causal = (
        build_session_order_and_prev_trial(full_frame, x_causal_pre5)[3][dev_mask]
        if x_causal_pre5 is not None
        else None
    )

    baselines: dict[str, dict[str, Any]] = {}
    pmaps_va: dict[str, dict[str, float]] = {}

    def _record(
        name: str,
        scores_dict: dict[str, np.ndarray],
        *,
        eval_mask: np.ndarray | None = None,
        note: str = "",
        n_features: int = 0,
    ) -> None:
        entry: dict[str, Any] = {"n_features": n_features, "note": note}
        for hk, arr in scores_dict.items():
            res_h, pm_h = _evaluate_by_cohort(dev_df, arr, mask=eval_mask, draws=draws)
            entry[hk] = res_h
            if hk == "stress_va":
                pmaps_va[name] = pm_h
        baselines[name] = entry
        va_auc = entry.get("stress_va", {}).get("overall", {}).get("within_id_auc")
        c12 = (
            entry.get("stress_va", {})
            .get("by_order_protocol", {})
            .get("cohorts_1_2_fully_random", {})
            .get("within_id_auc")
        )
        c35 = (
            entry.get("stress_va", {})
            .get("by_order_protocol", {})
            .get("cohorts_3_5_randomized_blocks", {})
            .get("within_id_auc")
        )
        print(f"  [P3] {name}: overall={va_auc} | C1-2={c12} | C3-5={c35}", flush=True)

    # Reference S07 models by cohort
    _record(
        "ref_S07_with_post",
        s07_with_post_oofs,
        n_features=feat_df.shape[1],
        note="Canonical 4-phase Ridge [-3, 12) s",
    )
    _record(
        "ref_S07_no_post",
        s07_no_post_oofs,
        n_features=len(no_post_cols),
        note="3-phase Ridge [-3, 10) s without post block",
    )
    _record(
        "ref_S07_no_post_nonfirst",
        s07_no_post_oofs,
        eval_mask=non_first,
        n_features=len(no_post_cols),
        note="3-phase Ridge evaluated on non-first trials in session (i >= 1)",
    )

    # 1. Index only (raw monotonic + linear + poly3 + nonlinear GBDT)
    _record(
        "p3a_raw_idx_norm_direct",
        {"stress_va": idx_norm, "neg_idx_norm": -idx_norm},
        n_features=1,
        note="Raw normalized within-session trial index [0, 1], reset per (ID, day)",
    )
    _record(
        "p3a_index_linear",
        run_r4_3x3(dev_df, x_idx_lin, max_pca=None, fit_logreg=True),
        n_features=1,
        note="OOF linear model on normalized session trial index",
    )
    _record(
        "p3a_index_poly3",
        run_r4_3x3(dev_df, x_idx_poly3, max_pca=None, fit_logreg=True),
        n_features=3,
        note="OOF cubic polynomial of session trial index",
    )
    _record(
        "p3a_index_gbdt",
        _run_gbdt_r4_3x3(dev_df, x_idx_gbdt),
        n_features=2,
        note="OOF nonlinear GBDT on (idx_raw, idx_norm)",
    )

    # 2. Cohort x index interaction (linear/poly + GBDT)
    _record(
        "p3b_cohort_x_index_poly",
        run_r4_3x3(dev_df, x_coh_x_idx, max_pca=None, fit_logreg=True),
        n_features=x_coh_x_idx.shape[1],
        note="OOF cohort x (idx_norm, idx_norm^2) interaction",
    )
    _record(
        "p3b_cohort_x_index_gbdt",
        _run_gbdt_r4_3x3(dev_df, x_coh_idx_gbdt),
        n_features=3,
        note="OOF nonlinear GBDT on (cohort, idx_raw, idx_norm)",
    )

    # 3. Strictly pre-stimulus (< 0 s) physiology
    if strict_pre0_body_cols:
        _record(
            "p3c_pre0_strict_eda_resp",
            run_r4_3x3(
                dev_df,
                dev_df[strict_pre0_body_cols].to_numpy(dtype=np.float32),
                max_pca=None,
                fit_logreg=True,
            ),
            n_features=len(strict_pre0_body_cols),
            note="Nominal pre-0 CURATED EDA/respiration; archival quality gates can use data through +5 s.",
        )
    if pre0_body_ecg_cols:
        _record(
            "p3c_pre0_body_all",
            run_r4_3x3(
                dev_df,
                dev_df[pre0_body_ecg_cols].to_numpy(dtype=np.float32),
                max_pca=None,
                fit_logreg=True,
            ),
            n_features=len(pre0_body_ecg_cols),
            note="Pre-0 body [-3.5, 0) s (EDA+Resp+ECG; note ECG R-peak prominence uses [-5, 5) s)",
        )
    if pre0_eeg_cols:
        all_pre0 = pre0_body_ecg_cols + pre0_eeg_cols
        _record(
            "p3c_pre0_multimodal",
            run_r4_3x3(
                dev_df,
                dev_df[all_pre0].to_numpy(dtype=np.float32),
                max_pca=64,
                fit_logreg=True,
            ),
            n_features=len(all_pre0),
            note="Pre-0 body + 30 e05_ref EEG features [-3, 0) s (EEG from HABS PREPROCESSED)",
        )

    # 4. Previous-trial unlabeled physiology (strictly within session & day boundary, zero labels)
    x_prev_s07_flag = np.column_stack([x_prev_s07, is_first.astype(np.float32)])
    oofs_prev_s07 = run_r4_3x3(dev_df, x_prev_s07_flag, max_pca=64, fit_logreg=True)
    _record(
        "p3d_prev_trial_s07_all_trials",
        oofs_prev_s07,
        n_features=x_prev_s07_flag.shape[1],
        note="Unlabeled S07_no_post physiology of trial i-1 (+ is_first flag), evaluated on all trials",
    )
    _record(
        "p3d_prev_trial_s07_nonfirst",
        oofs_prev_s07,
        eval_mask=non_first,
        n_features=x_prev_s07_flag.shape[1],
        note="Same prev-trial model restricted to non-first trials in session (i >= 1)",
    )
    if x_prev_causal is not None:
        x_prev_c_flag = np.column_stack([x_prev_causal, is_first.astype(np.float32)])
        oofs_prev_c = run_r4_3x3(dev_df, x_prev_c_flag, max_pca=64, fit_logreg=True)
        _record(
            "p3d_prev_trial_causal_nonfirst",
            oofs_prev_c,
            eval_mask=non_first,
            n_features=x_prev_c_flag.shape[1],
            note="Unlabeled causal pre-5s physiology of trial i-1, evaluated on non-first trials",
        )

    # Paired comparisons: S07_no_post minus each confound baseline
    paired_vs_s07_no_post = {}
    for bname, pmap_b in pmaps_va.items():
        if bname.startswith("ref_"):
            continue
        ref_key = "ref_S07_no_post_nonfirst" if bname.endswith("_nonfirst") else "ref_S07_no_post"
        paired_vs_s07_no_post[f"S07_no_post_minus_{bname}"] = paired_auc_diff(
            pmaps_va[ref_key], pmap_b, draws=draws
        )

    return {
        "n_dev_trials": int(len(dev_df)),
        "n_first_trials_in_session": int(is_first.sum()),
        "n_nonfirst_trials_in_session": int(non_first.sum()),
        "baselines": baselines,
        "paired_S07_no_post_minus_baseline": paired_vs_s07_no_post,
    }


# ---------------------------------------------------------------------------
# 6. Figure 1A exact per-bin finite-ID coverage on BOTH salts
# ---------------------------------------------------------------------------


def compute_fig1_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute exact per-bin finite contributing IDs and curves on both salts."""
    report: dict[str, Any] = {}
    for name, salt in (
        ("original_timecourse", TIMECOURSE_SALT),
        ("audited_decoding", STUDY09_BASE_SALT),
    ):
        full = assign_splits(frame, salt)
        dev = full.loc[full["split"] == "dev"]
        pos = stress_label(dev).to_numpy(bool)
        neg = ~pos
        ppl = dev["participant_id"].astype(str).to_numpy()
        entry: dict[str, Any] = {
            "salt": salt,
            "development_rows": int(len(dev)),
            "development_ids": int(dev["participant_id"].nunique()),
            "development_themes": int(dev["theme"].nunique()),
            "signals": {},
        }
        for sig in ("hr", "resp_amp", "sc", "eeg_central_delta"):
            cols = [f"tc_{sig}_{k}" for k in range(15)]
            if not all(c in dev.columns for c in cols):
                continue
            vals = dev[cols].to_numpy(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = vals - np.nanmean(vals[:, :3], axis=1, keepdims=True)
            rows = []
            for pid in np.unique(ppl):
                m = ppl == pid
                p_arr, n_arr = corr[m & pos], corr[m & neg]
                if len(p_arr) < MIN_EACH or len(n_arr) < MIN_EACH:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pooled = np.sqrt(
                        (
                            np.nanvar(p_arr, axis=0, ddof=1) * (len(p_arr) - 1)
                            + np.nanvar(n_arr, axis=0, ddof=1) * (len(n_arr) - 1)
                        )
                        / (len(p_arr) + len(n_arr) - 2)
                    )
                    rows.append((np.nanmean(p_arr, axis=0) - np.nanmean(n_arr, axis=0)) / pooled)
            eff = np.array(rows, dtype=float)
            rng = np.random.default_rng(0)
            bidx = rng.integers(0, len(eff), (DRAWS, len(eff)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                boot = np.nanmean(eff[bidx], axis=1)
                lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
                mean = np.nanmean(eff, axis=0)
            entry["signals"][sig] = {
                "eligible_ids": int(len(eff)),
                "finite_ids_per_bin": np.isfinite(eff).sum(axis=0).tolist(),
                "d": np.round(mean, 4).tolist(),
                "low": np.round(lo, 4).tolist(),
                "high": np.round(hi, 4).tolist(),
            }
        report[name] = entry
    return report


# ---------------------------------------------------------------------------
# 7. Table loading & CLI entrypoint
# ---------------------------------------------------------------------------


def load_cached_tables(folder: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load and 1:1-merge cached HABS private Parquet tables."""
    table = pd.read_parquet(folder / "trials.parquet")
    if (folder / "body.parquet").is_file():
        body = pd.read_parquet(folder / "body.parquet")
        table = table.drop(columns=[c for c in body if c != "trial_id" and c in table])
        table = table.merge(body, on="trial_id", how="left", validate="1:1")
    for fname in ("timecourse.parquet", "rich.parquet", "study09_causal.parquet"):
        p = folder / fname
        if p.is_file():
            extra = pd.read_parquet(p)
            keep = ["trial_id"] + [c for c in extra.columns if c not in table.columns]
            table = table.merge(extra[keep], on="trial_id", how="left", validate="1:1")
    table = assign_splits(table, STUDY09_BASE_SALT)

    aux: dict[str, pd.DataFrame] = {}
    for key, fname in (
        ("reve_primary", "study09_reve_primary.parquet"),
        ("reve_extra", "study09_reve_extra.parquet"),
    ):
        p = folder / fname
        if p.is_file():
            df_x = pd.read_parquet(p)
            aux[key] = table[["trial_id"]].merge(df_x, on="trial_id", how="left", validate="1:1")
    return table, aux


def _extract_reve_vid10(aux: dict[str, pd.DataFrame], mask: np.ndarray) -> np.ndarray | None:
    for key, pfx in (("reve_extra", "reve_cs_vid10_"), ("reve_primary", "reve_cs_ctx8_early_")):
        df_x = aux.get(key)
        if df_x is None:
            continue
        cols = [c for c in df_x.columns if c.startswith(pfx)]
        if cols:
            return df_x.loc[mask, cols].to_numpy(dtype=np.float32)
    return None


def run_all(folder: Path, *, stage: str, n_repeats: int = 10, draws: int = DRAWS) -> dict[str, Any]:
    """Execute Priorities 1, 2, 3 and Figure 1 coverage from cached trial tables."""
    t0 = time.perf_counter()
    table, aux = load_cached_tables(folder)
    dev_mask = (table["split"] == "dev").to_numpy(bool)
    dev_df = table.loc[dev_mask].reset_index(drop=True)

    feat_df = build_s07_feature_df(dev_df)
    no_post_cols, _ = split_post_columns(list(feat_df.columns))

    print(
        f"Loaded dev: {len(dev_df)} trials, {dev_df['participant_id'].nunique()} IDs, "
        f"{dev_df['theme'].nunique()} themes, {feat_df.shape[1]} S07 features "
        f"({len(no_post_cols)} without post).",
        flush=True,
    )

    if stage in {"ridge", "reve"}:
        matrix = feat_df.to_numpy(dtype=np.float32) if stage == "ridge" else _extract_reve_vid10(aux, dev_mask)
        if matrix is None:
            raise ValueError("Missing REVE features for the predeclared model comparison")
        name = "S07_with_post" if stage == "ridge" else "REVE_frozen_vid10"
        return run_priority1_matched_ladder(dev_df, {name: matrix}, n_repeats=n_repeats, draws=draws)

    # Reuse the same R4 out-of-fold predictions in the order and history controls.
    p2, p2_oofs = run_priority2_post_ablation(dev_df, feat_df, draws=draws)

    # Priority 3
    p3 = run_priority3_confound_controls(
        dev_df,
        feat_df,
        p2_oofs["all_no_post"],
        p2_oofs["all_with_post"],
        full_frame=table,
        dev_mask=dev_mask,
        draws=draws,
    )

    # Figure 1A exact per-bin finite coverage on both salts
    fig1_cov = compute_fig1_coverage(table)

    return {
        "study": "pub_revision_p123",
        "salt": STUDY09_BASE_SALT,
        "dev_summary": {
            "n_trials": int(len(dev_df)),
            "n_ids": int(dev_df["participant_id"].nunique()),
            "n_themes": int(dev_df["theme"].nunique()),
            "stress_prevalence": round(float(stress_label(dev_df).mean()), 4),
        },
        "priority2_post_ablation": p2,
        "priority3_order_and_confound_controls": p3,
        "figure1_per_bin_coverage": fig1_cov,
        "elapsed_cpu_seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("TRIAL_DATA", "/tmp/trials")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--stage", choices=["controls", "ridge", "reve"], required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    result = run_all(args.data, stage=args.stage, n_repeats=args.repeats, draws=DRAWS)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "pub_revision_p123_metrics.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e3:.1f} KB).", flush=True)


if __name__ == "__main__":
    main()
