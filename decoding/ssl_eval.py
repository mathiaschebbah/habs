"""Shared Study-08 evaluation, probing, and representation-geometry diagnostics.

Strict zero-leakage protocol
----------------------------
- All development evaluation uses 3 x 3 blocked person x theme cross-fitting
  (identical to ``decoding.common``): block ``(p, t)`` is predicted by probes
  fitted strictly on ``(person_fold != p) & (theme_fold != t)``.
- Every metric reports both point estimate and 95% percentile bootstrap CIs
  resampled over participants (within-person metrics) or videos (same-video
  appraisal metrics), never over correlated trials.
- Representation diagnostics quantify static participant identity (within-day
  and cross-day subject ID, ICC / variance explained by person) alongside
  temporal neurophysiological dynamics (pre-vs-video onset, early-vs-late phase,
  arrow-of-time ordering, effective rank, and temporal-shuffle degradation).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding import common
from decoding.stress import MIN_EACH, contrasts, label

DRAWS = 2000
MIN_VA_TRIALS = 5
MIN_SAME_VIDEO = 3


def boot_ci(values, *, draws: int = DRAWS, seed: int = 0) -> dict:
    """Mean and 95% percentile bootstrap CI over independent units (people/videos)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (draws, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(float(arr.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n": int(len(arr)),
    }


def paired_gain_ci(
    target_map: dict[str, float],
    ref_map: dict[str, float],
    *,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict:
    """Paired bootstrap 95% CI of ``target - ref`` over matched units."""
    keys = sorted(
        k
        for k in set(target_map) & set(ref_map)
        if np.isfinite(target_map[k]) and np.isfinite(ref_map[k])
    )
    if len(keys) < 3:
        return {"gain": None, "ci95": [None, None], "n": int(len(keys))}
    diffs = np.array([target_map[k] - ref_map[k] for k in keys], dtype=float)
    res = boot_ci(diffs, draws=draws, seed=seed)
    return {"gain": res["mean"], "ci95": res["ci95"], "n": res["n"]}


def person_stress_aucs(
    frame: pd.DataFrame, scores: pd.Series, pos_mask: pd.Series, neg_mask: pd.Series
) -> dict[str, float]:
    """Per-person ROC AUC dictionary for an arbitrary binary contrast."""
    out: dict[str, float] = {}
    for person, group in frame.groupby("participant_id"):
        idx = group.index
        p_m = pos_mask.loc[idx].to_numpy(bool)
        n_m = neg_mask.loc[idx].to_numpy(bool)
        s = scores.loc[idx].to_numpy(float)
        ok = np.isfinite(s)
        p_m, n_m = p_m & ok, n_m & ok
        if p_m.sum() < MIN_EACH or n_m.sum() < MIN_EACH:
            continue
        keep = p_m | n_m
        out[str(person)] = float(roc_auc_score(p_m[keep].astype(int), s[keep]))
    return out


def same_video_auc(
    frame: pd.DataFrame,
    scores: pd.Series,
    pos_mask: pd.Series,
    neg_mask: pd.Series,
    *,
    draws: int = DRAWS,
) -> dict:
    """Same-video ROC AUC (pair-weighted and unweighted) with video bootstrap CI."""
    per_video = []
    for vid, group in frame.groupby("video"):
        idx = group.index
        p_m = pos_mask.loc[idx].to_numpy(bool)
        n_m = neg_mask.loc[idx].to_numpy(bool)
        s = scores.loc[idx].to_numpy(float)
        ok = np.isfinite(s)
        p_m, n_m = p_m & ok, n_m & ok
        n_pos, n_neg = int(p_m.sum()), int(n_m.sum())
        if n_pos < MIN_SAME_VIDEO or n_neg < MIN_SAME_VIDEO:
            continue
        keep = p_m | n_m
        auc = float(roc_auc_score(p_m[keep].astype(int), s[keep]))
        per_video.append((str(vid), auc, n_pos * n_neg))

    if not per_video:
        return {
            "auc_weighted": None,
            "ci95_weighted": [None, None],
            "auc_unweighted": None,
            "ci95_unweighted": [None, None],
            "videos": 0,
            "pairs": 0,
        }
    aucs = np.array([x[1] for x in per_video], dtype=float)
    weights = np.array([x[2] for x in per_video], dtype=float)
    w_auc = float(np.sum(aucs * weights) / np.sum(weights))
    u_auc = float(aucs.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(aucs), (draws, len(aucs)))
    bw = np.sum(aucs[idx] * weights[idx], axis=1) / np.sum(weights[idx], axis=1)
    bu = aucs[idx].mean(axis=1)
    lo_w, hi_w = np.percentile(bw, [2.5, 97.5])
    lo_u, hi_u = np.percentile(bu, [2.5, 97.5])
    return {
        "auc_weighted": round(w_auc, 4),
        "ci95_weighted": [round(float(lo_w), 4), round(float(hi_w), 4)],
        "auc_unweighted": round(u_auc, 4),
        "ci95_unweighted": [round(float(lo_u), 4), round(float(hi_u), 4)],
        "videos": int(len(per_video)),
        "pairs": int(weights.sum()),
    }


def evaluate_stress(
    frame: pd.DataFrame,
    scores: pd.Series,
    *,
    draws: int = DRAWS,
    return_per_person: bool = False,
) -> dict:
    """Compute complete Study-07-compatible stress metrics on a data subset."""
    ctrs = contrasts(frame)
    v, a = frame["rating_valence"], frame["rating_arousal"]
    stressed = label(frame)
    calm_oklm = (v >= 5.0) & (a <= 5.0)

    out: dict = {}
    per_person_primary: dict[str, float] = {}
    for cname in (
        "stress",
        "strict",
        "vs_excitement",
        "vs_calm_negative",
        "vs_relaxed",
    ):
        pos_m, neg_m = ctrs[cname]
        pmap = person_stress_aucs(frame, scores, pos_m, neg_m)
        if cname == "stress":
            per_person_primary = pmap
        res = boot_ci(list(pmap.values()), draws=draws)
        out[cname] = {"auc": res["mean"], "ci95": res["ci95"], "people": res["n"]}

    out["same_video"] = same_video_auc(
        frame, scores, ctrs["stress"][0], ctrs["stress"][1], draws=draws
    )
    out["same_video_vs_calm"] = same_video_auc(
        frame, scores, stressed, calm_oklm, draws=draws
    )
    y = stressed.to_numpy(int)
    s = scores.to_numpy(float)
    ok = np.isfinite(s)
    if ok.sum() > 10 and len(np.unique(y[ok])) == 2:
        out["pooled_auc"] = round(float(roc_auc_score(y[ok], s[ok])), 4)
        out["average_precision"] = round(
            float(average_precision_score(y[ok], s[ok])), 4
        )
    else:
        out["pooled_auc"] = None
        out["average_precision"] = None
    out["n_trials"] = int(len(frame))
    if return_per_person:
        out["_per_person"] = per_person_primary
    return out


def evaluate_valence_arousal(
    frame: pd.DataFrame,
    val_pred: pd.Series,
    aro_pred: pd.Series,
    *,
    draws: int = DRAWS,
) -> dict:
    """Within-person Spearman rho and binary AUC for valence and arousal."""
    res: dict = {}
    rhos_both: dict[str, list[float]] = {"valence": [], "arousal": []}
    for name, col, pred, pos_fn, neg_fn in (
        (
            "valence",
            "rating_valence",
            val_pred,
            lambda s: s >= 6.0,
            lambda s: s <= 4.0,
        ),
        (
            "arousal",
            "rating_arousal",
            aro_pred,
            lambda s: s >= 6.0,
            lambda s: s <= 4.0,
        ),
    ):
        rhos, aucs = [], []
        for _, group in frame.groupby("participant_id"):
            idx = group.index
            y = group[col].to_numpy(float)
            p = pred.loc[idx].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            if (
                len(y) >= MIN_VA_TRIALS
                and len(np.unique(y)) > 1
                and len(np.unique(p)) > 1
            ):
                r = float(spearmanr(y, p).statistic)
                if np.isfinite(r):
                    rhos.append(r)
            pm, nm = (
                pos_fn(pd.Series(y)).to_numpy(bool),
                neg_fn(pd.Series(y)).to_numpy(bool),
            )
            if pm.sum() >= MIN_EACH and nm.sum() >= MIN_EACH:
                keep = pm | nm
                aucs.append(float(roc_auc_score(pm[keep].astype(int), p[keep])))
        rhos_both[name] = rhos
        rho_ci = boot_ci(rhos, draws=draws)
        auc_ci = boot_ci(aucs, draws=draws)
        y_all = frame[col].to_numpy(float)
        p_all = pred.to_numpy(float)
        ok_all = np.isfinite(y_all) & np.isfinite(p_all)
        pooled = (
            round(
                float(roc_auc_score((y_all[ok_all] > 5.0).astype(int), p_all[ok_all])),
                4,
            )
            if ok_all.sum() > 10 and len(np.unique(y_all[ok_all] > 5.0)) == 2
            else None
        )
        res[name] = {
            "within_person_rho": rho_ci["mean"],
            "rho_ci95": rho_ci["ci95"],
            "rho_people": rho_ci["n"],
            "within_person_auc": auc_ci["mean"],
            "auc_ci95": auc_ci["ci95"],
            "auc_people": auc_ci["n"],
            "pooled_gt5_auc": pooled,
        }
    v_r = res["valence"]["within_person_rho"]
    a_r = res["arousal"]["within_person_rho"]
    res["mean_rho"] = (
        round(0.5 * (v_r + a_r), 4) if (v_r is not None and a_r is not None) else None
    )
    return res


def person_center_matrix(
    x: np.ndarray,
    people: np.ndarray,
    *,
    prior_mean: np.ndarray | None = None,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Subtract each participant's column-wise mean within the given subset ONLY.

    WARNING (Study 09 audit)
    ------------------------
    Never call this across train and evaluation splits or across theme folds.
    For strict cold-start inference, do not apply per-subject batch centering at
    all; for causal unlabeled personalization, use ``causal_person_center`` which
    only accesses strictly past trials ``< i`` of the same participant.
    """
    arr = np.asarray(x, dtype=np.float64)
    df = pd.DataFrame(arr)
    grp = df.groupby(people, sort=False)
    means = grp.transform("mean").to_numpy(dtype=np.float64)
    if shrinkage > 0.0:
        counts = grp.transform("count").to_numpy(dtype=np.float64)
        mu0 = (
            np.asarray(prior_mean, dtype=np.float64)[None, :]
            if prior_mean is not None
            else np.nanmean(arr, axis=0, keepdims=True)
        )
        mu0 = np.where(np.isfinite(mu0), mu0, 0.0)
        means = np.where(np.isfinite(means), means, mu0)
        est = (counts * means + shrinkage * mu0) / np.maximum(counts + shrinkage, 1e-8)
        return (arr - est).astype(np.float32)
    return (arr - means).astype(np.float32)


def causal_person_center(
    x: np.ndarray,
    people: np.ndarray,
    order_keys: pd.DataFrame | None = None,
    *,
    prior_mean: np.ndarray | None = None,
    shrinkage: float = 5.0,
    rest_baseline: dict[str, np.ndarray] | None = None,
    rest_weight: float = 3.0,
) -> np.ndarray:
    """Strictly causal online Empirical-Bayes person centering (Protocol B1).

    For trial ``i`` of participant ``p`` (ordered chronologically), subtracts an
    unlabeled baseline estimated strictly from ``p``'s resting-state vector (if
    provided) and strictly past trials ``< i`` of ``p``, shrunk toward the
    training-fold population mean ``prior_mean``. Never reads trial ``i`` itself
    or any future trial ``j > i``.
    """
    arr = np.asarray(x, dtype=np.float64)
    n, d = arr.shape
    mu0 = (
        np.asarray(prior_mean, dtype=np.float64).reshape(d)
        if prior_mean is not None
        else np.zeros(d, dtype=np.float64)
    )
    mu0 = np.where(np.isfinite(mu0), mu0, 0.0)
    out = np.empty((n, d), dtype=np.float32)

    # Sort indices chronologically within each participant (stable tie-break).
    if order_keys is not None and len(order_keys) == n:
        sort_cols = [
            c
            for c in ("recorded_day", "event_order", "onset_s")
            if c in order_keys.columns
        ]
        if sort_cols:
            order_df = order_keys[sort_cols].copy()
            order_df["_pos"] = np.arange(n)
            order_df["_pid"] = np.asarray(people, dtype=str)
            grouped = order_df.sort_values(
                ["_pid", *sort_cols, "_pos"], kind="mergesort"
            ).groupby("_pid", sort=False)["_pos"]
        else:
            grouped = pd.Series(np.arange(n)).groupby(
                np.asarray(people, dtype=str), sort=False
            )
    else:
        grouped = pd.Series(np.arange(n)).groupby(
            np.asarray(people, dtype=str), sort=False
        )

    for pid, idx_series in grouped:
        idx = idx_series.to_numpy(dtype=int)
        running_sum = np.zeros(d, dtype=np.float64)
        running_cnt = np.zeros(d, dtype=np.float64)
        if rest_baseline is not None and str(pid) in rest_baseline:
            rb = np.asarray(rest_baseline[str(pid)], dtype=np.float64).reshape(d)
            ok_r = np.isfinite(rb)
            running_sum = np.where(ok_r, rest_weight * rb, 0.0)
            running_cnt = np.where(ok_r, rest_weight, 0.0)
        for pos in idx:
            # Estimate baseline strictly BEFORE observing trial ``pos``
            est = (running_sum + shrinkage * mu0) / np.maximum(
                running_cnt + shrinkage, 1e-8
            )
            row = arr[pos]
            out[pos] = (row - est).astype(np.float32)
            fin = np.isfinite(row)
            running_sum = np.where(
                fin, running_sum + np.where(fin, row, 0.0), running_sum
            )
            running_cnt = np.where(fin, running_cnt + 1.0, running_cnt)
    return out


def _fit_linear_probes(
    x_tr: np.ndarray,
    frame_tr: pd.DataFrame,
    x_ev: np.ndarray,
    *,
    max_pca: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit balanced LogReg (stress) + RidgeCV (valence, arousal, V&A stress).

    All scalers, imputers, PCA projections, and V&A standardization factors
    ``(sd_pv_tr, sd_pa_tr)`` are fitted strictly on ``x_tr`` (never using
    ``x_ev`` batch statistics), guaranteeing single-trial cold-start invariance.
    """
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if x_tr.shape[1] > max_pca and x_tr.shape[0] > max_pca + 10:
        steps.append(PCA(n_components=max_pca, random_state=0))
    pre = make_pipeline(*steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_tr = pre.fit_transform(x_tr)
        z_ev = pre.transform(x_ev)

        y_s = label(frame_tr).to_numpy(int)
        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 0, 7),
            cv=3,
            class_weight="balanced",
            max_iter=600,
            scoring="roc_auc",
        ).fit(z_tr, y_s)
        sc_log = clf.decision_function(z_ev)

        ppl = frame_tr["participant_id"]
        yv = frame_tr["rating_valence"] - frame_tr["rating_valence"].groupby(
            ppl
        ).transform("mean")
        ya = frame_tr["rating_arousal"] - frame_tr["rating_arousal"].groupby(
            ppl
        ).transform("mean")
        alphas = np.logspace(0, 4, 9)
        rv = RidgeCV(alphas=alphas).fit(z_tr, yv.to_numpy(float))
        ra = RidgeCV(alphas=alphas).fit(z_tr, ya.to_numpy(float))
        # Fit scaling strictly on training predictions (zero eval-batch leakage!)
        sd_pv = max(float(np.std(rv.predict(z_tr))), 1e-6)
        sd_pa = max(float(np.std(ra.predict(z_tr))), 1e-6)
        pv = rv.predict(z_ev)
        pa = ra.predict(z_ev)
        sc_va = pa / sd_pa - pv / sd_pv
    return sc_log, sc_va, pv, pa


def probe_representation_all_regimes(
    frame: pd.DataFrame,
    x_mat: np.ndarray,
    *,
    person_center: bool = True,
    normalization_protocol: str | None = None,
    eval_sealed: bool = True,
    max_pca: int = 128,
    draws: int = DRAWS,
    ref_per_person: dict[str, dict[str, float]] | None = None,
) -> tuple[dict, dict[str, pd.Series]]:
    """Run leak-free 3x3 dev cross-fit and (optionally) sealed-test linear probing.

    Normalization protocols
    -----------------------
    - ``"cold_start"`` (default if ``person_center=False``): zero per-subject
      history used at evaluation time (strictly invariant to single-trial eval).
    - ``"causal_online"``: strictly past-only Empirical-Bayes shrunk personal
      centering inside each fold (Protocol B1).
    - ``"fold_isolated_session"`` (default if ``person_center=True``): centers
      train strictly within train and eval strictly within each evaluation fold
      or regime (never crossing train/eval or theme folds; explicitly tagged as
      session-level unlabeled history, Protocol B2).
    - ``"legacy_global"``: reproduces Study-08 global transductive centering for
      audit comparisons only.
    """
    x_raw = np.asarray(x_mat, dtype=np.float32)
    proto = normalization_protocol
    if proto is None:
        proto = "fold_isolated_session" if person_center else "cold_start"

    def _prepare_fold(
        tr_idx: np.ndarray, ev_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if proto == "legacy_global":
            x_g = person_center_matrix(x_raw, frame["participant_id"].to_numpy())
            return x_g[tr_idx], x_g[ev_idx]
        if proto == "cold_start":
            return x_raw[tr_idx], x_raw[ev_idx]
        if proto == "causal_online":
            mu_tr = np.nanmean(x_raw[tr_idx], axis=0)
            mu_tr = np.where(np.isfinite(mu_tr), mu_tr, 0.0)
            x_tr_c = person_center_matrix(
                x_raw[tr_idx], frame.iloc[tr_idx]["participant_id"].to_numpy()
            )
            x_ev_c = causal_person_center(
                x_raw[ev_idx],
                frame.iloc[ev_idx]["participant_id"].to_numpy(),
                frame.iloc[ev_idx],
                prior_mean=mu_tr,
                shrinkage=5.0,
            )
            return x_tr_c, x_ev_c
        if proto == "fold_isolated_session":
            x_tr_c = person_center_matrix(
                x_raw[tr_idx], frame.iloc[tr_idx]["participant_id"].to_numpy()
            )
            x_ev_c = person_center_matrix(
                x_raw[ev_idx], frame.iloc[ev_idx]["participant_id"].to_numpy()
            )
            return x_tr_c, x_ev_c
        raise ValueError(f"Unknown normalization_protocol: {proto}")

    dev_mask = (frame["split"] == "dev").to_numpy()
    dev_idx = np.where(dev_mask)[0]
    dev_df = frame.iloc[dev_idx]

    dev_log = np.full(len(dev_idx), np.nan, dtype=float)
    dev_va = np.full(len(dev_idx), np.nan, dtype=float)
    dev_v = np.full(len(dev_idx), np.nan, dtype=float)
    dev_a = np.full(len(dev_idx), np.nan, dtype=float)

    pf = dev_df["person_fold"].to_numpy()
    tf = dev_df["theme_fold"].to_numpy()
    for p in range(3):
        for t in range(3):
            ev_local = (pf == p) & (tf == t)
            tr_local = (pf != p) & (tf != t)
            if ev_local.sum() == 0 or tr_local.sum() < 20:
                continue
            x_tr_f, x_ev_f = _prepare_fold(dev_idx[tr_local], dev_idx[ev_local])
            sl, sv, pv, pa = _fit_linear_probes(
                x_tr_f,
                dev_df.iloc[tr_local],
                x_ev_f,
                max_pca=max_pca,
            )
            dev_log[ev_local] = sl
            dev_va[ev_local] = sv
            dev_v[ev_local] = pv
            dev_a[ev_local] = pa

    out: dict = {
        "n_features": int(x_raw.shape[1]),
        "person_centered": bool(proto != "cold_start"),
        "normalization_protocol": proto,
    }
    preds_all: dict[str, pd.Series] = {}

    dev_s_log = pd.Series(dev_log, index=dev_df.index)
    dev_s_va = pd.Series(dev_va, index=dev_df.index)
    dev_pv = pd.Series(dev_v, index=dev_df.index)
    dev_pa = pd.Series(dev_a, index=dev_df.index)

    res_dev_log = evaluate_stress(
        dev_df, dev_s_log, draws=draws, return_per_person=True
    )
    res_dev_va = evaluate_stress(dev_df, dev_s_va, draws=draws, return_per_person=True)
    pp_dev_log = res_dev_log.pop("_per_person")
    pp_dev_va = res_dev_va.pop("_per_person")
    out["dev"] = {
        "stress_logreg": res_dev_log,
        "stress_ridge_va": res_dev_va,
        "affect": evaluate_valence_arousal(dev_df, dev_pv, dev_pa, draws=draws),
    }
    if ref_per_person and "dev" in ref_per_person:
        out["dev"]["gain_logreg_vs_ref"] = paired_gain_ci(
            pp_dev_log, ref_per_person["dev"], draws=draws
        )
        out["dev"]["gain_ridge_va_vs_ref"] = paired_gain_ci(
            pp_dev_va, ref_per_person["dev"], draws=draws
        )

    per_person_store = {"dev": pp_dev_va}
    preds_all["dev_logreg"] = dev_s_log
    preds_all["dev_ridge_va"] = dev_s_va
    preds_all["dev_valence"] = dev_pv
    preds_all["dev_arousal"] = dev_pa

    if eval_sealed:
        test_idx = np.where(~dev_mask)[0]
        test_df = frame.iloc[test_idx]
        ts_log = pd.Series(np.nan, index=test_df.index, dtype=float)
        ts_va = pd.Series(np.nan, index=test_df.index, dtype=float)
        tpv = pd.Series(np.nan, index=test_df.index, dtype=float)
        tpa = pd.Series(np.nan, index=test_df.index, dtype=float)

        # Evaluate each regime strictly isolated (never mixing test_new_both with
        # test_new_person or dev with test_new_video during centering!)
        for regime in common.REGIMES:
            reg_global = np.where((frame["split"] == regime).to_numpy())[0]
            if len(reg_global) == 0:
                continue
            x_tr_f, x_ev_f = _prepare_fold(dev_idx, reg_global)
            sl, sv, pv, pa = _fit_linear_probes(x_tr_f, dev_df, x_ev_f, max_pca=max_pca)
            reg_index = frame.index[reg_global]
            ts_log.loc[reg_index] = sl
            ts_va.loc[reg_index] = sv
            tpv.loc[reg_index] = pv
            tpa.loc[reg_index] = pa

        preds_all["test_logreg"] = ts_log
        preds_all["test_ridge_va"] = ts_va
        preds_all["test_valence"] = tpv
        preds_all["test_arousal"] = tpa

        for regime in common.REGIMES:
            rmask = test_df["split"] == regime
            sub = test_df[rmask]
            r_log = evaluate_stress(
                sub, ts_log[rmask], draws=draws, return_per_person=True
            )
            r_va = evaluate_stress(
                sub, ts_va[rmask], draws=draws, return_per_person=True
            )
            pp_log = r_log.pop("_per_person")
            pp_va = r_va.pop("_per_person")
            per_person_store[regime] = pp_va
            entry = {
                "stress_logreg": r_log,
                "stress_ridge_va": r_va,
                "affect": evaluate_valence_arousal(
                    sub, tpv[rmask], tpa[rmask], draws=draws
                ),
            }
            if ref_per_person and regime in ref_per_person:
                entry["gain_logreg_vs_ref"] = paired_gain_ci(
                    pp_log, ref_per_person[regime], draws=draws
                )
                entry["gain_ridge_va_vs_ref"] = paired_gain_ci(
                    pp_va, ref_per_person[regime], draws=draws
                )
            out[regime] = entry

    out["_per_person_va"] = per_person_store
    return out, preds_all


def probe_subject_identity(
    frame: pd.DataFrame,
    z_mat: np.ndarray,
    *,
    max_pca: int = 64,
) -> dict:
    """Measure static participant fingerprint in a representation.

    Reports:
    1. Variance fraction explained by participant ID (mean feature R^2 / ICC).
    2. Within-participant chronological split (first half -> second half) linear
       subject-ID top-1 and top-5 accuracy.
    3. Strict cross-day subject-ID accuracy (Day 1 -> Day 2 on multi-day people).
    4. Within-person vs between-person mean cosine similarity.
    """
    x = np.asarray(z_mat, dtype=float)
    ok_rows = np.isfinite(x).all(axis=1)
    if ok_rows.sum() < 50:
        return {"error": "insufficient_finite_rows"}
    sub = frame.iloc[np.where(ok_rows)[0]].reset_index(drop=True)
    x = x[ok_rows]
    # Standardize columns globally
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    xz = (x - mu) / sd

    # 1. Variance explained by participant_id (R^2_subject = Var(E[X|person]) / Var(X))
    df_x = pd.DataFrame(xz)
    person_means = (
        df_x.groupby(sub["participant_id"].to_numpy()).transform("mean").to_numpy()
    )
    r2_per_dim = np.var(person_means, axis=0) / np.maximum(np.var(xz, axis=0), 1e-8)
    subject_var_frac = float(np.mean(r2_per_dim))

    # Cosine similarity within vs between people (capped sample for speed)
    norms = np.linalg.norm(xz, axis=1, keepdims=True)
    xn = xz / np.maximum(norms, 1e-8)
    rng = np.random.default_rng(0)
    n_s = min(len(xn), 1500)
    sel = rng.choice(len(xn), size=n_s, replace=False)
    sim = xn[sel] @ xn[sel].T
    pids = sub["participant_id"].to_numpy()[sel]
    same = (pids[:, None] == pids[None, :]) & (~np.eye(n_s, dtype=bool))
    diff = pids[:, None] != pids[None, :]
    cos_same = float(sim[same].mean()) if same.any() else None
    cos_diff = float(sim[diff].mean()) if diff.any() else None

    # 2. Chronological split subject ID (first half -> second half per person)
    tr_idx, te_idx = [], []
    for _, grp in sub.groupby("participant_id"):
        if len(grp) < 10:
            continue
        order = grp.index.to_numpy()
        mid = len(order) // 2
        tr_idx.extend(order[:mid])
        te_idx.extend(order[mid:])

    def _id_eval(i_tr: list[int], i_te: list[int]) -> dict:
        if len(i_tr) < 20 or len(i_te) < 20:
            return {"top1": None, "top5": None, "n_people": 0, "chance": None}
        y_tr = sub.loc[i_tr, "participant_id"].to_numpy()
        y_te = sub.loc[i_te, "participant_id"].to_numpy()
        classes = np.unique(y_tr)
        if len(classes) < 3:
            return {
                "top1": None,
                "top5": None,
                "n_people": int(len(classes)),
                "chance": None,
            }
        n_comp = min(max_pca, xz.shape[1], len(i_tr) - 2)
        pipe = make_pipeline(
            PCA(n_components=n_comp, random_state=0),
            RidgeClassifier(alpha=10.0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(xz[i_tr], y_tr)
            dec = pipe.decision_function(xz[i_te])
            cls = pipe.named_steps["ridgeclassifier"].classes_
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        top1_pred = cls[np.argmax(dec, axis=1)]
        top1 = float((top1_pred == y_te).mean())
        k = min(5, len(cls))
        top5_idx = np.argsort(dec, axis=1)[:, -k:]
        top5 = float(np.mean([y_te[i] in cls[top5_idx[i]] for i in range(len(y_te))]))
        return {
            "top1": round(top1, 4),
            "top5": round(top5, 4),
            "n_people": int(len(classes)),
            "n_test_trials": int(len(i_te)),
            "chance": round(1.0 / len(classes), 4),
        }

    chrono = _id_eval(tr_idx, te_idx)

    # 3. Strict cross-day subject ID (Day 1 -> Day 2)
    cross_day = {"top1": None, "top5": None, "n_people": 0, "chance": None}
    if "recorded_day" in sub:
        d1_idx, d2_idx = [], []
        for _, grp in sub.groupby("participant_id"):
            g1 = grp[grp["recorded_day"] == 1]
            g2 = grp[grp["recorded_day"] == 2]
            if len(g1) >= 5 and len(g2) >= 5:
                d1_idx.extend(g1.index.tolist())
                d2_idx.extend(g2.index.tolist())
        cross_day = _id_eval(d1_idx, d2_idx)

    return {
        "subject_variance_r2": round(subject_var_frac, 4),
        "cosine_same_person": round(cos_same, 4) if cos_same is not None else None,
        "cosine_diff_person": round(cos_diff, 4) if cos_diff is not None else None,
        "within_session_id": chrono,
        "cross_day_id": cross_day,
    }


def probe_temporal_dynamics(
    frame: pd.DataFrame,
    z_ref: np.ndarray,
    z_early: np.ndarray,
    z_late: np.ndarray,
    *,
    draws: int = DRAWS,
) -> dict:
    """Test whether representations encode directed temporal physiological dynamics.

    All linear probes are trained strictly on ``dev`` participants and evaluated
    on held-out ``test_new_both + test_new_person`` participants (zero subject
    overlap!).

    Tests
    -----
    1. ``pre_vs_video``: distinguish pre-onset ``[-3, 0)`` vs stimulus ``[0, 10)``.
    2. ``early_vs_late``: distinguish early ``[0, 5)`` vs late ``[5, 10)`` stimulus.
    3. ``arrow_of_time``: given paired ``(z_early, z_late)`` vs reversed
       ``(z_late, z_early)``, classify forward vs reversed temporal order.
    4. ``effective_rank``: Roy-Vetterli effective rank of ``z_vid`` and ``z_delta``.
    """
    dev_m = (frame["split"] == "dev").to_numpy()
    # Evaluate on strictly unseen people (test_new_both + test_new_person)
    unseen_p_m = frame["split"].isin(["test_new_both", "test_new_person"]).to_numpy()

    zr = np.asarray(z_ref, dtype=float)
    ze = np.asarray(z_early, dtype=float)
    zl = np.asarray(z_late, dtype=float)
    zv = 0.5 * (ze + zl)
    ok = (
        np.isfinite(zr).all(axis=1)
        & np.isfinite(ze).all(axis=1)
        & np.isfinite(zl).all(axis=1)
    )

    def _binary_phase_probe(z0: np.ndarray, z1: np.ndarray) -> dict:
        tr_idx = np.where(dev_m & ok)[0]
        ev_idx = np.where(unseen_p_m & ok)[0]
        if len(tr_idx) < 40 or len(ev_idx) < 20:
            return {"within_person_auc": None, "ci95": [None, None], "people": 0}
        x_tr = np.vstack([z0[tr_idx], z1[tr_idx]])
        y_tr = np.concatenate([np.zeros(len(tr_idx)), np.ones(len(tr_idx))])
        x_ev = np.vstack([z0[ev_idx], z1[ev_idx]])
        y_ev = np.concatenate([np.zeros(len(ev_idx)), np.ones(len(ev_idx))])
        p_ev = np.concatenate(
            [
                frame["participant_id"].to_numpy()[ev_idx],
                frame["participant_id"].to_numpy()[ev_idx],
            ]
        )
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-3, 0, 5), cv=3, max_iter=400),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(x_tr, y_tr)
            sc = pipe.decision_function(x_ev)
        aucs = []
        for pid in np.unique(p_ev):
            m = p_ev == pid
            if m.sum() >= 10 and len(np.unique(y_ev[m])) == 2:
                aucs.append(float(roc_auc_score(y_ev[m], sc[m])))
        ci = boot_ci(aucs, draws=draws)
        return {
            "within_person_auc": ci["mean"],
            "ci95": ci["ci95"],
            "people": ci["n"],
            "pooled_auc": round(float(roc_auc_score(y_ev, sc)), 4),
        }

    # Arrow of time: forward [ze, zl, zl - ze] vs reversed [zl, ze, ze - zl]
    fwd = np.hstack([ze, zl, zl - ze])
    rev = np.hstack([zl, ze, ze - zl])

    # Effective rank via numpy SVD
    def _erank(mat: np.ndarray) -> float | None:
        m = mat[ok]
        if len(m) < 10:
            return None
        m = m - m.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(m[: min(len(m), 4000)], compute_uv=False)
        p = (sv**2) / max(float(np.sum(sv**2)), 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        return round(float(np.exp(-np.sum(p * np.log(p)))), 2)

    return {
        "pre_vs_video": _binary_phase_probe(zr, zv),
        "early_vs_late": _binary_phase_probe(ze, zl),
        "arrow_of_time": _binary_phase_probe(rev, fwd),
        "effective_rank_vid": _erank(zv),
        "effective_rank_delta": _erank(zv - zr),
        "embed_dim": int(zv.shape[1]),
    }


def probe_cross_modal_forecast(
    frame: pd.DataFrame,
    z_eeg_early: np.ndarray,
    body_targets: dict[str, np.ndarray],
    *,
    draws: int = DRAWS,
) -> dict:
    """Test whether early EEG latents [0, 5) s predict future late body [5, 10) s.

    Fits RidgeCV strictly on ``dev`` participants and computes within-person
    Pearson r on strictly unseen participants (``test_new_both + test_new_person``).
    """
    dev_m = (frame["split"] == "dev").to_numpy()
    unseen_p_m = frame["split"].isin(["test_new_both", "test_new_person"]).to_numpy()
    x = np.asarray(z_eeg_early, dtype=float)
    x_ok = np.isfinite(x).all(axis=1)
    out = {}
    for tname, y_raw in body_targets.items():
        y = np.asarray(y_raw, dtype=float)
        ok = x_ok & np.isfinite(y)
        tr = np.where(dev_m & ok)[0]
        ev = np.where(unseen_p_m & ok)[0]
        if len(tr) < 50 or len(ev) < 20:
            out[tname] = {"within_person_r": None, "ci95": [None, None], "people": 0}
            continue
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(x[tr], y[tr])
            pred = pipe.predict(x[ev])
        rs = []
        ev_df = pd.DataFrame(
            {"p": frame["participant_id"].to_numpy()[ev], "y": y[ev], "pr": pred}
        )
        for _, g in ev_df.groupby("p"):
            if len(g) >= MIN_VA_TRIALS and g["y"].std() > 0 and g["pr"].std() > 0:
                rs.append(float(np.corrcoef(g["y"], g["pr"])[0, 1]))
        ci = boot_ci(rs, draws=draws)
        out[tname] = {
            "within_person_r": ci["mean"],
            "ci95": ci["ci95"],
            "people": ci["n"],
        }
    return out
