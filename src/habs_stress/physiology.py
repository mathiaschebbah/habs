"""Acute stress definition, physiological effect sizes, ECG RR-control, and appraisal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from habs_stress.protocol import MIN_EACH, paired_person_bootstrap


def stress_label(frame: pd.DataFrame) -> pd.Series:
    """Acute stress distress quadrant: valence <= 4 and arousal >= 6."""
    return (frame["rating_valence"] <= 4) & (frame["rating_arousal"] >= 6)


def stress_contrasts(frame: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Primary stress contrast plus specificity contrasts against arousal and valence."""
    v, a = frame["rating_valence"], frame["rating_arousal"]
    stressed = stress_label(frame)
    return {
        "stress": (stressed, ~stressed),
        "strict": ((v <= 3) & (a >= 7), ~((v <= 3) & (a >= 7))),
        "vs_excitement": (stressed, (v >= 6) & (a >= 6)),
        "vs_calm_negative": (stressed, (v <= 4) & (a <= 4)),
        "vs_relaxed": (stressed, (v >= 6) & (a <= 4)),
    }


def within_person_cohens_d(
    values: pd.Series,
    positive: pd.Series,
    negative: pd.Series,
    people: pd.Series,
) -> dict[str, Any]:
    """Mean within-person standardized effect size (Cohen's d) with 95% bootstrap CI."""
    ds = []
    for _, idx in people[positive | negative].groupby(people[positive | negative]).groups.items():
        pos = values.loc[idx][positive.loc[idx]].dropna()
        neg = values.loc[idx][negative.loc[idx]].dropna()
        if len(pos) < MIN_EACH or len(neg) < MIN_EACH:
            continue
        pooled = np.sqrt(
            (pos.var(ddof=1) * (len(pos) - 1) + neg.var(ddof=1) * (len(neg) - 1))
            / (len(pos) + len(neg) - 2)
        )
        if pooled > 1e-12:
            ds.append(float((pos.mean() - neg.mean()) / pooled))
    return paired_person_bootstrap(np.array(ds, dtype=float))


def within_person_auc(
    scores: pd.Series,
    positive: pd.Series,
    negative: pd.Series,
    people: pd.Series,
) -> dict[str, Any]:
    """Mean within-person ROC-AUC across participants with >= 3 trials per class."""
    aucs = []
    for _, idx in people.groupby(people).groups.items():
        p_m, n_m = positive.loc[idx], negative.loc[idx]
        keep = (p_m | n_m) & scores.loc[idx].notna()
        y = p_m[keep].to_numpy(int)
        s = scores.loc[idx][keep].to_numpy(float)
        if y.sum() >= MIN_EACH and (1 - y).sum() >= MIN_EACH:
            aucs.append(float(roc_auc_score(y, s)))
    return paired_person_bootstrap(np.array(aucs, dtype=float))


def rr_adjusted_ecg_effect(
    frame: pd.DataFrame,
    interval_col: str,
    rr_col: str = "delta_rr_ms",
) -> dict[str, Any]:
    """Test whether an ECG interval change persists after regressing out delta-RR.

    Used to verify that Bazett QTc shortening during acute stress is a formula
    artefact of RR lengthening (sinus vagal deceleration) rather than intrinsic
    repolarization or AV-node (PR segment) modulation.
    """
    stressed = stress_label(frame)
    plain_d, adj_d = [], []
    for _, g in frame.groupby("participant_id"):
        sub = g[[interval_col, rr_col]].dropna()
        if len(sub) < 2 * MIN_EACH:
            continue
        st = stressed.loc[sub.index]
        if st.sum() < MIN_EACH or (~st).sum() < MIN_EACH:
            continue
        y = sub[interval_col].to_numpy(float)
        x = sub[rr_col].to_numpy(float)
        x_c = x - x.mean()
        slope = float((x_c @ (y - y.mean())) / (x_c @ x_c)) if (x_c @ x_c) > 1e-12 else 0.0
        resid = y - slope * x_c
        for arr, target_list in ((y, plain_d), (resid, adj_d)):
            pos, neg = arr[st.to_numpy()], arr[~st.to_numpy()]
            pooled = np.sqrt(
                (pos.var(ddof=1) * (len(pos) - 1) + neg.var(ddof=1) * (len(neg) - 1))
                / (len(pos) + len(neg) - 2)
            )
            if pooled > 1e-12:
                target_list.append(float((pos.mean() - neg.mean()) / pooled))
    return {
        "plain_d": paired_person_bootstrap(np.array(plain_d)),
        "rr_adjusted_d": paired_person_bootstrap(np.array(adj_d)),
    }


def fixed_video_appraisal_contrast(
    frame: pd.DataFrame,
    score_col: str,
    *,
    strict_calm: bool = True,
) -> dict[str, Any]:
    """Compare stressed vs. calm viewers watching the exact same video clip.

    Holding the stimulus fixed removes shared visual/acoustic effects and
    isolates subjective stress appraisal.
    """
    stressed = stress_label(frame)
    calm = (
        (frame["rating_valence"] >= 6) & (frame["rating_arousal"] <= 4)
        if strict_calm
        else ~stressed
    )
    aucs, weights = [], []
    for _, g in frame.groupby("video"):
        s = g[score_col]
        p_m, n_m = stressed.loc[g.index] & s.notna(), calm.loc[g.index] & s.notna()
        n_pos, n_neg = int(p_m.sum()), int(n_m.sum())
        if n_pos < MIN_EACH or n_neg < MIN_EACH:
            continue
        keep = p_m | n_m
        aucs.append(float(roc_auc_score(p_m[keep].to_numpy(int), s[keep].to_numpy(float))))
        weights.append(n_pos * n_neg)
    if not aucs:
        return {"auc_weighted": np.nan, "auc_unweighted": np.nan, "n_videos": 0}
    w = np.array(weights, dtype=float)
    a = np.array(aucs, dtype=float)
    return {
        "auc_weighted": float(np.sum(w * a) / np.sum(w)),
        "auc_unweighted": float(np.mean(a)),
        "n_videos": int(len(a)),
    }
