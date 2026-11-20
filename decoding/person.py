"""Study 06 analyses: does physiology read the person or the video.

Many people watched the same videos. For each trial we split a rating, or a
physiological response, into what the other viewers of the same video showed
(the shared, stimulus-driven part) and the rest (the person's own part). We
then ask which part each body signal follows, and whether physiology can tell
who, among people watching the same video, felt stressed.

Usage: ``python -m decoding.person --split audit`` (counts only, run before
freezing the protocol), ``--split main`` (the frozen analyses) or
``--split explore``. Every output is an aggregate; see
journal/06_person_or_video.md for the protocol.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from decoding import common, timecourse
from decoding.stress import label, person_stats

SALT = "habs-person-2026-11-20"
MIN_OTHERS = 9  # other viewers needed for a video mean
MIN_TRIALS = 10  # complete trials per person for the within-person regressions
MIN_EACH = 3
DRAWS = 2000
ONSET = timecourse.ONSET
# response = mean baseline-corrected change over seconds [start, stop) after
# onset; windows follow the development curves of study 05
RESPONSES = {
    "heart": (("hr",), 1, 8),
    "skin": (("sc",), 4, 10),
    "breathing": (("resp_amp",), 2, 6),
    "delta": (tuple(f"eeg_{r}_delta" for r in common.REGIONS), 0, 2),
    "alpha": (tuple(f"eeg_{r}_alpha" for r in common.REGIONS), 0, 10),
}
RATINGS = {"valence": "rating_valence", "arousal": "rating_arousal"}
# a stressed trial should show a slower heart, a larger skin rise, shallower
# breathing, more delta and less alpha (study 05); the sign makes higher = stress
STRESS_SIGN = {"heart": -1, "skin": 1, "breathing": -1, "delta": 1, "alpha": -1}


def load(folder=None):
    """Trial table joined to the second-by-second bins, with person folds."""
    folder = Path(folder or os.environ["TRIAL_DATA"])
    table = common.load(SALT, folder)
    bins = pd.read_parquet(folder / "timecourse.parquet")
    return table.merge(bins, on="trial_id", how="inner", validate="1:1")


def response(frame, name):
    """Mean baseline-corrected change of one response over its window."""
    signals, start, stop = RESPONSES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.nanmean([timecourse.corrected(frame, s) for s in signals], axis=0)[
            :, ONSET + start : ONSET + stop
        ]
        return pd.Series(np.nanmean(values, axis=1), index=frame.index)


def components(frame, values):
    """Split person-centred values into the other viewers' mean and the rest.

    Returns three Series: the person-centred value, the mean person-centred
    value of the *other* viewers of the same video (NaN with fewer than
    ``MIN_OTHERS`` of them) and the remainder, centred again per person.
    """
    person, video = frame["participant_id"], frame["video"]
    values = pd.Series(values, index=frame.index, dtype=float)
    own = values - values.groupby(person).transform("mean")
    total = own.groupby(video).transform("sum")
    count = own.notna().groupby(video).transform("sum")
    others = (total - own.fillna(0)) / (count - own.notna())
    others = others.where(count - own.notna() >= MIN_OTHERS)
    rest = own - others
    return own, others, rest - rest.groupby(person).transform("mean")


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * np.nan


def _person_betas(frame, physiology, rating):
    """Within-person regression of a response on the shared and own rating parts.

    One row per person with at least ``MIN_TRIALS`` complete trials: the
    standardized coefficients of the shared part (other viewers' mean rating of
    the video) and of the person's own part, plus the three simple
    correlations of the response with the person-centred rating, the shared
    part and the own part.
    """
    x, _, _ = components(frame, physiology)
    y, shared, own = components(frame, rating)
    data = pd.DataFrame({"x": x, "y": y, "shared": shared, "own": own})
    data = data.dropna().assign(person=frame["participant_id"])
    rows = {}
    for person, group in data.groupby("person"):
        if len(group) < MIN_TRIALS:
            continue
        z = group[["x", "y", "shared", "own"]].apply(_z)
        if z.isna().any().any():
            continue
        design = np.column_stack([np.ones(len(z)), z["shared"], z["own"]])
        beta = np.linalg.lstsq(design, z["x"].to_numpy(), rcond=None)[0]
        rows[person] = {
            "beta_shared": beta[1],
            "beta_own": beta[2],
            "r_total": z["x"].corr(z["y"]),
            "r_shared": z["x"].corr(z["shared"]),
            "r_own": z["x"].corr(z["own"]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _mean_ci(stats, seed=0):
    """Mean over people per column with a 95% person-bootstrap interval."""
    values = stats.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (DRAWS, len(values)))
    boot = np.nanmean(values[idx], axis=1)
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {
        c: [round(float(m), 4), round(float(lo), 4), round(float(hi), 4)]
        for c, m, lo, hi in zip(stats, np.nanmean(values, 0), low, high, strict=True)
    } | {"people": len(values)}


def coupling(frame):
    """Shared and own coupling of every response with valence and arousal."""
    out = {}
    for name in RESPONSES:
        physiology = response(frame, name)
        out[name] = {
            rating: _mean_ci(_person_betas(frame, physiology, frame[column]))
            for rating, column in RATINGS.items()
        }
        # how stimulus-driven the response itself is: within-person correlation
        # with the other viewers' mean response to the same video
        own, others, _ = components(frame, physiology)
        data = pd.DataFrame({"own": own, "others": others}).dropna()
        data = data.assign(person=frame["participant_id"])
        r = {
            p: {"r": g["own"].corr(g["others"])}
            for p, g in data.groupby("person")
            if len(g) >= MIN_TRIALS and g["own"].std() > 0 and g["others"].std() > 0
        }
        out[name]["stimulus_locking"] = _mean_ci(pd.DataFrame.from_dict(r, "index"))
    return out


def _stratified_auc(frame, score):
    """AUC over stressed/non-stressed pairs that watched the same video.

    Returns the pooled pair-weighted AUC, a 95% interval from a bootstrap over
    videos, and the per-video counts used.
    """
    stressed = label(frame).to_numpy()
    rows = []
    for _, index in frame.groupby("video").indices.items():
        pos, neg = stressed[index], ~stressed[index]
        s = score.to_numpy()[index]
        ok = np.isfinite(s)
        if (pos & ok).sum() < MIN_EACH or (neg & ok).sum() < MIN_EACH:
            continue
        a, b = s[pos & ok], s[neg & ok]
        diff = a[:, None] - b[None, :]
        greater = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        rows.append((greater, len(a) * len(b)))
    if not rows:
        return {"auc": None, "ci95": None, "videos": 0, "pairs": 0}
    rows = np.array(rows, float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(rows), (DRAWS, len(rows)))
    boot = rows[idx, 0].sum(1) / rows[idx, 1].sum(1)
    return {
        "auc": round(float(rows[:, 0].sum() / rows[:, 1].sum()), 4),
        "ci95": [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])],
        "videos": len(rows),
        "pairs": int(rows[:, 1].sum()),
    }


def _model_scores(frame):
    """Study-05 fixed model on the first 10 s, blind to each row's person and theme."""
    blocks = [(p, t) for p in range(3) for t in range(3)]

    def one(p, t):
        rows = (frame["person_fold"] == p) & (frame["theme_fold"] == t)
        train = frame[(frame["person_fold"] != p) & (frame["theme_fold"] != t)]
        return timecourse._fit_score(train, frame[rows], "all", 10)

    parts = Parallel(n_jobs=-1)(delayed(one)(p, t) for p, t in blocks)
    return pd.concat(parts).loc[frame.index]


def _within_person_auc(frame, score):
    stats = person_stats(frame, score.to_frame("stress"))
    return {
        "auc": round(float(np.nanmean(stats["stress"])), 4),
        "ci95": [
            round(x, 4)
            for x in common.bootstrap(
                stats, lambda s: float(np.nanmean(s["stress"])), draws=DRAWS
            )
        ],
        "people": int(stats["stress"].notna().sum()),
    }


def room(frame, model=None):
    """Tell who, among the viewers of one video, felt stressed.

    Compares the other viewers' stress rate for the video (the stimulus alone),
    single responses and the fixed model, by within-person AUC and by AUC over
    pairs of viewers of the same video.
    """
    stressed = label(frame).astype(float)
    total = stressed.groupby(frame["video"]).transform("sum")
    count = stressed.groupby(frame["video"]).transform("count")
    # the other viewers' raw stress rate: the prior a clinician knowing the video has
    consensus = ((total - stressed) / (count - 1)).where(count - 1 >= MIN_OTHERS)
    model = _model_scores(frame) if model is None else model
    scores = {"consensus": consensus, "model": model}
    for name, sign in STRESS_SIGN.items():
        scores[name] = sign * response(frame, name)
    out = {}
    for name, score in scores.items():
        keep = score.notna()
        out[name] = {
            "within_person": _within_person_auc(frame[keep], score[keep]),
            "same_video": _stratified_auc(frame[keep], score[keep]),
        }
    return out, model


def cohorts(frame, model):
    """Primary measures within each acquisition cohort (robustness)."""
    groups = {
        "cohort_1": frame["cohort"] == 1,
        "cohort_2": frame["cohort"] == 2,
        "cohorts_3_5": frame["cohort"] >= 3,
    }
    out = {}
    for name, rows in groups.items():
        part = frame[rows]
        heart = _person_betas(part, response(part, "heart"), part["rating_valence"])
        skin = _person_betas(part, response(part, "skin"), part["rating_arousal"])
        out[name] = {
            "heart_valence": _mean_ci(heart[["beta_shared", "beta_own"]]),
            "skin_arousal": _mean_ci(skin[["beta_shared", "beta_own"]]),
            "model_same_video": _stratified_auc(part, model[rows]),
        }
    return out


def audit(frame):
    """Count what the protocol needs; compute no outcome."""
    stressed = label(frame)
    viewers = frame.groupby("video")["participant_id"].nunique()
    per_person = frame.groupby("participant_id").size()
    by_video = stressed.groupby(frame["video"]).agg(["sum", "count"])
    eligible = (by_video["sum"] >= MIN_EACH) & (
        by_video["count"] - by_video["sum"] >= MIN_EACH
    )
    cohorts_per_video = frame.groupby("video")["cohort"].nunique()
    _, others, _ = components(frame, frame["rating_valence"])
    complete = frame.assign(ok=others.notna()).groupby("participant_id")["ok"].sum()
    responses = {n: float(response(frame, n).notna().mean()) for n in RESPONSES}
    return {
        "trials": len(frame),
        "people": int(frame["participant_id"].nunique()),
        "videos": int(frame["video"].nunique()),
        "viewers_per_video_quantiles": viewers.quantile([0, 0.1, 0.5, 0.9, 1]).tolist(),
        "videos_with_enough_others": int((viewers > MIN_OTHERS).sum()),
        "trials_with_video_mean": float(others.notna().mean()),
        "people_with_min_trials": int((complete >= MIN_TRIALS).sum()),
        "trials_per_person_median": float(per_person.median()),
        "stressed_fraction": float(stressed.mean()),
        "videos_eligible_same_video_auc": int(eligible.sum()),
        "stressed_pairs_same_video": int(
            (by_video["sum"] * (by_video["count"] - by_video["sum"]))[eligible].sum()
        ),
        "videos_in_several_cohorts": int((cohorts_per_video > 1).sum()),
        "trials_by_cohort": frame["cohort"].value_counts().sort_index().to_dict(),
        "response_coverage": responses,
        "dominance_coverage": float(frame["rating_dominance"].notna().mean()),
    }


def explore(frame):
    """Exploratory: cardiac response of stress trials by perceived control.

    Freezing-like bradycardia should be strongest when the threat feels
    uncontrollable (low dominance).
    """
    heart = response(frame, "heart")
    stressed = label(frame)
    dominance = frame["rating_dominance"]
    low, high = stressed & (dominance <= 4), stressed & (dominance >= 6)
    rest = ~stressed
    rows = {}
    for person, group in frame.groupby("participant_id"):
        g, i = heart.loc[group.index], group.index
        lo, hi, re = g[low.loc[i]], g[high.loc[i]], g[rest.loc[i]]
        if min(lo.count(), hi.count()) >= 2 and re.count() >= MIN_EACH:
            rows[person] = {
                "low_control": lo.mean() - re.mean(),
                "high_control": hi.mean() - re.mean(),
                "low_minus_high": lo.mean() - hi.mean(),
            }
    return {"heart_bpm_by_control": _mean_ci(pd.DataFrame.from_dict(rows, "index"))}


def main():
    """Run one analysis and write aggregate JSON only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("audit", "main", "explore"), required=True)
    parser.add_argument("--model", nargs="*")  # accepted for the shared workflow
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = load()
    if args.split == "audit":
        result = {"audit": audit(table)}
    elif args.split == "explore":
        result = {"explore": explore(table)}
    else:
        rooms, model = room(table)
        result = {
            "coupling": coupling(table),
            "room": rooms,
            "cohorts": cohorts(table, model),
        }
    text = json.dumps(result, indent=1, default=lambda x: None)
    print(text[:6000])
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(text)


if __name__ == "__main__":
    main()
