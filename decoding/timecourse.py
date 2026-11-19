"""Study 05 analyses: time course and detection latency of stress reactions.

Usage: ``python -m decoding.timecourse --split curves`` (or ``latency_test``,
``latency_dev``, ``emotions``, ``habituation``, ``hrv``). Every output is an
aggregate over people; see journal/05_stress_timecourse.md for the frozen
protocol.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding import common
from decoding.stress import contrasts, label, person_stats

SALT = "habs-timecourse-2026-11-20"
BINS, ONSET = 15, 3  # bin k covers [k - 3, k - 2) s; bins 0-2 are the baseline
BODY = {"heart": ("hr",), "skin": ("sc",), "breathing": ("resp_amp", "resp_rate")}
EEG = tuple(f"eeg_{r}_{b}" for r in common.REGIONS for b in common.BANDS)
FAMILIES = {
    **BODY,
    "eeg": EEG,
    "body": ("hr", "sc", "resp_amp", "resp_rate"),
    "all": ("hr", "sc", "resp_amp", "resp_rate", *EEG),
}
EMOTIONS = ("fear", "disgust", "sadness", "anger")
MIN_EACH, DRAWS = 3, 2000


def load(folder=None):
    """Trial table with splits, joined to the second-by-second bins."""
    folder = Path(folder or os.environ["TRIAL_DATA"])
    table = common.load(SALT, folder)
    bins = pd.read_parquet(folder / "timecourse.parquet")
    return table.merge(bins, on="trial_id", how="inner", validate="1:1")


def corrected(frame, signal):
    """Bins of one signal minus the trial's 3-second pre-onset mean."""
    values = frame[[f"tc_{signal}_{k}" for k in range(BINS)]].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return values - np.nanmean(values[:, :ONSET], axis=1, keepdims=True)


def _person_d(values, positive, negative, people):
    """Within-person Cohen's d per column; one row per eligible person.

    Returns the array of d values and the matching person IDs.
    """
    rows, kept = [], []
    for person in np.unique(people[positive | negative]):
        mine = people == person
        pos, neg = values[mine & positive], values[mine & negative]
        if len(pos) < MIN_EACH or len(neg) < MIN_EACH:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pooled = np.sqrt(
                (
                    np.nanvar(pos, axis=0, ddof=1) * (len(pos) - 1)
                    + np.nanvar(neg, axis=0, ddof=1) * (len(neg) - 1)
                )
                / (len(pos) + len(neg) - 2)
            )
            rows.append((np.nanmean(pos, axis=0) - np.nanmean(neg, axis=0)) / pooled)
            kept.append(person)
    return np.array(rows), np.array(kept)


def _summary(per_person, seed=0):
    """Mean over people with a vectorised 95% bootstrap interval per column."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_person), (DRAWS, len(per_person)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        boot = np.nanmean(per_person[idx], axis=1)
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        mean = np.nanmean(per_person, axis=0)
    return mean, low, high


def onset(low, high):
    """First post-onset second where the interval excludes 0 twice in a row."""
    away = (low > 0) | (high < 0)
    for k in range(ONSET, BINS - 1):
        if away[k] and away[k + 1]:
            return k - ONSET
    return None


def curves(table):
    """Per-signal effect-size curves and onset latencies, dev and test people."""
    groups = {
        "dev": table["split"] == "dev",
        "test_people": table["split"].isin(["test_new_both", "test_new_person"]),
    }
    signals = ("hr", "sc", "resp_amp", "resp_rate", *EEG)
    out = {}
    for group, rows in groups.items():
        frame = table[rows]
        people = frame["participant_id"].to_numpy()
        out[group] = {}
        for name in ("stress", "vs_excitement", "vs_calm_negative"):
            positive, negative = (m.to_numpy() for m in contrasts(frame)[name])
            out[group][name] = {}
            for signal in signals:
                d, _ = _person_d(corrected(frame, signal), positive, negative, people)
                mean, low, high = _summary(d)
                out[group][name][signal] = {
                    "d": mean.round(4).tolist(),
                    "low": low.round(4).tolist(),
                    "high": high.round(4).tolist(),
                    "onset_s": onset(low, high),
                    "people": len(d),
                }
    return out


def _features(frame, family, seconds):
    """Baseline-corrected bins from onset to ``seconds``, centred per person."""
    columns = []
    for signal in FAMILIES[family]:
        values = corrected(frame, signal)[:, ONSET : ONSET + seconds]
        columns.append(pd.DataFrame(values, index=frame.index).add_prefix(f"{signal}_"))
    x = pd.concat(columns, axis=1)
    return x - x.groupby(frame["participant_id"].to_numpy()).transform("mean")


def _model():
    """Build the fixed detection model of the protocol."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-4, 0, 9),
            l1_ratios=(0,),
            scoring="neg_log_loss",
            use_legacy_attributes=False,
            max_iter=2000,
        ),
    )


def _fit_score(train, test, family, seconds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model().fit(_features(train, family, seconds), label(train))
        scores = model.decision_function(_features(test, family, seconds))
    return pd.Series(scores, index=test.index)


def latency_test(table):
    """Fit on development, score the sealed test for every family and second."""
    dev, test = table[table["split"] == "dev"], table[table["split"] != "dev"]
    tasks = [(f, t) for f in FAMILIES for t in range(1, 11)]
    scores = Parallel(n_jobs=-1)(delayed(_fit_score)(dev, test, f, t) for f, t in tasks)
    out = {}
    primary = test["split"] == "test_new_both"
    for (family, seconds), score in zip(tasks, scores, strict=True):
        frame = test[primary]
        stats = person_stats(frame, score[primary].to_frame("stress"))
        result = {
            "auc": float(np.nanmean(stats["stress"])),
            "ci95": common.bootstrap(stats, lambda s: float(np.nanmean(s["stress"]))),
            "vs_excitement": float(np.nanmean(stats["vs_excitement"])),
            "vs_calm_negative": float(np.nanmean(stats["vs_calm_negative"])),
            "people": int(stats["stress"].notna().sum()),
        }
        # post hoc (added after the primary cell turned out to hold 9 people):
        # intervals for the two larger regimes; the model is fixed, nothing is tuned
        for regime in ("test_new_person", "test_new_video"):
            rows = test["split"] == regime
            other = person_stats(test[rows], score[rows].to_frame("stress"))
            result[regime] = {
                "auc": float(np.nanmean(other["stress"])),
                "ci95": common.bootstrap(
                    other, lambda s: float(np.nanmean(s["stress"])), draws=2000
                ),
                "vs_excitement": float(np.nanmean(other["vs_excitement"])),
                "people": int(other["stress"].notna().sum()),
            }
        out.setdefault(family, {})[seconds] = result
    for by_time in out.values():
        reliable = [t for t, r in by_time.items() if r["ci95"][0] > 0.5]
        by_time["earliest_reliable_s"] = min(reliable) if reliable else None
    return out


def latency_dev(table):
    """Development cross-fit AUC per family and second (descriptive)."""
    dev = table[table["split"] == "dev"]
    blocks = [(p, t) for p in range(3) for t in range(3)]
    tasks = [(f, s, b) for f in FAMILIES for s in range(1, 11) for b in blocks]

    def one(family, seconds, block):
        p, t = block
        rows = (dev["person_fold"] == p) & (dev["theme_fold"] == t)
        train = dev[(dev["person_fold"] != p) & (dev["theme_fold"] != t)]
        return _fit_score(train, dev[rows], family, seconds)

    scores = Parallel(n_jobs=-1)(delayed(one)(*task) for task in tasks)
    out = {}
    for f in FAMILIES:
        for s in range(1, 11):
            parts = [
                sc
                for (ff, ss, _), sc in zip(tasks, scores, strict=True)
                if ff == f and ss == s
            ]
            score = pd.concat(parts).loc[dev.index]
            stats = person_stats(dev, score.to_frame("stress"))
            out.setdefault(f, {})[s] = {
                "auc": float(np.nanmean(stats["stress"])),
                "ci95": common.bootstrap(
                    stats, lambda x: float(np.nanmean(x["stress"])), draws=1000
                ),
                "vs_excitement": float(np.nanmean(stats["vs_excitement"])),
            }
    return out


def emotions(table):
    """Heart and skin curves of stress trials by reported emotion, per person."""
    out = {}
    emotion = table["final_emotion_label"].astype(str).str.strip().str.lower()
    groups = {
        "dev": table["split"] == "dev",
        "test_people": table["split"].isin(["test_new_both", "test_new_person"]),
    }
    for group, rows in groups.items():
        frame, feeling = table[rows], emotion[rows]
        stressed = label(frame).to_numpy()
        people = frame["participant_id"].to_numpy()
        out[group] = {"counts": feeling[stressed].value_counts().head(12).to_dict()}
        for signal in ("hr", "sc"):
            values = corrected(frame, signal)
            out[group][signal] = {}
            for name in EMOTIONS:
                curves_ = []
                chosen = stressed & (feeling == name).to_numpy()
                for person in np.unique(people[chosen]):
                    mine = people == person
                    target, rest = values[mine & chosen], values[mine & ~stressed]
                    if len(target) < 2 or len(rest) < MIN_EACH:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        curves_.append(np.nanmean(target, 0) - np.nanmean(rest, 0))
                curves_ = np.array(curves_)
                if len(curves_) < 5:
                    continue
                mean, low, high = _summary(curves_)
                window = np.nanmean(curves_[:, ONSET + 1 : ONSET + 6], axis=1)
                w_mean, w_low, w_high = _summary(window[:, None])
                out[group][signal][name] = {
                    "curve": mean.round(4).tolist(),
                    "low": low.round(4).tolist(),
                    "high": high.round(4).tolist(),
                    "change_1_6s": [
                        float(w_mean[0]),
                        float(w_low[0]),
                        float(w_high[0]),
                    ],
                    "people": len(curves_),
                }
    return out


def habituation(table):
    """Stress effect in the first vs second half of sessions and day 1 vs day 2."""
    measures = {
        "hr_1_6s": lambda f: np.nanmean(
            corrected(f, "hr")[:, ONSET + 1 : ONSET + 6], 1
        ),
        "sc_1_6s": lambda f: np.nanmean(
            corrected(f, "sc")[:, ONSET + 1 : ONSET + 6], 1
        ),
        "alpha_0_10s": lambda f: np.nanmean(
            np.mean([corrected(f, f"eeg_{r}_alpha") for r in common.REGIONS], axis=0)[
                :, ONSET : ONSET + 10
            ],
            1,
        ),
    }
    session = [table["participant_id"], table["recorded_day"]]
    late = table["event_order"] > table.groupby(session)["event_order"].transform(
        "median"
    )
    splits = {
        "half": (~late.to_numpy(), late.to_numpy()),
        "day": (
            (table["recorded_day"] == 1).to_numpy(),
            (table["recorded_day"] == 2).to_numpy(),
        ),
    }
    stressed = label(table).to_numpy()
    people = table["participant_id"].to_numpy()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, measure in measures.items():
            values = measure(table)[:, None]
            out[name] = {}
            for split, (early, later) in splits.items():
                d_early, p_early = _person_d(
                    values, stressed & early, ~stressed & early, people
                )
                d_late, p_late = _person_d(
                    values, stressed & later, ~stressed & later, people
                )
                late_by = (
                    dict(zip(p_late, d_late[:, 0], strict=True)) if len(p_late) else {}
                )
                paired = np.array(
                    [
                        [late_by[p] - d]
                        for p, d in zip(p_early, d_early[:, 0], strict=True)
                        if p in late_by
                    ]
                )
                out[name][split] = {
                    "early": [float(x[0]) for x in _summary(d_early)],
                    "late": [float(x[0]) for x in _summary(d_late)],
                    "late_minus_early": [float(x[0]) for x in _summary(paired)],
                    "people_paired": len(paired),
                }
    return out


def hrv(folder=None):
    """Agreement of 5-30 s heart rate and RMSSD with the same 60 s window."""
    folder = Path(folder or os.environ["TRIAL_DATA"])
    windows = pd.read_parquet(folder / "hrv_windows.parquet")
    out = {"windows": len(windows), "recordings": int(windows["recording"].nunique())}
    for measure in ("hr", "rmssd"):
        reference = windows[f"{measure}_60"]
        for seconds in (5, 10, 20, 30):
            short = windows[f"{measure}_{seconds}"]
            ok = short.notna() & reference.notna()
            diff = (short - reference)[ok]
            within = [
                np.corrcoef(g[f"{measure}_{seconds}"], g[f"{measure}_60"])[0, 1]
                for _, g in windows[ok].groupby("recording")
                if len(g) >= 5 and g[f"{measure}_{seconds}"].std() > 0
            ]
            log_r = np.corrcoef(np.log(short[ok]), np.log(reference[ok]))[0, 1]
            out[f"{measure}_{seconds}"] = {
                "pearson": float(np.corrcoef(short[ok], reference[ok])[0, 1]),
                "pearson_log": float(log_r) if measure == "rmssd" else None,
                "bias": float(diff.mean()),
                "limits": [
                    float(diff.mean() - 1.96 * diff.std()),
                    float(diff.mean() + 1.96 * diff.std()),
                ],
                "within_recording_r": float(np.nanmedian(within)),
                "valid": float(ok.mean()),
            }
    return out


def main():
    """Run one analysis and write aggregate JSON only."""
    parser = argparse.ArgumentParser()
    parts = ("curves", "latency_test", "latency_dev", "emotions", "habituation", "hrv")
    parser.add_argument("--split", choices=parts, required=True)
    parser.add_argument("--model", nargs="*")  # accepted for the shared workflow
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.split == "hrv":
        result = hrv()
    else:
        table = load()
        result = {
            "splits": common.summary(table),
            args.split: globals()[args.split](table),
        }
    text = json.dumps(result, indent=1, default=lambda x: None)
    print(text[:4000])
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(text)


if __name__ == "__main__":
    main()
