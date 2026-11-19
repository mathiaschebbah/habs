"""Shared scoring code: trial table, person x theme splits, cross-fitting, bootstrap.

This code stays fixed while we search over models. The CI workflow takes the
model file from each experiment commit and everything else from ``main``, so a
model variant cannot change how it is scored.

Protocol
--------
Participants and video themes are each hashed into five buckets, with a salt
chosen per study. Bucket 0 of both is the sealed test, scored once at the end.
Test people with other themes and other people with test themes form two
secondary test regimes. Everything else is the development set, scored by a
3 x 3 blocked cross-fit: block (people fold p, theme fold t) is predicted by a
model trained on development rows from the other people folds *and* the other
theme folds. Every development prediction is therefore about a new person
watching a new video theme, and the stimulus columns are hidden from the rows a
model predicts.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BUDGET = 420  # seconds of fitting across all development blocks
# start_marker encodes the intended emotion of the video (markers.csv), so it is
# stimulus identity too.
STIMULUS = ("video", "video_theme", "theme", "default_emotion_label", "start_marker")
RATINGS = ("rating_valence", "rating_arousal", "rating_dominance")
HIDDEN = (*RATINGS, *STIMULUS, "recording", "onset_s")
REGIMES = ("test_new_both", "test_new_person", "test_new_video")
REGIONS = {
    "frontal": ("Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "F1", "F2", "F3", "F4", "Fz"),
    "left_temporal": ("F7", "FT7", "T7", "TP7", "F9", "T9", "P9"),
    "right_temporal": ("F8", "FT8", "T8", "TP8", "F10", "T10", "P10"),
    "central": ("FC1", "FC2", "FCz", "C1", "C2", "C3", "C4", "CP1", "CP2"),
    "parietal": ("P1", "P2", "P3", "P4", "Pz", "P5", "P6", "P7", "P8", "CP3", "CP4"),
    "occipital": ("O1", "O2", "PO3", "PO4", "PO7", "PO8", "POz", "PO5", "PO6"),
}
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
PERIPHERAL = ("ecg_", "eda_", "resp_")


def regional(frame, kind="bp_vid", bands=BANDS):
    """Mean over the channels of each region, per band (or evoked bin)."""
    return pd.DataFrame(
        {
            f"{kind}_{region}_{band}": frame[
                [f"{kind}_{ch}_{band}" for ch in chs if f"{kind}_{ch}_{band}" in frame]
            ].mean(axis=1)
            for region, chs in REGIONS.items()
            for band in bands
        },
        index=frame.index,
    )


def _bucket(keys, salt, k):
    return np.array(
        [
            int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest(), 16) % k
            for key in keys
        ]
    )


def assign_splits(table, salt):
    """Deterministic person x theme allocation; see the module docstring."""
    theme = table["video_theme"].astype("string")
    stem = table["video"].astype(str).str.replace(r"_\d+(\.\w+)?$", "", regex=True)
    table = table.assign(theme=theme.fillna(stem).astype(str))
    people = table["participant_id"].astype(str)
    p5 = _bucket(people, salt + "|people", 5)
    t5 = _bucket(table["theme"], salt + "|themes", 5)
    split = np.where(
        p5 == 0, np.where(t5 == 0, "test_new_both", "test_new_person"), "dev"
    )
    split = np.where((p5 != 0) & (t5 == 0), "test_new_video", split)
    return table.assign(
        split=split,
        person_fold=_bucket(people, salt + "|people-cv", 3),
        theme_fold=_bucket(table["theme"], salt + "|themes-cv", 3),
    )


def load(salt, path=None):
    """Read ``$TRIAL_DATA/trials.parquet`` and attach the study's splits."""
    if path is None:
        path = Path(os.environ["TRIAL_DATA"]) / "trials.parquet"
    return assign_splits(pd.read_parquet(path), salt)


def visible(frame):
    """Columns a model may read for the rows it predicts."""
    return frame.drop(columns=[c for c in HIDDEN if c in frame])


def _predict(fit_predict, train, evaluated, outputs):
    out = fit_predict(
        train.drop(columns=["split"]), visible(evaluated).drop(columns=["split"])
    )
    out = pd.DataFrame(out, index=evaluated.index)
    if (
        any(c not in out for c in outputs)
        or len(out) != len(evaluated)
        or not np.isfinite(out[list(outputs)]).all().all()
    ):
        raise ValueError("fit_predict must return finite predictions for every output.")
    return out[list(outputs)]


def cross_fit(table, fit_predict, outputs, *, budget=TIME_BUDGET):
    """Predict every development row from models blind to its person and theme."""
    dev = table[table["split"] == "dev"]
    predictions, started = [], time.perf_counter()
    for p in range(3):
        for t in range(3):
            block = (dev["person_fold"] == p) & (dev["theme_fold"] == t)
            train = dev[(dev["person_fold"] != p) & (dev["theme_fold"] != t)]
            predictions.append(_predict(fit_predict, train, dev[block], outputs))
            if time.perf_counter() - started > budget:
                raise TimeoutError(f"fit_predict exceeded the {budget}s budget.")
    return dev, pd.concat(predictions).loc[dev.index], time.perf_counter() - started


def sealed_test(table, fit_predict, outputs):
    """Fit once on all development rows; predict every sealed-test row."""
    dev, test = table[table["split"] == "dev"], table[table["split"] != "dev"]
    started = time.perf_counter()
    predictions = _predict(fit_predict, dev, test, outputs)
    return test, predictions, time.perf_counter() - started


def bootstrap(stats, score, reference=None, *, draws=5000, seed=0):
    """95% percentile interval over people for a score or a paired gain.

    Parameters
    ----------
    stats : pandas.DataFrame
        One row per person, as returned by a study's ``person_stats``.
    score : callable
        Maps such a table to the study's scalar score.
    reference : pandas.DataFrame, optional
        Same people for a baseline model; the interval is then for the gain.
    """
    if reference is not None:
        reference = reference.loc[stats.index]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(stats), len(stats))
        value = score(stats.iloc[idx])
        if reference is not None:
            value -= score(reference.iloc[idx])
        values.append(value)
    return [float(np.percentile(values, q)) for q in (2.5, 97.5)]


def summary(table):
    """Aggregate split sizes, safe to publish."""
    return {
        split: {
            "trials": int(len(rows)),
            "people": int(rows["participant_id"].nunique()),
            "themes": int(rows["theme"].nunique()),
        }
        for split, rows in table.groupby("split")
    }


def load_model(path):
    """Import ``fit_predict`` from a model file."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_predict


def run(study):
    """Command line shared by the studies; prints and saves aggregates only.

    ``--split dev`` cross-fits the development set, the search objective.
    ``--split test`` is the one-time sealed evaluation: every listed model is
    fitted on all development rows, and later models are compared with the
    first by a paired bootstrap over test people. Studies may add extra splits
    through ``study.extra_splits``.
    """
    parser = argparse.ArgumentParser()
    choices = ("dev", "test", *getattr(study, "EXTRA_SPLITS", ()))
    parser.add_argument("--split", choices=choices, default="dev")
    parser.add_argument("--model", nargs="+", default=[study.DEFAULT_MODEL])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = load(study.SALT)
    result = {"splits": summary(table), "models": {}}
    if args.split not in ("dev", "test"):
        result[args.split] = study.extra_splits(args.split, table)
    first = None
    for path in args.model if args.split in ("dev", "test") else ():
        fit_predict = load_model(path)
        if args.split == "dev":
            frame, predictions, elapsed = cross_fit(table, fit_predict, study.OUTPUTS)
            stats = study.person_stats(frame, predictions)
            scores = {"dev": study.metrics(frame, predictions, stats)}
            scores["dev"]["ci95"] = bootstrap(stats, study.score, draws=1000)
        else:
            frame, predictions, elapsed = sealed_test(table, fit_predict, study.OUTPUTS)
            scores, kept = {}, {}
            for regime in REGIMES:
                rows = frame["split"] == regime
                stats = study.person_stats(frame[rows], predictions[rows])
                kept[regime] = stats
                scores[regime] = study.metrics(frame[rows], predictions[rows], stats)
                scores[regime]["ci95"] = bootstrap(stats, study.score)
                if first is not None:
                    reference = first[regime]
                    scores[regime]["gain_vs_first"] = scores[regime]["score"] - (
                        study.score(reference)
                    )
                    scores[regime]["gain_ci95"] = bootstrap(
                        stats, study.score, reference
                    )
            first = first or kept
        scores["fit_seconds"] = round(elapsed, 1)
        result["models"][path] = scores
        head = scores["dev" if args.split == "dev" else "test_new_both"]
        print("---")
        for key, value in head.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
        print(f"fit_seconds: {elapsed:.1f}")
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(
            json.dumps(result, indent=2)
        )
