"""Fixed autoresearch harness: data access, splits, time budget and metric.

Do not modify during a run. The loop edits ``autoresearch/train.py`` only, and
the CI workflow enforces that by taking ``train.py`` from the experiment commit
and everything else from ``main``.

Protocol
--------
Participants and video themes are each hashed into five buckets. Bucket 0 of
both is the sealed test: it is scored once, at the end. The remaining
participants x themes form the development set, scored by 3 x 3 blocked
cross-fitting: each block (people fold p, theme fold t) is predicted by a model
trained on development rows from the other people folds *and* the other theme
folds. Every development prediction is therefore for a new person watching a
new video theme, so a model cannot score by recognising stimuli it was trained
on.

Metric
------
For valence and for arousal, Spearman correlation between predicted and
reported rating within each person, averaged over people. ``score`` is the mean
of the two. Chance is 0. Person-level offsets (how someone uses the scale)
cannot inflate it. Pooled AUC of ``rating > 5`` is printed for comparison with
other teams but is not the objective.
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SALT = "habs-autoresearch-2026-11-19"
TIME_BUDGET = 420  # seconds of fit_predict across all development blocks
TARGETS = ("rating_valence", "rating_arousal")
# Never shown to the model at prediction time. start_marker encodes the intended
# emotion of the video (markers.csv), so it is stimulus identity too.
STIMULUS = ("video", "video_theme", "theme", "default_emotion_label", "start_marker")
HIDDEN = (*TARGETS, "rating_dominance", *STIMULUS, "recording", "onset_s")
MIN_TRIALS = 5


def _bucket(keys, salt, k):
    return np.array(
        [
            int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest(), 16) % k
            for key in keys
        ]
    )


def load(path=None):
    """Read the trial table and attach split columns.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Parquet file written by ``autoresearch.extract``. Defaults to
        ``$AUTORESEARCH_DATA/trials.parquet``.

    Returns
    -------
    pandas.DataFrame
        One row per trial with ``theme``, ``person_fold``, ``theme_fold`` and
        ``split`` (``dev`` or one of the three test regimes).
    """
    if path is None:
        path = Path(os.environ["AUTORESEARCH_DATA"]) / "trials.parquet"
    table = pd.read_parquet(path)
    return assign_splits(table)


def assign_splits(table):
    """Deterministic person x theme allocation; see the module docstring."""
    theme = table["video_theme"].astype("string")
    stem = table["video"].astype(str).str.replace(r"_\d+(\.\w+)?$", "", regex=True)
    table = table.assign(theme=theme.fillna(stem).astype(str))
    people = table["participant_id"].astype(str)
    p5 = _bucket(people, SALT + "|people", 5)
    t5 = _bucket(table["theme"], SALT + "|themes", 5)
    split = np.where(
        p5 == 0, np.where(t5 == 0, "test_new_both", "test_new_person"), "dev"
    )
    split = np.where((p5 != 0) & (t5 == 0), "test_new_video", split)
    return table.assign(
        split=split,
        person_fold=_bucket(people, SALT + "|people-cv", 3),
        theme_fold=_bucket(table["theme"], SALT + "|themes-cv", 3),
    )


def visible(frame):
    """Columns a model may read for the rows it predicts."""
    return frame.drop(columns=[c for c in HIDDEN if c in frame], errors="ignore")


def person_rhos(frame, predictions):
    """Within-person Spearman rho per target, NaN when undefined.

    A person needs at least five trials and non-constant ratings and predictions.
    """
    rows = {}
    for person, group in frame.groupby("participant_id"):
        values = {}
        for target in TARGETS:
            y, p = group[target], predictions.loc[group.index, target]
            ok = len(group) >= MIN_TRIALS and y.nunique() > 1 and p.nunique() > 1
            values[target] = spearmanr(y, p).statistic if ok else np.nan
        rows[person] = values
    return pd.DataFrame.from_dict(rows, orient="index")


def _score(rhos):
    return float(np.mean([np.nanmean(rhos[t]) for t in TARGETS]))


def metrics(frame, predictions):
    """Score predictions for the rows of ``frame``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluated rows, with targets and participant IDs.
    predictions : pandas.DataFrame
        Same index, one column per target.

    Returns
    -------
    dict
        ``score`` (the objective) plus per-target within-person rho and AUC.
    """
    rhos = person_rhos(frame, predictions)
    result = {"score": _score(rhos)}
    for target in TARGETS:
        name = target.removeprefix("rating_")
        result[f"{name}_rho"] = float(np.nanmean(rhos[target]))
        result[f"{name}_auc"] = float(
            roc_auc_score(frame[target] > 5, predictions[target])
        )
    result["people"] = int(rhos.notna().any(axis=1).sum())
    result["n"] = len(frame)
    return result


def bootstrap(frame, predictions, baseline=None, *, draws=5000, seed=0):
    """95% percentile interval over people for the score or a paired gain."""
    rhos = person_rhos(frame, predictions)
    if baseline is not None:
        reference = person_rhos(frame, baseline).loc[rhos.index]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(rhos), len(rhos))
        value = _score(rhos.iloc[idx])
        if baseline is not None:
            value -= _score(reference.iloc[idx])
        values.append(value)
    return [float(np.percentile(values, q)) for q in (2.5, 97.5)]


def _predict(fit_predict, train, evaluated):
    out = fit_predict(
        train.drop(columns=["split"]), visible(evaluated).drop(columns=["split"])
    )
    out = pd.DataFrame(out, index=evaluated.index)
    missing = [t for t in TARGETS if t not in out]
    if (
        missing
        or len(out) != len(evaluated)
        or not np.isfinite(out[list(TARGETS)]).all().all()
    ):
        raise ValueError("fit_predict must return finite predictions for every target.")
    return out[list(TARGETS)]


def cross_fit(table, fit_predict, *, budget=TIME_BUDGET):
    """Predict every development row from models blind to its person and theme."""
    dev = table[table["split"] == "dev"]
    predictions, started = [], time.perf_counter()
    for p in range(3):
        for t in range(3):
            block = (dev["person_fold"] == p) & (dev["theme_fold"] == t)
            train = dev[(dev["person_fold"] != p) & (dev["theme_fold"] != t)]
            predictions.append(_predict(fit_predict, train, dev[block]))
            if time.perf_counter() - started > budget:
                raise TimeoutError(f"fit_predict exceeded the {budget}s budget.")
    return dev, pd.concat(predictions).loc[dev.index], time.perf_counter() - started


REGIMES = ("test_new_both", "test_new_person", "test_new_video")


def sealed_test(table, fit_predict):
    """Fit once on all development rows; predict every sealed-test row."""
    dev, test = table[table["split"] == "dev"], table[table["split"] != "dev"]
    started = time.perf_counter()
    predictions = _predict(fit_predict, dev, test)
    return test, predictions, time.perf_counter() - started


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


def _load_model(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_predict


def main():
    """Score one or more ``train.py`` files; print aggregate metrics only.

    ``--split dev`` cross-fits the development set (the loop's objective).
    ``--split test`` is the one-time sealed evaluation: every listed model is
    fitted on all development rows, and each later model is compared with the
    first one by a paired bootstrap over test people.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--train", nargs="+", default=["autoresearch/train.py"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = load()
    result = {"splits": summary(table), "models": {}}
    runs = []
    for path in args.train:
        fit_predict = _load_model(path)
        if args.split == "dev":
            frame, predictions, elapsed = cross_fit(table, fit_predict)
            scores = {"dev": metrics(frame, predictions)}
        else:
            frame, predictions, elapsed = sealed_test(table, fit_predict)
            scores = {}
            for regime in REGIMES:
                rows = frame["split"] == regime
                scores[regime] = metrics(frame[rows], predictions[rows])
                scores[regime]["ci95"] = bootstrap(frame[rows], predictions[rows])
                if runs:
                    scores[regime]["gain_vs_first"] = (
                        scores[regime]["score"] - runs[0][2][regime]["score"]
                    )
                    scores[regime]["gain_ci95"] = bootstrap(
                        frame[rows], predictions[rows], runs[0][1][rows]
                    )
        scores["fit_seconds"] = round(elapsed, 1)
        runs.append((path, predictions, scores))
        result["models"][path] = scores
        print("---")
        head = scores["dev" if args.split == "dev" else "test_new_both"]
        for key in (
            "score",
            "valence_rho",
            "arousal_rho",
            "valence_auc",
            "arousal_auc",
        ):
            print(f"{key}: {head[key]:.6f}")
        print(f"fit_seconds: {elapsed:.1f}")
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(
            json.dumps(result, indent=2)
        )


if __name__ == "__main__":
    main()
