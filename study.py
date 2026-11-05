"""Choose the study split, tune on development people, and summarize paired errors."""

import hashlib
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.model_selection import GroupKFold

from model import (
    CalibratedArousal,
    InsufficientSupport,
    VideoReference,
    attach_calibration,
    calibrated,
    population,
    reference_only,
    select_grid,
    valid_rows,
)

SEED = 20261105
PENALTIES = (0.1, 1.0, 10.0, 100.0, 1000.0)


def session_scenario(frame, minimum=20):
    """Use the first two adequately observed sessions and retain their true days."""
    frame = frame.copy()
    frame["recorded_day"] = frame.day
    valid = valid_rows(frame)
    counts = valid.groupby(["participant_id", "day"]).size()
    mapping = {}
    for person in frame.participant_id.unique():
        if person not in counts.index.get_level_values(0):
            continue
        days = sorted(counts.loc[person].index[counts.loc[person] >= minimum])
        for position, day in enumerate(days[:2], 1):
            mapping[(person, day)] = position
    frame["day"] = [
        mapping.get((p, d), 0)
        for p, d in zip(frame.participant_id, frame.recorded_day, strict=True)
    ]
    return frame


def participant_scores(y, predictions, people, include_rank=True):
    """Give each person equal weight and retain uncentered accuracy separately."""
    y, predictions, people = np.asarray(y), np.asarray(predictions), np.asarray(people)
    records = []
    for person in np.unique(people):
        mask = people == person
        truth, pred = y[mask], predictions[mask]
        errors = truth - pred
        rho = (
            float(spearmanr(truth, pred).statistic)
            if include_rank and np.std(truth) and np.std(pred)
            else np.nan
        )
        records.append(
            {
                "participant_id": person,
                "n": int(mask.sum()),
                "centered_mse": float(np.mean((errors - errors.mean()) ** 2)),
                "mae": float(np.mean(np.abs(errors))),
                "spearman": rho,
                "bias": float(errors.mean()),
            }
        )
    return pd.DataFrame(records)


def paired_people(frame, threshold=0.05, minimum=20):
    """Find participants with sufficient measured and rated trials on both days."""
    counts = valid_rows(frame, threshold).groupby(["participant_id", "day"]).size()
    counts = counts.unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    eligible = counts.index[(counts[1] >= minimum) & (counts[2] >= minimum)]
    return sorted(
        eligible,
        key=lambda p: hashlib.sha256(f"habs-arousal-v1|{p}".encode()).hexdigest(),
    )


def fold_context(
    frame,
    train_people,
    validation_people,
    paired,
    threshold=0.05,
    minimum_sources=10,
    minimum=20,
):
    """Recompute stimulus support without using validation people as sources."""
    background = reference_only(frame, paired, threshold)
    train_people = sorted(train_people)
    valid = valid_rows(frame, threshold)
    valid = valid.loc[valid.day.isin([1, 2])]
    while True:
        source = pd.concat(
            [valid.loc[valid.participant_id.isin(train_people)], background]
        )
        counts = source.groupby("video").participant_id.nunique()
        videos = sorted(counts.index[counts >= minimum_sources])
        train_grid = select_grid(frame, train_people, videos, threshold, minimum)
        remaining = sorted(train_grid.participant_id.unique())
        if remaining == train_people:
            break
        train_people = remaining
    test_grid = select_grid(frame, validation_people, videos, threshold, minimum)
    if len(train_people) < 2:
        raise InsufficientSupport(
            "Fewer than two development participants after stimulus support"
        )
    return {
        "train_people": train_people,
        "videos": videos,
        "background": background,
        "train_grid": train_grid,
        "test_grid": test_grid,
        "X_train": attach_calibration(
            train_grid.drop(columns="rating_arousal"), frame, videos, threshold
        ),
        "y_train": train_grid.rating_arousal.to_numpy(),
        "X_test": attach_calibration(
            test_grid.drop(columns="rating_arousal"), frame, videos, threshold
        ),
        "y_test": test_grid.rating_arousal.to_numpy(),
    }


def split_study(frame, threshold=0.05, minimum_sources=10, minimum=20):
    """Create the frozen person-level allocation before fitting any model."""
    people = paired_people(frame, threshold, minimum)
    n_test = (len(people) + 2) // 3
    context = fold_context(
        frame,
        people[n_test:],
        people[:n_test],
        people,
        threshold,
        minimum_sources,
        minimum,
    )
    context["paired_people"] = people
    context["assigned_train"] = people[n_test:]
    context["assigned_test"] = people[:n_test]
    if context["test_grid"].participant_id.nunique() < 2:
        raise InsufficientSupport(
            "Fewer than two test participants after stimulus support"
        )
    return context


def tuning_contexts(frame, split, threshold=0.05, minimum_sources=10, minimum=20):
    """Hold out whole development people in each internal validation fold."""
    grid = split["train_grid"]
    cv = GroupKFold(
        n_splits=min(5, grid.participant_id.nunique()), shuffle=True, random_state=SEED
    )
    result = []
    for train, validation in cv.split(grid, groups=grid.participant_id):
        context = fold_context(
            frame,
            grid.iloc[train].participant_id.unique(),
            grid.iloc[validation].participant_id.unique(),
            split["paired_people"],
            threshold,
            minimum_sources,
            minimum,
        )
        if len(context["test_grid"]):
            result.append(context)
    if len(result) < 2:
        raise InsufficientSupport(
            "Insufficient stimulus support in internal validation"
        )
    return result


def tune(
    frame,
    split,
    threshold=0.05,
    minimum_sources=10,
    minimum=20,
    modes=("eda", "eeg", "ecg", "all", "video", "missing"),
    verbose=True,
):
    """Select calibration and correction penalties without evaluating final test."""
    contexts = tuning_contexts(frame, split, threshold, minimum_sources, minimum)
    base_scores = {
        params: []
        for params in product(PENALTIES, (1.0, 10.0, 100.0), (1.0, 10.0, 100.0))
    }
    for fold in contexts:
        source = pd.concat(
            [population(fold["X_train"], fold["y_train"]), fold["background"]]
        )
        for alpha in PENALTIES:
            reference = VideoReference(alpha).fit(source)
            for level, slope in product((1.0, 10.0, 100.0), repeat=2):
                prediction = calibrated(reference, fold["X_test"], level, slope, True)
                scores = participant_scores(
                    fold["y_test"],
                    prediction,
                    fold["X_test"].participant_id,
                    include_rank=False,
                )
                base_scores[(alpha, level, slope)].extend(scores.centered_mse.tolist())
    best = min(
        base_scores,
        key=lambda key: (round(np.mean(base_scores[key]), 6), *(-v for v in key)),
    )
    parameters = dict(
        zip(["base_alpha", "level_penalty", "slope_penalty"], best, strict=True)
    )
    parameters.update(minimum_sources=minimum_sources, minimum_trials=minimum)
    records = [
        {
            "mode": "affine",
            "base_alpha": key[0],
            "level_penalty": key[1],
            "slope_penalty": key[2],
            "score": float(np.mean(values)),
            "people_scored": len(values),
        }
        for key, values in base_scores.items()
    ]
    if verbose:
        print("Calibration penalties selected on development participants", flush=True)
    selected = {}
    for mode in modes:
        candidates = {alpha: [] for alpha in PENALTIES}
        for fold in contexts:
            estimator = CalibratedArousal(
                mode=mode, signal_alpha=PENALTIES[0], **parameters
            )
            estimator.fit(
                fold["X_train"], fold["y_train"], background=fold["background"]
            )
            for alpha in PENALTIES:
                estimator.refit_correction(alpha)
                prediction = estimator.predict(fold["X_test"])
                table = participant_scores(
                    fold["y_test"],
                    prediction,
                    fold["X_test"].participant_id,
                    include_rank=False,
                )
                candidates[alpha].extend(table.centered_mse.tolist())
        for alpha, scores in candidates.items():
            candidates[alpha] = float(np.mean(scores))
            records.append(
                {
                    "mode": mode,
                    "signal_alpha": alpha,
                    "score": candidates[alpha],
                    "people_scored": len(scores),
                    **parameters,
                }
            )
        selected[mode] = min(candidates, key=lambda a: (round(candidates[a], 6), -a))
        if verbose:
            print(f"Correction selected: {mode}", flush=True)
    return parameters, selected, records


def effect_summary(reference, candidate, seed=SEED):
    """Bootstrap paired participant differences conditional on the fitted models."""
    joined = reference.merge(
        candidate,
        on="participant_id",
        suffixes=("_ref", "_model"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    results = {}
    for metric in ("centered_mse", "mae", "spearman"):
        difference = (joined[f"{metric}_ref"] - joined[f"{metric}_model"]).to_numpy()
        if metric == "spearman":
            difference = -difference
        difference = difference[np.isfinite(difference)]
        if not len(difference):
            results[metric] = {"n": 0}
            continue
        samples = rng.choice(
            difference, size=(5000, len(difference)), replace=True
        ).mean(axis=1)
        results[metric] = {
            "n": len(difference),
            "mean": float(difference.mean()),
            "median": float(np.median(difference)),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "positive": int((difference > 0).sum()),
            "quantiles": np.quantile(difference, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        }
        half_width = (
            (
                float(t.ppf(0.975, len(difference) - 1))
                * float(np.std(difference, ddof=1))
                / np.sqrt(len(difference))
            )
            if len(difference) > 1
            else None
        )
        results[metric]["ci95_student"] = (
            [
                float(difference.mean() - half_width),
                float(difference.mean() + half_width),
            ]
            if half_width is not None
            else [None, None]
        )
    return results
