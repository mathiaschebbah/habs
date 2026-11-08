"""Describe conditional physiology and profile stability after model settings freeze."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from habs.data import target_values
from habs.features import PHYSIOLOGY
from habs.profiles import PhysiologyProfiles

MARKERS = ("eeg_frontal_alpha", "ecg_hr", "eda_log_level")


def _mean_interval(values, *, minimum=5, seed=20261108):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < minimum:
        return {"people": len(values), "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {
        "people": len(values),
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def _within_rank(table, prediction):
    correlations = []
    truth = target_values(table).to_numpy()
    predicted = np.asarray(prediction, dtype=float)
    for person in sorted(set(table["participant_id"])):
        mask = (table["participant_id"] == person).to_numpy()
        valid = mask & np.isfinite(truth) & np.isfinite(predicted)
        if (
            valid.sum() < 10
            or np.std(truth[valid]) == 0
            or np.std(predicted[valid]) == 0
        ):
            continue
        correlations.append(float(spearmanr(truth[valid], predicted[valid]).statistic))
    return _mean_interval(correlations)


def _normalized_physiology(development, history, test):
    video_means = development.groupby("video")[list(MARKERS)].mean()
    global_means = development[list(MARKERS)].mean()

    def residual(table):
        baseline = video_means.reindex(table["video"]).fillna(global_means).to_numpy()
        return pd.DataFrame(
            table[list(MARKERS)].to_numpy() - baseline,
            index=table.index,
            columns=MARKERS,
        )

    previous = residual(history)
    previous["participant_id"] = history["participant_id"].to_numpy()
    personal = previous.groupby("participant_id")[list(MARKERS)].median()
    training = (
        residual(development).to_numpy()
        - personal.reindex(development["participant_id"]).to_numpy()
    )
    scales = np.asarray(
        [
            np.std(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
            for column in training.T
        ]
    )
    scales[scales < 1e-9] = np.nan
    current = (
        residual(test).to_numpy() - personal.reindex(test["participant_id"]).to_numpy()
    ) / scales
    return pd.DataFrame(current, index=test.index, columns=MARKERS)


def _specificity(test, normalized):
    differences = {name: [] for name in MARKERS}
    pair_counts = {name: 0 for name in MARKERS}
    arousal_gaps = []
    for _, group in test.groupby("participant_id"):
        negative = group.loc[
            (group["rating_valence"] <= 4) & (group["rating_arousal"] >= 6)
        ]
        positive = group.loc[
            (group["rating_valence"] >= 6) & (group["rating_arousal"] >= 6)
        ]
        candidates = sorted(
            (abs(float(n.rating_arousal) - float(p.rating_arousal)), n.Index, p.Index)
            for n in negative.itertuples()
            for p in positive.itertuples()
        )
        used_negative, used_positive, pairs = set(), set(), []
        for gap, neg, pos in candidates:
            if gap > 0.5:
                break
            if neg in used_negative or pos in used_positive:
                continue
            pairs.append((neg, pos))
            arousal_gaps.append(gap)
            used_negative.add(neg)
            used_positive.add(pos)
        for name in MARKERS:
            values = [
                normalized.loc[neg, name] - normalized.loc[pos, name]
                for neg, pos in pairs
                if np.isfinite(normalized.loc[[neg, pos], name]).all()
            ]
            if values:
                differences[name].append(float(np.mean(values)))
                pair_counts[name] += len(values)
    return {
        "caliper_arousal_units": 0.5,
        "matched_pairs": len(arousal_gaps),
        "mean_absolute_arousal_gap": float(np.mean(arousal_gaps))
        if arousal_gaps
        else None,
        "markers": {
            name: {**_mean_interval(values), "pairs": pair_counts[name]}
            for name, values in differences.items()
        },
    }


def _dose_response(test, normalized):
    score = target_values(test).to_numpy()
    masks = [score == 0] + [
        (score > low) & (score <= high)
        for low, high in (
            (0, 0.125),
            (0.125, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1),
        )
    ]
    centers = [0.0, 0.0625, 0.1875, 0.375, 0.625, 0.875]
    result = {}
    for name in MARKERS:
        points = []
        for center, mask in zip(centers, masks, strict=True):
            selected = pd.DataFrame(
                {
                    "person": test.loc[mask, "participant_id"],
                    "value": normalized.loc[mask, name],
                }
            )
            means = selected.groupby("person")["value"].mean().dropna().to_numpy()
            points.append(
                {
                    "score_center": center,
                    "trials": int(mask.sum()),
                    **_mean_interval(means),
                }
            )
        result[name] = {
            "points": points,
            "within_person_rank": _within_rank(test, normalized[name].to_numpy()),
        }
    return result


def _profile_analysis(development, history, test, count):
    training_ids = set(development["participant_id"])
    training_history = history.loc[history["participant_id"].isin(training_ids)]
    model = PhysiologyProfiles(count).fit(training_history)
    if not model.enabled_:
        return {"status": "insufficient_training_support"}, None
    test_ids = set(test["participant_id"])
    previous = history.loc[history["participant_id"].isin(test_ids)]
    first = model.summarize(previous)
    second = model.summarize(test)
    common = sorted(set(first) & set(second))
    if len(common) < 5:
        stability = {
            "status": "insufficient_two_session_support",
            "people": len(common),
        }
    else:

        def full_histogram(vector):
            complete = np.concatenate([vector, [max(0.0, 1 - np.sum(vector))]])
            return complete / complete.sum()

        j1 = np.vstack([full_histogram(first[person]) for person in common])
        j2 = np.vstack([full_histogram(second[person]) for person in common])
        similarity = 1 - np.sqrt(
            np.sum((np.sqrt(j1)[:, None] - np.sqrt(j2)[None]) ** 2, axis=2) / 2
        )
        matched = np.diag(similarity)
        off_diagonal = ~np.eye(len(common), dtype=bool)
        observed = float(matched.mean() - similarity[off_diagonal].mean())
        rng = np.random.default_rng(20261108)
        null, contrasts = [], []
        counts = (
            test.loc[np.isfinite(test[list(PHYSIOLOGY)]).all(axis=1)]
            .groupby("participant_id")
            .size()
        )
        ordered = sorted(range(len(common)), key=lambda i: counts[common[i]])
        strata = np.array_split(ordered, max(1, len(common) // 5))
        for _ in range(2000):
            perm = np.arange(len(common))
            for stratum in strata:
                perm[stratum] = rng.permutation(stratum)
            null.append(float(similarity[np.arange(len(common)), perm].mean()))
            sample = rng.integers(0, len(common), size=len(common))
            unequal = sample[:, None] != sample[None, :]
            if unequal.any():
                contrasts.append(
                    float(
                        matched[sample].mean()
                        - similarity[sample][:, sample][unequal].mean()
                    )
                )
        stability = {
            "status": "evaluated",
            "people": len(common),
            "matched_similarity": float(matched.mean()),
            "between_person_similarity": float(similarity[off_diagonal].mean()),
            "matched_minus_between": observed,
            "ci95": [float(v) for v in np.quantile(contrasts, [0.025, 0.975])],
            "identity_shuffle_reference95": [
                float(v) for v in np.quantile(null, [0.025, 0.975])
            ],
            "anonymous_matched_similarities": matched.tolist(),
            "identity_shuffle_means": null,
            "scope": (
                "Observational identity-shuffle reference within trial-count strata."
            ),
        }
    complete_train = training_history.loc[
        np.isfinite(training_history[list(PHYSIOLOGY)]).all(axis=1)
    ]
    complete_test = test.loc[np.isfinite(test[list(PHYSIOLOGY)]).all(axis=1)]
    z_train = model.scaler_.transform(model._residuals(complete_train))
    z_test = (
        model.scaler_.transform(model._residuals(complete_test))
        if len(complete_test)
        else np.empty((0, len(PHYSIOLOGY)))
    )
    projection = PCA(n_components=2, svd_solver="full").fit(z_train)
    structure = {
        "development_coordinates": projection.transform(z_train),
        "development_cluster": model.clusters_.predict(z_train),
        "test_coordinates": projection.transform(z_test)
        if len(z_test)
        else np.empty((0, 2)),
        "test_score": target_values(complete_test).to_numpy()
        if len(complete_test)
        else np.asarray([]),
        "centroids": projection.transform(model.clusters_.cluster_centers_),
        "explained_variance": projection.explained_variance_ratio_,
    }
    summary = {
        "status": "evaluated",
        "clusters": count,
        "training_complete_trials": len(complete_train),
        "test_complete_trials": len(complete_test),
        "stability": stability,
        "projection_variance": [float(x) for x in projection.explained_variance_ratio_],
    }
    return summary, structure


def describe_responses(development, history, test, *, n_clusters=None):
    """Describe reserved responses under fixed marker and matching definitions.

    Parameters
    ----------
    development : pandas.DataFrame
        Population development trials only, for stimulus baselines and scales.
    history : pandas.DataFrame
        First-session history, including the specified calibration participants.
    test : pandas.DataFrame
        Reserved novel-video future trials after the settings freeze.
    n_clusters : int or None, default=None
        Development-selected cluster count, or no discrete representation.

    Returns
    -------
    summary : dict
        Aggregate matching, dose-response and profile-stability statistics.
    structure : dict or None
        In-memory projection inputs for binned figures, never exported as raw rows.
    """
    normalized = _normalized_physiology(development, history, test)
    summary = {
        "specificity": _specificity(test, normalized),
        "dose_response": _dose_response(test, normalized),
    }
    structure = None
    if n_clusters is not None:
        summary["profiles"], structure = _profile_analysis(
            development, history, test, n_clusters
        )
    else:
        summary["profiles"] = {"status": "unsupported_on_development"}
    return summary, structure
