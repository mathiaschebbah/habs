"""Standardize both measurements with each person's already observed J1."""

import math
from collections import Counter

import numpy as np
import pandas as pd

from habs.temporal.data import _valid, attach_history, transition_table

CATEGORICAL = (
    "video",
    "previous_video",
    "expected_emotion",
    "previous_expected_emotion",
    "expected_emotion_source",
)
NUMERIC = (
    "lag_arousal_z",
    "lag_eeg_z",
    "order",
    "order_squared",
    "time",
    "time_squared",
    "log_gap",
    "log_gap_squared",
    "arousal_gap",
    "eeg_gap",
    "scale_ratio",
    "arousal_scale_ratio",
)


def j1_statistics(features):
    """Compute only first-session personal reference means and standard deviations.

    Parameters
    ----------
    features : pandas.DataFrame
        Native features with recorded days, source and ratings.

    Returns
    -------
    dict
        Per-person J1 arousal/EEG statistics; zero-variance people are omitted.
    """
    groups = {}
    for row in features.to_dict("records"):
        if int(row["recorded_day"]) == 1 and row["cohort"] == 2 and _valid(row):
            groups.setdefault(str(row["participant_id"]), []).append(
                (
                    float(row["rating_arousal"]),
                    -math.log(float(row["eeg_posterior_alpha"])),
                )
            )
    result = {}
    for person, rows in groups.items():
        values = np.asarray(rows)
        means, scales = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
        if len(values) < 20 or not np.isfinite(scales).all() or min(scales) < 1e-8:
            continue
        result[person] = {
            "mean_a": float(means[0]),
            "sd_a": float(scales[0]),
            "mean_p": float(means[1]),
            "sd_p": float(scales[1]),
            "n": len(values),
        }
    return result


def prepare_data(
    features, events, *, min_pairs=20, stable_scales=False, population_scales=False
):
    """Select actual novel-video pairs and retain their standardized responses.

    Parameters
    ----------
    features, events : pandas.DataFrame
        Audited native observations and complete event inventory.
    min_pairs : int, default=20
        Required usable transitions in each recorded day.
    stable_scales : bool, default=False
        Prespecified sensitivity excluding very small J1 standard deviations.
    population_scales : bool, default=False
        Use common J1 within-person scales as a normalization sensitivity.

    Returns
    -------
    tuple
        Selected dataframe, private J1 statistics and publication-safe support.
    """
    stats = j1_statistics(features)
    if stable_scales:
        stats = {
            p: s for p, s in stats.items() if s["sd_a"] >= 0.25 and s["sd_p"] >= 0.05
        }
    table = transition_table(features, events)
    rows = [
        r
        for r in table.to_dict("records")
        if r["pair_eligible"]
        and r["novel_to_person"]
        and r["participant_id"] in stats
        and r["recorded_day"] in (1, 2)
    ]
    counts = Counter((r["participant_id"], r["recorded_day"]) for r in rows)
    people = sorted(
        p for p in stats if all(counts[(p, d)] >= min_pairs for d in (1, 2))
    )
    selected = set(people)
    rows = [r for r in rows if r["participant_id"] in selected]
    if len(people) < 5:
        raise ValueError("Too few people with paired usable temporal measurements.")
    if population_scales:
        global_a = math.sqrt(float(np.mean([stats[p]["sd_a"] ** 2 for p in people])))
        global_p = math.sqrt(float(np.mean([stats[p]["sd_p"] ** 2 for p in people])))
        stats = {p: {**s, "sd_a": global_a, "sd_p": global_p} for p, s in stats.items()}
    for row in rows:
        s = stats[row["participant_id"]]
        row["arousal_z"] = (float(row["rating_arousal"]) - s["mean_a"]) / s["sd_a"]
        row["eeg_z"] = (-math.log(float(row["eeg_posterior_alpha"])) - s["mean_p"]) / s[
            "sd_p"
        ]
        row["difference"] = row["arousal_z"] - row["eeg_z"]
        row["lag_arousal_z"] = (row["lag_arousal"] - s["mean_a"]) / s["sd_a"]
        row["lag_eeg_z"] = (row["lag_eeg"] - s["mean_p"]) / s["sd_p"]
    result = pd.DataFrame(rows).sort_values(
        ["participant_id", "recorded_day", "event_order"]
    )
    gaps = np.asarray([r["gap_s"] for r in rows if r["recorded_day"] == 1])
    metadata = {
        "people": len(people),
        "transitions": len(rows),
        "by_day": dict(Counter(str(r["recorded_day"]) for r in rows)),
        "minimum_pairs": min_pairs,
        "population_scales": population_scales,
        "reference_gap_s": float(np.median(gaps)),
        "j1_gap_iqr": np.quantile(gaps, [0.25, 0.75]).tolist(),
        "j1_scale_ranges": {
            k: np.quantile(
                [stats[p][k] for p in people], [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            for k in ("sd_a", "sd_p")
        },
        "scope": (
            "Exploratory cohort-2 identifiers, fixed video catalogue; "
            "novel current videos for each person."
        ),
    }
    return result, stats, metadata


def _numeric(row, stats, reference_gap):
    s = stats[str(row["participant_id"])]
    a = (float(row["lag_arousal"]) - s["mean_a"]) / s["sd_a"]
    p = (float(row["lag_eeg"]) - s["mean_p"]) / s["sd_p"]
    a, p = (a if math.isfinite(a) else 0.0), (p if math.isfinite(p) else 0.0)
    gap = row["gap_s"]
    gap = float(gap) if math.isfinite(float(gap)) and gap >= 0 else reference_gap
    g = math.log1p(gap) - math.log1p(reference_gap)
    order, time = float(row["event_order"]) / 50, float(row["onset_s"]) / 1200
    ratio = s["sd_a"] / s["sd_p"]
    return (
        a,
        p,
        order,
        order**2,
        time,
        time**2,
        g,
        g**2,
        a * g,
        p * g,
        ratio,
        a * ratio,
    )


def predictor_frame(grid, features, events, reference_gap):
    """Derive only known context and predecessor values after the grid marker.

    Parameters
    ----------
    grid : pandas.DataFrame
        Physical trial grid, without response columns or an in-grid history buffer.
    features, events : pandas.DataFrame
        Upstream observed-history sources.
    reference_gap : float
        Frozen first-session reference gap in seconds.

    Returns
    -------
    pandas.DataFrame
        Exactly one permitted predictor row per query.
    """
    history = attach_history(grid, features, events)
    stats = j1_statistics(features)
    rows = []
    for row in history.to_dict("records"):
        if row["participant_id"] not in stats:
            raise ValueError("Prediction requires an observed, variable J1 reference.")
        rows.append(
            {
                "participant_id": row["participant_id"],
                "recorded_day": row["recorded_day"],
                **{k: row[k] for k in CATEGORICAL},
                **dict(zip(NUMERIC, _numeric(row, stats, reference_gap), strict=True)),
            }
        )
    return pd.DataFrame(rows, index=grid.index)


def grid_target(grid, features):
    """Read current standardized difference on the response branch only.

    Parameters
    ----------
    grid, features : pandas.DataFrame
        Query IDs and source observations with J1 normalization support.

    Returns
    -------
    numpy.ndarray
        Current standardized arousal minus negative-log-alpha, in grid order.
    """
    stats = j1_statistics(features)
    values = {r["trial_id"]: r for r in features.to_dict("records")}
    target = []
    for key in grid["trial_id"]:
        row = values[key]
        s = stats[str(row["participant_id"])]
        target.append(
            (float(row["rating_arousal"]) - s["mean_a"]) / s["sd_a"]
            - (-math.log(float(row["eeg_posterior_alpha"])) - s["mean_p"]) / s["sd_p"]
        )
    return np.asarray(target)


def calibration_frame(features, events, reference_gap):
    """Materialize the already observed J1 transitions used for personal fitting.

    Parameters
    ----------
    features, events : pandas.DataFrame
        Full source tables; only valid first-session pairs are selected.
    reference_gap : float
        Frozen J1 gap reference.

    Returns
    -------
    pandas.DataFrame
        Permitted predictors plus prior J1 target difference.
    """
    stats = j1_statistics(features)
    table = transition_table(features, events)
    grid = table.loc[
        table["pair_eligible"]
        & (table["recorded_day"] == 1)
        & table["participant_id"].isin(stats),
        ["trial_id", "participant_id", "recorded_day"],
    ]
    result = predictor_frame(grid, features, events, reference_gap)
    result["target_difference"] = grid_target(grid, features)
    return result
