"""Reconstruct real video predecessors before applying rating and EEG exclusions."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import skrub

from habs.data import load_table

KEYS = ("participant_id", "recorded_day", "start_marker", "onset_s")
MAX_GAP = 180.0


def physical_id(row):
    """Identify an event using the audited recording clock.

    Parameters
    ----------
    row : dict
        Participant, recorded day, marker and segment-relative onset.

    Returns
    -------
    str
        Physical-trial identifier used by the existing feature cache.
    """
    value = (
        f"{row['participant_id']}|{int(row['recorded_day'])}|"
        f"{int(row['start_marker'])}|{float(row['onset_s']):.6f}"
    )
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid(row):
    if row is None:
        return False
    alpha = row.get("eeg_posterior_alpha")
    arousal = row.get("rating_arousal")
    return (
        str(row.get("source")) == "CURATED"
        and str(row.get("eeg_valid")).casefold() in {"true", "1"}
        and _finite(alpha)
        and 0 < float(alpha) <= 1
        and _finite(arousal)
        and 1 <= float(arousal) <= 9
    )


def _event_map(events):
    records = events.to_dict("records")
    ordered = sorted(
        records,
        key=lambda r: (
            str(r["participant_id"]),
            int(r["recorded_day"]),
            float(r["onset_s"]),
        ),
    )
    result, previous, exposed = {}, {}, set()
    for row in ordered:
        key = physical_id(row)
        if key in result:
            raise ValueError("Duplicate physical event in the recorded inventory.")
        group = (str(row["participant_id"]), int(row["recorded_day"]))
        before = previous.get(group)
        exposure = (str(row["participant_id"]), str(row["video"]))
        result[key] = {
            "event": row,
            "previous": before,
            "novel": exposure not in exposed,
        }
        exposed.add(exposure)
        previous[group] = row
    return result


def attach_history(grid, features, events):
    """Attach the actual predecessor without dropping rows from the query grid.

    Parameters
    ----------
    grid : pandas.DataFrame
        Requested physical trial IDs and participant/day descriptors.
    features : pandas.DataFrame
        Audited full-trial observations, including earlier ratings and EEG.
    events : pandas.DataFrame
        Complete recorded video inventory, including unusable/unrated events.

    Returns
    -------
    pandas.DataFrame
        One row per query, with lagged values and explicit eligibility reasons.

    Raises
    ------
    ValueError
        If physical IDs are duplicated, inconsistent or absent from the inventory.
    """
    inventory = _event_map(events)
    observations = {}
    for row in features.to_dict("records"):
        key = physical_id(row)
        if key != row["trial_id"] or key in observations:
            raise ValueError("The feature cache has inconsistent physical trial IDs.")
        observations[key] = row
    rows = []
    for query in grid.to_dict("records"):
        key = str(query["trial_id"])
        if key not in inventory:
            raise ValueError("A query trial is absent from the recorded inventory.")
        item = inventory[key]
        event, before = item["event"], item["previous"]
        current = observations.get(key)
        row = {
            "trial_id": key,
            "participant_id": str(event["participant_id"]),
            "recorded_day": int(event["recorded_day"]),
            "video": str(event["video"]),
            "event_order": int(event["event_order"]),
            "onset_s": float(event["onset_s"]),
            "recording": str(event["recording"]),
            "novel_to_person": bool(item["novel"]),
            "previous_id": "",
            "previous_video": "missing",
            "gap_s": np.nan,
            "lag_arousal": np.nan,
            "lag_eeg": np.nan,
            "expected_emotion_source": str(
                (current or {}).get("expected_emotion_source", "missing")
            ),
            "expected_emotion": str(
                (current or {}).get("expected_emotion_label", "missing")
            ),
            "previous_expected_emotion": "missing",
        }
        reason = "ok"
        if before is None:
            reason = "first_video_in_session"
        else:
            previous_id = physical_id(before)
            prior = observations.get(previous_id)
            gap = float(event["onset_s"]) - (
                float(before["onset_s"]) + float(before["event_duration_s"])
            )
            row.update(
                previous_id=previous_id,
                previous_video=str(before["video"]),
                gap_s=gap,
                previous_expected_emotion=str(
                    (prior or {}).get("expected_emotion_label", "missing")
                ),
            )
            if _valid(prior):
                row.update(
                    lag_arousal=float(prior["rating_arousal"]),
                    lag_eeg=-math.log(float(prior["eeg_posterior_alpha"])),
                )
            if int(event["event_order"]) != int(before["event_order"]) + 1:
                reason = "unrecorded_intermediate_video"
            elif str(event["recording"]) != str(before["recording"]):
                reason = "recording_boundary"
            elif float(event["onset_s"]) < float(before["onset_s"]) + 10:
                reason = "overlapping_signal_windows"
            elif not _finite(gap) or gap < 0 or gap > MAX_GAP:
                reason = "invalid_or_long_gap"
            elif not _valid(prior):
                reason = "previous_rating_or_curated_eeg_unusable"
        if reason == "ok" and not _valid(current):
            reason = "current_rating_or_curated_eeg_unusable"
        row["pair_reason"] = reason
        row["pair_eligible"] = reason == "ok"
        rows.append(row)
    return pd.DataFrame(rows, index=grid.index)


def transition_table(features, events):
    """Join historical descriptors and current responses for structural analysis.

    Parameters
    ----------
    features, events : pandas.DataFrame
        Audited feature cache and complete recorded video inventory.

    Returns
    -------
    pandas.DataFrame
        All original cohort-2 query rows with response and pair eligibility.
    """
    selected = features.loc[features["cohort"] == 2].copy()
    descriptors = attach_history(selected[["trial_id"]], features, events)
    outcome = selected[
        ["trial_id", "rating_arousal", "rating_valence", "eeg_posterior_alpha"]
    ]
    return descriptors.merge(outcome, on="trial_id", validate="one_to_one")


def build_transition_graph():
    """Declare a source-bound reconstruction of the cohort-2 transition table.

    Returns
    -------
    skrub.DataOp
        Unfitted transformation bound by features_path and events_path.
    """
    features = skrub.var("features_path").skb.apply_func(load_table, no_cache=True)
    events = skrub.var("events_path").skb.apply_func(load_table, no_cache=True)
    return features.skb.apply_func(transition_table, events)


def feasibility_summary(table, input_dir):
    """Summarize structural transition support without fitting target associations.

    Parameters
    ----------
    table : pandas.DataFrame
        Cohort-2 physical trials with pair eligibility.
    input_dir : str or pathlib.Path
        Private audited input directory used for provenance hashes.

    Returns
    -------
    dict
        Publication-safe counts, gap distributions and context support.
    """
    records = table.to_dict("records")
    usable = [r for r in records if r["pair_eligible"] and r["recorded_day"] in (1, 2)]
    counts = Counter((r["participant_id"], r["recorded_day"]) for r in usable)
    people = {r["participant_id"] for r in usable}
    supported = {
        str(n): sorted(
            pid for pid in people if counts[(pid, 1)] >= n and counts[(pid, 2)] >= n
        )
        for n in (10, 15, 20, 25, 30)
    }
    previous_by_video = defaultdict(set)
    people_by_video = defaultdict(set)
    edges = Counter()
    for r in usable:
        previous_by_video[r["video"]].add(r["previous_video"])
        people_by_video[r["video"]].add(r["participant_id"])
        edges[(r["previous_video"], r["video"])] += 1
    gap = np.asarray([r["gap_s"] for r in usable])
    files = (
        "eda_summary.json",
        "recorded_events.csv.gz",
        "features.csv.gz",
        "feature_manifest.json",
    )
    fingerprints = {
        name: hashlib.sha256((Path(input_dir) / name).read_bytes()).hexdigest()
        for name in files
    }
    return {
        "status": "structural_feasibility_complete",
        "cohort": 2,
        "query_trials": len(records),
        "pair_reasons": dict(Counter(r["pair_reason"] for r in records)),
        "eligible_transitions": len(usable),
        "eligible_by_day": dict(Counter(str(r["recorded_day"]) for r in usable)),
        "paired_people_by_minimum_transitions": {
            n: len(v) for n, v in supported.items()
        },
        "paired_transitions_by_minimum": {
            n: sum(r["participant_id"] in set(v) for r in usable)
            for n, v in supported.items()
        },
        "novel_day2_transitions": sum(
            r["recorded_day"] == 2 and r["novel_to_person"] for r in usable
        ),
        "gap_quantiles_seconds": dict(
            zip(
                ("min", "q10", "q25", "median", "q75", "q90", "max"),
                np.quantile(gap, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist(),
                strict=True,
            )
        )
        if len(gap)
        else {},
        "current_videos": len(previous_by_video),
        "distinct_video_transitions": len(edges),
        "videos_by_predecessor_diversity": dict(
            Counter(str(len(v)) for v in previous_by_video.values())
        ),
        "videos_seen_by_at_least_people": {
            str(n): sum(len(v) >= n for v in people_by_video.values())
            for n in (2, 5, 10)
        },
        "input_sha256": fingerprints,
        "target_associations_fitted": False,
        "note": (
            "Counts concern actual consecutive recorded videos; "
            "no clinical or causal interpretation."
        ),
    }


def verify_inputs(input_dir):
    """Verify the saved feature cache against its audited manifest.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Completed HABS job's private data output directory.

    Returns
    -------
    dict
        Verified source feature manifest.

    Raises
    ------
    ValueError
        If features, native extraction code or annotation provenance differ.
    """
    directory = Path(input_dir)
    manifest = json.loads((directory / "feature_manifest.json").read_text())
    evidence = json.loads((directory / "eda_summary.json").read_text())
    code = Path(__file__).resolve().parents[1]
    algorithm = hashlib.sha256(
        (code / "features.py").read_bytes() + (code / "signals.py").read_bytes()
    ).hexdigest()
    if (
        manifest["features_sha256"]
        != hashlib.sha256((directory / "features.csv.gz").read_bytes()).hexdigest()
    ):
        raise ValueError("The saved features differ from their manifest.")
    if (
        manifest["algorithm_sha256"] != algorithm
        or manifest["labels_sha256"] != evidence["labels_sha256"]
    ):
        raise ValueError("The native-feature code or annotation release differs.")
    return manifest
