"""Count actual person-video repetitions after reconciling recording sessions."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from alignment import event_index, reconcile


def count_repetitions(frame):
    """Separate cross-session repetitions from repeated rows within one session."""
    counts = frame.groupby(["participant_id", "video", "day"]).size()
    sessions = frame.groupby("participant_id").day.nunique()
    repeated = []
    unique_pairs = []
    changed = {0.0: set(), 0.5: set(), 1.0: set()}
    for (person, video), rows in frame.groupby(["participant_id", "video"]):
        days = sorted(rows.day.unique())
        if len(days) < 2:
            continue
        repeated.append((person, video, len(days)))
        for first, second in combinations(days, 2):
            left, right = rows.loc[rows.day == first], rows.loc[rows.day == second]
            if len(left) != 1 or len(right) != 1:
                continue
            unique_pairs.append((person, video, first, second))
            if "rating_arousal" in rows:
                difference = abs(
                    float(right.rating_arousal.iloc[0] - left.rating_arousal.iloc[0])
                )
                for threshold in changed:
                    if difference > 1e-6 and difference >= threshold:
                        changed[threshold].add(person)
    result = {
        "trials": len(frame),
        "people": int(frame.participant_id.nunique()),
        "people_with_multiple_sessions": int((sessions >= 2).sum()),
        "people_with_cross_session_repetitions": len({row[0] for row in repeated}),
        "person_video_groups_across_sessions": len(repeated),
        "session_pairs": sum(n * (n - 1) // 2 for _, _, n in repeated),
        "pairs_with_one_trial_in_each_session": len(unique_pairs),
        "person_video_session_groups_with_multiple_trials": int((counts > 1).sum()),
    }
    if "rating_arousal" in frame:
        result["people_with_changed_note_on_unique_pairs"] = {
            str(threshold): len(people) for threshold, people in changed.items()
        }
    return result


def run(data_dir, out_dir):
    """Audit available events and numeric notes without fitting predictive models."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    frame = reconcile(data_dir, out_dir)
    linked = frame.loc[
        frame.alignment.isin(["exact", "day_corrected", "day_recovered"])
    ]
    if (frame.groupby("participant_id").cohort.nunique() != 1).any():
        raise ValueError("An identifier belongs to multiple acquisition cohorts")
    cohorts = frame.drop_duplicates("participant_id").set_index("participant_id").cohort
    events = [
        {
            "participant_id": event["participant_id"],
            "video": event["video"],
            "day": event["day"],
            "cohort": int(cohorts[event["participant_id"]]),
        }
        for candidates in event_index(data_dir).values()
        for event in candidates
        if event["participant_id"] in cohorts.index
        and abs(event["duration"] - 10) <= 0.001
        and any(info["bounded"] for info in event["variants"].values())
    ]
    recorded = pd.DataFrame(
        events, columns=["participant_id", "video", "day", "cohort"]
    )
    result = {
        "labels_sha256": hashlib.sha256(
            (data_dir / "labels_train.csv").read_bytes()
        ).hexdigest(),
        "identity": (
            "Normalized video filename and participant ID; "
            "distinct recorded session numbers"
        ),
        "numeric_rating_identifiers": int(frame.participant_id.nunique()),
        "aligned_notes": count_repetitions(linked),
        "recorded_events_for_numeric_rating_identifiers": count_repetitions(recorded),
        "cohorts": {},
    }
    for cohort in sorted(frame.cohort.unique()):
        result["cohorts"][str(int(cohort))] = {
            "aligned_notes": count_repetitions(linked.loc[linked.cohort == cohort]),
            "recorded_events": count_repetitions(
                recorded.loc[recorded.cohort == cohort]
            ),
        }
    (out_dir / "repetitions.json").write_text(json.dumps(result, indent=2) + "\n")
    print("HABS_REPETITIONS=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result
