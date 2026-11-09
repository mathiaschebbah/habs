"""Check that unusable videos cannot manufacture sequential-emotion pairs."""

import pandas as pd
import pytest

from habs.temporal.data import (
    attach_history,
    build_transition_graph,
    physical_id,
    transition_table,
)


def _tables():
    events, features = [], []
    for day in (1, 2):
        for order in range(4):
            row = {
                "participant_id": "person",
                "recorded_day": day,
                "start_marker": 100 + order,
                "onset_s": float(20 * order),
                "event_duration_s": 10.0,
                "video": f"video{day}_{order}",
                "event_order": order,
                "recording": f"CURATED/person/day{day}.bdf",
                "source": "CURATED",
            }
            events.append(row)
            features.append(
                {
                    **row,
                    "trial_id": physical_id(row),
                    "cohort": 2,
                    "rating_arousal": float(2 + order),
                    "rating_valence": 5.0,
                    "eeg_valid": True,
                    "eeg_posterior_alpha": 0.2 + 0.01 * order,
                    "expected_emotion_source": "default",
                }
            )
    return pd.DataFrame(features), pd.DataFrame(events)


def test_missing_middle_trial_does_not_create_adjacency():
    """Keep the actual missing predecessor rather than the last retained row."""
    features, events = _tables()
    features = features.drop(index=2)
    result = transition_table(features, events)
    last = result.loc[(result["recorded_day"] == 1) & (result["event_order"] == 3)]
    assert last.iloc[0]["pair_reason"] == "previous_rating_or_curated_eeg_unusable"
    assert not last.iloc[0]["pair_eligible"]
    assert int(result["pair_eligible"].sum()) == 4


def test_inventory_order_gap_and_day_boundary_are_not_bridged():
    """Detect unrecorded intermediate videos and the first trial of each day."""
    features, events = _tables()
    events = events.drop(index=2)
    features = features.drop(index=2)
    result = transition_table(features, events)
    assert set(result.loc[result["event_order"] == 0, "pair_reason"]) == {
        "first_video_in_session"
    }
    row = result.loc[(result["recorded_day"] == 1) & (result["event_order"] == 3)]
    assert row.iloc[0]["pair_reason"] == "unrecorded_intermediate_video"


def test_current_and_future_responses_are_not_history_features():
    """Changing the queried response cannot change its predictor history."""
    features, events = _tables()
    grid = features.iloc[[1]][["trial_id"]]
    before = attach_history(grid, features, events)
    changed = features.copy()
    changed.loc[changed.index >= 1, "rating_arousal"] = 9.0
    changed.loc[changed.index >= 1, "eeg_posterior_alpha"] = 0.7
    after = attach_history(grid, changed, events)
    assert before.iloc[0]["lag_arousal"] == after.iloc[0]["lag_arousal"]
    assert before.iloc[0]["lag_eeg"] == after.iloc[0]["lag_eeg"]
    assert "rating_arousal" not in before
    assert "eeg_posterior_alpha" not in before


def test_curated_and_timing_requirements_are_enforced():
    """Reject a preprocessed predecessor, overlap and excessive interruption."""
    features, events = _tables()
    features.loc[0, "source"] = "PREPROCESSED"
    events.loc[2, "onset_s"] = 25.0
    features.loc[2, "onset_s"] = 25.0
    features.loc[2, "trial_id"] = physical_id(features.iloc[2].to_dict())
    events.loc[7, "onset_s"] = 1000.0
    features.loc[7, "onset_s"] = 1000.0
    features.loc[7, "trial_id"] = physical_id(features.iloc[7].to_dict())
    result = transition_table(features, events)
    reasons = set(result["pair_reason"])
    assert "previous_rating_or_curated_eeg_unusable" in reasons
    assert "overlapping_signal_windows" in reasons
    assert "invalid_or_long_gap" in reasons


def test_duplicate_inventory_rejected_and_source_graph_replays(tmp_path):
    """Replay source-bound reconstruction and reject ambiguous physical events."""
    features, events = _tables()
    with pytest.raises(ValueError, match="Duplicate physical event"):
        transition_table(features, pd.concat([events, events.iloc[[0]]]))
    feature_path, event_path = tmp_path / "features.csv", tmp_path / "events.csv"
    features.to_csv(feature_path, index=False)
    events.to_csv(event_path, index=False)
    result = build_transition_graph().skb.eval(
        {"features_path": feature_path, "events_path": event_path}
    )
    assert len(result) == 8
    assert int(result["pair_eligible"].sum()) == 6
