"""Distinguish exact stimulus repetitions from duplicate annotations and sessions."""

import pandas as pd

from data.repetition_audit import count_repetitions, run
from tests.test_alignment import write_labels, write_session


def test_corrected_session_can_restore_a_real_repetition(tmp_path):
    """Different markers identify two actual sessions despite one wrong label day."""
    write_session(tmp_path, 1, marker=100)
    write_session(tmp_path, 2, marker=200)
    write_labels(tmp_path, [1, 1])
    labels = pd.read_csv(tmp_path / "labels_train.csv")
    labels.loc[1, ["start_marker", "rating_arousal"]] = [200, 5.0]
    labels.to_csv(tmp_path / "labels_train.csv", index=False)
    result = run(tmp_path, tmp_path / "output")
    assert result["aligned_notes"]["pairs_with_one_trial_in_each_session"] == 1
    assert (
        result["aligned_notes"]["people_with_changed_note_on_unique_pairs"]["1.0"] == 1
    )


def test_repeated_rows_in_one_session_are_not_cross_session_pairs():
    """A duplicated stimulus within a day is not a repeated-session observation."""
    frame = pd.DataFrame(
        {
            "participant_id": ["P1"] * 3,
            "video": ["v1"] * 3,
            "day": [1, 1, 2],
            "rating_arousal": [2.0, 3.0, 4.0],
        }
    )
    result = count_repetitions(frame)
    assert result["person_video_groups_across_sessions"] == 1
    assert result["pairs_with_one_trial_in_each_session"] == 0
    assert result["person_video_session_groups_with_multiple_trials"] == 1


def test_stable_notes_remain_in_the_repetition_sample():
    """Repeated trials must not be selected by whether their ratings changed."""
    frame = pd.DataFrame(
        {
            "participant_id": ["P1"] * 2,
            "video": ["v1"] * 2,
            "day": [1, 2],
            "rating_arousal": [4.0, 4.0],
        }
    )
    result = count_repetitions(frame)
    assert result["pairs_with_one_trial_in_each_session"] == 1
    assert result["people_with_changed_note_on_unique_pairs"]["0.0"] == 0
