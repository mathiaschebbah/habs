"""Check that session recovery uses event identity and never rating agreement."""

import json

import pandas as pd

from alignment import reconcile


def write_session(root, day, marker=100, source="CURATED"):
    """Create metadata for a unique ten-second video event."""
    folder = root / source / "P001"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"P001_day-{day}_task-emotion-video_multimodal.bdf"
    (folder / name).touch()
    obj = {
        "dataset_export": {
            "participant_id": "P001",
            "day": f"day{day}",
            "segments": {
                "emotion-video": {
                    "file": name,
                    "metadata_time_origin_s": 100,
                    "duration_s": 20,
                }
            },
        },
        "events": [
            {
                "trial_type": "emotion_video",
                "marker_id": marker,
                "media_file": "a.mp4",
                "onset_s": 100,
                "duration_s": 10,
            }
        ],
    }
    (folder / f"P001_day-{day}_metadata.json").write_text(json.dumps(obj))


def write_labels(root, days):
    """Write deliberately misdated labels for deterministic matching tests."""
    pd.DataFrame(
        [
            {
                "participant_id": "P001",
                "cohort": 4,
                "day": day,
                "start_marker": 100,
                "video": "a.mp4",
                "rating_arousal": 4.0,
            }
            for day in days
        ]
    ).to_csv(root / "labels_train.csv", index=False)


def test_unique_event_recovers_recorded_day(tmp_path):
    """A mislabeled day can be corrected when event identity is unambiguous."""
    write_session(tmp_path, 2)
    write_session(tmp_path, 2, source="PREPROCESSED")
    write_labels(tmp_path, [1])
    result = reconcile(tmp_path, tmp_path / "out")
    assert result.alignment.tolist() == ["day_corrected"]
    assert result.day.tolist() == [2]
    assert result.label_day.tolist() == [1]
    assert result.signal_source.tolist() == ["CURATED"]


def test_ambiguous_event_is_not_guessed(tmp_path):
    """Two candidate sessions cannot be distinguished by the response value."""
    write_session(tmp_path, 1)
    write_session(tmp_path, 2)
    write_labels(tmp_path, [3])
    result = reconcile(tmp_path, tmp_path / "out")
    assert result.alignment.tolist() == ["ambiguous"]
    assert result.event_uid.isna().all()


def test_two_annotations_cannot_reuse_one_recorded_trial(tmp_path):
    """Reject both rows if a repair would double count the same measurement."""
    write_session(tmp_path, 2)
    write_labels(tmp_path, [1, 2])
    result = reconcile(tmp_path, tmp_path / "out")
    assert result.alignment.tolist() == ["duplicate_annotation", "duplicate_annotation"]
