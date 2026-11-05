"""Verify trial matching and the score used for individual fluctuations."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from data.inventory import extent, inventory, number
from data.planning import centered_mse


class PreparationTests(unittest.TestCase):
    """Exercise ambiguous events, missing labels and constant corrections."""

    def test_even_sample_median_uses_both_central_values(self):
        """A summary of an even sample must not select the upper middle value."""
        self.assertEqual(extent([1, 9])["median"], 5)

    def test_constant_person_correction_cannot_improve_centered_score(self):
        """A stable personal offset must not count as within-person information."""
        observed = [2.0, 8.0, 4.0, 7.0]
        predicted = [3.0, 5.0, 6.0, 6.0]
        expected = centered_mse(observed, predicted)
        for offset in (-100.0, -1.5, 0.0, 4.0, 100.0):
            shifted = [value + offset for value in predicted]
            self.assertAlmostEqual(centered_mse(observed, shifted), expected)
        self.assertEqual(centered_mse(observed, observed), 0)

    def test_undefined_scores_are_rejected(self):
        """Missing ratings and mismatched trial lists cannot produce a score."""
        for observed, predicted in [
            ([1], [2]),
            ([1, 2], [3]),
            ([1, float("nan")], [2, 3]),
        ]:
            with self.assertRaises(ValueError):
                centered_mse(observed, predicted)
        self.assertIsNone(number("TESTSUBJECT"))
        self.assertIsNone(number("nan"))

    def test_join_requires_day_marker_video_and_complete_window(self):
        """Reject absent, ambiguous, mismatched and truncated trial events."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "CURATED" / "P001"
            folder.mkdir(parents=True)
            bdf = "P001_day-1_task-emotion-video_multimodal.bdf"
            (folder / bdf).touch()
            events = [
                {
                    "marker_id": code,
                    "onset_s": onset,
                    "duration_s": 10,
                    "trial_type": "emotion_video",
                    "media_file": video,
                    "arousal_score": 5,
                    "rating_tool": "default",
                }
                for code, onset, video in [
                    (1, 100, "a.mp4"),
                    (2, 110, "other.mp4"),
                    (3, 125, "a.mp4"),
                    (4, 100, "a.mp4"),
                    (4, 110, "a.mp4"),
                ]
            ]
            obj = {
                "dataset_export": {
                    "participant_id": "P001",
                    "day": "day1",
                    "segments": {
                        "emotion-video": {
                            "file": bdf,
                            "metadata_time_origin_s": 100,
                            "duration_s": 30,
                        }
                    },
                },
                "events": events,
            }
            (folder / "P001_day-1_metadata.json").write_text(json.dumps(obj))
            fields = [
                "cohort",
                "participant_id",
                "day",
                "start_marker",
                "video",
                "rating_arousal",
                "flagged_outliers",
                "final_emotion_label",
            ]
            with (root / "labels_train.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for marker in range(1, 7):
                    writer.writerow(
                        {
                            "cohort": "2",
                            "participant_id": "P001",
                            "day": "1",
                            "start_marker": marker,
                            "video": "a.mp4",
                            "rating_arousal": "TESTSUBJECT" if marker == 6 else 5,
                            "flagged_outliers": 0,
                            "final_emotion_label": "joy",
                        }
                    )
            result = inventory(root, root / "out")
            self.assertEqual(result["masked_rows"], 1)
            self.assertEqual(result["arousal_rows"], 5)
            self.assertEqual(
                result["joins"],
                {
                    "matched": 1,
                    "video_mismatch": 1,
                    "missing_bdf_or_outside_segment": 1,
                    "ambiguous_event": 1,
                    "no_event": 1,
                },
            )
            self.assertEqual(result["cohorts"]["2"]["both_days_people"], 0)
