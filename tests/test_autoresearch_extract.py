"""Run the autoresearch feature extractor on the real P001 sample recording."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from autoresearch.extract import COV, extract

SAMPLE = Path(__file__).resolve().parents[1] / "sample"
STEM = "P001_day-1_task-{}_multimodal"


def _as_video_segment(root):
    """Expose the sample resting-state file under the emotion-video name."""
    for source in ("CURATED", "PREPROCESSED"):
        folder = root / source / "P001"
        folder.mkdir(parents=True)
        for suffix in (".bdf", ".record_map"):
            shutil.copy(
                SAMPLE / source / "P001" / (STEM.format("resting-state") + suffix),
                folder / (STEM.format("emotion-video") + suffix),
            )
        meta = json.loads(
            (SAMPLE / source / "P001" / "P001_day-1_metadata.json").read_text()
        )
        segments = meta["dataset_export"]["segments"]
        segments["emotion-video"] = segments["resting-state"]
        (folder / "P001_day-1_metadata.json").write_text(json.dumps(meta))


def test_extract_on_sample(tmp_path):
    """Extract three windows, one of which leaves the recording."""
    _as_video_segment(tmp_path)
    trials = pd.DataFrame(
        {
            "participant_id": "P001",
            "recorded_day": 1,
            "start_marker": [1, 2, 3],
            "onset_s": [5.0, 40.0, 90.0],
            "recording": f"CURATED/P001/{STEM.format('emotion-video')}.bdf",
        }
    )
    for column in ("cohort", "event_order", "video", "video_theme"):
        trials[column] = 0
    for column in ("default_emotion_label", "rating_valence", "rating_arousal"):
        trials[column] = 0
    trials[["rating_dominance", "age", "gender"]] = 0
    table = extract(trials, tmp_path, workers=1)
    assert len(table) == 3
    assert (table["eeg_source"] == "PREPROCESSED").all()
    assert table.filter(like="bp_vid_").shape[1] == 64 * 5 - 2 * 5  # T9/T10 absent
    assert table.filter(like="cov_").shape[1] == len(COV) * (len(COV) + 1) // 2
    inside = table.iloc[:2]
    assert np.isfinite(inside.filter(like="bp_vid_Fz")).all().all()
    assert np.isfinite(inside.filter(like="cov_")).all().all()
    assert table.iloc[2].filter(like="bp_vid_").isna().all()  # window leaves the file
