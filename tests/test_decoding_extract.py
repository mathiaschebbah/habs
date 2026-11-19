"""Run the feature extractor on the real P001 sample recording."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from decoding.extract import COV, _erp, _sympathetic_only, extract

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
    rr = inside["hrv_rr_mean"]
    assert rr.between(0.4, 1.5).all() and inside["breath_rate"].between(6, 40).all()


def test_erp_with_non_integer_rate():
    """A 999.2 Hz recording yields five finite evoked bins."""
    fs = 999.2
    x = np.r_[np.zeros(round(5 * fs)), np.ones(round(5 * fs))]
    erp = _erp(x, fs, np.array([5.0]))
    assert erp.shape == (1, 5) and np.allclose(erp, 1.0, atol=0.01)


def test_sympathetic_only_matches_full_extraction(tmp_path):
    """The three-channel pass reproduces the full pass's body features."""
    _as_video_segment(tmp_path)
    trial = {"participant_id": "P001", "recorded_day": 1, "start_marker": 1}
    recording = f"CURATED/P001/{STEM.format('emotion-video')}.bdf"
    trials = [{**trial, "onset_s": 8.0, "recording": recording}]
    fast = _sympathetic_only((str(tmp_path), trials))
    full = extract(
        pd.DataFrame(trials).assign(
            **{c: 0 for c in ("cohort", "event_order", "video", "video_theme")},
            **{c: 0 for c in ("default_emotion_label", "rating_valence")},
            **{c: 0 for c in ("rating_arousal", "rating_dominance", "age", "gender")},
        ),
        tmp_path,
        workers=1,
    )
    columns = [c for c in fast if c != "trial_id"]
    assert np.allclose(
        fast[columns].to_numpy(), full[columns].to_numpy(), equal_nan=True
    )


def test_timecourse_on_sample(tmp_path):
    """Second-by-second bins are finite inside the recording, NaN outside."""
    from decoding.timecourse_extract import BINS, _recording

    _as_video_segment(tmp_path)
    recording = f"CURATED/P001/{STEM.format('emotion-video')}.bdf"
    trials = [
        {
            "participant_id": "P001",
            "recorded_day": 1,
            "start_marker": k,
            "onset_s": onset,
            "recording": recording,
        }
        for k, onset in enumerate((20.0, 50.0, 93.0))
    ]
    table, windows = _recording((str(tmp_path), trials))
    inside = table.iloc[:2]
    for signal in ("hr", "sc", "resp_amp", "eeg_frontal_alpha"):
        values = inside[[f"tc_{signal}_{k}" for k in range(BINS)]]
        assert np.isfinite(values.to_numpy()).all(), signal
    assert inside.filter(like="tc_hr_").stack().between(40, 150).all()
    assert table.iloc[2].filter(like="tc_eeg_").isna().any()  # runs past the end
    assert {"rmssd_10", "rmssd_60", "hr_5"} <= set(windows.columns)
