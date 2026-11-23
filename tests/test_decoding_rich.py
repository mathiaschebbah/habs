"""Check Study-07 enriched extraction on the real P001 sample recording."""

import json
import shutil
from pathlib import Path

import numpy as np

from decoding.rich_extract import _recording, delineate_ecg

SAMPLE = Path(__file__).resolve().parents[1] / "sample"
STEM = "P001_day-1_task-{}_multimodal"


def _as_video_segment(root):
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


def test_rich_extract_on_sample(tmp_path):
    """4-phase ECG P-QRS-T, EEG spatial homogeneity, and raw CURATED HEP run."""
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
        for k, onset in enumerate((15.0, 45.0, 93.0))
    ]
    frame = _recording((str(tmp_path), trials))
    assert len(frame) == 3
    inside = frame.iloc[:2]

    # ECG morphology inside physiological human ranges
    assert (inside["ecg_polarity"] == -1.0).all()
    assert inside["ecg_pr_pk_vid"].between(90, 200).all()
    assert inside["ecg_pr_on_vid"].between(100, 220).all()
    assert inside["ecg_pr_seg_vid"].between(20, 120).all()
    assert inside["ecg_qrs_vid"].between(45, 120).all()
    assert inside["ecg_qt_vid"].between(280, 460).all()
    assert inside["ecg_qtc_baz_vid"].between(320, 480).all()
    assert inside["ecg_rt_pk_vid"].between(180, 340).all()
    assert inside["ecg_t_amp_vid"].between(0.1, 1.5).all()
    assert inside["ecg_pr_pk_early"].notna().all()
    assert inside["ecg_pr_pk_late"].notna().all()

    # EEG 4-phase, spatial homogeneity, asymmetries, temporal variation
    assert inside["eeg_ep_early_frontal_alpha"].notna().all()
    assert inside["eeg_sp_std_early_alpha"].gt(0).all()
    assert inside["eeg_sp_ent_early_alpha"].between(0.5, 1.0).all()
    assert inside["eeg_faa_early_alpha"].notna().all()
    assert inside["eeg_tvar_early_alpha"].gt(0).all()
    assert inside["eeg_tslope_early_alpha"].notna().all()

    # Raw CURATED HEP finite and reasonable beat count
    assert inside["hep_raw_video_frontal_10"].notna().all()
    assert inside["hep_raw_video_central_10"].notna().all()
    assert inside["hep_raw200_video_frontal_10"].notna().all()
    assert (inside["hep_raw_beats_video"] >= 5).all()

    # Boundary trial running past the recording end yields NaN for late/post phases
    assert (
        frame.iloc[2]["eeg_ep_late_frontal_alpha"]
        != frame.iloc[2]["eeg_ep_late_frontal_alpha"]
    )


def test_delineate_handles_missing_or_flat_ecg():
    """Missing, short, or flat ECG returns empty arrays without raising."""
    for bad in (None, np.zeros(100), np.zeros(10_000), np.full(10_000, np.nan)):
        out = delineate_ecg(bad, 1000.0)
        assert len(out["times"]) == 0
        assert np.isnan(out["good_qrs_frac"])
