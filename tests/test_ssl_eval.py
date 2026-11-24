"""Unit tests for Study-08 evaluation and representation-geometry probes."""

import numpy as np
import pandas as pd

from decoding import common, ssl_eval


def _synthetic_frame(n_people=30, n_trials_per_person=18):
    rng = np.random.default_rng(42)
    rows = []
    for p in range(n_people):
        for t in range(n_trials_per_person):
            v = float(rng.uniform(1.0, 9.0))
            a = float(rng.uniform(1.0, 9.0))
            rows.append(
                {
                    "trial_id": f"P{p:02d}-{t:02d}",
                    "participant_id": f"P{p:02d}",
                    "recorded_day": 1 if t < n_trials_per_person // 2 else 2,
                    "video": f"theme{t % 10}_{t}.mp4",
                    "video_theme": f"theme{t % 10}",
                    "rating_valence": v,
                    "rating_arousal": a,
                }
            )
    return common.assign_splits(pd.DataFrame(rows), "test-study08-salt")


def test_probe_representation_and_diagnostics():
    """Verify stress/affect probing, subject-ID probe, and temporal dynamics probes."""
    rng = np.random.default_rng(123)
    frame = _synthetic_frame()
    n = len(frame)
    d = 16

    # Inject a real signal correlated with arousal - valence + person identity offset
    pids = frame["participant_id"].astype("category").cat.codes.to_numpy()
    person_fp = rng.normal(0, 2.0, (pids.max() + 1, d))[pids]
    aff_sig = (frame["rating_arousal"] - frame["rating_valence"]).to_numpy(float)[
        :, None
    ] * rng.normal(0, 0.35, (1, d))
    z_ref = person_fp + rng.normal(0, 0.5, (n, d))
    z_early = (
        person_fp + 0.5 * aff_sig + np.array([1.0] * d) + rng.normal(0, 0.5, (n, d))
    )
    z_late = (
        person_fp + 1.0 * aff_sig + np.array([2.0] * d) + rng.normal(0, 0.5, (n, d))
    )
    x_delta = np.hstack([z_early - z_ref, z_late - z_ref])

    metrics, preds = ssl_eval.probe_representation_all_regimes(
        frame, x_delta, person_center=True, eval_sealed=True, draws=40
    )
    assert metrics["dev"]["stress_ridge_va"]["stress"]["auc"] > 0.70
    assert metrics["dev"]["affect"]["mean_rho"] > 0.30
    assert "test_new_both" in metrics
    assert "dev_ridge_va" in preds

    id_raw = ssl_eval.probe_subject_identity(frame, z_late)
    id_delta = ssl_eval.probe_subject_identity(frame, z_late - z_ref)
    assert id_raw["subject_variance_r2"] > id_delta["subject_variance_r2"]
    assert id_raw["within_session_id"]["top1"] > id_delta["within_session_id"]["top1"]

    dyn = ssl_eval.probe_temporal_dynamics(frame, z_ref, z_early, z_late, draws=40)
    assert dyn["pre_vs_video"]["within_person_auc"] > 0.80
    assert dyn["early_vs_late"]["within_person_auc"] > 0.80
    assert dyn["arrow_of_time"]["within_person_auc"] > 0.80
    assert dyn["effective_rank_vid"] > 1.0

    # Cross-modal forecast probe
    body = {"hr_late": (z_early[:, 0] - z_ref[:, 0]) + rng.normal(0, 0.3, n)}
    cm = ssl_eval.probe_cross_modal_forecast(frame, z_early - z_ref, body, draws=40)
    assert cm["hr_late"]["within_person_r"] > 0.40
