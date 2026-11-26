"""End-to-end synthetic tests for physiology, forecasting, EB-MAP, Varimax, monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from habs_stress.forecasting import evaluate_causal_forecast, residualize_artefacts
from habs_stress.generalization import (
    coral_align_target,
    eb_map_adapt_ridge,
    passive_monitoring_curve,
    subjective_autonomic_coherence,
)
from habs_stress.physiology import (
    fixed_video_appraisal_contrast,
    rr_adjusted_ecg_effect,
)


def test_rr_adjusted_ecg_removes_rr_confound():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(12):
        for i in range(20):
            stressed = i < 10
            d_rr = (8.0 if stressed else 0.0) + rng.normal(0, 2.0)
            qtc_fake = -0.5 * d_rr + rng.normal(0, 0.5)
            rows.append(
                {
                    "participant_id": f"P{p:02d}",
                    "rating_valence": 2.0 if stressed else 7.0,
                    "rating_arousal": 8.0 if stressed else 3.0,
                    "delta_rr_ms": d_rr,
                    "delta_qtc": qtc_fake,
                }
            )
    df = pd.DataFrame(rows)
    res = rr_adjusted_ecg_effect(df, "delta_qtc", "delta_rr_ms")
    assert res["plain_d"]["mean"] < -1.0
    assert abs(res["rr_adjusted_d"]["mean"]) < 0.3


def test_fixed_video_appraisal_contrast():
    rng = np.random.default_rng(1)
    rows = []
    for v in range(6):
        for p in range(20):
            st = p < 10
            rows.append(
                {
                    "video": f"V{v}",
                    "rating_valence": 3.0 if st else 7.0,
                    "rating_arousal": 7.0 if st else 3.0,
                    "score": (0.8 if st else -0.8) + rng.normal(0, 0.5),
                }
            )
    res = fixed_video_appraisal_contrast(pd.DataFrame(rows), "score", strict_calm=True)
    assert res["n_videos"] == 6 and res["auc_weighted"] > 0.85


def test_causal_forecast_detects_eeg_signal_and_artefact_residualization():
    rng = np.random.default_rng(2)
    n_p, n_t = 18, 12
    n = n_p * n_t
    df = pd.DataFrame(
        {
            "participant_id": np.repeat([f"P{i:02d}" for i in range(n_p)], n_t),
            "theme": np.tile([f"T{j:02d}" for j in range(n_t)], n_p),
            "person_fold": np.repeat(np.arange(n_p) % 3, n_t),
            "theme_fold": np.tile(np.arange(n_t) % 3, n_p),
        }
    )
    x_body = rng.standard_normal((n, 4))
    x_eeg = rng.standard_normal((n, 6))
    y = 0.7 * x_body[:, 0] + 0.5 * x_eeg[:, 0] + rng.normal(0, 0.3, n)
    res = evaluate_causal_forecast(df, x_body, x_eeg, y, eeg_pca_grid=(None,))
    assert res["M2_minus_M0"]["mean"] > 0.05

    art = rng.standard_normal((n, 2))
    eeg_dirty = x_eeg + 2.0 * art[:, [0]]
    clean_tr, clean_ev = residualize_artefacts(eeg_dirty[:150], art[:150], eeg_dirty[150:], art[150:])
    assert abs(np.corrcoef(clean_ev[:, 0], art[150:, 0])[0, 1]) < 0.25


def test_coral_ebmap_varimax_and_passive_monitoring():
    rng = np.random.default_rng(3)
    zs = rng.standard_normal((240, 4))
    zt = rng.standard_normal((240, 4)) * 2.5 + 3.0
    aligned = coral_align_target(zs, zt[:120], zt[120:])
    assert np.allclose(aligned.mean(axis=0), zs.mean(axis=0), atol=0.5)

    w_prior = np.array([1.0, -0.5, 0.0, 0.0])
    z_cal = rng.standard_normal((10, 4))
    y_cal = z_cal @ w_prior + rng.normal(0, 0.2, 10)
    preds = eb_map_adapt_ridge(z_cal, y_cal, w_prior, zs[:10], prior_lambda=1000.0)
    assert np.corrcoef(preds, zs[:10] @ w_prior)[0, 1] > 0.99

    df = pd.DataFrame(
        {
            "participant_id": np.repeat([f"P{i:02d}" for i in range(10)], 20),
            "v1": rng.standard_normal(200),
            "v2": rng.standard_normal(200),
            "v3": rng.standard_normal(200),
            "v4": rng.standard_normal(200),
        }
    )
    coh = subjective_autonomic_coherence(df, ["v1", "v2", "v3", "v4"], n_factors=2)
    assert len(coh["explained_variance_ratio"]) == 2

    pos = np.tile([True] * 10 + [False] * 10, 10)
    scores = np.where(pos, 0.4, -0.4) + rng.normal(0, 1.0, 200)
    curve = passive_monitoring_curve(df, scores, pos, ~pos, k_grid=(1, 5), n_draws=60)
    assert curve[5]["auc"]["mean"] > curve[1]["auc"]["mean"]
