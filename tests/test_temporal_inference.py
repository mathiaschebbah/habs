"""Validate uncertainty calculations on known simulated effects and fixed designs."""

import numpy as np

from habs.temporal.inference import (
    day_geometry,
    estimate_day,
    fit_variance_components,
)
from habs.temporal.preprocess import predictor_frame, prepare_data


def test_projection_removes_nuisance_and_covariance_is_positive(temporal_tables):
    """Changing fixed personal levels must not alter adjusted history slopes."""
    features, events = temporal_tables()
    table, stats, metadata = prepare_data(features, events)
    people = sorted(stats)
    day = table.loc[table["recorded_day"] == 1]
    geometry = day_geometry(day, people, stats, metadata["reference_gap_s"])
    estimate = estimate_day(geometry)
    changed = day.copy()
    changed["arousal_z"] += 15.0
    changed["eeg_z"] -= 7.0
    other = estimate_day(
        day_geometry(changed, people, stats, metadata["reference_gap_s"])
    )
    assert np.max(np.abs(estimate["delta"] - other["delta"])) < 1e-9
    covariance = estimate["sampling_covariance"]
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.max(np.abs(covariance - covariance.T)) < 1e-10
    assert geometry.rank == 2 * len(people)


def test_reml_recovers_simulated_covariance_with_shared_scale_control():
    """A known nuisance scaling pattern must not masquerade as random stability."""
    rng = np.random.default_rng(928)
    n = 90
    ratio = rng.uniform(1, 4, size=n)
    standardized = (ratio - ratio.mean()) / ratio.std()
    omega = np.asarray([[0.12, 0.06], [0.06, 0.15]])
    sampling = np.eye(2 * n) * 0.002
    counts = np.full((2, n), 40.0)
    u = rng.multivariate_normal([0, 0], omega, size=n)
    values = np.r_[
        0.2 + 0.4 * standardized + u[:, 0], -0.1 + 0.2 * standardized + u[:, 1]
    ]
    values += rng.normal(scale=np.sqrt(0.002), size=2 * n)
    result = fit_variance_components(values, sampling, counts, scale_ratio=ratio)
    assert result["optimizer_success"]
    assert abs(result["covariance"] - 0.06) < 0.04
    assert np.max(np.abs(result["sd"] - np.sqrt(np.diag(omega)))) < 0.12
    zero = fit_variance_components(np.zeros(2 * n), sampling, counts, scale_ratio=ratio)
    assert zero["near_zero_variance"]


def test_current_response_cannot_enter_predictor_frame(temporal_tables):
    """A cold J2 query must depend only on context and genuinely earlier responses."""
    features, events = temporal_tables()
    table, _, metadata = prepare_data(features, events)
    grid = table.loc[
        table["recorded_day"] == 2, ["trial_id", "participant_id", "recorded_day"]
    ].iloc[[0]]
    before = predictor_frame(grid, features, events, metadata["reference_gap_s"])
    changed = features.copy()
    changed.loc[changed["trial_id"].isin(grid["trial_id"]), "rating_arousal"] = 9.0
    changed.loc[changed["trial_id"].isin(grid["trial_id"]), "eeg_posterior_alpha"] = 0.8
    after = predictor_frame(grid, changed, events, metadata["reference_gap_s"])
    assert before.to_dict("records") == after.to_dict("records")
    assert not set(before) & {"arousal_z", "eeg_z", "difference", "rating_arousal"}
