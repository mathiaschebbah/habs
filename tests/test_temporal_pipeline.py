"""Verify history routing and saved evaluation on explicitly simulated unit tables."""

import importlib

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from habs.temporal.model import TemporalForecaster
from habs.temporal.pipeline import build_learner
from habs.temporal.preprocess import (
    calibration_frame,
    predictor_frame,
    prepare_data,
)


@pytest.fixture
def temporal_bindings(tmp_path, temporal_tables):
    """Bind synthetic native-shaped sources and separate future query files.

    Returns
    -------
    tuple
        Source bindings, future grid and J1 median gap for unit verification.
    """
    features, events = temporal_tables()
    table, _, metadata = prepare_data(features, events)
    grid = table.loc[
        table["recorded_day"] == 2, ["trial_id", "participant_id", "recorded_day"]
    ]
    people = sorted(set(grid["participant_id"]))
    train = grid.loc[grid["participant_id"].isin(people[:-2])]
    future = grid.loc[grid["participant_id"].isin(people[-2:])].iloc[:17]
    paths = {
        name: tmp_path / (name + ".csv")
        for name in ("features", "events", "grid", "future")
    }
    for name, frame in (
        ("features", features),
        ("events", events),
        ("grid", train),
        ("future", future),
    ):
        frame.to_csv(paths[name], index=False)
    return paths, future, metadata["reference_gap_s"]


@pytest.mark.parametrize("mode", ["full", "ablation", "refit"])
def test_graph_replays_a_cold_query_file(temporal_bindings, mode):
    """Keep every future row while reading predecessor support upstream."""
    paths, future, gap = temporal_bindings
    learner = build_learner(splitter=GroupKFold(2), reference_gap=gap, mode=mode)
    bindings = {
        "grid_path": paths["grid"],
        "features_path": paths["features"],
        "events_path": paths["events"],
    }
    learner.fit(bindings)
    prediction = learner.predict({**bindings, "grid_path": paths["future"]})
    assert len(prediction) == len(future)
    assert np.isfinite(prediction).all()


def test_population_fit_excludes_other_peoples_j1(temporal_tables):
    """Allowed test-person calibration must not enter population transforms."""
    features, events = temporal_tables()
    table, _, metadata = prepare_data(features, events)
    grid = table.loc[
        table["recorded_day"] == 2, ["trial_id", "participant_id", "recorded_day"]
    ]
    inputs = predictor_frame(grid, features, events, metadata["reference_gap_s"])
    past = calibration_frame(features, events, metadata["reference_gap_s"])
    people = sorted(set(grid["participant_id"]))
    mask = inputs["participant_id"].isin(people[:-1])
    target = table.loc[table["recorded_day"] == 2, "difference"].to_numpy()
    a = TemporalForecaster().fit(inputs.loc[mask], target[mask], calibration=past)
    changed = past.copy()
    outside = ~changed["participant_id"].isin(people[:-1])
    changed.loc[outside, "target_difference"] += 100
    changed.loc[outside, "lag_arousal_z"] += 100
    b = TemporalForecaster().fit(inputs.loc[mask], target[mask], calibration=changed)
    assert np.max(np.abs(a.regressor_.coef_ - b.regressor_.coef_)) < 1e-12
    bad = past.copy()
    bad.loc[bad.index[0], "recorded_day"] = 2
    with pytest.raises(ValueError, match="first-session-only"):
        TemporalForecaster().fit(inputs.loc[mask], target[mask], calibration=bad)


def test_skore_grouped_report_and_default_metrics(temporal_bindings):
    """Exercise the sole producer on unit data, retaining cached fold predictions."""
    paths, _, gap = temporal_bindings
    experiment = importlib.import_module("experiments.02_emotional_carryover")
    bindings = {
        "grid_path": paths["grid"],
        "features_path": paths["features"],
        "events_path": paths["events"],
    }
    report = experiment._evaluate_forecast(bindings, gap, "full", n_splits=2)
    grid = importlib.import_module("habs.data").load_table(paths["grid"])
    prediction = experiment._oof_predictions(report, len(grid))
    assert prediction.shape == (len(grid),)
    for child in report.reports_:
        assert np.isfinite(float(child.metrics.get("participant_macro_mae")))
        assert np.isfinite(float(child.metrics.r2()))
        assert np.isfinite(float(child.metrics.get("default_score")))


def test_saved_cv_requires_bindings_but_never_refits(
    temporal_bindings, tmp_path, monkeypatch
):
    """Restore required CV source bindings without re-fitting a cached estimator."""
    paths, _, gap = temporal_bindings
    experiment = importlib.import_module("experiments.02_emotional_carryover")
    bindings = {
        "grid_path": paths["grid"],
        "features_path": paths["features"],
        "events_path": paths["events"],
    }
    report = experiment._evaluate_forecast(bindings, gap, "ablation", n_splits=2)
    grid = importlib.import_module("habs.data").load_table(paths["grid"])
    prediction = experiment._oof_predictions(report, len(grid))
    for child in report.reports_:
        child.metrics.summarize().frame()
        child.get_predictions(data_source="train")
    project = experiment._persist_forecast(tmp_path / "reports", "unit_cv_only", report)
    frame = project.summarize().frame().reset_index()
    column = "id" if "id" in frame else "report_id"
    report_id = str(frame.iloc[0][column])
    for path in paths.values():
        path.rename(path.with_name(path.name + ".unavailable"))
    # skore 0.29.1 reconstructs the source-bound CV split while opening a report.
    with pytest.raises(FileNotFoundError):
        project.get(report_id)
    for path in paths.values():
        path.with_name(path.name + ".unavailable").rename(path)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Opening cached predictions must not fit a forecaster.")

    monkeypatch.setattr(TemporalForecaster, "fit", forbid_fit)
    restored = project.get(report_id)
    actual = experiment._oof_predictions(restored, len(grid))
    assert np.max(np.abs(actual - prediction)) < 1e-12
    for child in restored.reports_:
        assert np.isfinite(float(child.metrics.get("participant_macro_mae")))
        assert np.isfinite(float(child.metrics.get("default_score")))
