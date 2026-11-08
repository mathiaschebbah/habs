"""Unit fixtures test binding and leakage behavior, not scientific performance."""

import importlib

import numpy as np
import pandas as pd
import pytest

from habs.evaluate import SessionGroupedSplit, participant_macro_mae
from habs.features import PHYSIOLOGY
from habs.pipeline import build_learner


@pytest.fixture
def bound_tables(tmp_path):
    """Provide small deterministic unit tables with distinct future-grid files."""
    rows = []
    for person in range(8):
        for day in (1, 2):
            for trial in range(6):
                rows.append(
                    {
                        "participant_id": f"unit{person}",
                        "recorded_day": day,
                        "evaluation_eligible": day == 2,
                        "event_order": trial,
                        "video": f"video{trial}",
                        "expected_emotion_label": "neutral",
                        "expected_emotion_source": "default",
                        "rating_valence": 2 + (person + trial) % 6,
                        "rating_arousal": 2 + trial,
                        **{
                            name: 0.1 * (i + 1) + person * 0.01 + trial * 0.005
                            for i, name in enumerate(PHYSIOLOGY)
                        },
                    }
                )
    table = pd.DataFrame(rows)
    train = table.loc[~table["participant_id"].isin(["unit6", "unit7"])]
    history = table.loc[table["recorded_day"] == 1]
    future = table.loc[
        (table["recorded_day"] == 2) & table["participant_id"].isin(["unit6", "unit7"])
    ]
    paths = {name: tmp_path / f"{name}.csv" for name in ("train", "history", "future")}
    for name, frame in (("train", train), ("history", history), ("future", future)):
        frame.to_csv(paths[name], index=False)
    return paths, future


@pytest.mark.parametrize("modality", ["context", "missing", "all", "no_eda"])
def test_new_grid_needs_no_current_response(bound_tables, modality):
    """Keep every future row and never evaluate its response branch at predict."""
    paths, future = bound_tables
    learner = build_learner(
        modality=modality,
        splitter=SessionGroupedSplit([f"unit{i}" for i in range(6)], 3),
    )
    learner.fit({"trials_path": paths["train"], "calibration_path": paths["history"]})
    bindings = {"trials_path": paths["future"], "calibration_path": paths["history"]}
    predicted = learner.predict(bindings)
    future.drop(columns=["rating_valence", "rating_arousal"]).to_csv(
        paths["future"], index=False
    )
    without_response = learner.predict(bindings)
    assert len(without_response) == len(future)
    assert np.isfinite(without_response).all()
    assert np.max(np.abs(predicted - without_response)) < 1e-12


def test_future_outcomes_cannot_be_supplied_as_calibration(bound_tables):
    """Fail closed when the history source includes a future session."""
    paths, _ = bound_tables
    with pytest.raises(ValueError, match="first-session-only"):
        build_learner(
            splitter=SessionGroupedSplit([f"unit{i}" for i in range(6)], 3)
        ).fit({"trials_path": paths["train"], "calibration_path": paths["future"]})


def test_missingness_control_does_not_read_observed_signal_values(bound_tables):
    """Change finite physiology without changing the control prediction."""
    paths, future = bound_tables
    learner = build_learner(
        modality="missing",
        splitter=SessionGroupedSplit([f"unit{i}" for i in range(6)], 3),
    )
    learner.fit({"trials_path": paths["train"], "calibration_path": paths["history"]})
    bindings = {"trials_path": paths["future"], "calibration_path": paths["history"]}
    original = learner.predict(bindings)
    future.loc[:, list(PHYSIOLOGY)] *= 1000
    future.to_csv(paths["future"], index=False)
    assert np.max(np.abs(original - learner.predict(bindings))) < 1e-12


def test_grouped_report_routes_independent_history(bound_tables):
    """Exercise skore through the experiment's producer on unit data only."""
    paths, _ = bound_tables
    experiment = importlib.import_module("experiments.01_aversive_reactivity")
    splitter = SessionGroupedSplit([f"unit{i}" for i in range(6)], 3)
    report = experiment._evaluate_report(
        paths["train"], paths["history"], splitter, {"modality": "all"}
    )
    assert len(report.reports_) == 3
    assert all(
        len(fold.get_predictions(data_source="test")) == 12 for fold in report.reports_
    )


@pytest.mark.parametrize(
    "configuration", [{"nonlinear": True}, {"profile": "cluster"}, {"profile": "pca"}]
)
def test_additional_representations_preserve_new_grid(bound_tables, configuration):
    """Check the spline and low-support profile paths before remote execution."""
    paths, future = bound_tables
    learner = build_learner(
        modality="all",
        splitter=SessionGroupedSplit([f"unit{i}" for i in range(6)], 3),
        **configuration,
    )
    learner.fit({"trials_path": paths["train"], "calibration_path": paths["history"]})
    predicted = learner.predict(
        {"trials_path": paths["future"], "calibration_path": paths["history"]}
    )
    assert len(predicted) == len(future)
    assert np.isfinite(predicted).all()


def test_final_freeze_and_saved_report_cache(bound_tables, tmp_path):
    """Require the freeze and distinguish raw paths from workspace references."""
    import json

    paths, future = bound_tables
    future = future.iloc[:-3].copy()
    future.to_csv(paths["future"], index=False)
    experiment = importlib.import_module("experiments.01_aversive_reactivity")
    folder = tmp_path / "analysis" / "cohort_2" / "full"
    folder.mkdir(parents=True)
    population = {
        "folder": folder,
        "train_path": paths["train"],
        "reserved_path": paths["future"],
        "history_path": paths["history"],
        "splitter": SessionGroupedSplit([f"unit{i}" for i in range(6)], 3),
    }
    with pytest.raises(RuntimeError, match="settings freeze"):
        experiment._final_report(population, {"modality": "all"})
    (folder.parents[1] / "frozen.json").write_text(
        json.dumps({"status": "frozen_before_reserved_scoring"})
    )
    report = experiment._final_report(population, {"modality": "all"})
    original = report.get_predictions(data_source="test")
    expected = float(experiment._person_errors(future, original).mean())
    assert float(report.metrics.get("participant_macro_mae")) == pytest.approx(expected)
    project = experiment._persist_report(tmp_path / "reports", "unit_only", report)
    frame = project.summarize().frame()
    print("SKORE_SUMMARY_SCHEMA", frame.index.names, list(frame.columns))
    normalized = frame.reset_index()
    column = "id" if "id" in normalized else "report_id"
    report_id = str(normalized.iloc[0][column])
    for path in paths.values():
        path.rename(path.with_name(path.name + ".unavailable"))
    relocated = tmp_path / "relocated_reports"
    (tmp_path / "reports").rename(relocated)
    reopened = experiment.skore.Project(
        "habs-aversive-reactivity", mode="local", workspace=str(relocated.resolve())
    )
    # Skore 0.29.1 stores absolute dataset references inside the workspace.
    with pytest.raises(FileNotFoundError):
        reopened.get(report_id)
    relocated.rename(tmp_path / "reports")
    restored_project = experiment.skore.Project(
        "habs-aversive-reactivity",
        mode="local",
        workspace=str((tmp_path / "reports").resolve()),
    )
    retrieved = restored_project.get(report_id)
    restored = retrieved.get_predictions(data_source="test")
    assert len(restored) == len(future)
    assert np.max(np.abs(original - restored)) < 1e-12
    assert float(retrieved.metrics.get("participant_macro_mae")) == pytest.approx(
        expected
    )


def test_participant_metric_is_not_trial_weighted():
    """A participant with two trials does not receive twice the influence."""
    assert participant_macro_mae([0, 0, 1], [1, 1, 1], groups=["a", "a", "b"]) == 0.5
