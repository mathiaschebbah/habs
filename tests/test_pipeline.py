"""Unit fixtures test binding and leakage behavior, not scientific performance."""

import numpy as np
import pandas as pd
import pytest

from habs.evaluate import SessionGroupedSplit
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
