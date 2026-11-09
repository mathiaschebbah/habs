"""Replay a real disjoint J2 grid with no in-grid predecessors or current ratings."""

import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from habs.data import load_table
from habs.temporal.pipeline import build_learner
from habs.temporal.preprocess import prepare_data


def test_real_cold_j2_grid(tmp_path):
    """Preserve all future query rows using separately bound real history sources."""
    source = os.environ.get("HABS_TEMPORAL_INPUTS")
    if source is None or not (Path(source) / "features.csv.gz").is_file():
        pytest.skip("Restricted real native feature/event tables are required.")
    source = Path(source)
    features, events = (
        load_table(source / "features.csv.gz"),
        load_table(source / "recorded_events.csv.gz"),
    )
    table, _, metadata = prepare_data(features, events)
    people = sorted(set(table["participant_id"]))
    grid = table.loc[
        table["recorded_day"] == 2, ["trial_id", "participant_id", "recorded_day"]
    ]
    train = grid.loc[grid["participant_id"].isin(people[:4])]
    future = grid.loc[grid["participant_id"] == people[4]].head(17)
    assert len(future) > 0
    assert set(train["trial_id"]).isdisjoint(future["trial_id"])
    train_path, future_path = tmp_path / "fit_grid.csv", tmp_path / "cold_grid.csv"
    train.to_csv(train_path, index=False)
    future.to_csv(future_path, index=False)
    learner = build_learner(
        splitter=GroupKFold(2), reference_gap=metadata["reference_gap_s"]
    )
    bindings = {
        "grid_path": train_path,
        "features_path": source / "features.csv.gz",
        "events_path": source / "recorded_events.csv.gz",
    }
    learner.fit(bindings)
    prediction = learner.predict({**bindings, "grid_path": future_path})
    assert len(prediction) == len(future)
    assert np.isfinite(prediction).all()
    # A performance threshold would need a measured CV result, unavailable here.
