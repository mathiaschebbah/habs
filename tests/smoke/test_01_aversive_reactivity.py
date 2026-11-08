"""Replay a disjoint real future grid through the approved calibrated graph."""

import os
from pathlib import Path

import numpy as np
import pytest

from habs.data import load_table
from habs.evaluate import SessionGroupedSplit
from habs.pipeline import build_learner


def test_real_disjoint_prediction_grid(tmp_path):
    """Preserve future rows without response columns or in-grid history buffers."""
    source = os.environ.get("HABS_REAL_FEATURES")
    if source is None or not Path(source).is_file():
        pytest.skip("The real labelled feature table is available in the HABS job.")
    table = load_table(source)
    table = table.loc[
        (table["cohort"] == 2) & (table["role"] == "development") & table["eeg_valid"]
    ]
    history_counts = (
        table.loc[table["recorded_day"] == 1].groupby("participant_id").size()
    )
    available = sorted(
        set(
            table.loc[
                (table["recorded_day"] == 2) & table["evaluation_eligible"],
                "participant_id",
            ]
        )
    )
    available = [pid for pid in available if history_counts.get(pid, 0) >= 5]
    if len(available) < 5:
        pytest.fail(
            "Fewer than five real development people are usable for the smoke test."
        )
    fit_people, predict_person = available[:4], available[4]
    train = table.loc[table["participant_id"].isin(fit_people)]
    history = table.loc[
        (table["recorded_day"] == 1)
        & table["participant_id"].isin(fit_people + [predict_person])
    ]
    future = table.loc[
        (table["participant_id"] == predict_person)
        & (table["recorded_day"] == 2)
        & table["evaluation_eligible"]
    ].head(17)
    assert len(future) > 0
    assert set(train["trial_id"]).isdisjoint(future["trial_id"])
    future = future.drop(columns=["rating_valence", "rating_arousal", "aversive_score"])
    train_path, history_path, future_path = (
        tmp_path / name for name in ("train.csv", "history.csv", "future.csv")
    )
    train.to_csv(train_path, index=False)
    history.to_csv(history_path, index=False)
    future.to_csv(future_path, index=False)
    learner = build_learner(
        modality="all", splitter=SessionGroupedSplit(fit_people, n_splits=2)
    )
    learner.fit({"trials_path": train_path, "calibration_path": history_path})
    prediction = learner.predict(
        {"trials_path": future_path, "calibration_path": history_path}
    )
    assert len(prediction) == len(future)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    # No performance threshold is asserted before a measured CV headline exists.
