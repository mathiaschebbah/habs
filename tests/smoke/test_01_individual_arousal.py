"""Check prediction replay on actual development trials with future labels hidden."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from model import CalibratedArousal, build_learner
from study import session_scenario, split_study, tuning_contexts


def verify_prediction_grid(
    feature_file, frame, split, out_dir, minimum_sources=10, minimum=20
):
    """Fit on one development fold and replay only its held-out participants."""
    fold = tuning_contexts(
        frame, split, minimum_sources=minimum_sources, minimum=minimum
    )[0]
    parameters = {"minimum_sources": minimum_sources, "minimum_trials": minimum}
    learner = build_learner(
        CalibratedArousal(mode="all", **parameters), GroupKFold(5), minimum=minimum
    )
    common = {
        "feature_file": str(feature_file),
        "videos": fold["videos"],
        "paired_people": split["paired_people"],
    }
    learner.fit({**common, "people": fold["train_people"]})
    held = sorted(fold["test_grid"].participant_id.unique())
    prediction = learner.predict({**common, "people": held})
    isolated = frame.loc[frame.participant_id.isin(held)].copy()
    isolated.loc[isolated.day == 2, "rating_arousal"] = np.nan
    target = Path(out_dir) / "smoke_replay.csv.gz"
    isolated.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
    replay = learner.predict({**common, "feature_file": str(target), "people": held})
    assert len(replay) == len(prediction) == len(fold["test_grid"])
    assert np.isfinite(replay).all()
    np.testing.assert_allclose(replay, prediction)
    return {
        "development_people": len(fold["train_people"]),
        "held_development_people": len(held),
        "prediction_rows": len(replay),
        "blind_replay_equal": True,
    }


def test_real_prediction_grid(tmp_path):
    """Use the gated feature table when it is available in the environment."""
    import pytest

    path = os.environ.get("HABS_FEATURES")
    if not path:
        pytest.skip("Set HABS_FEATURES to the full private feature table")
    frame = session_scenario(pd.read_csv(path).query("cohort == 2"))
    scenario = tmp_path / "scenario.csv.gz"
    frame.to_csv(scenario, index=False, compression={"method": "gzip", "mtime": 0})
    verify_prediction_grid(scenario, frame, split_study(frame), tmp_path)
