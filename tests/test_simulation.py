"""Verify that validation simulations cannot consult real reserved outcomes."""

from pathlib import Path

import numpy as np

from simulation import generated_responses, initialize
from study import session_scenario, split_study
from tests.test_model import synthetic_trials


def test_generated_responses_ignore_reserved_real_ratings(tmp_path):
    """Changing unseen human outcomes cannot change the synthetic generator."""
    frame = synthetic_trials()
    frame["cohort"] = 2
    frame["eeg_front_alpha"] = 0.3 + 0.05 * frame.eda_change
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    split = split_study(
        session_scenario(frame, minimum=3), minimum_sources=1, minimum=3
    )
    initialize(path, 2, 1, 3)
    original = generated_responses("individual_signal", 123)
    frame.loc[
        frame.participant_id.isin(split["assigned_test"]) & (frame.day == 2),
        "rating_arousal",
    ] = 10 - frame.rating_arousal
    frame.to_csv(path, index=False)
    initialize(path, 2, 1, 3)
    replay = generated_responses("individual_signal", 123)
    np.testing.assert_allclose(original, replay)
    assert np.isfinite(replay).all()
    assert replay.min() >= 1 and replay.max() <= 9


def test_individual_injection_removes_source_stimulus_means(tmp_path):
    """The individual mechanism must not retain a rescaled shared component."""
    from simulation import _STATE

    frame = synthetic_trials()
    frame["cohort"] = 2
    frame["eeg_front_alpha"] = 0.3 + 0.05 * frame.eda_change
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    initialize(path, 2, 1, 3)
    source = _STATE["frame"].loc[_STATE["source_rows"], ["video"]].copy()
    source["individual"] = (_STATE["z"] - _STATE["shared"])[_STATE["source_rows"]]
    np.testing.assert_allclose(source.groupby("video").individual.mean(), 0, atol=1e-12)


def checkpoint_fixture(tmp_path, duplicate=False):
    """Make a completed metadata fixture without requiring a worker process."""
    import json

    import pandas as pd

    from simulation import RECORD_COLUMNS, SCENARIOS, configuration
    from study import SEED

    feature_file = tmp_path / "features.csv"
    feature_file.write_text("unused by a completed checkpoint\n")
    previous = tmp_path / "previous"
    previous.mkdir()
    config = configuration(feature_file, 2, 10, 20, 1)
    (previous / "simulation_config.json").write_text(json.dumps(config))
    records = []
    for index, scenario in enumerate(SCENARIOS):
        record = {name: 0 for name in RECORD_COLUMNS}
        record.update(
            scenario=scenario,
            repetition=0,
            seed=SEED + 100000 * index,
            test_people=26,
            seconds=0.1,
            positive_vs_reference=False,
            positive_after_controls=False,
        )
        records.append(record)
    pd.DataFrame(records * (2 if duplicate else 1)).to_csv(
        previous / "simulation_replicates.csv", index=False
    )
    return feature_file, previous / "simulation_replicates.csv"


def test_completed_resume_copies_replication_records(tmp_path):
    """A completed checkpoint must remain resumable in its new destination."""
    import pandas as pd

    from simulation import run

    feature_file, checkpoint = checkpoint_fixture(tmp_path)
    destination = tmp_path / "next"
    summary = run(feature_file, destination, replicates=1, workers=1, resume=checkpoint)
    assert summary["complete"]
    assert len(pd.read_csv(destination / "simulation_replicates.csv")) == 4


def test_fresh_run_cannot_relabel_existing_replications(tmp_path):
    """Reject an occupied directory before replacing its configuration."""
    import pytest

    from simulation import run

    feature_file, checkpoint = checkpoint_fixture(tmp_path)
    config = checkpoint.parent / "simulation_config.json"
    before = config.read_bytes()
    with pytest.raises(FileExistsError):
        run(feature_file, checkpoint.parent, replicates=1, workers=1)
    assert config.read_bytes() == before


def test_duplicate_replications_are_rejected(tmp_path):
    """Duplicate rows cannot inflate the completed Monte-Carlo sample size."""
    import pytest

    from simulation import run

    feature_file, checkpoint = checkpoint_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        run(feature_file, tmp_path / "next", replicates=1, workers=1, resume=checkpoint)


def test_entrypoint_prefers_its_own_checkpoint(tmp_path, monkeypatch):
    """An interrupted local batch resumes its own compatible output first."""
    import process as entry

    root = tmp_path / "repository"
    (root / "results").mkdir(parents=True)
    (root / "results/simulation_replicates.csv").write_text("older checkpoint\n")
    output = tmp_path / "output"
    (output / "validation").mkdir(parents=True)
    local = output / "validation/simulation_replicates.csv"
    local.write_text("local checkpoint\n")
    (output / "trials.csv.gz").write_bytes(b"already extracted")
    monkeypatch.setattr(entry, "__file__", str(root / "process.py"))
    commands = []
    monkeypatch.setattr(
        entry.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )

    def unexpected_extraction(*_args):
        raise AssertionError("Cached features should be reused")

    monkeypatch.setattr(entry, "extract", unexpected_extraction)
    monkeypatch.chdir(tmp_path)
    entry.process(Path("data"), Path("output"))
    assert commands[0][commands[0].index("--resume") + 1] == str(local)
