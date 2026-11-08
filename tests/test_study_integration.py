"""Exercise the entire analysis on unit data; no scientific performance is claimed."""

import importlib
import json

import numpy as np
import pandas as pd

from habs.features import PHYSIOLOGY


def test_complete_reserved_analysis_and_figures_on_unit_data(tmp_path, monkeypatch):
    """Catch integration failures before spending the real reserved evaluation."""
    experiment = importlib.import_module("experiments.01_aversive_reactivity")
    monkeypatch.setattr(experiment, "ALPHAS", (10.0,))
    rng = np.random.default_rng(20261108)
    rows = []
    for person in range(20):
        for day in (1, 2):
            for trial in range(25):
                physiology = {
                    name: float(rng.normal() + person / 3) for name in PHYSIOLOGY
                }
                rows.append(
                    {
                        "participant_id": f"unit{person:02}",
                        "trial_id": f"{person}_{day}_{trial}",
                        "cohort": 2,
                        "recorded_day": day,
                        "source": "CURATED",
                        "event_order": trial,
                        "video": f"v{day}_{trial}",
                        "expected_emotion_label": "neutral",
                        "expected_emotion_source": "default",
                        "evaluation_eligible": day == 2,
                        "role": "development" if person < 15 else "final_test",
                        "eligible_pair": True,
                        "rating_valence": (2, 4, 6, 8)[trial % 4],
                        "rating_arousal": 5 + trial % 5,
                        "eeg_valid": True,
                        "early_eeg_valid": True,
                        **physiology,
                        **{"early_" + k: v + 0.01 for k, v in physiology.items()},
                    }
                )
    frame = pd.DataFrame(rows)
    frame["aversive_score"] = experiment.target_values(frame)
    (tmp_path / "public").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/feature_manifest.json").write_text(
        json.dumps({"source": "unit_fixture"})
    )
    result, _, _, _ = experiment._run_study(frame, tmp_path)
    assert result["status"] == "reserved_evaluation_complete"
    assert result["primary"]["test_people"] == 5
    assert result["primary"]["test_trials"] == 125
    assert result["profiles"]["status"] == "supported"
    assert result["descriptions"]["profiles"]["stability"]["status"] == "evaluated"
    assert result["extreme_gradient"]["status"] == "evaluated"
    assert (tmp_path / "frozen.json").is_file()
    assert len(list((tmp_path / "public/figures").glob("*.png"))) == 7
