"""Check the autoresearch harness on a simulated table (never research evidence)."""

import numpy as np
import pandas as pd
import pytest

from autoresearch import prepare
from autoresearch.train import fit_predict


@pytest.fixture
def table():
    """Simulate trials where one feature carries the within-person valence."""
    rng = np.random.default_rng(0)
    rows = []
    for person in range(60):
        offset = rng.normal()
        for theme in range(40):
            valence = rng.integers(1, 10)
            rows.append(
                {
                    "trial_id": f"{person}-{theme}",
                    "participant_id": f"P{person:03d}",
                    "cohort": 1,
                    "recorded_day": 1,
                    "event_order": theme,
                    "start_marker": 100 + theme,
                    "onset_s": 12.0 * theme,
                    "video": f"theme{theme}_1.mp4",
                    "video_theme": f"theme{theme}",
                    "default_emotion_label": "joy",
                    "rating_valence": valence,
                    "rating_arousal": rng.integers(1, 10),
                    "rating_dominance": np.nan,
                    "recording": "x.bdf",
                    "age": 30.0,
                    "gender": "F",
                    "eeg_source": "PREPROCESSED",
                    "bp_vid_Fz_alpha": valence + offset + rng.normal(),
                    "ecg_hr": rng.normal(70, 5),
                }
            )
    return prepare.assign_splits(pd.DataFrame(rows))


def test_splits_are_disjoint(table):
    """No development person or theme appears in the new-both test regime."""
    dev = table[table["split"] == "dev"]
    both = table[table["split"] == "test_new_both"]
    assert not set(dev["participant_id"]) & set(both["participant_id"])
    assert not set(dev["theme"]) & set(both["theme"])
    assert set(table["split"]) == {"dev", *prepare.REGIMES}


def test_cross_fit_hides_stimulus_and_scores(table):
    """Each block is predicted blind to its people, themes and stimulus columns."""
    seen = []

    def spy(train, test):
        assert not set(prepare.HIDDEN) & set(test.columns)
        assert not set(train["participant_id"]) & set(test["participant_id"])
        seen.append(len(test))
        return fit_predict(train, test)

    dev, predictions, _ = prepare.cross_fit(table, spy)
    assert sum(seen) == len(dev) and len(seen) == 9
    result = prepare.metrics(dev, predictions)
    assert result["valence_rho"] > 0.3  # the simulated signal is recovered
    assert abs(result["arousal_rho"]) < 0.15  # no signal, no score
    low, high = prepare.bootstrap(dev, predictions, draws=200)
    assert low < result["score"] < high


def test_sealed_test_regimes(table):
    """One development fit predicts all three test regimes."""
    test, predictions, _ = prepare.sealed_test(table, fit_predict)
    assert len(predictions) == len(test) == (table["split"] != "dev").sum()
    gain = prepare.bootstrap(test, predictions, predictions, draws=50)
    assert gain == [0.0, 0.0]


def test_rejects_incomplete_predictions(table):
    """A model that forgets a target cannot produce a score."""
    with pytest.raises(ValueError, match="finite predictions"):
        prepare.cross_fit(
            table, lambda train, test: fit_predict(train, test).iloc[:, :1]
        )
