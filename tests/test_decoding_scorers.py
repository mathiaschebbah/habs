"""Check the shared scoring code and both studies on simulated tables.

The simulated tables are never research evidence.
"""

import numpy as np
import pandas as pd
import pytest

from decoding import affect, common, stress
from decoding.models.affect import fit_predict as affect_model
from decoding.models.stress import fit_predict as stress_model

# The simulated table has one EEG channel, so most regional means are empty.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Skipping features without any observed values"),
    pytest.mark.filterwarnings("ignore:divide by zero encountered in log10"),
    pytest.mark.filterwarnings("ignore:Mean of empty slice"),
    pytest.mark.filterwarnings("ignore:invalid value encountered"),
]


@pytest.fixture
def table():
    """Simulate trials where one feature carries within-person valence."""
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
                    "bp_vid_Fz_alpha": -valence + offset + rng.normal(),
                    "ecg_hr": rng.normal(70, 5),
                }
            )
    return pd.DataFrame(rows)


def test_splits_are_disjoint(table):
    """No development person or theme appears in the new-both test regime."""
    split = common.assign_splits(table, stress.SALT)
    dev = split[split["split"] == "dev"]
    both = split[split["split"] == "test_new_both"]
    assert not set(dev["participant_id"]) & set(both["participant_id"])
    assert not set(dev["theme"]) & set(both["theme"])
    assert set(split["split"]) == {"dev", *common.REGIMES}


def test_cross_fit_hides_stimulus_and_scores_affect(table):
    """Each block is predicted blind to its people, themes and stimulus columns."""
    seen = []

    def spy(train, test):
        assert not set(common.HIDDEN) & set(test.columns)
        assert not set(train["participant_id"]) & set(test["participant_id"])
        seen.append(len(test))
        return affect_model(train, test)

    split = common.assign_splits(table, affect.SALT)
    dev, predictions, _ = common.cross_fit(split, spy, affect.OUTPUTS)
    assert sum(seen) == len(dev) and len(seen) == 9
    stats = affect.person_stats(dev, predictions)
    result = affect.metrics(dev, predictions, stats)
    assert result["valence_rho"] > 0.3  # the simulated signal is recovered
    assert abs(result["arousal_rho"]) < 0.15  # no signal, no score
    low, high = common.bootstrap(stats, affect.score, draws=200)
    assert low < result["score"] < high
    assert common.bootstrap(stats, affect.score, stats, draws=50) == [0.0, 0.0]


def test_stress_scores_and_contrasts(table):
    """The stress scorer recovers the simulated valence signal on the sealed test."""
    split = common.assign_splits(table, stress.SALT)
    test, predictions, _ = common.sealed_test(split, stress_model, stress.OUTPUTS)
    both = test["split"] == "test_new_both"
    stats = stress.person_stats(test[both], predictions[both])
    result = stress.metrics(test[both], predictions[both], stats)
    assert result["score"] > 0.6 and result["vs_excitement"] > 0.6
    assert 0 < result["base_rate"] < 1


def test_correlates_find_the_simulated_feature(table):
    """Alpha is higher on stressed trials in the simulation, heart rate is not."""
    split = common.assign_splits(table, stress.SALT)
    out = stress.extra_splits("correlates", split)["stress"]
    assert out["bp_vid_frontal_alpha"]["ci95"][0] > 0
    low, high = out["ecg_hr"]["ci95"]
    assert low < 0 < high


def test_rejects_incomplete_predictions(table):
    """A model that forgets an output cannot produce a score."""
    split = common.assign_splits(table, affect.SALT)
    with pytest.raises(ValueError, match="finite predictions"):
        common.cross_fit(
            split, lambda tr, te: affect_model(tr, te).iloc[:, :1], affect.OUTPUTS
        )
