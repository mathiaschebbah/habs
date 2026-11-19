"""Check the study-05 analyses on a simulated table (never research evidence)."""

import numpy as np
import pandas as pd
import pytest

from decoding import timecourse

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def folder(tmp_path):
    """Simulate trials where heart rate drops 2 s after onset on stressed trials."""
    rng = np.random.default_rng(1)
    trials, bins = [], []
    for person in range(50):
        for theme in range(24):
            valence, arousal = rng.integers(1, 10), rng.integers(1, 10)
            trial = f"{person}-{theme}"
            trials.append(
                {
                    "trial_id": trial,
                    "participant_id": f"P{person}",
                    "cohort": 1,
                    "recorded_day": 1 + theme % 2,
                    "event_order": theme,
                    "start_marker": 1,
                    "onset_s": 1.0,
                    "video": f"t{theme}_1.mp4",
                    "video_theme": f"t{theme}",
                    "default_emotion_label": "fear",
                    "rating_valence": valence,
                    "rating_arousal": arousal,
                    "rating_dominance": np.nan,
                    "recording": "x",
                    "age": 30.0,
                    "gender": "F",
                    "eeg_source": "PREPROCESSED",
                }
            )
            stressed = valence <= 4 and arousal >= 6
            row = {
                "trial_id": trial,
                "final_emotion_label": "Fear" if stressed else "joy",
            }
            for signal in ("hr", "sc", "resp_amp", "resp_rate", *timecourse.EEG):
                values = rng.normal(size=timecourse.BINS)
                if signal == "hr" and stressed:
                    values[timecourse.ONSET + 2 :] -= 1.5
                row.update({f"tc_{signal}_{k}": v for k, v in enumerate(values)})
            bins.append(row)
    pd.DataFrame(trials).to_parquet(tmp_path / "trials.parquet")
    pd.DataFrame(bins).to_parquet(tmp_path / "timecourse.parquet")
    windows = pd.DataFrame(
        {"recording": np.repeat(["a", "b"], 10), "window": np.tile(range(10), 2)}
    )
    for seconds in (5, 10, 20, 30, 60):
        windows[f"hr_{seconds}"] = 70 + rng.normal(size=20)
        windows[f"rmssd_{seconds}"] = 0.04 + 0.01 * rng.random(20)
    windows.to_parquet(tmp_path / "hrv_windows.parquet")
    return tmp_path


def test_curves_find_the_cardiac_drop(folder):
    """The simulated drop starts 2 s after onset; skin shows nothing."""
    out = timecourse.curves(timecourse.load(folder))["dev"]["stress"]
    assert out["hr"]["onset_s"] == 2
    assert out["sc"]["onset_s"] is None


def test_latency_and_other_parts_run(folder):
    """Detection improves once the drop is inside the window; other parts run."""
    table = timecourse.load(folder)
    result = timecourse.latency_test(table)
    heart = result["heart"]
    assert heart[4]["test_new_person"]["auc"] > heart[1]["test_new_person"]["auc"] + 0.1
    emotions = timecourse.emotions(table)
    assert emotions["dev"]["hr"]["fear"]["change_1_6s"][0] < 0
    assert set(timecourse.habituation(table)["hr_1_6s"]) == {"half", "day"}
    assert timecourse.hrv(folder)["rmssd_10"]["valid"] == 1.0
