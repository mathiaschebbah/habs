"""Check the study-06 analyses on simulated data (never research evidence)."""

import numpy as np
import pandas as pd
import pytest

from decoding import hep_extract, person, timecourse

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def folder(tmp_path):
    """Heart follows the person's own valence; skin follows only the video."""
    rng = np.random.default_rng(2)
    video_valence = rng.normal(5, 2, 30)
    video_skin = rng.normal(0, 1, 30)
    trials, bins = [], []
    for p in range(40):
        for v in range(30):
            own = rng.normal(0, 1.5)
            valence = float(np.clip(np.round(video_valence[v] + own), 1, 9))
            arousal = float(rng.integers(1, 10))
            trial = f"{p}-{v}"
            trials.append(
                {
                    "trial_id": trial,
                    "participant_id": f"P{p}",
                    "cohort": 1 + p % 3,
                    "recorded_day": 1,
                    "event_order": v,
                    "video": f"t{v % 10}_{v}.mp4",
                    "video_theme": f"t{v % 10}",
                    "rating_valence": valence,
                    "rating_arousal": arousal,
                    "rating_dominance": float(rng.integers(1, 10)),
                }
            )
            response = {
                "hr": 0.8 * own + rng.normal(0, 1),
                "sc": video_skin[v] + rng.normal(0, 0.3),
            }
            row = {"trial_id": trial}
            for signal in ("hr", "sc", "resp_amp", "resp_rate", *timecourse.EEG):
                values = rng.normal(0, 0.1, timecourse.BINS)
                values[timecourse.ONSET :] += response.get(signal, 0.0)
                row.update({f"tc_{signal}_{k}": x for k, x in enumerate(values)})
            bins.append(row)
    pd.DataFrame(trials).to_parquet(tmp_path / "trials.parquet")
    pd.DataFrame(bins).to_parquet(tmp_path / "timecourse.parquet")
    return tmp_path


def test_components_leave_the_trial_out(folder):
    """The shared part of a trial never uses that trial's own value."""
    table = person.load(folder)
    values = pd.Series(0.0, index=table.index)
    values.iloc[0] = 100.0
    own, others, rest = person.components(table, values)
    assert others.iloc[0] < 1.0
    assert rest.groupby(table["participant_id"]).mean().abs().max() < 1e-9


def test_coupling_separates_shared_from_own(folder):
    """Heart tracks the person's own valence, skin tracks the video only."""
    table = person.load(folder)
    out = person.coupling(table)
    heart, skin = out["heart"]["valence"], out["skin"]["valence"]
    assert heart["beta_own"][1] > 0.2 and abs(heart["beta_shared"][0]) < 0.1
    assert skin["beta_own"][2] < 0.1 + abs(skin["beta_own"][0])
    assert out["skin"]["stimulus_locking"]["r"][0] > 0.8
    assert out["heart"]["stimulus_locking"]["r"][0] < 0.2


def test_room_and_audit_run(folder):
    """Same-video AUC sees the heart's personal signal; the audit only counts."""
    table = person.load(folder)
    model = pd.Series(np.random.default_rng(0).normal(size=len(table)), table.index)
    rooms, _ = person.room(table, model)
    assert set(rooms) == {"consensus", "model", *person.STRESS_SIGN}
    assert rooms["model"]["same_video"]["auc"] == pytest.approx(0.5, abs=0.1)
    assert set(person.cohorts(table, model)) == {"cohort_1", "cohort_2", "cohorts_3_5"}
    audit = person.audit(table)
    assert audit["people"] == 40 and audit["videos"] == 30
    assert "heart_bpm_by_control" in person.explore(table)


def test_heartbeat_locking_recovers_a_template():
    """A bump 300 ms after every beat comes back in the right 50 ms bin."""
    fs = 1000.0
    beats = np.arange(1.0, 59.0, 0.8)
    x = np.zeros(60_000)
    for b in beats:
        x[int((b + 0.3) * fs) : int((b + 0.35) * fs)] += 5.0
    x[int(20.5 * fs)] = 500.0  # one artefact, rejected by amplitude
    mean, count = hep_extract._locked(x, fs, beats, 10.0, 30.0, reject=100.0)
    assert np.argmax(mean) == 10  # bin 10 covers +300 to +350 ms
    assert mean[10] == pytest.approx(5.0)
    assert count == 24
