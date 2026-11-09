"""Provide explicitly simulated tables for software verification of study 02."""

import math

import numpy as np
import pandas as pd
import pytest

from habs.temporal.data import physical_id


@pytest.fixture
def temporal_tables():
    """Create synthetic panels; these observations are never research evidence.

    Returns
    -------
    callable
        Factory returning native-feature-shaped observations and event metadata.
    """

    def create(people=10, trials=40, seed=73):
        rng = np.random.default_rng(seed)
        features, events = [], []
        for person in range(people):
            pid = f"synthetic{person}"
            for day in (1, 2):
                order_videos = rng.permutation(trials)
                arousal, brain, onset = 0.0, 0.0, 0.0
                for order, video in enumerate(order_videos):
                    arousal = (
                        0.25 * arousal + 0.4 * math.sin(video) + rng.normal(scale=0.5)
                    )
                    brain = 0.1 * arousal + 0.2 * brain + rng.normal(scale=0.5)
                    row = {
                        "participant_id": pid,
                        "recorded_day": day,
                        "start_marker": 100 + order,
                        "onset_s": onset,
                        "event_duration_s": 10.0,
                        "video": f"video{day}_{video}",
                        "event_order": order,
                        "recording": f"CURATED/{pid}/day{day}.bdf",
                        "source": "CURATED",
                    }
                    events.append(row)
                    features.append(
                        {
                            **row,
                            "trial_id": physical_id(row),
                            "cohort": 2,
                            "rating_arousal": float(np.clip(5 + arousal, 1, 9)),
                            "rating_valence": 5.0,
                            "eeg_valid": True,
                            "eeg_posterior_alpha": math.exp(
                                -2 - 0.3 * float(np.clip(brain, -3, 3))
                            ),
                            "expected_emotion_source": "default",
                            "expected_emotion_label": "neutral"
                            if video % 2
                            else "fear",
                        }
                    )
                    onset += 10 + float(rng.uniform(5, 25))
        return pd.DataFrame(features), pd.DataFrame(events)

    return create
