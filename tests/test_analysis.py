"""Check matching and label-free profile descriptions on explicit unit fixtures."""

import numpy as np
import pandas as pd

from habs.analysis import MARKERS, _specificity
from habs.features import PHYSIOLOGY
from habs.profiles import PhysiologyProfiles


def test_specificity_matching_never_reuses_a_trial():
    """Six people with two possible pairs each contribute twelve pairs, not 24."""
    rows = []
    values = []
    for person in range(6):
        for valence, arousal in ((2, 7.0), (3, 7.2), (7, 7.1), (8, 7.3)):
            rows.append(
                {
                    "participant_id": f"unit{person}",
                    "rating_valence": valence,
                    "rating_arousal": arousal,
                }
            )
            values.append([1.0 if valence < 5 else 0.0] * len(MARKERS))
    result = _specificity(pd.DataFrame(rows), pd.DataFrame(values, columns=MARKERS))
    assert result["matched_pairs"] == 12
    assert result["mean_absolute_arousal_gap"] <= 0.5
    for marker in MARKERS:
        assert result["markers"][marker]["people"] == 6
        assert result["markers"][marker]["pairs"] == 12
        assert result["markers"][marker]["mean"] == 1.0


def test_supported_profiles_ignore_response_columns():
    """Exercise enabled profiles and verify their construction is label-free."""
    rng = np.random.default_rng(20261108)
    rows = []
    for person in range(16):
        for trial in range(10):
            rows.append(
                {
                    "participant_id": f"unit{person}",
                    "video": f"v{trial}",
                    "rating_valence": 1 + (trial % 9),
                    **{name: float(rng.normal() + person / 4) for name in PHYSIOLOGY},
                }
            )
    table = pd.DataFrame(rows)
    first = PhysiologyProfiles(3).fit(table)
    second = PhysiologyProfiles(3).fit(table.drop(columns="rating_valence"))
    assert first.enabled_
    assert (
        np.max(
            np.abs(first.clusters_.cluster_centers_ - second.clusters_.cluster_centers_)
        )
        < 1e-12
    )
    descriptors = second.summarize(table)
    assert len(descriptors) == 16
    assert all(len(vector) == 2 for vector in descriptors.values())
