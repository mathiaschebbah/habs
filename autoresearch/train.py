"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = ("rating_valence", "rating_arousal")


def person_z(x, people):
    """Z-score each feature within person, from that person's own trials."""
    groups = x.groupby(people.to_numpy())
    return (x - groups.transform("mean")) / groups.transform("std").replace(0, np.nan)


PREFIXES = ("bp_vid_", "ecg_", "eda_", "resp_")


def features(frame):
    """Selected feature groups, z-scored within person."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    return person_z(x, frame["participant_id"])


def fit_predict(train, test):
    """Fit on labelled ``train`` rows and predict both ratings for ``test``."""
    x_train, x_test = features(train), features(test)
    out = {}
    for target in TARGETS:
        y = train[target] - train[target].groupby(train["participant_id"]).transform(
            "mean"
        )
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=0
        )
        model.fit(x_train, y)
        out[target] = model.predict(x_test)
    return pd.DataFrame(out, index=test.index)
