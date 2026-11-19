"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("rating_valence", "rating_arousal")


PREFIXES = ("bp_vid_", "ecg_", "eda_", "resp_")


def features(frame):
    """Selected feature groups."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    return x


def fit_predict(train, test):
    """Fit on labelled ``train`` rows and predict both ratings for ``test``."""
    x_train, x_test = features(train), features(test)
    out = {}
    for target in TARGETS:
        y = train[target] - train[target].groupby(train["participant_id"]).transform(
            "mean"
        )
        pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge()
        )
        # choose alpha on held-out people, not leave-one-trial-out
        model = GridSearchCV(
            pipe,
            {"ridge__alpha": np.logspace(0, 6, 13)},
            cv=GroupKFold(3),
            scoring="r2",
        )
        model.fit(x_train, y, groups=train["participant_id"])
        out[target] = model.predict(x_test)
    return pd.DataFrame(out, index=test.index)
