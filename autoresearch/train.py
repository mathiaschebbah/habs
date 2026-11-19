"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("rating_valence", "rating_arousal")
PREFIXES = ("bp_vid_", "ecg_", "eda_", "resp_")


def features(frame):
    """Video-window band powers and peripheral summaries."""
    return frame[[c for c in frame if c.startswith(PREFIXES)]]


def fit_predict(train, test):
    """Fit on labelled ``train`` rows and predict both ratings for ``test``.

    Parameters
    ----------
    train : pandas.DataFrame
        Development rows with features, metadata, ratings and stimulus columns.
    test : pandas.DataFrame
        Rows to predict: features and metadata only, from people and video
        themes absent from ``train``.

    Returns
    -------
    pandas.DataFrame
        Index of ``test``, one column per target.
    """
    out = {}
    for target in TARGETS:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(0, 4, 9)),
        )
        model.fit(features(train), train[target])
        out[target] = model.predict(features(test))
    return pd.DataFrame(out, index=test.index)
