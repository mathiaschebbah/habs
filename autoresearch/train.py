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
    """Selected feature groups."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    bp = frame.filter(like="bp_vid_")
    channel = bp.columns.str.rsplit("_", n=1).str[0]
    total = np.log10((10**bp).T.groupby(channel).transform("sum").T)
    x = x.drop(columns=bp.columns)
    return pd.concat([x, (bp - total).add_prefix("rel_")], axis=1)


def fit_predict(train, test):
    """Fit on labelled ``train`` rows and predict both ratings for ``test``."""
    x_train, x_test = features(train), features(test)
    out = {}
    for target in TARGETS:
        y = train[target] - train[target].groupby(train["participant_id"]).transform(
            "mean"
        )
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(0, 4, 9)),
        )
        model.fit(x_train, y)
        out[target] = model.predict(x_test)
    return pd.DataFrame(out, index=test.index)
