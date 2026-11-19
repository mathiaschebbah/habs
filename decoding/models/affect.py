"""Study 03 final model (search commit b800330).

Two ridge regressions on person-centred ratings, one on regional absolute and relative
band power, one on channel-level band power, both with cardiac, skin and
respiratory summaries; their standardised predictions are averaged.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding.common import PERIPHERAL, regional

TARGETS = ("rating_valence", "rating_arousal")
PREFIXES = PERIPHERAL

ALPHAS = np.logspace(0, 4, 9)


def features(frame):
    """Regional absolute and relative band power plus peripheral summaries."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    power = regional(frame)
    region = power.columns.str.rsplit("_", n=1).str[0]
    total = np.log10((10**power).T.groupby(region).transform("sum").T)
    return pd.concat([x, power, (power - total).add_prefix("rel_")], axis=1)


def channels(frame):
    """Channel-level video band power plus peripheral summaries."""
    return frame[[c for c in frame if c.startswith(("bp_vid_",) + PREFIXES)]]


def ridge(x_train, y, x_test):
    """Median-impute, standardise, ridge with GCV-chosen penalty."""
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS)
    )
    return model.fit(x_train, y).predict(x_test)


def centred(frame, target):
    """Ratings minus each training person's mean rating."""
    return frame[target] - frame[target].groupby(frame["participant_id"]).transform(
        "mean"
    )


VIEWS = (features, channels)


def fit_predict(train, test):
    """Average standardised ridge predictions over several feature views."""
    out = {}
    for target in TARGETS:
        y = centred(train, target)
        preds = [ridge(view(train), y, view(test)) for view in VIEWS]
        # scale each view by its spread over the predicted batch, then average
        out[target] = sum(p / p.std() for p in preds) / len(preds)
    return pd.DataFrame(out, index=test.index)
