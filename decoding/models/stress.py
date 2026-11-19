"""Study 04 model: the one file we change during the stress search."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from decoding.common import PERIPHERAL, regional
from decoding.stress import SYMPATHETIC, label

BODY = PERIPHERAL + SYMPATHETIC
ALPHAS = np.logspace(0, 4, 9)


def regional_view(frame):
    """Regional absolute and relative band power plus body signals."""
    power = regional(frame)
    region = power.columns.str.rsplit("_", n=1).str[0]
    total = np.log10((10**power).T.groupby(region).transform("sum").T)
    body = frame[[c for c in frame if c.startswith(BODY)]]
    return pd.concat([body, power, (power - total).add_prefix("rel_")], axis=1)


def channel_view(frame):
    """Channel-level video band power plus body signals."""
    return frame[[c for c in frame if c.startswith(("bp_vid_",) + BODY)]]


VIEWS = (regional_view,)


def ridge(x_train, y, x_test):
    """Median-impute, standardise, then k nearest neighbours."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=200, weights="distance"),
    )
    return model.fit(x_train, y).predict(x_test)


def centred(values, people):
    """Values minus each person's mean."""
    return values - values.groupby(people).transform("mean")


def ensemble(train, y, test):
    """Average the views' standardised ridge predictions."""
    preds = [ridge(view(train), y, view(test)) for view in VIEWS]
    return sum(p / p.std() for p in preds) / len(preds)


def target(train):
    """Person-centred stress label."""
    return centred(label(train).astype(float), train["participant_id"])


def fit_predict(train, test):
    """Predict valence and arousal, then score distress as arousal minus valence."""
    people = train["participant_id"]
    valence = ensemble(train, centred(train["rating_valence"], people), test)
    arousal = ensemble(train, centred(train["rating_arousal"], people), test)
    return pd.DataFrame({"stress": arousal - valence}, index=test.index)
