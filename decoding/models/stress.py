"""Study 04 final model (search commit 3b4b57f).

Two ridge regressions, one for person-centred valence and one for
person-centred arousal, on regional absolute and relative EEG band power plus
heart, skin and breathing features, each centred on the person's own mean.
The stress score is standardised arousal minus standardised valence: high when
a trial is predicted both activating and negative. On the sealed test it did
not beat the starting logistic regression (docs/research/stress/RESULTS.md).
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding.common import PERIPHERAL, regional
from decoding.stress import SYMPATHETIC

BODY = PERIPHERAL + SYMPATHETIC
ALPHAS = np.logspace(0, 4, 9)


def features(frame):
    """Regional power, relative power and body signals, centred per person."""
    power = regional(frame)
    region = power.columns.str.rsplit("_", n=1).str[0]
    total = np.log10((10**power).T.groupby(region).transform("sum").T)
    body = frame[[c for c in frame if c.startswith(BODY)]]
    x = pd.concat([body, power, (power - total).add_prefix("rel_")], axis=1)
    # label-free: uses only the person's own trials, as a batch at test time
    return x - x.groupby(frame["participant_id"].to_numpy()).transform("mean")


def ridge(train, y, test):
    """Median-impute, standardise, ridge with GCV-chosen penalty; scaled output."""
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS)
    )
    prediction = model.fit(features(train), y).predict(features(test))
    return prediction / prediction.std()


def fit_predict(train, test):
    """Score distress as predicted arousal minus predicted valence."""
    people = train["participant_id"]
    centred = {
        r: train[r] - train[r].groupby(people).transform("mean")
        for r in ("rating_valence", "rating_arousal")
    }
    valence = ridge(train, centred["rating_valence"], test)
    arousal = ridge(train, centred["rating_arousal"], test)
    return pd.DataFrame({"stress": arousal - valence}, index=test.index)
