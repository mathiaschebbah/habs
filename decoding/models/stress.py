"""Study 04 model: the one file we change during the stress search."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding.common import PERIPHERAL
from decoding.stress import SYMPATHETIC, label


def features(frame):
    """Channel-level video band power plus peripheral summaries."""
    return frame[
        [c for c in frame if c.startswith(("bp_vid_",) + PERIPHERAL + SYMPATHETIC)]
    ]


def fit_predict(train, test):
    """Return a stress score per ``test`` row (higher means more likely)."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-4, 0, 9),
            l1_ratios=(0,),
            scoring="neg_log_loss",
            use_legacy_attributes=False,
            max_iter=2000,
        ),
    )
    model.fit(features(train), label(train))
    return pd.DataFrame(
        {"stress": model.predict_proba(features(test))[:, 1]}, index=test.index
    )
