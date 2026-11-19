"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("rating_valence", "rating_arousal")


PREFIXES = ("bp_vid_", "ecg_", "eda_", "resp_")

PAIRS = [
    ("Fp1", "Fp2"),
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("C3", "C4"),
    ("T7", "T8"),
    ("CP5", "CP6"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("O1", "O2"),
    ("F1", "F2"),
    ("F5", "F6"),
    ("PO3", "PO4"),
]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def features(frame):
    """Selected feature groups."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    asym = {
        f"asym_{left}_{right}_{band}": frame[f"bp_vid_{right}_{band}"]
        - frame[f"bp_vid_{left}_{band}"]
        for left, right in PAIRS
        for band in BANDS
        if f"bp_vid_{left}_{band}" in frame and f"bp_vid_{right}_{band}" in frame
    }
    return pd.concat([x, pd.DataFrame(asym, index=frame.index)], axis=1)


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
