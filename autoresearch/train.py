"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("rating_valence", "rating_arousal")


PREFIXES = ("ecg_", "eda_", "resp_")

REGIONS = {
    "frontal_left": ("Fp1", "AF3", "AF7", "F1", "F3", "F5"),
    "frontal_mid": ("Fz", "FCz"),
    "frontal_right": ("Fp2", "AF4", "AF8", "F2", "F4", "F6"),
    "left_temporal": ("F7", "FT7", "T7", "TP7", "F9", "T9", "P9"),
    "right_temporal": ("F8", "FT8", "T8", "TP8", "F10", "T10", "P10"),
    "central_left": ("FC1", "FC3", "FC5", "C1", "C3", "C5", "CP1", "CP3", "CP5"),
    "central_right": ("FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2", "CP4", "CP6"),
    "parietal_left": ("P1", "P3", "P5", "P7"),
    "parietal_right": ("P2", "P4", "P6", "P8"),
    "midline_posterior": ("Pz", "POz"),
    "occipital_left": ("O1", "PO3", "PO5", "PO7"),
    "occipital_right": ("O2", "PO4", "PO6", "PO8"),
}
BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def regional(frame, kind="bp_vid", bands=BANDS):
    """Mean over the channels of each region, per band (or evoked bin)."""
    return pd.DataFrame(
        {
            f"{kind}_{region}_{band}": frame[
                [f"{kind}_{ch}_{band}" for ch in chs if f"{kind}_{ch}_{band}" in frame]
            ].mean(axis=1)
            for region, chs in REGIONS.items()
            for band in bands
        },
        index=frame.index,
    )


def features(frame):
    """Selected feature groups."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    return pd.concat([x, regional(frame)], axis=1)


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
