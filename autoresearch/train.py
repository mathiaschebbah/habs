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
    "frontal": ("Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "F1", "F2", "F3", "F4", "Fz"),
    "left_temporal": ("F7", "FT7", "T7", "TP7", "F9", "T9", "P9"),
    "right_temporal": ("F8", "FT8", "T8", "TP8", "F10", "T10", "P10"),
    "central": ("FC1", "FC2", "FCz", "C1", "C2", "C3", "C4", "CP1", "CP2"),
    "parietal": ("P1", "P2", "P3", "P4", "Pz", "P5", "P6", "P7", "P8", "CP3", "CP4"),
    "occipital": ("O1", "O2", "PO3", "PO4", "PO7", "PO8", "POz", "PO5", "PO6"),
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
    power = regional(frame)
    region = power.columns.str.rsplit("_", n=1).str[0]
    total = np.log10((10**power).T.groupby(region).transform("sum").T)
    return pd.concat([x, power, (power - total).add_prefix("rel_")], axis=1)


def channels(frame):
    """Channel-level video band power plus peripheral summaries."""
    return frame[[c for c in frame if c.startswith(("bp_vid_",) + PREFIXES)]]


def ridge(x_train, y, x_test, alphas=np.logspace(0, 4, 9)):
    """Median-impute, standardise, ridge with GCV-chosen penalty."""
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=alphas)
    )
    return model.fit(x_train, y).predict(x_test)


def centred(frame, target):
    """Ratings minus each training person's mean rating."""
    return frame[target] - frame[target].groupby(frame["participant_id"]).transform(
        "mean"
    )


def peripheral(frame):
    """Ablation: cardiac, skin and respiratory summaries only."""
    return frame[[c for c in frame if c.startswith(PREFIXES)]]


VIEWS = (peripheral,)


def fit_predict(train, test):
    """Average standardised ridge predictions over several feature views."""
    out = {}
    for target in TARGETS:
        y = centred(train, target)
        preds = [ridge(view(train), y, view(test)) for view in VIEWS]
        out[target] = sum(p / p.std() for p in preds) / len(preds)
    return pd.DataFrame(out, index=test.index)
