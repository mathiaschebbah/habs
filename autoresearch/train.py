"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("rating_valence", "rating_arousal")


def tangent(frame):
    """Log-Euclidean tangent vectors of the 20-channel covariance."""
    cov = frame.filter(like="cov_")
    names = sorted(
        {c.split("_")[1] for c in cov},
        key=lambda n: list(cov.columns).index(f"cov_{n}_{n}"),
    )
    k = len(names)
    iu = np.triu_indices(k)
    mats = np.full((len(frame), k, k), np.nan)
    mats[:, iu[0], iu[1]] = cov.to_numpy()
    mats[:, iu[1], iu[0]] = cov.to_numpy()
    ok = np.isfinite(mats).all(axis=(1, 2))
    out = np.full((len(frame), len(iu[0])), np.nan)
    if ok.any():
        m = mats[ok] + 1e-3 * np.trace(mats[ok], axis1=1, axis2=2)[
            :, None, None
        ] / k * np.eye(k)
        w, v = np.linalg.eigh(m)
        logm = (v * np.log(np.clip(w, 1e-12, None))[:, None, :]) @ v.transpose(0, 2, 1)
        weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
        out[ok] = logm[:, iu[0], iu[1]] * weight
    return pd.DataFrame(out, index=frame.index).add_prefix("ts_")


PREFIXES = ("bp_vid_", "ecg_", "eda_", "resp_")


def features(frame):
    """Selected feature groups."""
    x = frame[[c for c in frame if c.startswith(PREFIXES)]]
    return pd.concat([x, tangent(frame)], axis=1)


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
