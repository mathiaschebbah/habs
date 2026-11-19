"""The one file the autoresearch loop edits. Rules: autoresearch/program.md."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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
    return pd.concat([x, regional(frame)], axis=1)


def fit_predict(train, test):
    """Fit on labelled ``train`` rows and predict both ratings for ``test``."""
    x_train = features(train).dropna(axis=1, how="all")
    x_test = features(test)[x_train.columns]
    out = {}
    for target in TARGETS:
        y = train[target] - train[target].groupby(train["participant_id"]).transform(
            "mean"
        )
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=0,
        )
        model.fit(x_train, y)
        out[target] = model.predict(x_test)
    return pd.DataFrame(out, index=test.index)
