"""Read materialized HABS trials without hiding selection or temporal features."""

import numpy as np
import pandas as pd


def load_table(path):
    """Read an analysis table with stable participant identifiers.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV or compressed CSV produced by the annotation or signal audit.

    Returns
    -------
    pandas.DataFrame
        All rows and columns as stored, with no selection or imputation.
    """
    return pd.read_csv(path, na_values=["TESTSUBJECT"], dtype={"participant_id": str})


def target_values(table, target="score"):
    """Derive a response solely from the trial's declared ratings.

    Parameters
    ----------
    table : pandas.DataFrame
        Trial table containing valence and arousal ratings.
    target : str, default="score"
        Primary score, normalized valence/arousal, negative valence, or the
        prespecified square-root-arousal sensitivity score.

    Returns
    -------
    pandas.Series
        Continuous response in [0, 1], retaining the input index.

    Raises
    ------
    ValueError
        If the target name is unsupported or a rating is invalid.
    """
    valence = pd.to_numeric(table["rating_valence"], errors="raise")
    arousal = pd.to_numeric(table["rating_arousal"], errors="raise")
    if not (valence.between(1, 9).all() and arousal.between(1, 9).all()):
        raise ValueError("Targets require valid, observed valence and arousal.")
    negative = ((5 - valence) / 4).clip(lower=0)
    activation = (arousal - 1) / 8
    options = {
        "score": negative * activation,
        "valence": (valence - 1) / 8,
        "arousal": activation,
        "negative": negative,
        "alternative": negative * np.sqrt(activation),
    }
    if target not in options:
        raise ValueError(f"Unsupported target {target!r}.")
    return options[target].rename(target)
