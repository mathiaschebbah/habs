"""Study 03 scorer: valence and arousal ratings for new people and new themes.

Score: for valence and for arousal, Spearman correlation between predicted and
reported rating within each person (at least five trials, non-constant ratings
and predictions), averaged over people, then over the two ratings. Chance is 0
and a person's offset or scale use cannot raise it. Pooled AUC of rating > 5 is
reported for comparison with other teams only.

Usage: ``python -m decoding.affect --split dev --model decoding/models/affect.py``.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from decoding.common import run

SALT = "habs-autoresearch-2026-11-19"  # frozen value used for study 03's splits
OUTPUTS = ("rating_valence", "rating_arousal")
DEFAULT_MODEL = "decoding/models/affect.py"
MIN_TRIALS = 5


def person_stats(frame, predictions):
    """Within-person Spearman rho per rating, NaN when undefined."""
    rows = {}
    for person, group in frame.groupby("participant_id"):
        values = {}
        for target in OUTPUTS:
            y, p = group[target], predictions.loc[group.index, target]
            ok = len(group) >= MIN_TRIALS and y.nunique() > 1 and p.nunique() > 1
            values[target] = spearmanr(y, p).statistic if ok else np.nan
        rows[person] = values
    return pd.DataFrame.from_dict(rows, orient="index")


def score(stats):
    """Mean over the two ratings of the mean within-person rho."""
    return float(np.mean([np.nanmean(stats[t]) for t in OUTPUTS]))


def metrics(frame, predictions, stats):
    """Score plus per-rating rho and pooled AUC."""
    result = {"score": score(stats)}
    for target in OUTPUTS:
        name = target.removeprefix("rating_")
        result[f"{name}_rho"] = float(np.nanmean(stats[target]))
        result[f"{name}_auc"] = float(
            roc_auc_score(frame[target] > 5, predictions[target])
        )
    result["people"] = int(stats.notna().any(axis=1).sum())
    result["n"] = len(frame)
    return result


if __name__ == "__main__":
    run(__import__(__name__))
