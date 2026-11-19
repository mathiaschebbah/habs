"""Study 04 scorer: detecting stress reactions in new people watching new videos.

The release has no stress questionnaire, cortisol or stress manipulation, so we
define stress per trial from the participant's own ratings: a negative
(valence <= 4) and activated (arousal >= 6) reaction, the "distress" quadrant of
the circumplex model. All other trials count as not stressed.

Score: area under the ROC curve computed within each person (at least three
stressed and three non-stressed trials), averaged over people. Chance is 0.5,
and a person's general physiology or rating habits cannot raise it. We also
report pooled AUC and two specificity contrasts:

- ``vs_excitement``: among activated trials (arousal >= 6), stressed versus
  positive (valence >= 6). Can physiology tell stress from excitement?
- ``vs_calm_negative``: among negative trials (valence <= 4), activated versus
  calm (arousal <= 4). Does it track the activation part?

``--split correlates`` describes, on development data only, how each
interpretable physiological feature differs within person between stressed and
non-stressed trials.

Usage: ``python -m decoding.stress --split dev --model decoding/models/stress.py``.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from decoding.common import PERIPHERAL, bootstrap, regional, run

SALT = "habs-stress-2026-11-19"
OUTPUTS = ("stress",)
DEFAULT_MODEL = "decoding/models/stress.py"
EXTRA_SPLITS = ("correlates",)
MIN_EACH = 3


def label(frame):
    """Stressed trial: valence <= 4 and arousal >= 6."""
    return (frame["rating_valence"] <= 4) & (frame["rating_arousal"] >= 6)


def contrasts(frame):
    """Each contrast as (positive mask, negative mask) on ``frame``."""
    v, a, stressed = frame["rating_valence"], frame["rating_arousal"], label(frame)
    return {
        "stress": (stressed, ~stressed),
        "strict": ((v <= 3) & (a >= 7), ~((v <= 3) & (a >= 7))),
        "vs_excitement": (stressed, (v >= 6) & (a >= 6)),
        "vs_calm_negative": (stressed, (v <= 4) & (a <= 4)),
    }


def _auc(score, positive, negative):
    if positive.sum() < MIN_EACH or negative.sum() < MIN_EACH:
        return np.nan
    keep = positive | negative
    return roc_auc_score(positive[keep], score[keep])


def person_stats(frame, predictions):
    """Within-person AUC for every contrast, NaN when a class is too small."""
    rows = {}
    for person, group in frame.groupby("participant_id"):
        p = predictions.loc[group.index, "stress"]
        rows[person] = {
            name: _auc(p, pos, neg) for name, (pos, neg) in contrasts(group).items()
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def score(stats):
    """Mean within-person AUC for stressed versus other trials."""
    return float(np.nanmean(stats["stress"]))


def metrics(frame, predictions, stats):
    """Score, specificity contrasts and pooled detection metrics."""
    y, p = label(frame), predictions["stress"]
    result = {"score": score(stats)}
    for name in ("strict", "vs_excitement", "vs_calm_negative"):
        result[name] = float(np.nanmean(stats[name]))
        result[f"{name}_people"] = int(stats[name].notna().sum())
    result["pooled_auc"] = float(roc_auc_score(y, p))
    result["average_precision"] = float(average_precision_score(y, p))
    result["base_rate"] = float(y.mean())
    result["people"] = int(stats["stress"].notna().sum())
    result["n"] = len(frame)
    return result


def interpretable(frame):
    """Regional log band power, relative power and peripheral summaries."""
    power = regional(frame)
    region = power.columns.str.rsplit("_", n=1).str[0]
    total = np.log10((10**power).T.groupby(region).transform("sum").T)
    peripheral = frame[
        [c for c in frame if c.startswith(PERIPHERAL) and not c.endswith("_valid")]
    ]
    return pd.concat([power, (power - total).add_prefix("rel_"), peripheral], axis=1)


def effect_sizes(features, positive, negative, people):
    """Within-person standardised mean difference per feature, averaged.

    For each person with at least three trials in each group: (mean of the
    positive trials minus mean of the negative trials) divided by the person's
    pooled within-group standard deviation.
    """
    rows = []
    for person in people[positive | negative].unique():
        mine = people == person
        pos, neg = features[mine & positive], features[mine & negative]
        if len(pos) < MIN_EACH or len(neg) < MIN_EACH:
            continue
        pooled = np.sqrt(
            (pos.var() * (len(pos) - 1) + neg.var() * (len(neg) - 1))
            / (len(pos) + len(neg) - 2)
        )
        rows.append((pos.mean() - neg.mean()) / pooled.replace(0, np.nan))
    return pd.DataFrame(rows)


def extra_splits(name, table):
    """Development-only physiological correlates of the stress label."""
    dev = table[table["split"] == "dev"]
    features = interpretable(dev)
    out = {}
    for contrast in ("stress", "vs_excitement", "vs_calm_negative"):
        positive, negative = contrasts(dev)[contrast]
        d = effect_sizes(features, positive, negative, dev["participant_id"])
        out[contrast] = {
            column: {
                "d": float(d[column].mean()),
                "ci95": bootstrap(
                    d[[column]].dropna(),
                    lambda s, c=column: float(s[c].mean()),
                    draws=2000,
                ),
                "people": int(d[column].notna().sum()),
            }
            for column in d
        }
    return out


if __name__ == "__main__":
    run(__import__(__name__))
