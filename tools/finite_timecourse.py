"""Estimate within-ID contrasts using the actual finite class counts in each bin."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from habs_stress.pub_revision import STUDY09_BASE_SALT, TIMECOURSE_SALT, assign_splits, stress_label


def contrast(a, b):
    """Compute d with at least three finite observations per class and positive pooled SD."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 3:
        return np.nan
    sd = np.sqrt(
        (np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2)) / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan


def run(folder):
    trial = pd.read_parquet(
        folder / "trials.parquet",
        columns=[
            "trial_id",
            "participant_id",
            "video",
            "video_theme",
            "rating_valence",
            "rating_arousal",
        ],
    )
    names = ["hr", "resp_amp", "sc", "eeg_central_delta"]
    cols = ["trial_id", *[f"tc_{name}_{i}" for name in names for i in range(15)]]
    table = trial.merge(
        pd.read_parquet(folder / "timecourse.parquet", columns=cols), on="trial_id", validate="1:1"
    )
    answer = {}
    for key, salt in [
        ("original_timecourse", TIMECOURSE_SALT),
        ("audited_decoding", STUDY09_BASE_SALT),
    ]:
        subset = assign_splits(table, salt)
        dev = subset[subset.split == "dev"].reset_index(drop=True)
        y = stress_label(dev).to_numpy()
        groups = [
            idx
            for idx in dev.groupby("participant_id").indices.values()
            if min(y[idx].sum(), (~y[idx]).sum()) >= 3
        ]
        indices = np.random.default_rng(0).integers(len(groups), size=(2000, len(groups)))
        signals = {}
        for name in names:
            arr = dev[[f"tc_{name}_{i}" for i in range(15)]].to_numpy(float)
            count = np.isfinite(arr[:, :3]).sum(axis=1)
            base = np.divide(
                np.nansum(arr[:, :3], axis=1), count, out=np.full(len(arr), np.nan), where=count > 0
            )
            arr = arr - base[:, None]
            effects = np.array(
                [
                    [contrast(arr[idx[y[idx]], i], arr[idx[~y[idx]], i]) for i in range(15)]
                    for idx in groups
                ]
            )
            boot = np.nanmean(effects[indices], axis=1)
            signals[name] = {
                "eligible_ids": len(groups),
                "finite_ids_per_bin": np.isfinite(effects).sum(axis=0).tolist(),
                "d": np.nanmean(effects, axis=0).round(4).tolist(),
                "low": np.quantile(boot, 0.025, axis=0).round(4).tolist(),
                "high": np.quantile(boot, 0.975, axis=0).round(4).tolist(),
            }
        answer[key] = {
            "development_rows": len(dev),
            "development_ids": dev.participant_id.nunique(),
            "signals": signals,
        }
    return answer


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(run(Path(sys.argv[1])), indent=2, allow_nan=False))
