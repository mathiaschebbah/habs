"""Check how pooling scores from different withheld theme folds affects the time baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from habs_stress.pub_revision import (
    _run_gbdt_r4_3x3,
    assign_splits,
    bootstrap_mean,
    build_session_order_and_prev_trial,
    person_stress_aucs,
    run_r4_3x3,
    stress_label,
)


def run(trials):
    full = assign_splits(trials)
    raw, rank, _, _ = build_session_order_and_prev_trial(full, np.zeros((len(full), 1)))
    dev = (full.split == "dev").to_numpy()
    frame = full.loc[dev].reset_index(drop=True)
    raw, rank = raw[dev], rank[dev]
    y = stress_label(frame).to_numpy()
    pf, tf = frame.person_fold.to_numpy(), frame.theme_fold.to_numpy()
    prior = np.empty(len(frame))
    for p in range(3):
        for t in range(3):
            prior[(pf == p) & (tf == t)] = y[(pf != p) & (tf != t)].mean()
    models = {
        "training_prevalence_only": prior,
        "rank": rank,
        "linear_rank": run_r4_3x3(frame, rank[:, None], max_pca=None, fit_logreg=True)[
            "stress_logreg"
        ],
        "tree_rank": _run_gbdt_r4_3x3(frame, np.column_stack([raw, rank]))["stress_clf"],
    }
    results = {}
    for name, pred in models.items():
        whole = person_stress_aucs(frame, pred, y, ~y)
        conditioned = []
        matched_whole = []
        for pid, pos in frame.groupby("participant_id").indices.items():
            if str(pid) not in whole:
                continue
            auc_weight, pairs = 0.0, 0
            for t in range(3):
                ids = pos[tf[pos] == t]
                npair = int(y[ids].sum()) * int((~y[ids]).sum())
                if npair:
                    auc_weight += npair * roc_auc_score(y[ids], pred[ids])
                    pairs += npair
            if pairs:
                conditioned.append(auc_weight / pairs)
                matched_whole.append(whole[str(pid)])
        results[name] = {
            "within_id": bootstrap_mean(np.asarray(matched_whole)),
            "within_id_and_same_outer_model": bootstrap_mean(np.asarray(conditioned)),
        }
    return results


if __name__ == "__main__":
    data = pd.read_parquet(Path(sys.argv[1]) / "trials.parquet")
    Path(sys.argv[2]).write_text(json.dumps(run(data), indent=2))
