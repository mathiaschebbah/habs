"""Tests for zero-leakage person x theme splits, causal centering, and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from habs_stress.protocol import (
    assign_splits,
    benjamini_hochberg,
    causal_person_center,
    paired_person_bootstrap,
    prepare_protocol_fold,
)


def _toy_table(n_people: int = 20, n_themes: int = 10) -> pd.DataFrame:
    rows = []
    for p in range(n_people):
        for t in range(n_themes):
            rows.append(
                {
                    "participant_id": f"P{p:03d}",
                    "video": f"clip_{t:02d}_1.mp4",
                    "video_theme": f"theme_{t:02d}",
                    "onset_s": float(t * 15),
                    "rating_valence": 3.0 if t % 3 == 0 else 7.0,
                    "rating_arousal": 7.0 if t % 3 == 0 else 3.0,
                }
            )
    return pd.DataFrame(rows)


def test_person_theme_split_disjointness():
    df = assign_splits(_toy_table(30, 15))
    dev = df[df["split"] == "dev"]
    test_both = df[df["split"] == "test_new_both"]
    assert set(dev["participant_id"]).isdisjoint(set(test_both["participant_id"]))
    assert set(dev["theme"]).isdisjoint(set(test_both["theme"]))


def test_causal_online_centering_ignores_future_trials():
    rng = np.random.default_rng(42)
    x_a = rng.standard_normal((10, 4))
    x_b = x_a.copy()
    x_b[6:] += 500.0  # perturb strictly future trials
    people = np.array(["P001"] * 10)
    order = np.arange(10, dtype=float)
    prior = np.zeros(4)

    c_a = causal_person_center(x_a, people, order, prior_mean=prior, shrinkage=5.0)
    c_b = causal_person_center(x_b, people, order, prior_mean=prior, shrinkage=5.0)
    assert np.allclose(c_a[:6], c_b[:6])


def test_cold_start_protocol_is_batch_invariant():
    rng = np.random.default_rng(1)
    df = _toy_table(6, 5)
    x = rng.standard_normal((len(df), 3)).astype(np.float32)
    _, ev_all = prepare_protocol_fold(x[:15], df.iloc[:15], x[15:], df.iloc[15:], protocol="cold_start")
    _, ev_one = prepare_protocol_fold(x[:15], df.iloc[:15], x[15:16], df.iloc[15:16], protocol="cold_start")
    assert np.array_equal(ev_all[0], ev_one[0])


def test_bootstrap_and_bh_fdr():
    a = np.linspace(0.6, 0.8, 30)
    b = np.linspace(0.5, 0.7, 30)
    res = paired_person_bootstrap(a, b, draws=500, seed=0)
    assert res["mean"] > 0.09 and res["ci95"][0] > 0.0 and res["frac_improved"] == 1.0
    q = benjamini_hochberg([0.001, 0.02, 0.50])
    assert q[0] <= q[1] <= q[2] and np.all((q >= 0) & (q <= 1))
