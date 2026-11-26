"""Unit tests for publication-revision Priorities 1-3 contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from habs_stress.pub_revision import (
    assign_splits,
    build_matched_cell_pools,
    build_session_order_and_prev_trial,
    run_priority1_matched_ladder,
    split_post_columns,
)


def _synthetic_dev(n_ids: int = 18, n_themes: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ids):
        pid = f"P{i:03d}"
        for day in (1, 2):
            for k in range(10):
                th = f"theme_{(i * 3 + k) % n_themes}"
                rows.append(
                    {
                        "participant_id": pid,
                        "recorded_day": day,
                        "event_order": k,
                        "onset_s": float(100 + 25 * k),
                        "video": f"{th}_{k % 3}",
                        "video_theme": th,
                        "cohort": 1 if i < 9 else 3,
                        "rating_valence": int(rng.integers(1, 10)),
                        "rating_arousal": int(rng.integers(1, 10)),
                    }
                )
    df = pd.DataFrame(rows)
    df = assign_splits(df)
    return df.reset_index(drop=True)


def test_post_column_exclusion_is_strict() -> None:
    cols = [
        "ecg_hr_e_chg",
        "ecg_hr_l_chg",
        "ecg_hr_p_chg",
        "ecg_hr_d_l_e",
        "ecg_hr_d_p_l",
        "ecg_qt_vid_chg",
        "eeg_frontal_delta_p_chg",
        "eeg_tvar_e_delta",
        "custom_post_feature",
    ]
    no_post, post = split_post_columns(cols)
    assert set(post) == {
        "ecg_hr_p_chg",
        "ecg_hr_d_p_l",
        "eeg_frontal_delta_p_chg",
        "custom_post_feature",
    }
    assert set(no_post) == {
        "ecg_hr_e_chg",
        "ecg_hr_l_chg",
        "ecg_hr_d_l_e",
        "ecg_qt_vid_chg",
        "eeg_tvar_e_delta",
    }


def test_matched_n_ladder_pools_share_test_cell_and_exact_train_size() -> None:
    df = _synthetic_dev()
    pf = df["person_fold"].to_numpy(int)
    tf = df["theme_fold"].to_numpy(int)

    for p in range(3):
        for t in range(3):
            ev_m, pools = build_matched_cell_pools(pf, tf, p, t)
            for name, pm in pools.items():
                assert not (ev_m & pm).any(), f"{name} overlaps test cell!"
            assert pools["R4_double_unseen"].sum() <= pools["R3_new_id_seen_themes"].sum()
            assert pools["R4_double_unseen"].sum() <= pools["R2_seen_id_new_themes"].sum()
            assert pools["R1_random_pool"].sum() == (~ev_m).sum()

    rng = np.random.default_rng(7)
    x = rng.normal(size=(len(df), 8)).astype(np.float32)
    res = run_priority1_matched_ladder(
        df, {"toy": x}, n_repeats=2, max_pca=6, draws=50
    )
    audit = res["pool_audit_mean_across_9_cells"]
    n_r4 = audit["matched_R4_double_unseen"]["n_rows"]
    for r in ("R1_random_pool", "R2_seen_id_new_themes", "R3_new_id_seen_themes"):
        assert audit[f"matched_{r}"]["n_rows"] == n_r4
    assert audit["matched_R1_random_pool"]["frac_same_id"] > 0.0
    assert audit["matched_R1_random_pool"]["frac_same_theme"] > 0.0
    assert audit["matched_R4_double_unseen"]["frac_same_id"] == 0.0
    assert audit["matched_R4_double_unseen"]["frac_same_theme"] == 0.0


def test_session_order_resets_per_day_and_prev_trial_never_crosses_boundary() -> None:
    df = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P1", "P1", "P2", "P2"],
            "recorded_day": [1, 1, 2, 2, 1, 1],
            "event_order": [2, 1, 1, 2, 1, 2],
            "onset_s": [20.0, 10.0, 10.0, 20.0, 10.0, 20.0],
        }
    )
    x = np.arange(6, dtype=np.float32)[:, None] * 10.0
    idx_raw, idx_norm, is_first, x_prev = build_session_order_and_prev_trial(df, x)
    # Row 1 (P1 day 1 order 1) is first; row 0 (P1 day 1 order 2) gets row 1's x (=10.0)
    assert is_first[1] and not is_first[0]
    assert np.isnan(x_prev[1, 0]) and x_prev[0, 0] == 10.0
    # Row 2 (P1 day 2 order 1) resets to 0 and MUST NOT inherit from day 1!
    assert is_first[2] and idx_raw[2] == 0.0 and np.isnan(x_prev[2, 0])
    assert x_prev[3, 0] == 20.0
    # Row 4 (P2 day 1 order 1) resets and MUST NOT inherit from P1!
    assert is_first[4] and np.isnan(x_prev[4, 0]) and x_prev[5, 0] == 40.0
    assert np.allclose(idx_norm[[1, 0, 2, 3, 4, 5]], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
