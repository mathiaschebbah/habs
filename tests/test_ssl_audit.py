"""Tests for Study-08 feasibility and runtime audit."""

import numpy as np
import pandas as pd

from decoding import ssl_audit


def test_audit_splits_and_runtime(tmp_path):
    """Split counts and runtime probe return structured aggregate dictionaries."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(30):
        for v in range(20):
            rows.append(
                {
                    "participant_id": f"P{p:03d}",
                    "recorded_day": 1 if v < 10 else 2,
                    "video": f"theme{v % 10}_{v}.mp4",
                    "video_theme": f"theme{v % 10}",
                    "rating_valence": float(rng.uniform(1, 9)),
                    "rating_arousal": float(rng.uniform(1, 9)),
                }
            )
    trials = pd.DataFrame(rows)
    splits = ssl_audit.audit_splits(trials, salts=("test-salt-1",))
    assert "test-salt-1" in splits
    assert "dev" in splits["test-salt-1"]
    split_names = ("dev", "test_new_both", "test_new_person", "test_new_video")
    total = sum(
        splits["test-salt-1"][s]["trials"]
        for s in split_names
        if s in splits["test-salt-1"]
    )
    assert total == len(trials)

    rt = ssl_audit.probe_runtime(tmp_path)
    assert rt["cpu_count"] >= 1
    assert "packages" in rt
