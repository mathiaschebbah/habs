"""Stage the team's existing, aligned development grid inside its private GPU job."""

import os
import sys
from pathlib import Path

import pandas as pd
from habs_hack.store import BlobStore

from habs_stress.protocol import assign_splits


def main(destination: Path) -> None:
    """Select the same trials, labels, target and body features used in Study 09."""
    team = os.environ["HABS_KEY"]
    container = BlobStore(os.environ["HABS_STORAGE"]).service.get_container_client(team)
    tables = []
    for suffix, name in (
        ("20261119T195319-d95b8d", "trials"),
        ("20261124T102049-806609", "study09_causal"),
    ):
        local = Path(os.environ["RUNNER_TEMP"]) / f"{name}.parquet"
        blob = container.get_blob_client(f"out/{team}--{suffix}/features/{name}.parquet")
        with local.open("wb") as f:
            blob.download_blob().readinto(f)
        tables.append(pd.read_parquet(local))
    base, physio = tables
    keep = [
        "trial_id",
        "participant_id",
        "recorded_day",
        "recording",
        "onset_s",
        "video",
        "video_theme",
        "cohort",
        "rating_valence",
        "rating_arousal",
    ]
    fields = ["trial_id", "tgt_ecg_hr_chg_5_10", *sorted(c for c in physio if c.startswith("b05_"))]
    if len(fields) != 59:
        raise ValueError("Expected 57 archived body predictors, a target and a join key")
    table = assign_splits(base[keep].merge(physio[fields], on="trial_id", validate="1:1"))
    dev = table.loc[table["split"] == "dev"].copy()
    if (len(dev), dev["participant_id"].nunique(), dev["theme"].nunique()) != (12693, 228, 66):
        raise ValueError("Development grid differs from the prerevision data")
    dev.to_parquet(destination, index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
