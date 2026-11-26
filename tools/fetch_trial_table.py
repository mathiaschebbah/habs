"""Download private trial Parquet tables into an ephemeral CI runner."""

from __future__ import annotations

import os
import re
from pathlib import Path

from habs_hack.store import BlobStore

MAX_BYTES = 300e6


def main() -> None:
    """Copy every ``features/*.parquet`` of the listed jobs to ``$TRIAL_DATA``."""
    suffixes = os.environ["SOURCE_JOB"].split("+")
    if not all(re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}", s) for s in suffixes):
        raise ValueError("Invalid completed-job suffix.")
    team = os.environ["HABS_KEY"]
    container = BlobStore(os.environ["HABS_STORAGE"]).service.get_container_client(team)
    destination = Path(os.environ["TRIAL_DATA"])
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        prefix = f"out/{team}--{suffix}/features/"
        for blob in container.list_blobs(name_starts_with=prefix):
            name = blob.name[len(prefix) :]
            if "/" in name or not name.endswith(".parquet"):
                continue
            if blob.size > MAX_BYTES:
                raise ValueError("A trial table exceeds the agreed transfer size.")
            data = container.download_blob(blob.name).readall()
            (destination / name).write_bytes(data)
            print(f"Restored {name}, {blob.size / 1e6:.1f} MB.", flush=True)
    if not (destination / "trials.parquet").is_file():
        raise ValueError("The first job must provide trials.parquet.")


if __name__ == "__main__":
    main()
