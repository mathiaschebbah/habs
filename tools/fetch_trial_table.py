"""Download the private trial table into an ephemeral CI runner."""

import os
import re
from pathlib import Path

from habs_hack.store import BlobStore

MAX_BYTES = 300e6  # HABS confirmed 300 MB outputs are fine; habs get caps at 25 MB


def main():
    """Copy every ``features/*.parquet`` of the listed jobs to $TRIAL_DATA.

    ``SOURCE_JOB`` is one or more completed-job suffixes joined by ``+``. The
    first must hold ``trials.parquet``; later jobs add ``body.parquet``,
    ``timecourse.parquet`` or ``hrv_windows.parquet``.

    Raises
    ------
    ValueError
        If a job suffix is malformed or a table is too large.
    """
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
            print(f"Restored {name}, {blob.size / 1e6:.1f} MB.")
    if not (destination / "trials.parquet").is_file():
        raise ValueError("The first job must provide trials.parquet.")


if __name__ == "__main__":
    main()
