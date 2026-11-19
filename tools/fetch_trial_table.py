"""Download the private trial table into an ephemeral CI runner."""

import os
import re
from pathlib import Path

from habs_hack.store import BlobStore

MAX_BYTES = 300e6  # HABS confirmed 300 MB outputs are fine; habs get caps at 25 MB


def main():
    """Copy the trial table (and an optional body table) to $TRIAL_DATA.

    ``SOURCE_JOB`` is one completed-job suffix, or two joined by ``+``: the
    first holds ``features/trials.parquet``, the second ``features/body.parquet``.

    Raises
    ------
    ValueError
        If a job suffix is malformed, a table is absent or too large.
    """
    suffixes = os.environ["SOURCE_JOB"].split("+")
    if len(suffixes) > 2 or not all(
        re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}", s) for s in suffixes
    ):
        raise ValueError("Invalid completed-job suffix.")
    team = os.environ["HABS_KEY"]
    container = BlobStore(os.environ["HABS_STORAGE"]).service.get_container_client(team)
    destination = Path(os.environ["TRIAL_DATA"])
    destination.mkdir(parents=True, exist_ok=True)
    for suffix, name in zip(suffixes, ("trials.parquet", "body.parquet"), strict=False):
        blob = container.get_blob_client(f"out/{team}--{suffix}/features/{name}")
        size = blob.get_blob_properties().size
        if size > MAX_BYTES:
            raise ValueError("The trial table exceeds the agreed transfer size.")
        (destination / name).write_bytes(blob.download_blob().readall())
        print(f"Restored {name}, {size / 1e6:.1f} MB.")


if __name__ == "__main__":
    main()
