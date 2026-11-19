"""Download the private autoresearch trial table into an ephemeral CI runner."""

import os
import re
from pathlib import Path

from habs_hack.store import BlobStore

MAX_BYTES = 300e6  # HABS confirmed 300 MB outputs are fine; habs get caps at 25 MB


def main():
    """Copy ``autoresearch/trials.parquet`` of a completed job to $AUTORESEARCH_DATA.

    Raises
    ------
    ValueError
        If the job suffix is malformed, the table is absent or too large.
    """
    suffix = os.environ["SOURCE_JOB"]
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}", suffix):
        raise ValueError("Invalid completed-job suffix.")
    team = os.environ["HABS_KEY"]
    container = BlobStore(os.environ["HABS_STORAGE"]).service.get_container_client(team)
    blob = container.get_blob_client(
        f"out/{team}--{suffix}/autoresearch/trials.parquet"
    )
    size = blob.get_blob_properties().size
    if size > MAX_BYTES:
        raise ValueError("The trial table exceeds the agreed transfer size.")
    destination = Path(os.environ["AUTORESEARCH_DATA"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trials.parquet").write_bytes(blob.download_blob().readall())
    print(f"Restored the trial table, {size / 1e6:.1f} MB.")


if __name__ == "__main__":
    main()
