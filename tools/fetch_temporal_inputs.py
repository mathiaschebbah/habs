"""Read only the audited trial inputs needed for the sequential-emotion study."""

import json
import os
import re
from pathlib import Path

from habs_hack.store import BlobStore

FILES = (
    "eda_summary.json",
    "aligned_trials.csv.gz",
    "recorded_events.csv.gz",
    "recordings.csv",
    "allocation.csv",
    "features.csv.gz",
    "feature_manifest.json",
)


def main():
    """Download a bounded whitelist from the team's completed HABS job.

    Returns
    -------
    None
        Private input files are restored without printing individual observations.

    Raises
    ------
    ValueError
        If the job ID, required files or published transfer limit are invalid.
    """
    suffix = os.environ["JOB_SUFFIX"]
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}", suffix):
        raise ValueError("Invalid completed-job suffix.")
    team = os.environ["HABS_KEY"]
    store = BlobStore(os.environ["HABS_STORAGE"])
    container = store.service.get_container_client(team)
    prefix = f"out/{team}--{suffix}/data/"
    files = {
        blob.name[len(prefix) :]: int(blob.size)
        for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"])
        if not (getattr(blob, "metadata", None) or {}).get("hdi_isfolder")
    }
    if any(name not in files for name in FILES):
        raise ValueError("The completed job lacks required audited trial inputs.")
    total = sum(files[name] for name in FILES)
    limit = float(store.read_limits(team).get("max_download_mb", 25)) * 1e6
    if total > limit:
        raise ValueError("Selected input files exceed the published transfer limit.")
    destination = Path(os.environ["PRIVATE_INPUT_DEST"])
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (destination / name).write_bytes(
            container.download_blob(prefix + name).readall()
        )
    provenance = {"source_job_suffix": suffix, "files": len(FILES), "bytes": total}
    (destination / "temporal_input_source.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(f"Restored {len(FILES)} audited input files, {total / 1e6:.2f} MB.")


if __name__ == "__main__":
    main()
