"""Recover aggregate carryover outputs and optionally their private report evidence."""

import os
import re
from pathlib import Path, PurePosixPath

from habs_hack.store import BlobStore


def main():
    """Restore a bounded study output selection, excluding original trial inputs.

    Returns
    -------
    None
        Publication files and, for read-only audit, exact private saved reports.

    Raises
    ------
    ValueError
        If the source, required evidence, relative paths or size limit are invalid.
    """
    suffix = os.environ["JOB_SUFFIX"]
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}", suffix):
        raise ValueError("Invalid completed temporal-job suffix.")
    team = os.environ["HABS_KEY"]
    store = BlobStore(os.environ["HABS_STORAGE"])
    container = store.service.get_container_client(team)
    prefix = f"out/{team}--{suffix}/"
    private = os.environ.get("FETCH_TEMPORAL_REPORTS", "false").casefold() == "true"
    files = {
        b.name[len(prefix) :]: int(b.size)
        for b in container.list_blobs(name_starts_with=prefix, include=["metadata"])
        if not (getattr(b, "metadata", None) or {}).get("hdi_isfolder")
    }
    selected = {
        name
        for name in files
        if (name.startswith("public/") and name.endswith((".json", ".png")))
        or (
            private
            and (
                name.startswith("temporal/reports/")
                or name
                in {
                    "temporal/frozen.json",
                    "temporal/estimates.npz",
                    "temporal/real_smoke.log",
                    "temporal/j2_grid.csv",
                    "data/features.csv.gz",
                    "data/recorded_events.csv.gz",
                    "data/eda_summary.json",
                    "data/feature_manifest.json",
                }
            )
        )
    }
    if "public/temporal_results.json" not in selected:
        raise ValueError("Completed temporal scientific evidence was not found.")
    if (
        private
        and not {
            "temporal/frozen.json",
            "temporal/estimates.npz",
            "temporal/j2_grid.csv",
            "data/features.csv.gz",
            "data/recorded_events.csv.gz",
            "data/eda_summary.json",
            "data/feature_manifest.json",
        }
        <= selected
    ):
        raise ValueError("Required statistical audit evidence was not found.")
    total = sum(files[n] for n in selected)
    limit = float(store.read_limits(team).get("max_download_mb", 25)) * 1e6
    if total > limit:
        raise ValueError(
            "Selected temporal evidence exceeds the published transfer limit."
        )
    destination = Path(os.environ["PRIVATE_OUTPUT_DEST"]).resolve()
    for name in sorted(selected):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe completed-job output path.")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(container.download_blob(prefix + name).readall())
    print(f"Recovered {len(selected)} carryover evidence files, {total / 1e6:.2f} MB.")


if __name__ == "__main__":
    main()
