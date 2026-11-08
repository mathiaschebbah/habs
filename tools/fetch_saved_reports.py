"""Retrieve primary saved reports and their referenced datasets within team limits."""

import json
import os
import re
from pathlib import Path, PurePosixPath

from habs_hack.store import BlobStore


def main():
    """Download only report evidence, dependency files and publication-safe outputs.

    Returns
    -------
    None
        Selected files are restored below the private destination directory.

    Raises
    ------
    ValueError
        If identifiers, paths, report support or the published size limit fail.
    """
    suffix = os.environ["JOB_SUFFIX"]
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}", suffix):
        raise ValueError("Invalid completed-job suffix.")
    team = os.environ["HABS_KEY"]
    store = BlobStore(os.environ["HABS_STORAGE"])
    container = store.service.get_container_client(team)
    prefix = f"out/{team}--{suffix}/"
    files = {
        blob.name[len(prefix) :]: int(blob.size)
        for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"])
        if not (getattr(blob, "metadata", None) or {}).get("hdi_isfolder")
    }
    destination = Path(os.environ["PRIVATE_OUTPUT_DEST"]).resolve()
    roots = set()
    report_prefix = ("reports", "projects", "habs-aversive-reactivity", "reports")
    for name in files:
        parts = PurePosixPath(name).parts
        if (
            parts[:4] == report_prefix
            and len(parts) >= 6
            and "__id_" in parts[4]
            and "__01_aversive_reactivity__cohort2__" in parts[4]
        ):
            roots.add("/".join(parts[:5]) + "/")
    if len(roots) < 5:
        raise ValueError("Completed primary reports were not found.")
    selected = {
        name
        for name in files
        if name == "frozen.json"
        or name == "reports/.SKORE_WORKSPACE"
        or (name.startswith("public/") and name.endswith((".json", ".png")))
        or any(name.startswith(root) for root in roots)
    }
    limit = float(store.read_limits(team).get("max_download_mb", 25)) * 1e6
    downloaded = set()
    total = 0

    def download(name):
        nonlocal total
        if name in downloaded:
            return
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe output path.")
        if name not in files or total + files[name] > limit:
            raise ValueError(
                "Selected report evidence exceeds the published transfer limit."
            )
        target = destination.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(container.download_blob(prefix + name).readall())
        total += files[name]
        downloaded.add(name)

    for name in sorted(selected):
        download(name)
    dependencies = set()
    for name in selected:
        if "/data/" in name and name.endswith(".json") and name.startswith("reports/"):
            references = json.loads(
                destination.joinpath(*PurePosixPath(name).parts).read_text()
            )
            for reference in references.values():
                original = PurePosixPath(reference["file_path"])
                relative = original.relative_to(PurePosixPath("/out"))
                if relative.parts[:2] != ("reports", "datasets"):
                    raise ValueError("Unexpected saved dataset reference.")
                dependencies.add(str(relative))
    for name in sorted(dependencies):
        download(name)
    print(
        f"Restored {len(roots)} primary reports and their dependencies: "
        f"{len(downloaded)} files, {total / 1e6:.2f} MB, within published limit."
    )


if __name__ == "__main__":
    main()
