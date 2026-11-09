"""Audit the released HABS annotations and native recording coverage."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def process(data_dir: Path, out_dir: Path) -> None:
    """Audit the mounted dataset and retain the measured evidence.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS dataset root.
    out_dir : pathlib.Path
        Writable directory retained by the HABS platform.

    Returns
    -------
    None
        Summaries and detailed audit outputs are written below out_dir.

    Raises
    ------
    RuntimeError
        If any analysis cell fails or the final summary is absent.
    """
    root = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = out_dir / "eda_cells.md"
    environment = {
        **os.environ,
        "HABS_DATA_ROOT": str(data_dir.resolve()),
        "HABS_ANALYSIS_OUT": str(out_dir.resolve()),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    cached = root / ".habs_inputs" / "data"
    temporal = (root / "temporal_run.json").is_file()
    if temporal and not cached.is_dir():
        raise RuntimeError(
            "The temporal study requires the audited native feature cache."
        )
    if cached.is_dir():
        evidence = json.loads((cached / "eda_summary.json").read_text())
        label_hash = hashlib.sha256(
            (data_dir / "labels_train.csv").read_bytes()
        ).hexdigest()
        if label_hash != evidence["labels_sha256"]:
            raise RuntimeError(
                "The resumed audit belongs to a different label release."
            )
        shutil.copytree(cached, out_dir / "data", dirs_exist_ok=True)
        checkpoints = root / ".habs_inputs" / "feature_checkpoints"
        if checkpoints.is_dir():
            shutil.copytree(
                checkpoints, out_dir / "feature_checkpoints", dirs_exist_ok=True
            )
        print(
            "Reused the team's audited inputs after checking the label fingerprint.",
            flush=True,
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vendor.run_cells",
                str(root / "data" / "eda.py"),
                str(digest),
            ],
            env=environment,
            cwd=root,
            check=True,
        )
        if (
            "**error:**" in digest.read_text()
            or not (out_dir / "data" / "eda_summary.json").is_file()
        ):
            raise RuntimeError("The data audit failed; see eda_cells.md.")
    summary = json.loads((out_dir / "data" / "eda_summary.json").read_text())
    if summary["status"] == "sample_without_ratings":
        print("Sample audit complete; no labelled study is possible on this sample.")
        return
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.02_emotional_carryover"
            if temporal
            else "experiments.01_aversive_reactivity",
        ],
        cwd=root,
        env=environment,
        check=True,
    )
