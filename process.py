"""Extract the fixed study data and validate the primary learning procedure."""

import subprocess
import sys
from pathlib import Path

from features import extract


def process(data_dir: Path, out_dir: Path) -> None:
    """Run a bounded, resumable batch of synthetic validation experiments.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS release.
    out_dir : pathlib.Path
        Private result directory retained by HABS.
    """
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    feature_file = out_dir / "trials.csv.gz"
    if not feature_file.exists():
        extract(data_dir, out_dir)
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-m",
        "experiments.02_method_validation",
        str(feature_file),
        str(out_dir / "validation"),
        "--replicates",
        "1000",
        "--workers",
        "24",
        "--max-seconds",
        "2400",
        "--frozen",
        str(root / "results/development.json"),
    ]
    resume = out_dir / "validation/simulation_replicates.csv"
    if not resume.exists():
        resume = root / "results/simulation_replicates.csv"
    if resume.exists():
        command.extend(["--resume", str(resume)])
    subprocess.run(command, cwd=root, check=True)
