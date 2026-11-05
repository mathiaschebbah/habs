"""Extract HABS signals, validate the method and replay the frozen study."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from data.repetition_audit import run as audit_repetitions
from features import extract


def process(data_dir: Path, out_dir: Path, stage: str = "repetitions") -> None:
    """Run one study stage with private outputs and aggregate public summaries.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS release.
    out_dir : pathlib.Path
        Private result directory retained by HABS.
    stage : str
        Extract signals, tune development models, validate, or evaluate final models.
    """
    if stage not in {"repetitions", "extract", "development", "validation", "final"}:
        raise ValueError(f"Unknown study stage: {stage}")
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    if stage == "repetitions":
        audit_repetitions(data_dir, out_dir)
        return
    feature_file = out_dir / "trials.csv.gz"
    if not feature_file.exists():
        extract(data_dir, out_dir)
    root = Path(__file__).resolve().parent
    if stage == "extract":
        return
    if stage in {"development", "final"}:
        destination = out_dir / ("study" if stage == "final" else "development")
        command = [
            sys.executable,
            "-m",
            "experiments.01_individual_arousal",
            str(feature_file),
            str(destination),
            "--stage",
            stage,
        ]
        if stage == "final":
            command.extend(["--frozen", str(root / "results/development.json")])
        subprocess.run(command, cwd=root, check=True)
        if stage == "final":
            subprocess.run(
                [sys.executable, "-m", "audit.01_individual_arousal"],
                cwd=root,
                env={**os.environ, "HABS_RESULTS": str(destination)},
                check=True,
            )
        return
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--stage",
        choices=("repetitions", "extract", "development", "validation", "final"),
        default="repetitions",
    )
    arguments = parser.parse_args()
    process(arguments.data_dir, arguments.out_dir, arguments.stage)
