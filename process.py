"""Extract measurements and develop the calibrated HABS prediction models."""

import importlib
from pathlib import Path

from features import extract


def process(data_dir: Path, out_dir: Path) -> None:
    """Run development analyses without scoring the reserved final participants.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS release.
    out_dir : pathlib.Path
        Private result directory retained by HABS.
    """
    extract(data_dir, out_dir)
    experiment = importlib.import_module("experiments.01_individual_arousal")
    experiment.run(out_dir / "trials.csv.gz", out_dir / "study", stage="development")
