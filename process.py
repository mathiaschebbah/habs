"""Run the HABS signal extraction on the mounted dataset."""

from pathlib import Path

from features import extract


def process(data_dir: Path, out_dir: Path) -> None:
    """Extract complementary measurements from all reconciled cohorts.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS release.
    out_dir : pathlib.Path
        Private result directory retained by HABS.
    """
    extract(data_dir, out_dir)
