"""Run the current HABS study stage on the mounted dataset."""

from pathlib import Path

from features import extract


def process(data_dir: Path, out_dir: Path) -> None:
    """Extract trial features and save their quality report.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only dataset supplied by HABS.
    out_dir : pathlib.Path
        Private directory retained by the HABS platform.
    """
    extract(data_dir, out_dir)
