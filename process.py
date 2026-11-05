"""Audit the remaining annotation links before fixing the analysis population."""

from pathlib import Path

from alignment import reconcile


def process(data_dir: Path, out_dir: Path) -> None:
    """Audit unlinked videos without changing any uncertain match.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS release.
    out_dir : pathlib.Path
        Private result directory retained by HABS.
    """
    reconcile(data_dir, out_dir)
