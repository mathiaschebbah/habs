"""Run the current HABS study stage on the mounted dataset."""

from pathlib import Path

from alignment import reconcile


def process(data_dir: Path, out_dir: Path) -> None:
    """Reconcile all released arousal notes with their recorded events.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only dataset supplied by HABS.
    out_dir : pathlib.Path
        Private directory retained by the HABS platform.
    """
    reconcile(data_dir, out_dir)
