"""Check annotation coverage before the physiological arousal study."""

from pathlib import Path

from data.inventory import inventory


def process(data_dir: Path, out_dir: Path) -> None:
    """Summarize the mounted dataset without training a predictive model.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only dataset root supplied by the HABS platform.
    out_dir : pathlib.Path
        Directory retained by the platform for aggregate results.
    """
    inventory(data_dir, out_dir, probe_signals=True)
