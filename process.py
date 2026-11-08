"""Audit the released HABS annotations and native recording coverage."""

import os
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
    }
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
