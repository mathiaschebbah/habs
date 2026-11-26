"""Minimal HABS platform entrypoint.

Runs a signal-quality and polarity audit on any mounted BDF dataset (including
the local ``sample/`` participant) using ``habs_stress.signals``.
"""

from __future__ import annotations

import json
from pathlib import Path

from habs_stress.signals import read_bdf_summary


def process(data_dir: Path, out_dir: Path) -> None:
    """Audit mounted BDF recordings and write publication-safe summaries.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS dataset root (e.g. ``./sample`` or ``/data``).
    out_dir : pathlib.Path
        Writable directory where JSON audit summaries are stored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir = out_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    bdfs = sorted(data_dir.rglob("*.bdf"))
    if not bdfs:
        raise RuntimeError(f"No BDF files found under {data_dir}")

    summaries = [read_bdf_summary(p) for p in bdfs[:12]]
    report = {
        "data_dir": str(data_dir),
        "n_bdf_found": len(bdfs),
        "n_audited": len(summaries),
        "recordings": summaries,
    }
    (public_dir / "signal_audit.json").write_text(json.dumps(report, indent=2))
    print(f"Audited {len(summaries)}/{len(bdfs)} BDF files -> {public_dir / 'signal_audit.json'}")
