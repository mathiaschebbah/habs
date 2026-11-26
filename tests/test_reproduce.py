"""Contract tests verifying results/*.json integrity, figure generation, and process.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from habs_stress.figures import generate_all
from process import process


def test_results_and_figures_reproduce_cleanly(tmp_path: Path):
    sys.path.insert(0, str(Path("scripts").resolve()))
    import reproduce

    data = reproduce.load_results(Path("results"))
    assert len(data) == 5
    reproduce.print_summary_tables(data)

    fig_paths = generate_all(Path("results"), tmp_path)
    assert len(fig_paths) == 4
    for p in fig_paths:
        assert p.is_file() and p.stat().st_size > 20_000


def test_process_entrypoint_on_sample(tmp_path: Path):
    process(Path("sample"), tmp_path)
    audit_json = tmp_path / "public" / "signal_audit.json"
    assert audit_json.is_file()
    payload = json.loads(audit_json.read_text())
    assert payload["n_bdf_found"] >= 2 and payload["n_audited"] >= 2
