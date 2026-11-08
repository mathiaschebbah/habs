"""Check that a failed analysis cell cannot produce a successful HABS run."""

from pathlib import Path

import pytest

import process as entrypoint


def test_cell_error_fails_even_with_summary(tmp_path, monkeypatch):
    """Reject a partially successful output from the cell executor."""

    def write_failed_run(*args, **kwargs):
        (tmp_path / "eda_cells.md").write_text("**error:** `ValueError: bad input`")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "eda_summary.json").write_text("{}")

    monkeypatch.setattr(entrypoint.subprocess, "run", write_failed_run)
    with pytest.raises(RuntimeError, match="audit failed"):
        entrypoint.process(Path("sample"), tmp_path)


def test_missing_summary_fails(tmp_path, monkeypatch):
    """Reject an execution that did not complete the required analysis."""

    def write_incomplete_run(*args, **kwargs):
        (tmp_path / "eda_cells.md").write_text("No final output.")

    monkeypatch.setattr(entrypoint.subprocess, "run", write_incomplete_run)
    with pytest.raises(RuntimeError, match="audit failed"):
        entrypoint.process(Path("sample"), tmp_path)
