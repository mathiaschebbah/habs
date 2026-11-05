"""Check deterministic signal summaries before running the cohort extraction."""

import numpy as np

from features import describe_window, extract_session
from starting_notebooks.bdf_loader import uV_to_uS


def test_linear_window_preserves_change():
    """The two three-second medians must be seven seconds apart."""
    values = np.arange(10000) / 1000
    level, change, spread = describe_window(values, 1000)
    assert abs(level - 4.9995) < 1e-8
    assert abs(change - 7) < 1e-8
    assert abs(spread - 4.9995) < 1e-8


def test_converter_clipping_is_distinguishable_from_raw_flatness():
    """A varying voltage above calibration can yield a flat converted signal."""
    volts = np.linspace(2.1, 2.5, 10000)
    converted = uV_to_uS(volts * 1e6)
    assert np.max(volts) > np.min(volts)
    assert np.min(converted) == np.max(converted) == 40


def test_missing_metadata_preserves_annotation_rows(tmp_path):
    """Missing recordings must remain visible in the inclusion accounting."""
    rows = [{"participant_id": "P001", "day": 1, "video": "a.mp4", "start_marker": 100}]
    result = extract_session((tmp_path, "P001", 1, rows))
    assert len(result) == 1
    assert result[0]["join_status"] == "missing_metadata"
