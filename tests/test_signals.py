"""Exercise calibration provenance and failure behavior independently of labels."""

from pathlib import Path

import numpy as np
import pyedflib
import pytest

from habs.signals import CALIBRATION_VOLTS, eda_to_conductance


def test_calibration_endpoints_have_declared_units():
    """Convert the known calibration endpoints exactly, in volts."""
    converted, quality = eda_to_conductance(CALIBRATION_VOLTS, "V")
    assert list(converted) == [1.0, 2.0, 4.0, 8.0, 20.0, 40.0]
    assert quality["status"] == "ok"
    assert not quality["unit_mismatch"]


def test_outside_range_is_missing_instead_of_clamped():
    """Prevent an invalid voltage from becoming an apparently valid endpoint."""
    converted, quality = eda_to_conductance([-3e6, 3e6], "uV")
    assert not np.isfinite(converted).any()
    assert quality["status"] == "outside_calibration"


def test_plausible_conductance_is_not_converted_again():
    """Quarantine a unit convention not resolved by the release documentation."""
    converted, quality = eda_to_conductance([2.0, 3.0, 4.0], "uS")
    assert not np.isfinite(converted).any()
    assert quality["status"] == "ambiguous_native_unit"


def test_positive_raw_voltage_uses_the_same_documented_convention():
    """Avoid selecting GSR validity according to the sign of its voltage."""
    converted, quality = eda_to_conductance([1_700_740.0], "uS")
    assert abs(converted[0] - 20.0) < 1e-12
    assert quality["unit_mismatch"]


def test_documented_conversion_matches_provided_loader_on_real_sample():
    """Check the observed sample mismatch against HABS's own conversion."""
    from starting_notebooks.bdf_loader import uV_to_uS

    bdf = (
        Path(__file__).resolve().parents[1]
        / "sample/CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf"
    )
    if not bdf.is_file() or bdf.stat().st_size < 1_000_000:
        pytest.skip("The actual local BDF sample is not present.")
    with pyedflib.EdfReader(str(bdf)) as reader:
        index = reader.getSignalLabels().index("GSR")
        native = reader.readSignal(index, 0, 1000)
        unit = reader.getPhysicalDimension(index)
    converted, quality = eda_to_conductance(native, unit)
    assert quality["unit_mismatch"]
    assert quality["status"] == "ok"
    assert np.max(np.abs(converted - uV_to_uS(native))) < 1e-12
