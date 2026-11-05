"""Check deterministic signal summaries before running the cohort extraction."""

from pathlib import Path

import numpy as np

from features import describe_window, ecg_features, eeg_features, read_recording
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


def test_alpha_wave_is_measured_in_both_regions():
    """A known 10-Hz wave belongs to the alpha band irrespective of amplitude."""
    values = np.sin(2 * np.pi * 10 * np.arange(10000) / 1000)
    channels = {
        name: {"values": values * scale, "rate": 1000}
        for scale, name in enumerate(
            ["F3", "F4", "Fz", "Cz", "P3", "P4", "O1", "O2"], 1
        )
    }
    result = eeg_features(channels, 0)
    assert result["eeg_valid"]
    assert result["eeg_front_alpha"] > 0.99
    assert result["eeg_back_alpha"] > 0.99


def test_heart_rate_uses_intervals_not_voltage_units():
    """Changing the amplitude unit cannot change an interval-derived feature."""
    channel = {"values": np.ones(10000), "rate": 1000, "peaks": np.arange(0.5, 10, 1)}
    first = ecg_features(channel, 0)
    channel["values"] *= 1000
    assert first == ecg_features(channel, 0)
    assert first == {"ecg_valid": True, "ecg_rate": 60.0, "ecg_change": 0.0}


def test_real_sample_channels_and_annotations_are_readable():
    """Exercise the supplied BDF reader on the actual resting-state sample."""
    root = Path(__file__).resolve().parents[1]
    channels, markers = read_recording(
        root / "sample/CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf"
    )
    assert {"GSR", "ECG", "F3", "O2"} <= set(channels)
    assert markers
    assert channels["GSR"]["rate"] == 1000
