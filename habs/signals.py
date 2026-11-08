"""Convert HABS skin-sensor voltages while exposing unit and range uncertainty."""

import numpy as np

CALIBRATION_VOLTS = (-2.14, -1.99755, -1.59528, 0.0167, 1.70074, 2.00199)
CALIBRATION_KOHMS = (1000.0, 500.0, 250.0, 125.0, 50.0, 25.0)


def eda_to_conductance(values, header_unit, *, max_invalid_fraction=0.01):
    """Apply the supplied GSR calibration without silently clamping bad values.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        Native physical samples returned by pyEDFlib.
    header_unit : str
        Unit in the BDF header. The supplied sample labels raw microvolt-scale
        voltages as uS; this conflict is recorded rather than ignored.
    max_invalid_fraction : float, default=0.01
        Maximum fraction of non-finite or out-of-calibration samples. A window
        exceeding this fraction is returned entirely missing.

    Returns
    -------
    conductance : numpy.ndarray of shape (n_samples,)
        Conductance in microsiemens, or NaN for invalid/ambiguous samples.
    quality : dict
        Conversion convention, native unit, mismatch flag and invalid fraction.

    Raises
    ------
    ValueError
        If the samples are not one-dimensional or the allowed fraction is invalid.

    Notes
    -----
    The calibration is the resistance interpolation in the supplied HABS loader.
    A microvolt-scale signal labelled uS follows the documented raw
    microvolt convention, with an explicit mismatch flag. A plausible positive
    conductance signal labelled uS is ambiguous under the release documentation
    and is quarantined rather than converted twice. This is a provenance rule,
    not an independent validation of the sensor's physical calibration.
    """
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not 0 <= max_invalid_fraction <= 1:
        raise ValueError("Expected a vector and an invalid fraction between 0 and 1.")
    converted = np.full(raw.shape, np.nan)
    unit = str(header_unit).strip().replace("µ", "u").replace("μ", "u")
    quality = {"header_unit": str(header_unit), "unit_mismatch": False}
    finite = np.isfinite(raw)
    if not raw.size or not finite.any():
        return converted, {**quality, "status": "missing", "invalid_fraction": 1.0}
    if unit in {"V", "mV", "uV"}:
        voltage = raw * {"V": 1.0, "mV": 1e-3, "uV": 1e-6}[unit]
        convention = "header_voltage"
    elif unit == "uS" and np.median(np.abs(raw[finite])) > 1e4:
        voltage = raw * 1e-6
        convention = "documented_microvolts"
        quality["unit_mismatch"] = True
    else:
        return converted, {
            **quality,
            "status": "ambiguous_native_unit",
            "invalid_fraction": 1.0,
        }
    valid = (
        finite & (voltage >= CALIBRATION_VOLTS[0]) & (voltage <= CALIBRATION_VOLTS[-1])
    )
    invalid_fraction = float(1 - valid.mean())
    quality.update({"convention": convention, "invalid_fraction": invalid_fraction})
    if invalid_fraction > max_invalid_fraction:
        return converted, {**quality, "status": "outside_calibration"}
    resistance = np.interp(
        voltage, CALIBRATION_VOLTS, CALIBRATION_KOHMS, left=np.nan, right=np.nan
    )
    converted[valid] = 1000.0 / resistance[valid]
    return converted, {**quality, "status": "ok"}
