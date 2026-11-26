"""Boundary, electrode-coverage and frozen-regression checks on raw input."""

import numpy as np

from habs_stress.raw_eeg import ecg_design, prefix, reference_eeg, regress_ecg


def test_future_samples_do_not_change_values_or_masks():
    rng = np.random.default_rng(48)
    x = rng.normal(size=14001)
    for stop in (0.0, 5.0):
        y = x.copy()
        edge = int(np.ceil((6.0003 + stop) * 1000))
        y[edge::2] = np.nan
        y[edge + 1 :: 2] = 1e8
        np.testing.assert_array_equal(
            prefix(x, 1000, [6.0003], stop=stop),
            prefix(y, 1000, [6.0003], stop=stop),
        )


def test_frozen_ecg_projection_removes_injected_component():
    rng = np.random.default_rng(99)
    design = ecg_design(rng.normal(size=(2, 2000)))
    coef = rng.normal(size=(25, 5))
    component = np.einsum("ntl,lc->nct", design, coef)
    valid = np.ones((2, 5), dtype=bool)
    np.testing.assert_allclose(regress_ecg(component, valid, design, coef), 0.0, atol=1e-10)
    component[:, 2] = np.nan
    referenced, good = reference_eeg(component)
    assert not good[:, 2].any()
    np.testing.assert_allclose(referenced.sum(axis=1), 0.0, atol=1e-5)
