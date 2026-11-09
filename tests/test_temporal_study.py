"""Exercise the complete study components on explicitly simulated software data."""

import importlib

import numpy as np

from habs.temporal.figures import export_figures
from habs.temporal.inference import _mean_design, bootstrap_components
from habs.temporal.preprocess import prepare_data


def test_synthetic_components_export_and_keep_evidence_private(
    tmp_path, temporal_tables
):
    """Check integration and figure schemas without making an empirical claim."""
    features, events = temporal_tables()
    source, private = tmp_path / "data", tmp_path / "private"
    source.mkdir()
    private.mkdir()
    features.to_csv(source / "features.csv.gz", index=False)
    events.to_csv(source / "recorded_events.csv.gz", index=False)
    table, stats, metadata = prepare_data(features, events)
    experiment = importlib.import_module("experiments.02_emotional_carryover")
    geometry = experiment._geometry(table, stats, metadata["reference_gap_s"])
    estimated, values, sampling, counts = experiment._component_inputs(geometry)
    people = geometry[0].people
    ratio = np.asarray([stats[p]["sd_a"] / stats[p]["sd_p"] for p in people])
    components = bootstrap_components(
        values, sampling, counts, draws=3, scale_ratio=ratio
    )
    settings = {
        "seed": 20261109,
        "prediction_folds": 2,
        "predictive_bootstrap_draws": 50,
        "population_alpha": 10.0,
        "personal_penalty": 20.0,
        "reference_gap_s": metadata["reference_gap_s"],
    }
    prediction = experiment._forecast(table, source, private, settings)
    assert len(prediction["models"]) == 3
    assert prediction["people"] == len(people)
    assert not (source / "public").exists()
    adjusted = values - _mean_design(counts, True, ratio) @ np.asarray(
        components["mean_parameters"]
    )
    result = {
        "status": "synthetic_software_fixture_only",
        "structure": metadata,
        "variance_components": components,
        "prediction": prediction,
        "anonymous_delta_estimates": values.reshape(2, -1).tolist(),
        "anonymous_delta_adjusted_estimates": adjusted.reshape(2, -1).tolist(),
        "anonymous_delta_standard_errors": [
            np.sqrt(np.diag(e["sampling_covariance"])).tolist() for e in estimated
        ],
    }
    precision = {
        "scenarios": [
            {
                "scenario": "random_effect",
                "true_random_sd": 0.3,
                "true_correlation": 0.6,
                "true_covariance": 0.054,
                "estimated_covariance_quantiles": [0.01, 0.05, 0.11],
            }
        ]
    }
    q = np.quantile(table["gap_s"], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    feasibility = {
        "eligible_transitions": len(table),
        "gap_quantiles_seconds": dict(
            zip(("min", "q10", "q25", "median", "q75", "q90", "max"), q, strict=True)
        ),
    }
    paths = export_figures(
        result, precision, feasibility, tmp_path / "synthetic_layout_only"
    )
    assert len(paths) == 5
    assert all(__import__("pathlib").Path(p).stat().st_size > 10000 for p in paths)
