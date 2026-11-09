# %% [markdown]
# # Subjective and EEG carryover between recording days
#
# The scientific run follows structural and simulated-precision checks.
# Current responses stay on the target branch; J1 personal history is explicit.

# %%
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import skore
from scipy.linalg import block_diag
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold

from habs import PROJECT_ROOT
from habs.data import load_table
from habs.evaluate import participant_macro_mae
from habs.temporal.data import verify_inputs
from habs.temporal.inference import (
    _mean_design,
    bootstrap_components,
    covariance_profile,
    day_geometry,
    estimate_day,
)
from habs.temporal.pipeline import build_learner
from habs.temporal.preprocess import prepare_data

SEED = 20261109


def _evaluate_forecast(
    bindings,
    reference_gap,
    mode,
    *,
    n_splits=5,
    population_alpha=10.0,
    personal_penalty=20.0,
):
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    learner = build_learner(
        splitter=splitter,
        reference_gap=reference_gap,
        mode=mode,
        population_alpha=population_alpha,
        personal_penalty=personal_penalty,
    )
    grid = load_table(bindings["grid_path"])
    # skore 0.29.1 does not forward groups to an explicitly supplied CV object.
    # Materialize the genuine grouped splits rather than allowing a row split.
    folds = tuple(splitter.split(grid, groups=grid["participant_id"]))
    report = skore.evaluate(learner, data=bindings, splitter=folds, n_jobs=1)
    for child, (_, test) in zip(report.reports_, report.split_indices, strict=True):
        people = tuple(grid.iloc[test]["participant_id"])
        child.metrics.add(
            make_scorer(
                participant_macro_mae,
                greater_is_better=False,
                response_method="predict",
                groups=people,
            ),
            name="participant_macro_mae",
        )
    return report


def _oof_predictions(report, count):
    prediction = np.full(count, np.nan)
    visits = np.zeros(count, dtype=int)
    for (_, test), values in zip(
        report.split_indices, report.get_predictions(data_source="test"), strict=True
    ):
        prediction[test] = np.asarray(values)
        visits[test] += 1
    if not np.all(visits == 1) or not np.isfinite(prediction).all():
        raise ValueError("Every J2 row must have one finite out-of-person prediction.")
    return prediction


def _persist_forecast(workspace, key, report):
    report_workspace = Path(workspace)
    project = skore.Project(
        "habs-emotional-carryover",
        mode="local",
        workspace=str(report_workspace.resolve()),
    )
    project.put(key, report)
    return project


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def _code_hash():
    files = [
        p
        for p in sorted((PROJECT_ROOT / "habs/temporal").glob("*.py"))
        if p.name != "figures.py"
    ] + [
        PROJECT_ROOT / "experiments/02_emotional_carryover.py",
        PROJECT_ROOT / "temporal_config.json",
        PROJECT_ROOT / "docs/research/carryover/PROTOCOL.md",
    ]
    return hashlib.sha256(
        b"".join(
            str(p.relative_to(PROJECT_ROOT)).encode() + p.read_bytes() for p in files
        )
    ).hexdigest()


def _geometry_hash():
    package = PROJECT_ROOT / "habs/temporal"
    return hashlib.sha256(
        b"".join(
            n.encode() + (package / n).read_bytes()
            for n in ("data.py", "preprocess.py", "inference.py")
        )
    ).hexdigest()


def _geometry(table, stats, reference_gap):
    people = sorted(set(table["participant_id"]))
    return [
        day_geometry(
            table.loc[table["recorded_day"] == d], people, stats, reference_gap
        )
        for d in (1, 2)
    ]


def _component_inputs(geometries, *, hac_lags=2, hc3=True):
    estimates = [estimate_day(g, hac_lags=hac_lags, hc3=hc3) for g in geometries]
    values = np.concatenate([e["delta"] for e in estimates])
    sampling = block_diag(*(e["sampling_covariance"] for e in estimates))
    counts = np.asarray([e["counts"] for e in estimates])
    return estimates, values, sampling, counts


def _sensitivity_result(
    table,
    stats,
    reference_gap,
    *,
    covariance_factor=1.0,
    moderator=True,
    hac_lags=2,
    hc3=True,
):
    geometries = _geometry(table, stats, reference_gap)
    _, values, sampling, counts = _component_inputs(
        geometries, hac_lags=hac_lags, hc3=hc3
    )
    ratio = np.asarray(
        [stats[p]["sd_a"] / stats[p]["sd_p"] for p in geometries[0].people]
    )
    profile = covariance_profile(
        values,
        sampling * covariance_factor,
        counts,
        moderator=moderator,
        scale_ratio=ratio,
    )
    fit = profile["model"]
    return {
        "status": "completed",
        "people": len(ratio),
        "transitions": len(table),
        "covariance": fit["covariance"],
        "rho_point": fit["rho"],
        "rho_profile95_grid": profile["rho_profile95_grid"],
        "correlation_weakly_identified": profile["correlation_weakly_identified"],
        "random_effect_sd": fit["sd"].tolist(),
        "likelihood_ratio": profile["likelihood_ratio"],
    }


def _sensitivities(features, events, table, stats, metadata):
    reference = metadata["reference_gap_s"]
    results = {}
    for name, settings in (
        ("sampling_half", {"covariance_factor": 0.5}),
        ("sampling_double", {"covariance_factor": 2.0}),
        ("without_count_moderator", {"moderator": False}),
        ("hc1_no_serial_kernel", {"hac_lags": 0, "hc3": False}),
    ):
        results[name] = _sensitivity_result(table, stats, reference, **settings)
    for name, settings in (
        ("minimum_30", {"min_pairs": 30}),
        ("common_j1_scales", {"population_scales": True}),
        ("stable_j1_scales", {"stable_scales": True}),
    ):
        selected, normalization, _ = prepare_data(features, events, **settings)
        if name == "stable_j1_scales" and set(selected["trial_id"]) == set(
            table["trial_id"]
        ):
            results[name] = {
                "status": "identical_primary_grid_and_scales",
                "people": metadata["people"],
                "transitions": len(table),
            }
        else:
            try:
                results[name] = _sensitivity_result(selected, normalization, reference)
            except ValueError as error:
                results[name] = {"status": "unidentified_design", "reason": str(error)}
    low, high = metadata["j1_gap_iqr"]
    selected = table.loc[(table["gap_s"] >= low) & (table["gap_s"] <= high)].copy()
    counts = selected.groupby(["participant_id", "recorded_day"]).size()
    people = [
        p
        for p in set(selected["participant_id"])
        if all(counts.get((p, d), 0) >= 20 for d in (1, 2))
    ]
    selected = selected.loc[selected["participant_id"].isin(people)]
    if len(people) < 5:
        results["j1_gap_iqr"] = {
            "status": "insufficient_support",
            "people": len(people),
            "transitions": len(selected),
        }
    else:
        try:
            results["j1_gap_iqr"] = _sensitivity_result(selected, stats, reference)
        except ValueError as error:
            results["j1_gap_iqr"] = {
                "status": "unidentified_design",
                "reason": str(error),
            }
    return results


def _person_errors(grid, target, prediction, people):
    error = np.abs(target - prediction)
    groups = grid["participant_id"].to_numpy()
    return np.asarray([float(np.mean(error[groups == p])) for p in people])


def _paired(reference, candidate, seed, draws):
    delta = reference - candidate
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {
        "gain": float(np.mean(delta)),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "people_helped": int(np.sum(delta > 0)),
        "people": len(delta),
        "bootstrap_draws": draws,
    }


def _forecast(table, input_dir, private, settings):
    future = table.loc[table["recorded_day"] == 2].copy()
    grid = future[["trial_id", "participant_id", "recorded_day"]]
    grid_path = private / "j2_grid.csv"
    grid.to_csv(grid_path, index=False)
    target = future["difference"].to_numpy()
    people = sorted(set(grid["participant_id"]))
    errors, models = {}, {}
    bindings = {
        "grid_path": str(grid_path),
        "features_path": str(input_dir / "features.csv.gz"),
        "events_path": str(input_dir / "recorded_events.csv.gz"),
    }
    for mode in ("ablation", "full", "refit"):
        report = _evaluate_forecast(
            bindings,
            settings["reference_gap_s"],
            mode,
            n_splits=settings["prediction_folds"],
            population_alpha=settings["population_alpha"],
            personal_penalty=settings["personal_penalty"],
        )
        prediction = _oof_predictions(report, len(grid))
        errors[mode] = _person_errors(grid, target, prediction, people)
        fold_macros = []
        for child, (_, test) in zip(report.reports_, report.split_indices, strict=True):
            registered = float(child.metrics.get("participant_macro_mae"))
            expected = participant_macro_mae(
                target[test],
                prediction[test],
                groups=tuple(grid.iloc[test]["participant_id"]),
            )
            if abs(registered - expected) > 1e-10:
                raise RuntimeError(
                    "Cached registered and independently grouped errors differ."
                )
            fold_macros.append(registered)
            child.metrics.summarize().frame()
            child.get_predictions(data_source="train")
        _persist_forecast(
            private / "reports", "02_emotional_carryover__" + mode, report
        )
        models[mode] = {
            "participant_macro_mae": float(np.mean(errors[mode])),
            "fold_registered_macro_mae": fold_macros,
            "anonymous_person_errors": errors[mode].tolist(),
        }
    return {
        "people": len(people),
        "trials": len(grid),
        "models": models,
        "full_vs_ablation": _paired(
            errors["ablation"],
            errors["full"],
            settings["seed"],
            settings["predictive_bootstrap_draws"],
        ),
        "full_vs_refitted_restriction": _paired(
            errors["refit"],
            errors["full"],
            settings["seed"],
            settings["predictive_bootstrap_draws"],
        ),
        "scope": (
            "Grouped one-step J2 prediction with already observed predecessor "
            "responses and J1-only personal calibration."
        ),
        "macro_note": (
            "Global equal-person OOF error; parent fold-mean metrics "
            "can weight folds differently."
        ),
    }


def _real_smoke(input_dir, private):
    environment = {
        **os.environ,
        "HABS_TEMPORAL_INPUTS": str(input_dir),
        "HABS_REAL_FEATURES": str(input_dir / "features.csv.gz"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/smoke/", "-q"],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    (private / "real_smoke.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0 or "skipped" in result.stdout:
        raise RuntimeError(
            "A required real-data smoke failed; "
            "details are private in temporal/real_smoke.log."
        )
    return result.stdout.strip().splitlines()[-1]


def run_study(out_dir):
    """Execute the frozen cohort-2 study and retain auditable private reports.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Writable HABS output tree containing the verified audited input cache.

    Returns
    -------
    None
        Private statistical/report evidence and public aggregate exports are saved.

    Raises
    ------
    RuntimeError
        If source/structure changed after precision checking or a smoke test fails.
    """
    out = Path(out_dir)
    source, private, public = out / "data", out / "temporal", out / "public"
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    if (public / "temporal_results.json").exists():
        raise RuntimeError(
            "Completed scientific results already exist; audit rather than refit."
        )
    config = json.loads((PROJECT_ROOT / "temporal_config.json").read_text())
    manifest = verify_inputs(source)
    input_fingerprints = json.loads(
        (PROJECT_ROOT / "data/temporal_feasibility.json").read_text()
    )["input_sha256"]
    for name, expected in input_fingerprints.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                "A private source differs from the read-only feasibility audit."
            )
    structure = json.loads((PROJECT_ROOT / "data/temporal_structure.json").read_text())
    precision = json.loads((PROJECT_ROOT / "data/temporal_precision.json").read_text())
    if structure["geometry_inference_code_sha256"] != _geometry_hash():
        raise RuntimeError(
            "Precision evidence belongs to a different geometry/inference source."
        )
    features, events = (
        load_table(source / "features.csv.gz"),
        load_table(source / "recorded_events.csv.gz"),
    )
    table, stats, metadata = prepare_data(
        features, events, min_pairs=config["minimum_pairs_per_day"]
    )
    people = sorted(set(table["participant_id"]))
    hashes = {
        "person_selection_sha256": hashlib.sha256(
            "\n".join(people).encode()
        ).hexdigest(),
        "pair_grid_sha256": hashlib.sha256(
            "\n".join(table["trial_id"]).encode()
        ).hexdigest(),
        "input_features_sha256": hashlib.sha256(
            (source / "features.csv.gz").read_bytes()
        ).hexdigest(),
    }
    if any(hashes[k] != structure[k] for k in hashes):
        raise RuntimeError(
            "Input or pair selection differs from the measured precision plan."
        )
    geometries = _geometry(table, stats, metadata["reference_gap_s"])
    frozen = {
        "status": "frozen_before_empirical_carryover_fits",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": (PROJECT_ROOT / "source_revision.txt").read_text().strip(),
        "scientific_code_sha256": _code_hash(),
        "geometry_inference_code_sha256": _geometry_hash(),
        "feature_manifest": manifest,
        "settings": {**config, "reference_gap_s": metadata["reference_gap_s"]},
        "structure": metadata,
        "input_hashes": hashes,
        "source_files_sha256": input_fingerprints,
        "precision_status": precision["status"],
        "interpretation": (
            "Exploratory study after experiment-01 outcomes were opened; "
            "no independent confirmatory holdout is claimed."
        ),
    }
    _write_json(private / "frozen.json", frozen)
    _write_json(public / "temporal_frozen.json", frozen)
    smoke = _real_smoke(source, private)
    print(
        "Both real-data cold-grid smokes passed; fitting the frozen effects.",
        flush=True,
    )
    estimates, values, sampling, counts = _component_inputs(geometries)
    ratio = np.asarray([stats[p]["sd_a"] / stats[p]["sd_p"] for p in people])
    components = bootstrap_components(
        values,
        sampling,
        counts,
        draws=config["variance_bootstrap_draws"],
        seed=config["seed"],
        scale_ratio=ratio,
    )
    np.savez_compressed(
        private / "estimates.npz",
        values=values,
        sampling=sampling,
        counts=counts,
        scale_ratio=ratio,
        arousal_slopes=np.asarray([e["arousal_slopes"] for e in estimates]),
        eeg_slopes=np.asarray([e["eeg_slopes"] for e in estimates]),
        reml_parameters=np.asarray(components["reml_parameters"]),
        mean_parameters=np.asarray(components["mean_parameters"]),
    )
    prediction = _forecast(table, source, private, frozen["settings"])
    sensitivities = _sensitivities(features, events, table, stats, metadata)
    result = {
        "status": "empirical_carryover_complete",
        "structure": metadata,
        "geometry": [g.summary() for g in geometries],
        "variance_components": components,
        "observed_slope_correlation_descriptive_only": float(
            np.corrcoef(values[: len(people)], values[len(people) :])[0, 1]
        ),
        "sampling_standard_error_quantiles": [
            np.quantile(
                np.sqrt(np.diag(e["sampling_covariance"])), [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            for e in estimates
        ],
        "covariance_numerical_adjustments": [
            e["covariance_eigenvalue_adjustment"] for e in estimates
        ],
        "anonymous_delta_estimates": [e["delta"].tolist() for e in estimates],
        "anonymous_delta_standard_errors": [
            np.sqrt(np.diag(e["sampling_covariance"])).tolist() for e in estimates
        ],
        "anonymous_delta_adjusted_estimates": (
            values
            - _mean_design(counts, True, ratio)
            @ np.asarray(components["mean_parameters"])
        )
        .reshape(2, -1)
        .tolist(),
        "prediction": prediction,
        "sensitivities": sensitivities,
        "real_data_smoke": smoke,
        "frozen_sha256": hashlib.sha256(
            (public / "temporal_frozen.json").read_bytes()
        ).hexdigest(),
        "scientific_code_sha256": _code_hash(),
        "scope": frozen["interpretation"],
    }
    _write_json(public / "temporal_results.json", result)
    from habs.temporal.figures import export_figures

    paths = export_figures(
        result,
        precision,
        json.loads((PROJECT_ROOT / "data/temporal_feasibility.json").read_text()),
        public / "figures",
    )
    result["figures"] = [Path(p).name for p in paths]
    _write_json(public / "temporal_results.json", result)
    print(
        "Frozen carryover study complete; saved reports await read-only audit.",
        flush=True,
    )


if __name__ == "__main__":
    run_study(os.environ.get("HABS_ANALYSIS_OUT", "/out"))
