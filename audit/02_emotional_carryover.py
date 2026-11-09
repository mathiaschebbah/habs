# %% [markdown]
# # Saved carryover estimates and grouped J2 forecasts
#
# Read the completed source/freeze evidence and cached reports at their original
# absolute workspace path. No model is fitted and no report is overwritten.

# %%
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import skore
from scipy.linalg import cho_factor, cho_solve

from habs.evaluate import participant_macro_mae
from habs.temporal.data import verify_inputs
from habs.temporal.inference import _marginal, _mean_design

ROOT = Path(os.environ.get("HABS_ANALYSIS_OUT", "/out"))
audit_state = {
    "source": False,
    "likelihood": False,
    "predictions": False,
    "paired": False,
}
PUBLIC = ROOT / "public"
PRIVATE = ROOT / "temporal"
result = json.loads((PUBLIC / "temporal_results.json").read_text())
frozen = json.loads((PUBLIC / "temporal_frozen.json").read_text())
producer = importlib.import_module("experiments.02_emotional_carryover")
verify_inputs(ROOT / "data")
for name, expected in frozen["source_files_sha256"].items():
    assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected
assert (
    hashlib.sha256((ROOT / "data/features.csv.gz").read_bytes()).hexdigest()
    == frozen["input_hashes"]["input_features_sha256"]
)
assert frozen["status"] == "frozen_before_empirical_carryover_fits"
assert result["status"] == "empirical_carryover_complete"
assert (
    result["frozen_sha256"]
    == hashlib.sha256((PUBLIC / "temporal_frozen.json").read_bytes()).hexdigest()
)
assert (
    result["scientific_code_sha256"]
    == frozen["scientific_code_sha256"]
    == producer._code_hash()
)
audit_state["source"] = True
{
    "source_revision": frozen["source_revision"],
    "frozen_at_utc": frozen["frozen_at_utc"],
    "source_and_freeze_verified": True,
    "audited_result_sha256": hashlib.sha256(
        (PUBLIC / "temporal_results.json").read_bytes()
    ).hexdigest(),
    "audited_frozen_sha256": hashlib.sha256(
        (PUBLIC / "temporal_frozen.json").read_bytes()
    ).hexdigest(),
    "structure": result["structure"],
    "real_data_smoke": result["real_data_smoke"],
}

# %% [markdown]
# ## Likelihood and sampling-covariance reconstruction

# %%
saved = np.load(PRIVATE / "estimates.npz")
values, sampling, counts, ratio = (
    saved[name] for name in ("values", "sampling", "counts", "scale_ratio")
)
parameters, mean = saved["reml_parameters"], saved["mean_parameters"]
matrix, sd, rho, covariance = _marginal(parameters, sampling, None)
design = _mean_design(counts, True, ratio)
factor = cho_factor(matrix, lower=True)
information = design.T @ cho_solve(factor, design)
mean_factor = cho_factor(information, lower=True)
residual = values - design @ mean
objective = float(
    np.sum(np.log(np.diag(factor[0])))
    + np.sum(np.log(np.diag(mean_factor[0])))
    + 0.5 * residual @ cho_solve(factor, residual)
)
assert abs(objective - result["variance_components"]["reml_objective"]) < 1e-8
assert abs(covariance - result["variance_components"]["covariance"]) < 1e-12
assert (
    np.max(
        np.abs(values.reshape(2, -1) - np.asarray(result["anonymous_delta_estimates"]))
    )
    < 1e-12
)
assert np.linalg.eigvalsh(sampling).min() > 0
audit_state["likelihood"] = True
{
    "saved_likelihood_verified": True,
    "covariance": covariance,
    "covariance_ci95": result["variance_components"]["covariance_ci95"],
    "null_bootstrap_p_two_sided": result["variance_components"][
        "null_bootstrap_p_two_sided"
    ],
    "random_effect_sd": sd.tolist(),
    "rho_point": rho,
    "rho_profile95_grid": result["variance_components"]["rho_profile95_grid"],
    "correlation_weakly_identified": result["variance_components"][
        "correlation_weakly_identified"
    ],
}

# %% [markdown]
# ## Stored skore reports

# %%
report_workspace = PRIVATE / "reports"
project = skore.Project(
    "habs-emotional-carryover", mode="local", workspace=str(report_workspace.resolve())
)
summary = project.summarize()
summary.frame()

# %%
frame = summary.frame().reset_index()
id_column = "id" if "id" in frame else "report_id"
verified, checks, metrics = {}, {}, {}
for mode in ("ablation", "full", "refit"):
    rows = frame.loc[frame["key"] == "02_emotional_carryover__" + mode]
    assert len(rows) == 1
    report = project.get(str(rows.iloc[0][id_column]))
    grid, target = report.X, np.asarray(report.y)
    prediction = producer._oof_predictions(report, len(grid))
    groups = grid["participant_id"].to_numpy()
    people = sorted(set(groups))
    errors = producer._person_errors(grid, target, prediction, people)
    assert (
        np.max(
            np.abs(
                errors - result["prediction"]["models"][mode]["anonymous_person_errors"]
            )
        )
        < 1e-10
    )
    for child, (_, test) in zip(report.reports_, report.split_indices, strict=True):
        registered = float(child.metrics.get("participant_macro_mae"))
        reconstructed = participant_macro_mae(
            target[test], prediction[test], groups=groups[test]
        )
        assert abs(registered - reconstructed) < 1e-10
    verified[mode] = {
        "rows": len(grid),
        "people": len(people),
        "all_finite": bool(np.isfinite(prediction).all()),
        "participant_macro_mae": float(errors.mean()),
    }
    metrics[mode] = report.metrics.summarize().frame()
    try:
        checks[mode] = report.checks.summarize().frame()
    except (ValueError, TypeError, AttributeError, FileNotFoundError) as error:
        checks[mode] = {"status": "generic_checks_unavailable", "reason": str(error)}
audit_state["predictions"] = set(verified) == {"ablation", "full", "refit"}
verified

# %% [markdown]
# ## Metrics summary

# %%
metrics

# %% [markdown]
# ## Checks summary

# %%
checks

# %% [markdown]
# ## Paired forecast intervals and prespecified sensitivities

# %%
for name, reference in (
    ("full_vs_ablation", "ablation"),
    ("full_vs_refitted_restriction", "refit"),
):
    a = np.asarray(result["prediction"]["models"][reference]["anonymous_person_errors"])
    b = np.asarray(result["prediction"]["models"]["full"]["anonymous_person_errors"])
    actual = producer._paired(
        a,
        b,
        frozen["settings"]["seed"],
        frozen["settings"]["predictive_bootstrap_draws"],
    )
    expected = result["prediction"][name]
    assert abs(actual["gain"] - expected["gain"]) < 1e-12
    assert np.max(np.abs(np.asarray(actual["ci95"]) - expected["ci95"])) < 1e-12
audit_state["paired"] = True
assert all(audit_state.values())
completion = {
    "status": "complete_verified",
    "primary_verified": all(audit_state.values()),
    "verification_stages": audit_state,
    "audited_result_sha256": hashlib.sha256(
        (PUBLIC / "temporal_results.json").read_bytes()
    ).hexdigest(),
    "audited_frozen_sha256": hashlib.sha256(
        (PUBLIC / "temporal_frozen.json").read_bytes()
    ).hexdigest(),
}
completion_path = Path("scratch/audit/02_emotional_carryover/completion.json")
completion_path.parent.mkdir(parents=True, exist_ok=True)
completion_path.write_text(json.dumps(completion, indent=2))
{
    **completion,
    "prediction": result["prediction"],
    "sensitivities": result["sensitivities"],
    "figure_count": len(list((PUBLIC / "figures").glob("figure*.png"))),
    "scope": result["scope"],
}
