# %% [markdown]
# # Reserved aversive-reactivity reports
#
# Read the saved cohort-2 predictions and metrics without training or modifying
# the reports. The output tree is restored at its original absolute location.

# %%
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skore

from habs import PROJECT_ROOT
from habs.model import INPUT_COLUMNS

out_dir = Path(os.environ["HABS_ANALYSIS_OUT"]).resolve()
project = skore.Project(
    "habs-aversive-reactivity",
    mode="local",
    workspace=str((out_dir / "reports").resolve()),
)
summary = project.summarize().frame().reset_index()
summary[["id", "key", "report_type"]]

# %% [markdown]
# ## Input and parameter provenance

# %%
frozen = json.loads((out_dir / "frozen.json").read_text())
evidence = json.loads((out_dir / "public/study_summary.json").read_text())
source_files = sorted((PROJECT_ROOT / "habs").glob("*.py")) + [
    PROJECT_ROOT / "experiments/01_aversive_reactivity.py",
    PROJECT_ROOT / "run_config.json",
]
current_code = hashlib.sha256(
    b"".join(p.name.encode() + p.read_bytes() for p in source_files)
).hexdigest()
assert current_code == frozen["code_sha256"]
assert (
    hashlib.sha256((out_dir / "frozen.json").read_bytes()).hexdigest()
    == evidence["frozen_sha256"]
)
assert frozen["status"] == "frozen_before_reserved_scoring"
assert not frozen["selected"]["M4"].get("conditional_arousal", False)
assert not {
    "rating_valence",
    "rating_arousal",
    "final_emotion_label",
    "aversive_score",
}.intersection(INPUT_COLUMNS)
{
    "source_matches_frozen": True,
    "freeze_digest_matches": True,
    "prospective_response_columns_excluded": True,
    "allocation_sha256": frozen["feature_manifest"]["allocation_sha256"],
    "labels_sha256": frozen["feature_manifest"]["labels_sha256"],
    "test_grid_sha256": frozen["test_grid_sha256"],
}

# %% [markdown]
# ## Reloaded participant-macro metrics

# %%
reports = {}
verified = []
for name in frozen["selected"]:
    key = "01_aversive_reactivity__cohort2__" + name
    matching = summary.loc[summary["key"] == key]
    if len(matching) == 0:
        raise KeyError(f"Missing saved report for {key}")
    report = project.get(str(matching.iloc[0]["id"]))
    prediction = np.asarray(report.get_predictions(data_source="test"))
    assert np.isfinite(prediction).all()
    assert len(prediction) == evidence["primary"]["test_trials"]
    macro = float(report.metrics.get("participant_macro_mae"))
    expected = evidence["primary"]["models"][name]["macro_mae"]
    assert abs(macro - expected) < 1e-10
    reports[name] = report
    verified.append(
        {
            "model": name,
            "participant_macro_mae": macro,
            "predictions": len(prediction),
            "matches_export": True,
        }
    )
pd.DataFrame(verified)

# %% [markdown]
# ## Checks summary

# %%
reports["M4"].checks.summarize().frame()

# %% [markdown]
# ## Metrics summary

# %%
reports["M4"].metrics.summarize().frame()

# %% [markdown]
# ## Paired uncertainty and interpretation boundaries

# %%
contrast = evidence["primary"]["comparisons"]["M1_minus_M4"]
reference = np.asarray(contrast["anonymous_reference_errors"], dtype=float)
candidate = np.asarray(contrast["anonymous_candidate_errors"], dtype=float)
assert len(reference) == evidence["primary"]["test_people"]
assert len(candidate) == len(reference)
assert np.isfinite(reference).all() and np.isfinite(candidate).all()
assert (
    abs(reference.mean() - reports["M1"].metrics.get("participant_macro_mae")) < 1e-10
)
assert (
    abs(candidate.mean() - reports["M4"].metrics.get("participant_macro_mae")) < 1e-10
)
difference = reference - candidate
rng = np.random.default_rng(frozen["seed"])
means = difference[
    rng.integers(0, len(difference), size=(10000, len(difference)))
].mean(axis=1)
interval = np.quantile(means, [0.025, 0.975])
assert abs(difference.mean() - contrast["delta_mae"]) < 1e-12
assert np.max(np.abs(interval - np.asarray(contrast["ci95"]))) < 1e-12
{
    "primary_verified": True,
    "people": len(difference),
    "reference_macro_mae": float(reference.mean()),
    "multimodal_macro_mae": float(candidate.mean()),
    "paired_gain": float(difference.mean()),
    "ci95": interval.tolist(),
    "scope": (
        "Internal released-identifier hold-out, fixed stimulus catalogue, "
        "experimental proxy."
    ),
}
