# %% [markdown]
# # Individual aversive reactivity beyond the stimulus
#
# The analysis separates structural preparation from model selection and the
# final reserved evaluation. Individual measurements stay in private outputs.

# %%
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import skore
from sklearn.metrics import make_scorer, silhouette_score

from habs import PROJECT_ROOT
from habs.analysis import _within_rank, describe_responses
from habs.data import load_table, target_values
from habs.evaluate import SessionGroupedSplit, participant_macro_mae
from habs.features import FEATURE_VERSION, PHYSIOLOGY
from habs.figures import export_figures
from habs.model import INPUT_COLUMNS
from habs.pipeline import build_feature_graph, build_learner
from habs.profiles import PhysiologyProfiles

SEED = 20261108
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _evaluate_report(trials_path, history_path, splitter, parameters):
    learner = build_learner(splitter=splitter, **parameters)
    return skore.evaluate(
        learner,
        data={"trials_path": str(trials_path), "calibration_path": str(history_path)},
        splitter=splitter,
        n_jobs=1,
    )


def _persist_report(workspace, key, report):
    project = skore.Project(
        "habs-aversive-reactivity",
        mode="local",
        workspace=str(Path(workspace).resolve()),
    )
    project.put(key, report)
    return project


def _physical_ids(table):
    return [
        hashlib.sha256(
            f"{r.participant_id}|{int(r.recorded_day)}|{int(r.start_marker)}|{float(r.onset_s):.6f}".encode()
        ).hexdigest()[:24]
        for r in table.itertuples()
    ]


def _prepare(raw_dir, out_dir):
    started = time.monotonic()
    output = out_dir / "data"
    public = out_dir / "public"
    public.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((output / "eda_summary.json").read_text())
    actual_label_hash = hashlib.sha256(
        (raw_dir / "labels_train.csv").read_bytes()
    ).hexdigest()
    if actual_label_hash != evidence["labels_sha256"]:
        raise ValueError("The label release changed after the source audit.")
    allocation = load_table(output / "allocation.csv")
    reconstructed = allocation.sort_values("participant_id").to_dict("records")
    allocation_hash = hashlib.sha256(
        json.dumps(reconstructed, sort_keys=True).encode()
    ).hexdigest()
    if allocation_hash != evidence["allocation_sha256"]:
        raise ValueError("The protected allocation no longer matches the source audit.")
    aligned = load_table(output / "aligned_trials.csv.gz")
    events = load_table(output / "recorded_events.csv.gz")
    aligned = aligned.loc[aligned["cohort"].isin([1, 2, 4])].copy()
    aligned["trial_id"] = _physical_ids(aligned)
    events["trial_id"] = _physical_ids(events)
    chronological = events.sort_values(["participant_id", "recorded_day", "onset_s"])
    first_exposures = set(
        chronological.drop_duplicates(["participant_id", "video"])["trial_id"]
    )
    aligned["evaluation_eligible"] = aligned["trial_id"].isin(first_exposures)
    roles = allocation.set_index("participant_id")["role"].to_dict()
    paired = allocation.set_index("participant_id")["eligible_two_sessions"].to_dict()
    aligned["role"] = aligned["participant_id"].map(roles)
    aligned["eligible_pair"] = aligned["participant_id"].map(paired)
    for cohort in (1, 4):
        cohort_rows = aligned.loc[aligned["cohort"] == cohort]
        counts = cohort_rows.groupby(["participant_id", "recorded_day"]).size()
        eligible = sorted(
            pid
            for pid in set(cohort_rows["participant_id"])
            if counts.get((pid, 1), 0) >= 20 and counts.get((pid, 2), 0) >= 20
        )
        ranked = sorted(
            eligible,
            key=lambda pid: hashlib.sha256(f"{SEED}:{pid}".encode()).hexdigest(),
        )
        reserved = set(ranked[: (len(ranked) + 2) // 3])
        mask = aligned["cohort"] == cohort
        aligned.loc[mask, "eligible_pair"] = aligned.loc[mask, "participant_id"].isin(
            eligible
        )
        aligned.loc[mask, "role"] = aligned.loc[mask, "participant_id"].map(
            lambda pid, reserved=reserved: (
                "final_test" if pid in reserved else "development"
            )
        )
    prepared = output / "signal_input.csv.gz"
    aligned.to_csv(prepared, index=False)
    fingerprint = hashlib.sha256(prepared.read_bytes()).hexdigest()
    feature_path = output / "features.csv.gz"
    manifest_path = output / "feature_manifest.json"
    algorithm_hash = hashlib.sha256(
        (PROJECT_ROOT / "habs/features.py").read_bytes()
        + (PROJECT_ROOT / "habs/signals.py").read_bytes()
    ).hexdigest()
    inventory = [
        (name, (raw_dir / name).stat().st_size, (raw_dir / name).stat().st_mtime_ns)
        for name in sorted(set(aligned["recording"]))
    ]
    inventory_hash = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    reusable = False
    if feature_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        reusable = (
            manifest.get("algorithm_sha256") == algorithm_hash
            and manifest.get("labels_sha256") == actual_label_hash
            and manifest.get("allocation_sha256") == allocation_hash
            and manifest.get("raw_inventory_sha256") == inventory_hash
            and manifest.get("trial_ids_sha256")
            == hashlib.sha256("\n".join(aligned["trial_id"]).encode()).hexdigest()
        )
    if reusable:
        print(
            "Reusing audited features with matching code, release and trial IDs.",
            flush=True,
        )
        cached_features = load_table(feature_path)
        computed_columns = [
            name
            for name in cached_features
            if name == "trial_id" or name not in aligned.columns
        ]
        features = aligned.merge(
            cached_features[computed_columns],
            on="trial_id",
            how="left",
            validate="one_to_one",
        )
    else:
        features = build_feature_graph().skb.eval(
            {
                "aligned_path": prepared,
                "raw_dir": raw_dir,
                "cache_dir": out_dir / "feature_checkpoints",
            }
        )
        features.to_csv(feature_path, index=False)
    if list(features["trial_id"]) != list(aligned["trial_id"]):
        raise ValueError("Signal features do not match the physical trial grid.")
    manifest = {
        "feature_version": FEATURE_VERSION,
        "algorithm_sha256": algorithm_hash,
        "labels_sha256": actual_label_hash,
        "allocation_sha256": allocation_hash,
        "signal_input_sha256": fingerprint,
        "trial_ids_sha256": hashlib.sha256(
            "\n".join(aligned["trial_id"]).encode()
        ).hexdigest(),
        "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "raw_inventory_sha256": inventory_hash,
        "rows": len(features),
        "source_revision": (PROJECT_ROOT / "source_revision.txt").read_text().strip()
        if (PROJECT_ROOT / "source_revision.txt").is_file()
        else "local_worktree",
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "skrub",
                "skore",
                "pyedflib",
            )
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    quality = []
    for cohort, group in features.groupby("cohort"):
        quality.append(
            {
                "cohort": int(cohort),
                "trials": len(group),
                "people": group["participant_id"].nunique(),
                "eeg_usable": int(group["eeg_valid"].sum()),
                "ecg_usable": int(group["ecg_valid"].sum()),
                "eda_usable": int(group["eda_valid"].sum()),
                "all_usable": int(
                    (group["eeg_valid"] & group["ecg_valid"] & group["eda_valid"]).sum()
                ),
                "early_eeg_usable": int(group["early_eeg_valid"].sum()),
                "eda_reasons": {
                    str(k): int(v)
                    for k, v in group["eda_reason"].value_counts().items()
                },
            }
        )
    summary = {
        "status": "signal_preparation_complete",
        "manifest": manifest,
        "elapsed_seconds": time.monotonic() - started,
        "cohorts": quality,
        "reserved_response_values_inspected": False,
    }
    (public / "eda_summary.json").write_text(json.dumps(evidence, indent=2))
    (public / "feature_summary.json").write_text(json.dumps(summary, indent=2))
    print("SIGNAL_QUALITY=" + json.dumps(summary), flush=True)
    return features


def _real_smoke(out_dir):
    environment = {
        **os.environ,
        "HABS_REAL_FEATURES": str(out_dir / "data/features.csv.gz"),
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/smoke/test_01_aversive_reactivity.py",
            "-q",
            "--tb=short",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _write_table(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _person_errors(table, prediction, target="score"):
    prediction = np.asarray(prediction, dtype=float)
    if len(prediction) != len(table) or not np.isfinite(prediction).all():
        raise ValueError("Predictions do not cover the exact evaluation grid.")
    frame = pd.DataFrame(
        {
            "participant_id": table["participant_id"].to_numpy(),
            "absolute_error": np.abs(
                target_values(table, target).to_numpy() - prediction
            ),
        }
    )
    return frame.groupby("participant_id")["absolute_error"].mean()


def _paired_interval(reference, candidate):
    paired = pd.concat(
        [reference.rename("reference"), candidate.rename("candidate")], axis=1
    )
    if paired.isna().any().any() or len(paired) != len(reference):
        raise ValueError("A paired comparison changed the participant population.")
    difference = (paired["reference"] - paired["candidate"]).to_numpy()
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(difference), size=(10000, len(difference)))
    means = difference[sampled].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975])
    return {
        "delta_mae": float(np.mean(difference)),
        "ci95": [float(x) for x in interval],
        "people": len(difference),
        "fraction_helped": float(np.mean(difference > 0)),
        "useful_gain_001_excluded": bool(interval[1] < 0.01),
        "anonymous_reference_errors": paired["reference"].to_list(),
        "anonymous_candidate_errors": paired["candidate"].to_list(),
    }


def _cohort_population(features, cohort, out_dir, *, early=False):
    table = features.loc[features["cohort"] == cohort].copy()
    if early:
        for name in PHYSIOLOGY:
            table[name] = table["early_" + name]
    quality_column = "early_eeg_valid" if early else "eeg_valid"
    history = table.loc[(table["recorded_day"] == 1) & table[quality_column]].copy()
    history_counts = history.groupby("participant_id").size()
    grid = table.loc[table[quality_column] & (table["source"] == "CURATED")].copy()
    grid["observed_arousal"] = (grid["rating_arousal"] - 1) / 8
    grid = grid.loc[grid["participant_id"].map(history_counts).fillna(0) >= 5]
    development = grid.loc[grid["role"] == "development"].reset_index(drop=True)
    future_mask = (
        (grid["recorded_day"] == 2)
        & grid["evaluation_eligible"]
        & grid["eligible_pair"]
    )
    reserved = grid.loc[(grid["role"] == "final_test") & future_mask].reset_index(
        drop=True
    )
    validation = development.loc[
        (development["recorded_day"] == 2)
        & development["evaluation_eligible"]
        & development["eligible_pair"]
    ]
    people = sorted(set(validation["participant_id"]))
    if len(people) < 5 or reserved["participant_id"].nunique() < 5:
        return None
    folder = out_dir / f"cohort_{cohort}" / ("early" if early else "full")
    folder.mkdir(parents=True, exist_ok=True)
    train_path = _write_table(development, folder / "development.csv.gz")
    history_path = _write_table(history, folder / "j1_history.csv.gz")
    reserved_path = _write_table(reserved, folder / "reserved_responses.csv.gz")
    predictors = reserved.drop(
        columns=["rating_valence", "rating_arousal", "aversive_score"]
    )
    _write_table(predictors, folder / "reserved_predictors.csv.gz")
    splitter = SessionGroupedSplit(people)
    folds = []
    for index, (fit, validate) in enumerate(
        splitter.split(
            development[list(INPUT_COLUMNS)], groups=development["participant_id"]
        )
    ):
        fit_frame, val_frame = development.iloc[fit], development.iloc[validate]
        if not set(fit_frame["participant_id"]).isdisjoint(val_frame["participant_id"]):
            raise ValueError("A validation participant leaked into population fitting.")
        folds.append(
            (
                _write_table(fit_frame, folder / f"fold_{index}_train.csv.gz"),
                _write_table(val_frame, folder / f"fold_{index}_validate.csv.gz"),
                val_frame,
            )
        )
    return {
        "development": development,
        "history": history,
        "history_path": history_path,
        "train_path": train_path,
        "reserved_path": reserved_path,
        "test_people": reserved["participant_id"].nunique(),
        "test_trials": len(reserved),
        "splitter": splitter,
        "folds": folds,
        "folder": folder,
        "cohort": cohort,
        "validation_people": len(people),
        "test_grid_sha256": hashlib.sha256(
            "\n".join(reserved["trial_id"]).encode()
        ).hexdigest(),
    }


def _candidate_score(population, parameters):
    errors = []
    for train_path, val_path, validation in population["folds"]:
        learner = build_learner(splitter=population["splitter"], **parameters)
        learner.fit(
            {
                "trials_path": str(train_path),
                "calibration_path": str(population["history_path"]),
            }
        )
        prediction = learner.predict(
            {
                "trials_path": str(val_path),
                "calibration_path": str(population["history_path"]),
            }
        )
        errors.append(
            _person_errors(validation, prediction, parameters.get("target", "score"))
        )
    return float(pd.concat(errors).mean())


def _select(population, candidates, name):
    results = []
    for parameters in candidates:
        score = _candidate_score(population, parameters)
        results.append({"parameters": parameters, "development_macro_mae": score})
    best = min(
        results,
        key=lambda item: (
            item["development_macro_mae"],
            item["parameters"].get("calibrated", True),
            -item["parameters"]["alpha"],
        ),
    )
    print(
        f"Development selection {name}: {best['development_macro_mae']:.6f}", flush=True
    )
    return best["parameters"], {"selected": best, "candidates": results}


def _context_candidates(target="score", conditional=False):
    return [
        {
            "modality": "context",
            "alpha": alpha,
            "target": target,
            "conditional_arousal": conditional,
            "calibrated": calibrated,
            "shrinkage": shrinkage,
        }
        for alpha in ALPHAS
        for calibrated, shrinkage in (
            (False, 0.0),
            (True, 0.0),
            (True, 10.0),
            (True, 40.0),
        )
    ]


def _modality_candidates(reference, modality, **extra):
    return [
        {**reference, "modality": modality, "alpha": alpha, **extra} for alpha in ALPHAS
    ]


def _profile_choice(population):
    ids = set(population["development"]["participant_id"])
    history = population["history"].loc[
        population["history"]["participant_id"].isin(ids)
    ]
    complete = history.loc[np.isfinite(history[list(PHYSIOLOGY)]).all(axis=1)]
    if len(complete) < 200 or complete["participant_id"].nunique() < 15:
        return None, {
            "status": "insufficient_complete_J1_support",
            "trials": len(complete),
            "people": complete["participant_id"].nunique(),
        }
    choices = []
    for count in range(2, 6):
        model = PhysiologyProfiles(count).fit(history)
        standardized = model.scaler_.transform(model._residuals(complete))
        labels = model.clusters_.predict(standardized)
        frequencies = np.asarray([np.mean(labels == i) for i in range(count)])
        silhouette = float(
            silhouette_score(
                standardized,
                labels,
                sample_size=min(1500, len(labels)),
                random_state=SEED,
            )
        )
        choices.append(
            {
                "clusters": count,
                "silhouette": silhouette,
                "minimum_cluster_fraction": float(frequencies.min()),
            }
        )
    supported = [c for c in choices if c["minimum_cluster_fraction"] >= 0.05]
    chosen = max(supported or choices, key=lambda c: (c["silhouette"], -c["clusters"]))
    return chosen["clusters"], {
        "status": "supported",
        "selection": chosen,
        "candidates": choices,
        "complete_J1_trials": len(complete),
        "complete_J1_people": complete["participant_id"].nunique(),
    }


def _select_models(population, early_population):
    selected, development = {}, {}
    selected["M1"], development["M1"] = _select(population, _context_candidates(), "M1")
    uncalibrated = [
        item
        for item in development["M1"]["candidates"]
        if not item["parameters"]["calibrated"]
    ]
    m0 = min(uncalibrated, key=lambda item: item["development_macro_mae"])
    selected["M0"] = m0["parameters"]
    for name, modality in (
        ("M2", "eeg"),
        ("M3", "peripheral"),
        ("M4", "all"),
        ("missingness", "missing"),
        ("no_EDA", "no_eda"),
        ("history_only", "history"),
    ):
        selected[name], development[name] = _select(
            population, _modality_candidates(selected["M1"], modality), name
        )
    selected["nonlinear"], development["nonlinear"] = _select(
        population,
        _modality_candidates(selected["M1"], "all", nonlinear=True),
        "nonlinear",
    )
    clusters, profile_selection = _profile_choice(population)
    if clusters is not None:
        for name, kind in (("M5", "cluster"), ("continuous_profile", "pca")):
            selected[name], development[name] = _select(
                population,
                _modality_candidates(
                    selected["M1"], "all", profile=kind, n_clusters=clusters
                ),
                name,
            )
    sensitivities = {}
    for target, conditional in (
        ("valence", False),
        ("arousal", False),
        ("alternative", False),
        ("negative", True),
    ):
        reference, record = _select(
            population, _context_candidates(target, conditional), target + "_reference"
        )
        physiology, physiology_record = _select(
            population, _modality_candidates(reference, "all"), target + "_physiology"
        )
        sensitivities[target] = {
            "reference": reference,
            "physiology": physiology,
            "development": {"reference": record, "physiology": physiology_record},
        }
    early_reference, early_reference_record = _select(
        early_population, _context_candidates(), "early_M1"
    )
    early, early_record = _select(
        early_population, _modality_candidates(early_reference, "all"), "early_M4"
    )
    return (
        selected,
        development,
        sensitivities,
        {
            "reference": early_reference,
            "physiology": early,
            "development": {
                "reference": early_reference_record,
                "physiology": early_record,
            },
        },
        profile_selection,
    )


def _final_report(population, parameters):
    frozen_path = population["folder"].parents[1] / "frozen.json"
    if (
        not frozen_path.is_file()
        or json.loads(frozen_path.read_text()).get("status")
        != "frozen_before_reserved_scoring"
    ):
        raise RuntimeError("Reserved evaluation requires a completed settings freeze.")
    learner = build_learner(splitter=population["splitter"], **parameters)
    learner.fit(
        {
            "trials_path": str(population["train_path"]),
            "calibration_path": str(population["history_path"]),
        }
    )
    report = skore.evaluate(
        learner,
        data={
            "trials_path": str(population["reserved_path"]),
            "calibration_path": str(population["history_path"]),
        },
        splitter="prefit",
    )
    people = tuple(load_table(population["reserved_path"])["participant_id"])
    report.metrics.add(
        make_scorer(
            participant_macro_mae,
            greater_is_better=False,
            response_method="predict",
            groups=people,
        ),
        name="participant_macro_mae",
    )
    return report


def _evaluate_models(population, parameters, project, prefix):
    test = load_table(population["reserved_path"])
    errors, reports, model_results = {}, {}, {}
    for name, configuration in parameters.items():
        report = _final_report(population, configuration)
        prediction = np.asarray(report.get_predictions(data_source="test"))
        errors[name] = _person_errors(
            test, prediction, configuration.get("target", "score")
        )
        registered = float(report.metrics.get("participant_macro_mae"))
        if abs(registered - float(errors[name].mean())) > 1e-10:
            raise ValueError("Registered and paired participant MAE disagree.")
        model_results[name] = {
            "macro_mae": float(errors[name].mean()),
            "anonymous_person_errors": errors[name].to_list(),
            "people": len(errors[name]),
            "trials": len(test),
        }
        project.put(prefix + "__" + name, report)
        reports[name] = report
    response = target_values(population["development"]).to_numpy()
    for name, value in (
        ("constant_zero", 0.0),
        ("population_mean", float(np.mean(response))),
        ("population_median", float(np.median(response))),
    ):
        errors[name] = _person_errors(test, np.full(len(test), value))
        model_results[name] = {
            "macro_mae": float(errors[name].mean()),
            "anonymous_person_errors": errors[name].to_list(),
            "people": len(errors[name]),
            "trials": len(test),
        }
    comparisons = {}
    for reference, candidate in (
        ("M0", "M1"),
        ("M1", "M4"),
        ("missingness", "M4"),
        ("history_only", "M4"),
        ("no_EDA", "M4"),
        ("M4", "nonlinear"),
        ("M4", "M5"),
        ("continuous_profile", "M5"),
    ):
        if reference in errors and candidate in errors:
            comparisons[reference + "_minus_" + candidate] = _paired_interval(
                errors[reference], errors[candidate]
            )
    complete_mask = np.isfinite(test[list(PHYSIOLOGY)]).all(axis=1).to_numpy()
    complete_contrast = None
    if complete_mask.any() and "M1" in reports and "M4" in reports:
        subgroup = test.loc[complete_mask]
        if subgroup["participant_id"].nunique() >= 5:
            complete_contrast = _paired_interval(
                _person_errors(
                    subgroup,
                    reports["M1"].get_predictions(data_source="test")[complete_mask],
                ),
                _person_errors(
                    subgroup,
                    reports["M4"].get_predictions(data_source="test")[complete_mask],
                ),
            )
            complete_contrast["trials"] = len(subgroup)
    return (
        {
            "models": model_results,
            "comparisons": comparisons,
            "training_people": population["development"]["participant_id"].nunique(),
            "training_trials": len(population["development"]),
            "test_people": population["test_people"],
            "test_trials": population["test_trials"],
            "validation_people": population["validation_people"],
            "complete_observation_subgroup": complete_contrast,
            "unseen_population_video_trials": int(
                (~test["video"].isin(set(population["development"]["video"]))).sum()
            ),
        },
        reports,
        errors,
    )


def _run_study(features, out_dir):
    population = _cohort_population(features, 2, out_dir)
    early_population = _cohort_population(features, 2, out_dir, early=True)
    if population is None or early_population is None:
        raise ValueError(
            "The primary cohort cannot support the prespecified comparison."
        )
    selected, development, sensitivities, early, profile_selection = _select_models(
        population, early_population
    )
    source_files = sorted((PROJECT_ROOT / "habs").glob("*.py")) + [
        Path(__file__),
        PROJECT_ROOT / "run_config.json",
    ]
    code_hash = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in source_files)
    ).hexdigest()
    development_score = target_values(population["development"]).to_numpy()
    positive_development_score = development_score[development_score > 0]
    extreme_threshold = (
        float(np.quantile(positive_development_score, 0.75))
        if len(positive_development_score) >= 200
        else None
    )
    frozen = {
        "seed": SEED,
        "selected": selected,
        "development": development,
        "sensitivities": sensitivities,
        "early": early,
        "profiles": profile_selection,
        "test_grid_sha256": population["test_grid_sha256"],
        "code_sha256": code_hash,
        "minimum_useful_gain": 0.01,
        "feature_manifest": json.loads(
            (out_dir / "data/feature_manifest.json").read_text()
        ),
        "status": "frozen_before_reserved_scoring",
        "descriptive_markers": ["eeg_frontal_alpha", "ecg_hr", "eda_log_level"],
        "specificity_matching": {
            "negative_valence_max": 4,
            "positive_valence_min": 6,
            "arousal_min": 6,
            "arousal_caliper": 0.5,
        },
        "extreme_positive_quantile": 0.75,
        "extreme_high_threshold": extreme_threshold,
        "permutation_control": {
            "replicates": 100,
            "within": "participant and session",
            "joint_vector": True,
            "minimum_shift_when_possible": 5,
        },
    }
    (out_dir / "frozen.json").write_text(json.dumps(frozen, indent=2))
    (out_dir / "public/frozen.json").write_text(json.dumps(frozen, indent=2))
    print("FROZEN_BEFORE_RESERVED_SCORING=" + code_hash, flush=True)
    project = skore.Project(
        "habs-aversive-reactivity",
        mode="local",
        workspace=str((out_dir / "reports").resolve()),
    )
    # Selected development reports are persisted before the first reserved result.
    for name, configuration in selected.items():
        report = _evaluate_report(
            population["train_path"],
            population["history_path"],
            population["splitter"],
            configuration,
        )
        project.put("01_aversive_reactivity__development__" + name, report)
    primary, reports, errors = _evaluate_models(
        population, selected, project, "01_aversive_reactivity__cohort2"
    )
    secondary = {}
    test = load_table(population["reserved_path"])
    for target, configuration in sensitivities.items():
        pair = {}
        for name in ("reference", "physiology"):
            report = _final_report(population, configuration[name])
            project.put("01_aversive_reactivity__" + target + "__" + name, report)
            pair[name] = _person_errors(
                test, report.get_predictions(data_source="test"), target
            )
        secondary[target] = {
            "reference_mae": float(pair["reference"].mean()),
            "physiology_mae": float(pair["physiology"].mean()),
            "contrast": _paired_interval(pair["reference"], pair["physiology"]),
            "interpretation": "conditional explanatory analysis"
            if target == "negative"
            else "target sensitivity",
        }
    early_errors = {}
    for name in ("reference", "physiology"):
        early_report = _final_report(early_population, early[name])
        project.put("01_aversive_reactivity__early__" + name, early_report)
        early_errors[name] = _person_errors(
            load_table(early_population["reserved_path"]),
            early_report.get_predictions(data_source="test"),
        )
    early_result = {
        "reference_mae": float(early_errors["reference"].mean()),
        "physiology_mae": float(early_errors["physiology"].mean()),
        "contrast_vs_own_M1": _paired_interval(
            early_errors["reference"], early_errors["physiology"]
        ),
        "test_people": early_population["test_people"],
        "test_trials": early_population["test_trials"],
        "scope": (
            "Five-second signal eligibility and measurements; separately selected "
            "development settings. Later self-report, not stress onset."
        ),
    }
    replications = {}
    for cohort in (1, 4):
        replicate = _cohort_population(features, cohort, out_dir)
        if replicate is None:
            replications[str(cohort)] = {"status": "insufficient_usable_paired_support"}
            continue
        result, _, _ = _evaluate_models(
            replicate, selected, project, f"01_aversive_reactivity__cohort{cohort}"
        )
        replications[str(cohort)] = {
            "status": "exploratory_fixed_cohort2_settings",
            **result,
        }
    descriptions, structure = describe_responses(
        population["development"],
        population["history"],
        test,
        n_clusters=profile_selection.get("selection", {}).get("clusters"),
    )
    extremes = _extreme_gradient(population, selected, project, extreme_threshold)
    permutations = _shift_control(population, reports["M4"], errors["M1"])
    result = {
        "status": "reserved_evaluation_complete",
        "primary": primary,
        "sensitivities": secondary,
        "early": early_result,
        "replications": replications,
        "profiles": profile_selection,
        "descriptions": descriptions,
        "extreme_gradient": extremes,
        "shift_control": permutations,
        "code_sha256": code_hash,
        "frozen_sha256": hashlib.sha256(
            (out_dir / "frozen.json").read_bytes()
        ).hexdigest(),
        "warning": (
            "Experimental aversive-activation proxy, not validated stress. "
            "Secondary intervals are exploratory."
        ),
    }
    (out_dir / "public/study_summary.json").write_text(json.dumps(result, indent=2))
    figure_paths = export_figures(
        features, result, structure, out_dir / "public/figures"
    )
    print(f"Exported {len(figure_paths)} empirical figures.", flush=True)
    print(
        "PRIMARY_RESULT=" + json.dumps(primary["comparisons"]["M1_minus_M4"]),
        flush=True,
    )
    return result, reports, population, structure


def _extreme_gradient(population, selected, project, threshold):
    development = population["development"]
    score = target_values(development).to_numpy()
    if threshold is None:
        return {"status": "insufficient_development_support"}
    extreme = (score == 0) | (score >= threshold)
    train_path = _write_table(
        development.loc[extreme], population["folder"] / "extreme_train.csv.gz"
    )
    test = load_table(population["reserved_path"])
    test_score = target_values(test).to_numpy()
    middle = (test_score > 0) & (test_score < threshold)
    extreme_test = (test_score == 0) | (test_score >= threshold)
    if not extreme_test.any():
        return {"status": "no_reserved_extreme_trials", "high_threshold": threshold}
    models = {}
    for name in ("M0", "M4"):
        configuration = {
            **selected[name],
            "target": f"extreme:{threshold:.17g}",
            "calibrated": False,
        }
        changed = {
            **population,
            "train_path": train_path,
            "reserved_path": _write_table(
                test.loc[extreme_test], population["folder"] / "extreme_test.csv.gz"
            ),
        }
        report = _final_report(changed, configuration)
        project.put("01_aversive_reactivity__extreme__" + name, report)
        middle_path = _write_table(
            test.loc[middle], population["folder"] / "middle_prediction.csv.gz"
        )
        if middle.sum():
            prediction = report.estimator_.predict(
                {
                    "trials_path": str(middle_path),
                    "calibration_path": str(population["history_path"]),
                }
            )
            models[name] = {
                "within_person_rank_on_middle": _within_rank(
                    test.loc[middle], prediction
                )
            }
    return {
        "status": "evaluated",
        "high_threshold": threshold,
        "supervised_training_trials": int(extreme.sum()),
        "middle_test_trials": int(middle.sum()),
        "models": models,
        "scope": "binary-extreme ridge scores are not calibrated stress quantities",
    }


def _shift_control(population, report, reference_errors):
    test = load_table(population["reserved_path"])
    rng = np.random.default_rng(SEED)
    performances = []
    path = population["folder"] / "shift_control.csv.gz"
    for _ in range(100):
        shifted = test.copy()
        for _, group in test.groupby("participant_id"):
            count = len(group)
            if count < 2:
                continue
            offset = (
                int(rng.integers(5, count - 4))
                if count >= 10
                else int(rng.integers(1, count))
            )
            shifted.loc[group.index, list(PHYSIOLOGY)] = np.roll(
                group[list(PHYSIOLOGY)].to_numpy(), offset, axis=0
            )
        _write_table(shifted, path)
        prediction = report.estimator_.predict(
            {
                "trials_path": str(path),
                "calibration_path": str(population["history_path"]),
            }
        )
        performances.append(float(_person_errors(test, prediction).mean()))
    gain = float(reference_errors.mean()) - np.asarray(performances)
    return {
        "status": "diagnostic_not_exact_randomization_test",
        "replicates": 100,
        "shifted_macro_mae_mean": float(np.mean(performances)),
        "gain_vs_M1_mean": float(np.mean(gain)),
        "gain_reference95": [float(v) for v in np.quantile(gain, [0.025, 0.975])],
    }


def main():
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    config = json.loads((PROJECT_ROOT / "run_config.json").read_text())
    features = _prepare(raw_dir, out_dir)
    _real_smoke(out_dir)
    if config["stage"] == "prepare":
        print(
            "Real-data smoke passed. Reserved outcome evaluation has not run.",
            flush=True,
        )
        return
    if config["stage"] == "study":
        _run_study(features, out_dir)
        return
    raise ValueError("Unknown analysis stage.")


if __name__ == "__main__":
    main()
