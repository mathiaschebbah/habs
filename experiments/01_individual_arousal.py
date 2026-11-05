# %%
"""Evaluate calibrated arousal prediction on disjoint participants and sessions."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from model import CalibratedArousal, InsufficientSupport, build_learner
from study import (
    effect_summary,
    paired_people,
    participant_scores,
    session_scenario,
    split_study,
    tune,
)
from tests.smoke.test_01_individual_arousal import verify_prediction_grid


def finite_mean(values):
    """Represent an undefined aggregate explicitly rather than as zero."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if len(valid) else None


def run(
    feature_file,
    out_dir,
    stage="development",
    cohorts=(2, 1, 4, 3),
    minimum_sources=10,
    minimum=20,
):
    """Tune only on development people, then optionally evaluate frozen models.

    Parameters
    ----------
    feature_file : pathlib.Path
        Private reconciled feature table.
    out_dir : pathlib.Path
        Private directory for reports, manifests and predictions.
    stage : str
        'development' leaves final test performance unevaluated.
    cohorts : tuple of int
        Independently analyzed acquisition cohorts.
    minimum_sources : int
        Minimum source participants per known video.
    minimum : int
        Minimum trials in each selected session.

    Returns
    -------
    dict
        Aggregate results suitable for the scientific report.
    """
    feature_file, out_dir = Path(feature_file), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(feature_file)
    frame = frame.loc[frame.rating_arousal.between(1, 9)].copy()
    environment = {
        package: importlib.metadata.version(package)
        for package in (
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "skrub",
            "skore",
            "pyedflib",
        )
    }
    result = {
        "stage": stage,
        "cohorts": {},
        "python": platform.python_version(),
        "environment": environment,
        "feature_sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        "minimum_sources": minimum_sources,
        "minimum_trials": minimum,
    }
    with threadpool_limits(limits=1):
        for cohort in cohorts:
            case_dir = out_dir / f"cohort_{cohort}"
            case_dir.mkdir(exist_ok=True)
            data = session_scenario(frame.loc[frame.cohort == cohort], minimum)
            case = {
                "annotated_people": int(data.participant_id.nunique()),
                "paired_before_support": len(paired_people(data, minimum=minimum)),
            }
            result["cohorts"][str(cohort)] = case
            try:
                split = split_study(
                    data, minimum_sources=minimum_sources, minimum=minimum
                )
                scenario_file = case_dir / "scenario.csv.gz"
                data.to_csv(
                    scenario_file,
                    index=False,
                    compression={"method": "gzip", "mtime": 0},
                )
                smoke = verify_prediction_grid(
                    scenario_file,
                    data,
                    split,
                    case_dir,
                    minimum_sources=minimum_sources,
                    minimum=minimum,
                )
                parameters, signal_penalties, tuning = tune(
                    data, split, minimum_sources=minimum_sources, minimum=minimum
                )
            except InsufficientSupport as error:
                case.update(status="insufficient_support", reason=str(error))
                print(f"Cohort {cohort}: insufficient prespecified support", flush=True)
                (out_dir / "results.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                print(
                    "HABS_COHORT="
                    + json.dumps({"cohort": cohort, **case}, allow_nan=False),
                    flush=True,
                )
                continue
            case.update(
                status="development_complete",
                parameters=parameters,
                signal_penalties=signal_penalties,
                source_only_people=int(split["background"].participant_id.nunique()),
                development_people=len(split["train_people"]),
                test_people=int(split["test_grid"].participant_id.nunique()),
                development_trials=len(split["train_grid"]),
                test_trials=len(split["test_grid"]),
                supported_videos=len(split["videos"]),
                smoke_test=smoke,
            )
            pd.DataFrame(tuning).to_csv(
                case_dir / "development_scores.csv", index=False
            )
            roles = [
                {"participant_id": person, "role": role}
                for role, people in (
                    ("development", split["assigned_train"]),
                    ("test", split["assigned_test"]),
                )
                for person in people
            ]
            pd.DataFrame(roles).to_csv(case_dir / "allocation.csv", index=False)
            common = {
                "feature_file": str(scenario_file),
                "videos": split["videos"],
                "paired_people": split["paired_people"],
            }
            train_env = {**common, "people": split["train_people"]}
            test_env = {**common, "people": split["assigned_test"]}
            frozen = {
                "parameters": parameters,
                "signal_penalties": signal_penalties,
                "train": split["train_people"],
                "test": split["assigned_test"],
                "videos": split["videos"],
            }
            frozen_text = json.dumps(frozen, sort_keys=True)
            (case_dir / "frozen_plan.json").write_text(frozen_text + "\n")
            case["frozen_plan_sha256"] = hashlib.sha256(
                frozen_text.encode()
            ).hexdigest()
            if stage != "final":
                (out_dir / "results.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                print(
                    "HABS_COHORT="
                    + json.dumps({"cohort": cohort, **case}, allow_nan=False),
                    flush=True,
                )
                continue
            reports, scores = {}, {}
            predictions = split["test_grid"][
                ["participant_id", "video", "start_marker", "rating_arousal"]
            ].copy()
            case["models"] = {}
            for mode in (
                "video_only",
                "level",
                "affine",
                "eda",
                "eeg",
                "ecg",
                "all",
                "video",
                "missing",
                "person",
            ):
                estimator = CalibratedArousal(
                    mode=mode,
                    signal_alpha=signal_penalties.get(mode, 10.0),
                    **parameters,
                )
                learner = build_learner(estimator, GroupKFold(5), minimum=minimum)
                report = skore.EstimatorReport(
                    learner, train_data=train_env, test_data=test_env
                )
                prediction = np.asarray(report.get_predictions(data_source="test"))
                if (
                    len(prediction) != len(predictions)
                    or not np.isfinite(prediction).all()
                ):
                    raise RuntimeError("Prediction grid was not preserved")
                predictions[mode] = prediction
                scores[mode] = participant_scores(
                    predictions.rating_arousal, prediction, predictions.participant_id
                )
                scores[mode].to_csv(case_dir / f"scores_{mode}.csv", index=False)
                case["models"][mode] = {
                    metric: finite_mean(scores[mode][metric])
                    for metric in ("centered_mse", "mae", "spearman", "bias")
                }
                reports[mode] = report
            case["effects_vs_affine"] = {
                mode: effect_summary(scores["affine"], scores[mode])
                for mode in scores
                if mode != "affine"
            }
            case["individual_signal_controls"] = {
                mode: effect_summary(scores[mode], scores["all"])
                for mode in ("video", "missing", "person")
            }
            predictions.to_csv(
                case_dir / "predictions.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )
            if cohort == 2:
                project = skore.Project(
                    "habs-arousal", mode="local", workspace=out_dir / "reports"
                )
                project.put("01_individual_arousal", reports["all"])
            case["status"] = "complete"
            print(f"Cohort {cohort}: frozen evaluation completed", flush=True)
            (out_dir / "results.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        "HABS_STUDY=" + json.dumps(result, separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    return result


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_file", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--stage", choices=("development", "final"), default="development"
    )
    arguments = parser.parse_args()
    run(arguments.feature_file, arguments.out_dir, stage=arguments.stage)
