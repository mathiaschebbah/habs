"""Replay the learning procedure under explicit synthetic response mechanisms."""

import hashlib
import importlib.metadata
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t
from threadpoolctl import threadpool_limits

from model import CalibratedArousal, VideoReference, observed_features, population
from study import (
    SEED,
    effect_summary,
    participant_scores,
    session_scenario,
    split_study,
    tune,
)

SCENARIOS = ("context", "response_style", "shared_stimulus", "individual_signal")
RECORD_COLUMNS = (
    ["scenario", "repetition", "seed", "test_people", "seconds"]
    + [
        f"{statistic}_{model}"
        for model in ("affine", "video", "missing")
        for statistic in ("gain", "low", "high")
    ]
    + ["positive_vs_reference", "positive_after_controls"]
)
_STATE = {}


def interval_calibration(baseline, prediction, people, order, replications=1000):
    """Check interval coverage at an exact zero-gain boundary, without real outcomes.

    Synthetic outcomes lie at the midpoint of the two centered predictions plus
    zero-mean noise. The true centered-error difference is therefore zero.
    This checks conditional inference, not the realism of an emotion generator.
    """
    baseline, prediction = np.asarray(baseline), np.asarray(prediction)
    people, order = np.asarray(people), np.asarray(order)
    groups = np.unique(people)
    indices = np.random.default_rng(SEED).choice(
        len(groups), (5000, len(groups)), replace=True
    )
    result = {
        "replications": replications,
        "people": len(groups),
        "nominal_coverage": 0.95,
        "nominal_positive_tail": 0.025,
        "noise": {},
    }
    for rho in (0.0, 0.4):
        rng = np.random.default_rng(SEED + round(100 * rho))
        differences = np.zeros((replications, len(groups)))
        for column, person in enumerate(groups):
            positions = np.flatnonzero(people == person)
            positions = positions[np.argsort(order[positions])]
            q = prediction[positions] - baseline[positions]
            q = q - q.mean()
            errors = rng.normal(size=(replications, len(positions)))
            errors *= np.exp(rng.normal(0, 0.25, size=(replications, 1)))
            for j in range(1, len(positions)):
                errors[:, j] = (
                    rho * errors[:, j - 1] + np.sqrt(1 - rho**2) * errors[:, j]
                )
            differences[:, column] = 2 * np.mean(errors * q, axis=1)
        coverage = positive = student_coverage = 0
        for sample in differences:
            low, high = np.quantile(sample[indices].mean(axis=1), [0.025, 0.975])
            coverage += int(low <= 0 <= high)
            positive += int(low > 0)
            half = (
                float(t.ppf(0.975, len(groups) - 1))
                * sample.std(ddof=1)
                / np.sqrt(len(groups))
            )
            student_coverage += int(sample.mean() - half <= 0 <= sample.mean() + half)
        bounds = binomtest(coverage, replications).proportion_ci(method="wilson")
        result["noise"][str(rho)] = {
            "covered": coverage,
            "coverage": coverage / replications,
            "coverage_ci95": [bounds.low, bounds.high],
            "positive_rejections": positive,
            "student_coverage": student_coverage / replications,
        }
    return result


def initialize(feature_file, cohort, minimum_sources, minimum):
    """Prepare fixed covariates without using the reserved participants' outcomes."""
    frame = pd.read_csv(feature_file)
    frame = session_scenario(frame.loc[frame.cohort == cohort], minimum).reset_index(
        drop=True
    )
    split = split_study(frame, minimum_sources=minimum_sources, minimum=minimum)
    safe = pd.concat(
        [population(split["X_train"], split["y_train"]), split["background"]]
    )
    with threadpool_limits(limits=1):
        reference = VideoReference(10.0).fit(safe)
        baseline = reference.predict(frame)
    values = observed_features(frame, ["eeg_front_alpha"])[:, 0]
    source_values = observed_features(safe, ["eeg_front_alpha"])[:, 0]
    finite = source_values[np.isfinite(source_values)]
    fallback = float(finite.mean()) if len(finite) else 0.0
    values = np.where(np.isfinite(values), values, fallback)
    z = np.zeros(len(frame))
    traits = {}
    for person, group in frame.groupby("participant_id"):
        calibration = group.loc[group.day == 1]
        if not len(calibration):
            calibration = group.loc[group.recorded_day == group.recorded_day.min()]
        sample = values[calibration.index]
        traits[person] = float(sample.mean())
        z[group.index] = np.clip(
            (values[group.index] - sample.mean()) / max(sample.std(), 1e-6), -4, 4
        )
    trait_mean, trait_sd = np.mean(list(traits.values())), np.std(list(traits.values()))
    traits = {
        p: (value - trait_mean) / max(trait_sd, 1e-6) for p, value in traits.items()
    }
    source_people = set(safe.participant_id)
    source_rows = frame.participant_id.isin(source_people)
    shared = pd.DataFrame(
        {"video": frame.loc[source_rows, "video"], "z": z[source_rows]}
    )
    shared = shared.groupby("video").z.mean()
    shared = shared.reindex(frame.video).fillna(0).to_numpy()
    shared_standard = shared / max(shared[source_rows].std(), 1e-6)
    _STATE.update(
        frame=frame,
        baseline=baseline,
        z=z,
        traits=traits,
        shared=shared,
        shared_standard=shared_standard,
        source_rows=source_rows.to_numpy(),
        minimum_sources=minimum_sources,
        minimum=minimum,
        test_ids=sorted(split["test_grid"].participant_id.unique()),
    )


def generated_responses(scenario, seed):
    """Draw responses with person, session and autocorrelated trial components."""
    frame = _STATE["frame"]
    rng = np.random.default_rng(seed)
    response = np.zeros(len(frame))
    for person, group in frame.groupby("participant_id"):
        offset = rng.normal(0, 0.5)
        gain = np.exp(rng.normal(0, 0.2))
        if scenario == "response_style":
            offset += 0.4 * _STATE["traits"][person]
            gain = np.exp(np.clip(0.25 * _STATE["traits"][person], -0.7, 0.7))
        response[group.index] = (
            5 + offset + gain * (_STATE["baseline"][group.index] - 5)
        )
        noise_sd = np.exp(rng.normal(0, 0.2))
        for _, session in group.groupby("recorded_day"):
            indices = session.sort_values("order").index.to_numpy()
            innovations = rng.normal(0, noise_sd, len(indices))
            errors = innovations.copy()
            for i in range(1, len(errors)):
                errors[i] = 0.4 * errors[i - 1] + np.sqrt(1 - 0.4**2) * innovations[i]
            response[indices] += rng.normal(0, 0.25) + errors
    if scenario == "shared_stimulus":
        response += 0.5 * _STATE["shared_standard"]
    if scenario == "individual_signal":
        response += 0.5 * (_STATE["z"] - _STATE["shared"])
    return np.clip(response, 1, 9)


def one_replicate(task):
    """Refit and reselect all primary models for one synthetic response draw."""
    scenario, repetition = task
    seed = SEED + 100000 * SCENARIOS.index(scenario) + repetition
    start = time.monotonic()
    frame = _STATE["frame"].copy()
    frame["rating_arousal"] = generated_responses(scenario, seed)
    options = {
        "minimum_sources": _STATE["minimum_sources"],
        "minimum": _STATE["minimum"],
    }
    with threadpool_limits(limits=1):
        split = split_study(frame, **options)
        if sorted(split["test_grid"].participant_id.unique()) != _STATE["test_ids"]:
            raise RuntimeError("Synthetic labels changed the fixed test population")
        parameters, penalties, _ = tune(
            frame, split, modes=("all", "video", "missing"), verbose=False, **options
        )
        scores = {}
        for mode in ("affine", "all", "video", "missing"):
            model = CalibratedArousal(
                mode=mode, signal_alpha=penalties.get(mode, 10.0), **parameters
            )
            model.fit(
                split["X_train"], split["y_train"], background=split["background"]
            )
            prediction = model.predict(split["X_test"])
            scores[mode] = participant_scores(
                split["y_test"],
                prediction,
                split["X_test"].participant_id,
                include_rank=False,
            )
        effects = {
            name: effect_summary(scores[name], scores["all"], seed=seed)["centered_mse"]
            for name in ("affine", "video", "missing")
        }
    record = {
        "scenario": scenario,
        "repetition": repetition,
        "seed": seed,
        "test_people": len(scores["all"]),
        "seconds": time.monotonic() - start,
    }
    for name, result in effects.items():
        record[f"gain_{name}"] = result["mean"]
        record[f"low_{name}"], record[f"high_{name}"] = result["ci95"]
    record["positive_vs_reference"] = record["low_affine"] > 0
    record["positive_after_controls"] = all(
        record[f"low_{name}"] > 0 for name in effects
    )
    return record


def worker_limit(requested):
    """Respect the CPU quota of the execution container."""
    count = os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        maximum, period = quota.read_text().split()
        if maximum != "max":
            count = min(count, max(1, int(maximum) // int(period)))
    return min(requested, count)


def configuration(feature_file, cohort, minimum_sources, minimum, replicates):
    """Fingerprint every model-defining input to a resumable simulation."""
    root = Path(__file__).resolve().parent
    algorithm = b"".join(
        (root / name).read_bytes()
        for name in ("model.py", "study.py", "features.py", "simulation.py")
    )
    return {
        "feature_sha256": hashlib.sha256(Path(feature_file).read_bytes()).hexdigest(),
        "algorithm_sha256": hashlib.sha256(algorithm).hexdigest(),
        "cohort": cohort,
        "minimum_sources": minimum_sources,
        "minimum_trials": minimum,
        "scenarios": list(SCENARIOS),
        "target_per_scenario": replicates,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }


def save_records(records, out_dir):
    """Replace a checkpoint only after its complete CSV has been written."""
    temporary = out_dir / "simulation_replicates.csv.tmp"
    table = pd.DataFrame(records) if records else pd.DataFrame(columns=RECORD_COLUMNS)
    table.to_csv(temporary, index=False)
    temporary.replace(out_dir / "simulation_replicates.csv")


def run(
    feature_file,
    out_dir,
    replicates=1000,
    workers=8,
    max_seconds=2400,
    resume=None,
    cohort=2,
    minimum_sources=10,
    minimum=20,
    expected_feature_sha=None,
):
    """Run a bounded simulation batch and retain every completed replication.

    Parameters
    ----------
    feature_file : pathlib.Path
        Private observed covariates and source annotations.
    out_dir : pathlib.Path
        Destination for synthetic replication statistics.
    replicates : int
        Target replications per mechanism.
    workers : int
        Maximum independent CPU workers.
    max_seconds : float
        Stop dispatching tasks after this duration; running tasks finish.
    resume : pathlib.Path or None
        Previously completed synthetic statistics, with matching configuration.
    cohort : int
        Cohort whose observed design is replayed.
    minimum_sources, minimum : int
        The same support thresholds as the main analysis.
    expected_feature_sha : str or None
        Required fingerprint of the frozen observed covariates.

    Returns
    -------
    dict
        Replication counts and average gains under each mechanism.
    """
    feature_file, out_dir = Path(feature_file), Path(out_dir)
    if replicates < 1 or workers < 1 or max_seconds <= 0:
        raise ValueError("Replication, worker and time budgets must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied = any(
        (out_dir / name).exists()
        for name in ("simulation_config.json", "simulation_replicates.csv")
    )
    if occupied and (
        resume is None or out_dir.resolve() != Path(resume).resolve().parent
    ):
        raise FileExistsError(
            "Use a fresh output directory or resume its own checkpoint"
        )
    config = configuration(feature_file, cohort, minimum_sources, minimum, replicates)
    if expected_feature_sha and config["feature_sha256"] != expected_feature_sha:
        raise ValueError("Observed features differ from the frozen study")
    records = []
    if resume:
        resume = Path(resume)
        old = json.loads((resume.parent / "simulation_config.json").read_text())
        if any(
            old.get(key) != value
            for key, value in config.items()
            if key != "target_per_scenario"
        ):
            raise ValueError("Simulation resume does not match the data and algorithm")
        records = pd.read_csv(resume).to_dict("records")
    for record in records:
        scenario, repetition = record.get("scenario"), record.get("repetition")
        if (
            scenario not in SCENARIOS
            or not float(repetition).is_integer()
            or not 0 <= repetition < replicates
        ):
            raise ValueError("Checkpoint contains an invalid replication identifier")
        expected_seed = SEED + 100000 * SCENARIOS.index(scenario) + int(repetition)
        if record.get("seed") != expected_seed or not set(RECORD_COLUMNS) <= set(
            record
        ):
            raise ValueError("Checkpoint seed or schema does not match the simulation")
    completed = {(record["scenario"], int(record["repetition"])) for record in records}
    if len(completed) != len(records):
        raise ValueError("Checkpoint contains duplicate replications")
    tasks = iter(
        (scenario, repetition)
        for repetition in range(replicates)
        for scenario in SCENARIOS
        if (scenario, repetition) not in completed
    )
    save_records(records, out_dir)
    metadata_file = out_dir / "simulation_config.json.tmp"
    metadata_file.write_text(json.dumps(config, indent=2) + "\n")
    metadata_file.replace(out_dir / "simulation_config.json")
    print("HABS_SIM_CONFIG=" + json.dumps(config), flush=True)
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=worker_limit(workers),
        mp_context=get_context("spawn"),
        initializer=initialize,
        initargs=(feature_file, cohort, minimum_sources, minimum),
    ) as pool:
        pending = {
            pool.submit(one_replicate, task)
            for task in [next(tasks, None) for _ in range(worker_limit(workers))]
            if task is not None
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                records.append(record)
                print("HABS_SIM_RESULT=" + json.dumps(record), flush=True)
                save_records(records, out_dir)
                task = (
                    next(tasks, None)
                    if time.monotonic() - start < max_seconds
                    else None
                )
                if task is not None:
                    pending.add(pool.submit(one_replicate, task))
    table = pd.DataFrame(records)
    summary = {
        "configuration": config,
        "seconds": time.monotonic() - start,
        "scenarios": {},
        "workers": worker_limit(workers),
    }
    for scenario, group in table.groupby("scenario"):
        summary["scenarios"][scenario] = {
            "replications": len(group),
            "positive_vs_reference": int(group.positive_vs_reference.sum()),
            "positive_after_controls": int(group.positive_after_controls.sum()),
            "mean_gains": {
                name: float(group[f"gain_{name}"].mean())
                for name in ("affine", "video", "missing")
            },
            "median_seconds": float(group.seconds.median()),
        }
        for name in ("positive_vs_reference", "positive_after_controls"):
            count = int(group[name].sum())
            bounds = binomtest(count, len(group)).proportion_ci(method="wilson")
            summary["scenarios"][scenario][name + "_fraction"] = count / len(group)
            summary["scenarios"][scenario][name + "_ci95"] = [bounds.low, bounds.high]
    summary["complete"] = all(
        summary["scenarios"].get(s, {}).get("replications", 0) >= replicates
        for s in SCENARIOS
    )
    (out_dir / "simulation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("HABS_SIM_SUMMARY=" + json.dumps(summary), flush=True)
    return summary
