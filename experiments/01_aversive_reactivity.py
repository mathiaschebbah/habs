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

from habs import PROJECT_ROOT
from habs.data import load_table
from habs.features import FEATURE_VERSION
from habs.pipeline import build_feature_graph

SEED = 20261108


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


def main():
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    config = json.loads((PROJECT_ROOT / "run_config.json").read_text())
    _prepare(raw_dir, out_dir)
    _real_smoke(out_dir)
    if config["stage"] == "prepare":
        print(
            "Real-data smoke passed. Reserved outcome evaluation has not run.",
            flush=True,
        )
        return
    raise ValueError(
        "The requested study stage is not implemented in this source version."
    )


if __name__ == "__main__":
    main()
