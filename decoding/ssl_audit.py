"""Study-08 feasibility, runtime, channel, and unlabeled-data audit on HABS.

Runs once on the HABS H100 backend (~1 minute) before freezing the Study-08
protocol. Collects only aggregate counts and hardware/network telemetry — never
writes individual trial rows or inspects any model predictions.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pyedflib

from decoding.common import assign_splits
from decoding.extract import _eeg_source
from decoding.stress import MIN_EACH, label

CANONICAL_62 = (
    "Fp1",
    "Fp2",
    "F9",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "F10",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P9",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "P10",
    "O1",
    "O2",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FC3",
    "FCz",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CP4",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO5",
    "PO3",
    "PO4",
    "PO6",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "POz",
)

CANDIDATE_SALTS = (
    "habs-ssl-2026-11-24",
    "habs-ssl-2026-11-24-b",
    "habs-ssl-2026-11-24-c",
)

PACKAGES_TO_CHECK = (
    "torch",
    "torchaudio",
    "torchvision",
    "einops",
    "timm",
    "safetensors",
    "huggingface_hub",
    "transformers",
    "braindecode",
    "mne",
    "scipy",
    "sklearn",
)


def probe_runtime(out_dir: Path) -> dict:
    """Inspect GPU, CPU, RAM, disk, installed packages, and HuggingFace access."""
    info: dict = {
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["ram_gb"] = round(mem_bytes / (1024**3), 2)
    except (AttributeError, ValueError, OSError):
        info["ram_gb"] = None

    for tag, target in (("out_dir", out_dir), ("tmp", Path("/tmp"))):
        try:
            usage = shutil.disk_usage(target)
            info[f"disk_{tag}_free_gb"] = round(usage.free / (1024**3), 2)
        except OSError:
            info[f"disk_{tag}_free_gb"] = None

    for pkg in PACKAGES_TO_CHECK:
        try:
            mod = importlib.import_module(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            info["packages"][pkg] = None

    if info["packages"].get("torch"):
        import torch

        gpu: dict = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu["device_name"] = props.name
            gpu["vram_gb"] = round(props.total_memory / (1024**3), 2)
            gpu["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            gpu["device_count"] = int(torch.cuda.device_count())
        info["gpu"] = gpu

    # Non-destructive check for outbound HuggingFace Hub connectivity.
    url = "https://huggingface.co/braindecode/cbramod-pretrained/raw/main/config.json"
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "habs-study08-audit"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(512)
        info["huggingface_network"] = {
            "reachable": True,
            "status": int(resp.status),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "bytes_read": len(body),
        }
    except Exception as exc:
        info["huggingface_network"] = {
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    return info


def audit_splits(trials: pd.DataFrame, salts=CANDIDATE_SALTS) -> dict:
    """Count split sizes and eligible participants for candidate fresh salts."""
    out = {}
    for salt in salts:
        sp = assign_splits(trials, salt)
        stressed = label(sp)
        entry = {}
        for name, group in sp.groupby("split"):
            pos = stressed.loc[group.index]
            per_p = group.groupby("participant_id").apply(
                lambda g, p=pos: int(p.loc[g.index].sum() >= MIN_EACH)
                and int((~p.loc[g.index]).sum() >= MIN_EACH),
                include_groups=False,
            )
            multi_day = (
                int(
                    (
                        group.groupby("participant_id")["recorded_day"].nunique() >= 2
                    ).sum()
                )
                if "recorded_day" in group
                else 0
            )
            entry[name] = {
                "trials": int(len(group)),
                "people": int(group["participant_id"].nunique()),
                "themes": int(group["theme"].nunique()),
                "eligible_stress_people": int(per_p.sum()),
                "multi_day_people": multi_day,
            }
        dev = sp[sp["split"] == "dev"]
        entry["dev_person_folds"] = {
            str(k): int(v)
            for k, v in dev.groupby("person_fold")["participant_id"]
            .nunique()
            .sort_index()
            .items()
        }
        entry["dev_theme_folds"] = {
            str(k): int(v)
            for k, v in dev.groupby("theme_fold")["theme"]
            .nunique()
            .sort_index()
            .items()
        }
        out[salt] = entry
    return out


def audit_recordings(raw_dir: Path, trials: pd.DataFrame) -> dict:
    """Audit EEG 62-ch consistency, sampling rates, and unlabeled resting-state data."""
    canon_set = set(CANONICAL_62)
    unique_recs = (
        trials.groupby("recording", sort=True)
        .first()
        .reset_index()[["recording", "participant_id"]]
    )

    ev_stats = {
        "n_recordings": int(len(unique_recs)),
        "preprocessed_found": 0,
        "exact_canonical_62": 0,
        "exact_canonical_62_order": 0,
        "has_body_in_preprocessed": 0,
        "eeg_sfreq_counts": {},
        "total_duration_hours": 0.0,
        "missing_channel_hist": {},
        "errors": 0,
    }

    for row in unique_recs.to_dict("records"):
        eeg_path, _ = _eeg_source(raw_dir, row)
        is_prep = "/PREPROCESSED/" in str(eeg_path)
        if is_prep:
            ev_stats["preprocessed_found"] += 1
        try:
            with pyedflib.EdfReader(str(eeg_path)) as reader:
                labels = reader.getSignalLabels()
                dur_s = float(reader.getFileDuration())
                ev_stats["total_duration_hours"] += dur_s / 3600.0
                stop = min(
                    [labels.index(n) for n in ("ECG", "Resp", "GSR") if n in labels]
                    or [len(labels)]
                )
                eeg_chs = labels[:stop]
                if set(eeg_chs) == canon_set:
                    ev_stats["exact_canonical_62"] += 1
                if tuple(eeg_chs) == CANONICAL_62:
                    ev_stats["exact_canonical_62_order"] += 1
                missing = sorted(canon_set - set(eeg_chs))
                for ch in missing:
                    ev_stats["missing_channel_hist"][ch] = (
                        ev_stats["missing_channel_hist"].get(ch, 0) + 1
                    )
                if is_prep and {"ECG", "Resp", "GSR"} <= set(labels):
                    ev_stats["has_body_in_preprocessed"] += 1
                sf = str(int(round(reader.getSampleFrequency(0))))
                ev_stats["eeg_sfreq_counts"][sf] = (
                    ev_stats["eeg_sfreq_counts"].get(sf, 0) + 1
                )
        except Exception:
            ev_stats["errors"] += 1

    ev_stats["total_duration_hours"] = round(ev_stats["total_duration_hours"], 2)

    # Audit resting-state & artefact BDF inventory for unlabeled SSL pretraining
    sp_default = assign_splits(trials, CANDIDATE_SALTS[0])
    dev_mask = sp_default["split"] == "dev"
    dev_people = set(sp_default.loc[dev_mask, "participant_id"].astype(str))
    all_people = set(trials["participant_id"].astype(str))

    unlabeled = {}
    for task in ("resting-state", "artefact"):
        prep_files = sorted(
            (raw_dir / "PREPROCESSED").glob(f"P*/P*_task-{task}_multimodal.bdf")
        )
        cur_files = sorted(
            (raw_dir / "CURATED").glob(f"P*/P*_task-{task}_multimodal.bdf")
        )
        dev_dur_h = 0.0
        dev_n = 0
        dev_people_set = set()
        dev_exact62 = 0
        for f in prep_files:
            pid = f.parent.name
            if pid not in dev_people:
                continue
            dev_n += 1
            dev_people_set.add(pid)
            try:
                with pyedflib.EdfReader(str(f)) as reader:
                    dev_dur_h += float(reader.getFileDuration()) / 3600.0
                    labels = reader.getSignalLabels()
                    stop = min(
                        [labels.index(n) for n in ("ECG", "Resp", "GSR") if n in labels]
                        or [len(labels)]
                    )
                    if set(labels[:stop]) == canon_set:
                        dev_exact62 += 1
            except Exception:
                pass
        unlabeled[task] = {
            "total_preprocessed_files": len(prep_files),
            "total_curated_files": len(cur_files),
            "in_audited_people_files": sum(
                1 for f in prep_files if f.parent.name in all_people
            ),
            "dev_only_files": dev_n,
            "dev_only_people": len(dev_people_set),
            "dev_only_exact_62ch": dev_exact62,
            "dev_only_hours": round(dev_dur_h, 2),
        }

    return {
        "emotion_video": ev_stats,
        "unlabeled_tasks": unlabeled,
    }


def main() -> None:
    """Run Study-08 feasibility audit and save public JSON summary."""
    t0 = time.perf_counter()
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    trials = pd.read_csv(
        out_dir / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )

    report = {
        "study": "08_ssl_audit",
        "elapsed_seconds": None,
        "runtime": probe_runtime(out_dir),
        "splits": audit_splits(trials),
        "recordings": audit_recordings(raw_dir, trials),
        "n_trials_total": int(len(trials)),
        "n_people_total": int(trials["participant_id"].nunique()),
        "n_videos_total": int(trials["video"].nunique()),
        "trial_hours_10s": round(float(len(trials)) * 10.0 / 3600.0, 2),
        "trial_hours_15s": round(float(len(trials)) * 15.0 / 3600.0, 2),
    }
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    pub = out_dir / "public"
    pub.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    (pub / "ssl_audit.json").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
