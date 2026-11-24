"""Extract frozen phase-resolved embeddings from 5 pretrained EEG foundation models.

Runs once on the HABS H100 GPU backend. Walks the 508 audited emotion-video
recordings ONCE in parallel to load canonical 62-channel ICA-cleaned CAR EEG
at 256 Hz over ``[-3, +11) s`` (14 s), then runs batched GPU inference in
bfloat16 across all five publicly accessible EEG foundation models:

1. ``Signal-JEPA`` (``braindecode/signal-jepa``, 3.4M, 128 Hz, 47 aligned ch)
2. ``CBraMod`` (``braindecode/cbramod-pretrained``, 4.9M, 200 Hz, 62/62 ch)
3. ``LaBraM`` (``braindecode/labram-pretrained``, 5.8M, 200 Hz, 62/62 ch)
4. ``EEGPT`` (``braindecode/eegpt-pretrained``, 25.3M, 256 Hz, 54 pretrained ch)
5. ``REVE-base`` (``brain-bzh/reve-base``, 69.0M, 200 Hz, 62/62 ch)

For every model, saves one compact zstd-compressed Parquet table under
``/out/features/fm_<name>.parquet`` (< 150 MB each) holding ``trial_id``,
``eeg_valid``, and phase-resolved mean embeddings (``ref`` [-3, 0), ``early``
[0, 5), ``late`` [5, 10)), plus public runtime/VRAM/effective-rank telemetry
under ``/out/public/fm_extract.json``.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyedflib
import torch
import torchaudio.functional as TAF
from scipy.signal import butter, resample_poly, sosfiltfilt

from decoding.extract import EEG_UNITS, _eeg_source, _windows
from decoding.ssl_audit import CANONICAL_62
from decoding.ssl_models import count_parameters, effective_rank
from habs.temporal.data import physical_id

FM_WIN_START_S = -3.0
FM_WIN_SECONDS = 14
FM_BASE_FS = 256

# EEGPT's 58-channel pretraining montage excludes 4 HABS channels not in
# CHANNEL_DICT (F9, F10, P9, P10) and 4 channels present in CHANNEL_DICT whose
# embeddings were never trained during pretraining (AF7, AF8, PO5, PO6).
EEGPT_EXCLUDED = {"F9", "F10", "P9", "P10", "AF7", "AF8", "PO5", "PO6"}
EEGPT_54_CHANS = tuple(ch for ch in CANONICAL_62 if ch not in EEGPT_EXCLUDED)


def _resample_2d(
    x: np.ndarray, src_fs: float, dst_fs: int, target_len: int
) -> np.ndarray:
    """Polyphase resample ``(..., N)`` to exact ``target_len``."""
    up = int(round(dst_fs))
    down = max(1, int(round(src_fs)))
    g = int(np.gcd(up, down))
    y = resample_poly(x, up // g, down // g, axis=-1)
    if y.shape[-1] > target_len:
        y = y[..., :target_len]
    elif y.shape[-1] < target_len:
        pad = [(0, 0)] * (y.ndim - 1) + [(0, target_len - y.shape[-1])]
        y = np.pad(y, pad, mode="edge")
    return y


def _worker_eeg_256hz(args: tuple[str, list[dict]]) -> dict:
    """Read 62-ch CAR EEG at 256 Hz (float16, uV) + recording channel z-stats."""
    raw_dir, trials = args
    curated = Path(raw_dir) / trials[0]["recording"]
    onsets = np.array([float(t["onset_s"]) for t in trials], dtype=np.float64)
    trial_ids = [physical_id(t) for t in trials]
    n_trials = len(trials)
    n_ch = len(CANONICAL_62)
    n_samp = FM_WIN_SECONDS * FM_BASE_FS

    eeg_path, shift = _eeg_source(raw_dir, trials[0])
    candidates = [(eeg_path, shift)] + ([(curated, 0.0)] if eeg_path != curated else [])

    eeg_out = np.zeros((n_trials, n_ch, n_samp), dtype=np.float32)
    rec_mean = np.zeros((n_ch,), dtype=np.float32)
    rec_std = np.ones((n_ch,), dtype=np.float32)
    valid = np.zeros((n_trials,), dtype=bool)

    for path, offset in candidates:
        try:
            adj_onsets = onsets + offset
            ch_present = np.zeros(n_ch, dtype=bool)
            bad_ch_count = np.zeros(n_trials, dtype=np.int32)
            with pyedflib.EdfReader(str(path)) as reader:
                labels = reader.getSignalLabels()
                lbl_map = {lbl.upper(): i for i, lbl in enumerate(labels)}
                sos_cache: dict[int, np.ndarray] = {}
                for c_i, ch_name in enumerate(CANONICAL_62):
                    idx = lbl_map.get(ch_name.upper())
                    if idx is None:
                        continue
                    unit = (
                        reader.getPhysicalDimension(idx)
                        .replace("µ", "u")
                        .replace("μ", "u")
                    )
                    fs = float(reader.getSampleFrequency(idx))
                    if unit not in EEG_UNITS or fs < 100:
                        continue
                    sig = reader.readSignal(idx).astype(np.float64) * EEG_UNITS[unit]
                    if not np.isfinite(sig).all():
                        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
                    fs_key = int(round(fs))
                    if fs_key not in sos_cache:
                        hi = min(45.0, 0.45 * fs)
                        sos_cache[fs_key] = butter(
                            4, (0.5, hi), btype="bandpass", fs=fs, output="sos"
                        )
                    sig = sosfiltfilt(sos_cache[fs_key], sig)
                    # Robust recording stats (for REVE's session z-score protocol)
                    rec_mean[c_i] = float(np.mean(sig))
                    sd = float(np.std(sig))
                    rec_std[c_i] = sd if sd > 1e-3 else 1.0

                    win = _windows(
                        sig, fs, adj_onsets, FM_WIN_START_S, float(FM_WIN_SECONDS)
                    )
                    fin = np.isfinite(win).all(axis=1)
                    res = _resample_2d(np.nan_to_num(win), fs, FM_BASE_FS, n_samp)
                    ptp = np.ptp(res, axis=-1)
                    tsd = np.std(res, axis=-1)
                    bad = (~fin) | (ptp > 800.0) | (tsd < 0.01)
                    res[bad] = 0.0
                    eeg_out[:, c_i, :] = res.astype(np.float32)
                    ch_present[c_i] = True
                    bad_ch_count += bad.astype(np.int32)

            if ch_present.sum() >= 30:
                car = eeg_out[:, ch_present, :].mean(axis=1, keepdims=True)
                eeg_out[:, ch_present, :] -= car
                valid = bad_ch_count <= int(0.20 * ch_present.sum())
                break
        except (OSError, ValueError):
            continue

    return {
        "trial_ids": trial_ids,
        "eeg": np.clip(eeg_out, -500.0, 500.0).astype(np.float16),
        "rec_mean": rec_mean,
        "rec_std": rec_std,
        "valid": valid,
    }


def load_all_eeg_256hz(
    trials: pd.DataFrame, raw_dir: Path, *, workers: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load canonical 62-ch 256 Hz EEG for all audited trials into RAM (~10 GB)."""
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    id_to_pos = {tid: i for i, tid in enumerate(trials["trial_id"])}
    n = len(trials)
    n_ch = len(CANONICAL_62)
    n_samp = FM_WIN_SECONDS * FM_BASE_FS

    eeg = np.zeros((n, n_ch, n_samp), dtype=np.float16)
    rec_mean = np.zeros((n, n_ch), dtype=np.float32)
    rec_std = np.ones((n, n_ch), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)

    jobs = [
        (str(raw_dir), grp.to_dict("records"))
        for _, grp in trials.groupby("recording", sort=True)
    ]
    ctx = multiprocessing.get_context("spawn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for idx, part in enumerate(pool.map(_worker_eeg_256hz, jobs), 1):
            pos = np.array([id_to_pos[t] for t in part["trial_ids"]], dtype=int)
            eeg[pos] = part["eeg"]
            rec_mean[pos] = part["rec_mean"][None, :]
            rec_std[pos] = part["rec_std"][None, :]
            valid[pos] = part["valid"]
            if idx % 50 == 0 or idx == len(jobs):
                print(f"FM EEG load: {idx}/{len(jobs)} recordings", flush=True)
    return eeg, rec_mean, rec_std, valid


def make_chs_info(ch_names: list[str] | tuple[str, ...]) -> list[dict]:
    """Build MNE-compatible ``chs_info`` with 3D coordinates from standard_1005."""
    montage = mne.channels.make_standard_montage("standard_1005")
    pos_map = {k.upper(): v for k, v in montage.get_positions()["ch_pos"].items()}
    info = []
    for name in ch_names:
        loc = pos_map.get(name.upper(), np.zeros(3))
        info.append(
            {
                "ch_name": name,
                "kind": 2,
                "unit": 107,
                "loc": [float(loc[0]), float(loc[1]), float(loc[2])],
            }
        )
    return info


def save_fm_parquet(
    out_path: Path,
    trial_ids: list[str],
    valid: np.ndarray,
    prefix: str,
    z_ref: np.ndarray,
    z_early: np.ndarray,
    z_late: np.ndarray,
) -> dict:
    """Save ``(z_ref, z_early, z_late)`` to compact zstd Parquet and return stats."""
    d = z_ref.shape[1]
    cols: dict[str, object] = {
        "trial_id": trial_ids,
        f"{prefix}_valid": valid.astype(bool),
    }
    for phase_name, arr in (("ref", z_ref), ("early", z_early), ("late", z_late)):
        arr32 = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for j in range(d):
            cols[f"{prefix}_{phase_name}_{j}"] = arr32[:, j]
    df = pd.DataFrame(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd")
    size_mb = round(out_path.stat().st_size / 1e6, 2)

    z_vid = 0.5 * (z_early + z_late)
    z_delta = z_vid - z_ref
    ok = valid & np.isfinite(z_vid).all(axis=1) & np.isfinite(z_ref).all(axis=1)
    er_vid = (
        round(effective_rank(torch.from_numpy(z_vid[ok][:4000])), 2)
        if ok.sum() > 20
        else None
    )
    er_delta = (
        round(effective_rank(torch.from_numpy(z_delta[ok][:4000])), 2)
        if ok.sum() > 20
        else None
    )
    return {
        "parquet_file": out_path.name,
        "parquet_mb": size_mb,
        "embed_dim": int(d),
        "total_cols": int(df.shape[1]),
        "valid_fraction": round(float(ok.mean()), 4),
        "effective_rank_vid": er_vid,
        "effective_rank_delta": er_delta,
    }


def _extract_cbramod(
    eeg_256: np.ndarray, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract CBraMod (200 Hz, 62/62 ch, uV/100) phase embeddings."""
    from braindecode.models import CBraMod

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model = CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained",
        n_chans=62,
        n_times=13 * 200,
        sfreq=200,
        n_outputs=1,
        strict=False,
    )
    model = model.to(device).eval()
    n_params = count_parameters(model, trainable_only=False)
    n = len(eeg_256)
    z_ref = np.zeros((n, 200), dtype=np.float32)
    z_early = np.zeros((n, 200), dtype=np.float32)
    z_late = np.zeros((n, 200), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            x = torch.from_numpy(eeg_256[start:end, :, : 13 * 256]).to(
                device, dtype=torch.float32
            )
            x = TAF.resample(x, 256, 200)[..., : 13 * 200] / 100.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, return_features=True)
            toks = out["tokens"].float()  # (B, 62, 13, 200)
            ch_mean = toks.mean(dim=1)  # (B, 13, 200)
            z_ref[start:end] = ch_mean[:, 0:3, :].mean(dim=1).cpu().numpy()
            z_early[start:end] = ch_mean[:, 3:8, :].mean(dim=1).cpu().numpy()
            z_late[start:end] = ch_mean[:, 8:13, :].mean(dim=1).cpu().numpy()

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return (
        z_ref,
        z_early,
        z_late,
        {
            "repo": "braindecode/cbramod-pretrained",
            "n_params": n_params,
            "n_chans_used": 62,
            "sfreq": 200,
            "scaling": "uV / 100.0",
            "runtime_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "trials_per_s": round(n / max(elapsed, 1e-6), 1),
        },
    )


def _extract_labram(
    eeg_256: np.ndarray, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract LaBraM (200 Hz, 62/62 ch, uV/100) phase embeddings."""
    from braindecode.models import Labram

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    ch_upper = [c.upper() for c in CANONICAL_62]
    chs_info = make_chs_info(ch_upper)
    model = Labram.from_pretrained(
        "braindecode/labram-pretrained",
        n_chans=62,
        n_times=13 * 200,
        sfreq=200,
        chs_info=chs_info,
        n_outputs=0,
        strict=False,
    )
    model = model.to(device).eval()
    n_params = count_parameters(model, trainable_only=False)
    n = len(eeg_256)
    z_ref = np.zeros((n, 200), dtype=np.float32)
    z_early = np.zeros((n, 200), dtype=np.float32)
    z_late = np.zeros((n, 200), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            x = torch.from_numpy(eeg_256[start:end, :, : 13 * 256]).to(
                device, dtype=torch.float32
            )
            x = TAF.resample(x, 256, 200)[..., : 13 * 200] / 100.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, return_features=True, ch_names=ch_upper)
            toks = out["tokens"].float()  # (B, 62, 13, 200)
            ch_mean = toks.mean(dim=1)
            z_ref[start:end] = ch_mean[:, 0:3, :].mean(dim=1).cpu().numpy()
            z_early[start:end] = ch_mean[:, 3:8, :].mean(dim=1).cpu().numpy()
            z_late[start:end] = ch_mean[:, 8:13, :].mean(dim=1).cpu().numpy()

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return (
        z_ref,
        z_early,
        z_late,
        {
            "repo": "braindecode/labram-pretrained",
            "n_params": n_params,
            "n_chans_used": 62,
            "sfreq": 200,
            "scaling": "uV / 100.0",
            "runtime_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "trials_per_s": round(n / max(elapsed, 1e-6), 1),
        },
    )


def _extract_sjepa(
    eeg_256: np.ndarray, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract Signal-JEPA (128 Hz, 47 pretrain-aligned ch, raw uV) phase embeddings."""
    from braindecode.models import SignalJEPA
    from huggingface_hub import hf_hub_download

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    cfg_path = hf_hub_download("braindecode/signal-jepa", "config.json")
    cfg = json.loads(Path(cfg_path).read_text())
    pretrain_chs = {c["ch_name"].upper() for c in cfg["chs_info"]}
    keep_idx = [i for i, c in enumerate(CANONICAL_62) if c.upper() in pretrain_chs]
    keep_names = [CANONICAL_62[i] for i in keep_idx]
    chs_info = make_chs_info(keep_names)

    model = SignalJEPA.from_pretrained(
        "braindecode/signal-jepa",
        chs_info=chs_info,
        channel_embedding="pretrain_aligned",
        n_times=14 * 128,
        sfreq=128,
        strict=False,
    )
    model = model.to(device).eval()
    n_params = count_parameters(model, trainable_only=False)
    n = len(eeg_256)
    n_ch = len(keep_idx)
    d_model = 64
    z_ref = np.zeros((n, d_model), dtype=np.float32)
    z_early = np.zeros((n, d_model), dtype=np.float32)
    z_late = np.zeros((n, d_model), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            x = (
                torch.from_numpy(
                    np.ascontiguousarray(eeg_256[start:end, keep_idx, : 14 * 256])
                )
                .to(device, dtype=torch.float32)
                .contiguous()
            )
            x = TAF.resample(x, 256, 128)[..., : 14 * 128].contiguous()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, return_features=True)
            feats = out["features"].float()  # (B, n_ch * n_t, 64)
            n_t = feats.shape[1] // n_ch
            ch_mean = feats.reshape(end - start, n_ch, n_t, d_model).mean(dim=1)
            # 14 s -> 13 1-s tokens (0..2 ref, 3..7 early, 8..12 late)
            t_ref = min(3, n_t)
            t_early_end = min(8, n_t)
            z_ref[start:end] = ch_mean[:, :t_ref, :].mean(dim=1).cpu().numpy()
            z_early[start:end] = (
                ch_mean[:, t_ref:t_early_end, :].mean(dim=1).cpu().numpy()
            )
            z_late[start:end] = ch_mean[:, t_early_end:, :].mean(dim=1).cpu().numpy()

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return (
        z_ref,
        z_early,
        z_late,
        {
            "repo": "braindecode/signal-jepa",
            "n_params": n_params,
            "n_chans_used": n_ch,
            "sfreq": 128,
            "scaling": "uV (raw)",
            "runtime_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "trials_per_s": round(n / max(elapsed, 1e-6), 1),
        },
    )


def _extract_eegpt(
    eeg_256: np.ndarray, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract EEGPT (256 Hz, 54 genuinely pretrained ch, mV, 4-s windows)."""
    from braindecode.models import EEGPT

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    keep_idx = [CANONICAL_62.index(c) for c in EEGPT_54_CHANS]
    chs_info = make_chs_info([c.upper() for c in EEGPT_54_CHANS])

    model = EEGPT.from_pretrained(
        "braindecode/eegpt-pretrained",
        n_chans=len(keep_idx),
        chs_info=chs_info,
        n_times=1024,
        sfreq=256,
        chan_proj_type="none",
        n_outputs=1,
        strict=False,
    )
    model = model.to(device).eval()
    n_params = count_parameters(model, trainable_only=False)
    n = len(eeg_256)
    d = 512
    z_ref = np.zeros((n, d), dtype=np.float32)
    z_early = np.zeros((n, d), dtype=np.float32)
    z_late = np.zeros((n, d), dtype=np.float32)

    def _run_4s(x_4s: torch.Tensor, net: torch.nn.Module) -> np.ndarray:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x_4s, return_features=True)
        toks = out["tokens"].float()  # (B, n_patches=16, embed_num=4, 512)
        return toks.mean(dim=(1, 2)).cpu().numpy()

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            # Raw uV -> mV (EEGPT pretraining scale, CAR already applied)
            x = (
                torch.from_numpy(eeg_256[start:end, keep_idx, : 13 * 256]).to(
                    device, dtype=torch.float32
                )
                * 1e-3
            )
            # Strictly causal 3-s pre-onset [-3, 0) s left-padded by 1 s reflection
            x_pre3 = x[:, :, : 3 * 256]
            x_ref = torch.cat([x_pre3[:, :, :256].flip(-1), x_pre3], dim=-1)
            # Early [0, 4) s and late average of [4, 8) s and [6, 10) s
            x_early = x[:, :, 3 * 256 : 7 * 256]
            x_late_a = x[:, :, 7 * 256 : 11 * 256]
            x_late_b = x[:, :, 9 * 256 : 13 * 256]
            z_ref[start:end] = _run_4s(x_ref, model)
            z_early[start:end] = _run_4s(x_early, model)
            z_late[start:end] = 0.5 * (
                _run_4s(x_late_a, model) + _run_4s(x_late_b, model)
            )

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return (
        z_ref,
        z_early,
        z_late,
        {
            "repo": "braindecode/eegpt-pretrained",
            "n_params": n_params,
            "n_chans_used": len(keep_idx),
            "excluded_untrained_or_absent": sorted(EEGPT_EXCLUDED),
            "sfreq": 256,
            "scaling": "mV (uV * 1e-3), CAR",
            "runtime_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "trials_per_s": round(n / max(elapsed, 1e-6), 1),
        },
    )


def _extract_reve(
    eeg_256: np.ndarray,
    rec_mean: np.ndarray,
    rec_std: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract REVE-base (200 Hz, 62/62 ch, recording z-score clip +-15)."""
    from transformers import AutoModel

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions", trust_remote_code=True
    ).to(device)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).to(
        device
    )
    model.eval()
    n_params = count_parameters(model, trainable_only=False)
    base_pos = pos_bank(list(CANONICAL_62)).to(device)  # (62, 3)

    n = len(eeg_256)
    d = 512
    z_ref = np.zeros((n, d), dtype=np.float32)
    z_early = np.zeros((n, d), dtype=np.float32)
    z_late = np.zeros((n, d), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            bsz = end - start
            x = torch.from_numpy(eeg_256[start:end, :, : 13 * 256]).to(
                device, dtype=torch.float32
            )
            mu = torch.from_numpy(rec_mean[start:end]).to(device).unsqueeze(-1)
            sd = torch.from_numpy(rec_std[start:end]).to(device).unsqueeze(-1)
            x = torch.clamp((x - mu) / sd.clamp(min=1e-3), -15.0, 15.0)
            x = TAF.resample(x, 256, 200)[..., : 13 * 200]
            pos = base_pos.unsqueeze(0).expand(bsz, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model(x, pos)  # (B, 62, H=14, 512)
            ch_mean = toks.float().mean(dim=1)  # (B, H, 512)
            # 14 overlapping 1-s patches (step 0.9 s) across [-3, +10) s
            z_ref[start:end] = ch_mean[:, 0:3, :].mean(dim=1).cpu().numpy()
            z_early[start:end] = ch_mean[:, 3:8, :].mean(dim=1).cpu().numpy()
            z_late[start:end] = ch_mean[:, 8:, :].mean(dim=1).cpu().numpy()

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    del model, pos_bank
    gc.collect()
    torch.cuda.empty_cache()
    return (
        z_ref,
        z_early,
        z_late,
        {
            "repo": "brain-bzh/reve-base",
            "n_params": n_params,
            "n_chans_used": 62,
            "sfreq": 200,
            "scaling": "recording z-score, clip +-15",
            "runtime_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "trials_per_s": round(n / max(elapsed, 1e-6), 1),
        },
    )


def main() -> None:
    """Run all 5 foundation model extractions on H100 and save parquets + manifest."""
    t_start = time.perf_counter()
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trials = pd.read_csv(
        out_dir / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trial_ids = [physical_id(r) for r in trials.to_dict("records")]

    t_load0 = time.perf_counter()
    workers = min(20, os.cpu_count() or 8)
    eeg_256, rec_mean, rec_std, valid = load_all_eeg_256hz(
        trials, raw_dir, workers=workers
    )
    load_s = round(time.perf_counter() - t_load0, 2)
    print(
        f"Loaded {len(trials)} trials 62-ch 256 Hz EEG in {load_s}s "
        f"(valid={valid.mean():.4f})",
        flush=True,
    )

    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "study": "08_ssl_fm_extract",
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "n_trials": len(trials),
        "eeg_valid_fraction": round(float(valid.mean()), 4),
        "eeg_load_seconds": load_s,
        "models": {},
    }

    extractors = (
        ("sjepa", lambda: _extract_sjepa(eeg_256, device)),
        ("cbramod", lambda: _extract_cbramod(eeg_256, device)),
        ("labram", lambda: _extract_labram(eeg_256, device)),
        ("eegpt", lambda: _extract_eegpt(eeg_256, device)),
        ("reve", lambda: _extract_reve(eeg_256, rec_mean, rec_std, device)),
    )

    for name, fn in extractors:
        print(f"=== Extracting {name} ===", flush=True)
        try:
            zr, ze, zl, info = fn()
            pq_stats = save_fm_parquet(
                feat_dir / f"fm_{name}.parquet",
                trial_ids,
                valid,
                name,
                zr,
                ze,
                zl,
            )
            manifest["models"][name] = {"status": "ok", **info, **pq_stats}
            er_v = pq_stats["effective_rank_vid"]
            print(
                f"  OK {name}: {info['runtime_s']}s, {info['peak_vram_gb']} GB VRAM, "
                f"{pq_stats['parquet_mb']} MB, erank_vid={er_v}",
                flush=True,
            )
        except Exception as exc:
            manifest["models"][name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED {name}: {exc}", flush=True)
            traceback.print_exc()

    manifest["total_elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
    pub = out_dir / "public"
    pub.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2)
    (pub / "fm_extract.json").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
