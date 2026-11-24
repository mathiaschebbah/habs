"""Multimodal raw-signal loading and causal preprocessing for Study 08.

Reuses the audited HABS alignment and preprocessing primitives from
``decoding.extract`` and ``habs.signals``. Produces aligned 15-second
``[-3, +12) s`` multimodal windows (62-ch EEG at 128 Hz, 2-ch ECG at 128 Hz,
2-ch EDA at 16 Hz, 2-ch Respiration at 16 Hz) plus per-second validity masks,
and strictly dev-only unlabeled resting-state windows for self-supervised
pretraining.
"""

from __future__ import annotations

import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from decoding.extract import (
    EEG_UNITS,
    _eeg_source,
    _read_body,
    _windows,
)
from decoding.ssl_audit import CANONICAL_62
from habs.signals import eda_to_conductance
from habs.temporal.data import physical_id

WIN_START_S = -3.0
WIN_SECONDS = 15
EEG_FS = 128
ECG_FS = 128
SLOW_FS = 16


def _resample_2d(
    x: np.ndarray, src_fs: float, dst_fs: int, target_len: int
) -> np.ndarray:
    """Resample ``(..., N)`` safely to exact ``target_len`` samples."""
    up = int(round(dst_fs))
    down = max(1, int(round(src_fs)))
    g = int(np.gcd(up, down))
    y = resample_poly(x, up // g, down // g, axis=-1)
    if y.shape[-1] > target_len:
        y = y[..., :target_len]
    elif y.shape[-1] < target_len:
        pad_width = [(0, 0)] * (y.ndim - 1) + [(0, target_len - y.shape[-1])]
        y = np.pad(y, pad_width, mode="edge")
    return y


def _robust_scale(x: np.ndarray, floor: float = 1e-4) -> float:
    """IQR-based robust scale estimator resistant to brief recording artifacts."""
    finite = x[np.isfinite(x)]
    if len(finite) < 16:
        return 1.0
    q75, q25 = np.percentile(finite, [75.0, 25.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < floor:
        scale = float(np.std(finite))
    return scale if (np.isfinite(scale) and scale >= floor) else 1.0


def _extract_eeg_128hz(
    path: Path, onsets: np.ndarray, *, target_fs: int = EEG_FS
) -> tuple[np.ndarray, np.ndarray]:
    """Read canonical 62-ch EEG for ``onsets``, bandpass 0.5-45 Hz, resample.

    Returns
    -------
    eeg : np.ndarray
        Shape ``(N_trials, 62, WIN_SECONDS, target_fs)`` in microvolts (float32).
    valid : np.ndarray
        Boolean validity mask of shape ``(N_trials, WIN_SECONDS)``.
    """
    n_trials = len(onsets)
    n_ch = len(CANONICAL_62)
    eeg_out = np.zeros((n_trials, n_ch, WIN_SECONDS, target_fs), dtype=np.float32)
    ch_present = np.zeros(n_ch, dtype=bool)
    bad_sec = np.zeros((n_trials, WIN_SECONDS), dtype=bool)

    with pyedflib.EdfReader(str(path)) as reader:
        labels = reader.getSignalLabels()
        lbl_to_idx = {lbl.upper(): i for i, lbl in enumerate(labels)}
        n_raw_target = WIN_SECONDS * target_fs
        sos_cache: dict[int, np.ndarray] = {}

        for c_i, ch_name in enumerate(CANONICAL_62):
            idx = lbl_to_idx.get(ch_name.upper())
            if idx is None:
                continue
            unit = reader.getPhysicalDimension(idx).replace("µ", "u").replace("μ", "u")
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
            win = _windows(sig, fs, onsets, WIN_START_S, float(WIN_SECONDS))
            # Blank non-finite or saturated (>500 uV ptp in full window) trials
            finite_mask = np.isfinite(win).all(axis=1)
            win_clean = np.where(np.isfinite(win), win, 0.0)
            res = _resample_2d(win_clean, fs, target_fs, n_raw_target)
            res = res.reshape(n_trials, WIN_SECONDS, target_fs)
            # Flag per-second saturation (>500 uV ptp) or flatline (<0.01 uV std)
            ptp = np.ptp(res, axis=-1)
            sd = np.std(res, axis=-1)
            sec_bad = (~finite_mask[:, None]) | (ptp > 500.0) | (sd < 0.01)
            res[sec_bad] = 0.0
            eeg_out[:, c_i, :, :] = res.astype(np.float32)
            ch_present[c_i] = True
            bad_sec |= sec_bad

    # Common average reference across present channels (idempotent for PREPROCESSED)
    if ch_present.sum() >= 30:
        car = eeg_out[:, ch_present, :, :].mean(axis=1, keepdims=True)
        eeg_out[:, ch_present, :, :] -= car
        valid = ~bad_sec
    else:
        valid = np.zeros((n_trials, WIN_SECONDS), dtype=bool)
    return eeg_out, valid


def _extract_body_multirate(
    curated_path: Path, onsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 2-ch ECG (128 Hz), 2-ch EDA (16 Hz), 2-ch Resp (16 Hz) + masks.

    Returns
    -------
    ecg : ``(N, 2, WIN_SECONDS, 128)`` float32
    eda : ``(N, 2, WIN_SECONDS, 16)`` float32
    resp : ``(N, 2, WIN_SECONDS, 16)`` float32
    valid_body : ``(N, 3, WIN_SECONDS)`` bool for ``(ecg, eda, resp)``
    """
    n_trials = len(onsets)
    ecg_out = np.zeros((n_trials, 2, WIN_SECONDS, ECG_FS), dtype=np.float32)
    eda_out = np.zeros((n_trials, 2, WIN_SECONDS, SLOW_FS), dtype=np.float32)
    resp_out = np.zeros((n_trials, 2, WIN_SECONDS, SLOW_FS), dtype=np.float32)
    valid_body = np.zeros((n_trials, 3, WIN_SECONDS), dtype=bool)

    ecg, ecg_fs, resp, resp_fs, eda, eda_fs, eda_unit, _ = _read_body(curated_path)

    # 1. ECG: polarity-aligned 0.5-40 Hz waveform + 5-25 Hz rectified QRS envelope
    if ecg is not None and ecg_fs >= 100 and np.isfinite(ecg).all():
        y_ecg = sosfiltfilt(
            butter(3, (0.5, 40.0), btype="bandpass", fs=ecg_fs, output="sos"),
            ecg.astype(np.float64),
        )
        # Quick polarity alignment via 99th vs 1st percentile skew (R-wave upright)
        q99, q01 = np.percentile(y_ecg, [99.5, 0.5])
        if abs(q01) > abs(q99):
            y_ecg = -y_ecg
        y_ecg = np.clip(y_ecg / _robust_scale(y_ecg), -10.0, 10.0)

        y5 = sosfiltfilt(
            butter(2, (5.0, 25.0), btype="bandpass", fs=ecg_fs, output="sos"),
            ecg.astype(np.float64),
        )
        w_env = max(1, round(0.08 * ecg_fs))
        env = np.convolve(np.abs(y5), np.ones(w_env) / w_env, mode="same")
        env = np.clip((env - np.median(env)) / _robust_scale(env), -5.0, 10.0)

        w0 = _windows(y_ecg, ecg_fs, onsets, WIN_START_S, float(WIN_SECONDS))
        w1 = _windows(env, ecg_fs, onsets, WIN_START_S, float(WIN_SECONDS))
        ok = np.isfinite(w0).all(axis=1) & np.isfinite(w1).all(axis=1)
        r0 = _resample_2d(np.nan_to_num(w0), ecg_fs, ECG_FS, WIN_SECONDS * ECG_FS)
        r1 = _resample_2d(np.nan_to_num(w1), ecg_fs, ECG_FS, WIN_SECONDS * ECG_FS)
        r0 = r0.reshape(n_trials, WIN_SECONDS, ECG_FS)
        r1 = r1.reshape(n_trials, WIN_SECONDS, ECG_FS)
        r0[~ok] = 0.0
        r1[~ok] = 0.0
        ecg_out[:, 0] = r0.astype(np.float32)
        ecg_out[:, 1] = r1.astype(np.float32)
        valid_body[:, 0, :] = ok[:, None]

    # 2. EDA: calibrated conductance (uS) + phasic temporal derivative
    if eda is not None and eda_fs >= 10:
        cond, _ = eda_to_conductance(eda, eda_unit, max_invalid_fraction=1.0)
        w_sc = _windows(cond, eda_fs, onsets, WIN_START_S, float(WIN_SECONDS))
        finite_tr = np.isfinite(w_sc).all(axis=1) & (np.nanmin(w_sc, axis=1) > 0.0)
        if finite_tr.any():
            sc_scale = _robust_scale(cond[np.isfinite(cond)], floor=0.05)
            # Causal per-trial baseline-referenced log conductance + derivative
            r_sc = _resample_2d(
                np.nan_to_num(w_sc), eda_fs, SLOW_FS, WIN_SECONDS * SLOW_FS
            )
            r_sc = r_sc.reshape(n_trials, WIN_SECONDS, SLOW_FS)
            # Pre-stimulus baseline mean per trial (bins 0..2 = [-3, 0) s)
            pre_mean = r_sc[:, :3, :].mean(axis=(1, 2), keepdims=True)
            sc_norm = np.clip((r_sc - pre_mean) / sc_scale, -8.0, 8.0)
            flat = r_sc.reshape(n_trials, -1)
            dsc = np.diff(flat, axis=1, prepend=flat[:, :1]) * SLOW_FS
            dsc_norm = np.clip(dsc / max(sc_scale, 0.05), -8.0, 8.0).reshape(
                n_trials, WIN_SECONDS, SLOW_FS
            )
            sc_norm[~finite_tr] = 0.0
            dsc_norm[~finite_tr] = 0.0
            eda_out[:, 0] = sc_norm.astype(np.float32)
            eda_out[:, 1] = dsc_norm.astype(np.float32)
            valid_body[:, 1, :] = finite_tr[:, None]

    # 3. Respiration: 0.1-1.0 Hz waveform + Hilbert amplitude envelope
    if resp is not None and resp_fs >= 10 and np.isfinite(resp).all():
        y_r = sosfiltfilt(
            butter(2, (0.1, 1.0), btype="bandpass", fs=resp_fs, output="sos"),
            resp.astype(np.float64),
        )
        r_scale = _robust_scale(y_r)
        if r_scale > 0:
            y_rn = np.clip(y_r / r_scale, -8.0, 8.0)
            env_r = np.abs(hilbert(y_rn))
            env_rn = np.clip(
                (env_r - np.median(env_r)) / _robust_scale(env_r), -5.0, 8.0
            )
            wr0 = _windows(y_rn, resp_fs, onsets, WIN_START_S, float(WIN_SECONDS))
            wr1 = _windows(env_rn, resp_fs, onsets, WIN_START_S, float(WIN_SECONDS))
            ok_r = np.isfinite(wr0).all(axis=1) & np.isfinite(wr1).all(axis=1)
            rr0 = _resample_2d(
                np.nan_to_num(wr0), resp_fs, SLOW_FS, WIN_SECONDS * SLOW_FS
            ).reshape(n_trials, WIN_SECONDS, SLOW_FS)
            rr1 = _resample_2d(
                np.nan_to_num(wr1), resp_fs, SLOW_FS, WIN_SECONDS * SLOW_FS
            ).reshape(n_trials, WIN_SECONDS, SLOW_FS)
            rr0[~ok_r] = 0.0
            rr1[~ok_r] = 0.0
            resp_out[:, 0] = rr0.astype(np.float32)
            resp_out[:, 1] = rr1.astype(np.float32)
            valid_body[:, 2, :] = ok_r[:, None]

    return ecg_out, eda_out, resp_out, valid_body


def normalize_eeg_windows(
    eeg_uv: np.ndarray, *, mode: str = "pre_rms", return_scale: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Normalize ``(N, 62, T, 128)`` EEG microvolts without future leakage.

    Modes
    -----
    - ``pre_rms``: divide each channel of each trial by its own causal
      pre-stimulus ``[-3, 0) s`` RMS (floor 5 uV) and clip to ``[-10, 10]``.
      Removes static electrode impedance / baseline gain while preserving
      evoked amplitude shifts (e.g. alpha desynchronization, delta burst).
    - ``fixed``: divide uniformly by ``25.0 uV`` and clip to ``[-10, 10]``.
      Retains inter-subject baseline amplitude differences (for the Identity
      Shortcut ablation).
    """
    if mode == "fixed":
        arr = np.clip(eeg_uv / 25.0, -10.0, 10.0).astype(np.float16)
        if return_scale:
            scale = np.full((len(eeg_uv), eeg_uv.shape[1]), 25.0, dtype=np.float16)
            return arr, scale
        return arr
    if mode == "pre_rms":
        pre = eeg_uv[:, :, :3, :]  # (N, 62, 3, 128)
        rms = np.sqrt(np.mean(pre.astype(np.float32) ** 2, axis=(2, 3), keepdims=True))
        # Fall back to whole-window RMS only if pre window was blanked
        fallback = np.sqrt(
            np.mean(eeg_uv.astype(np.float32) ** 2, axis=(2, 3), keepdims=True)
        )
        scale = np.where(rms > 1.0, rms, fallback)
        scale = np.maximum(scale, 5.0)
        arr = np.clip(eeg_uv / scale, -10.0, 10.0).astype(np.float16)
        if return_scale:
            return arr, scale[:, :, 0, 0].astype(np.float16)
        return arr
    raise ValueError(f"Unknown EEG norm mode: {mode}")


def _worker_recording(args: tuple[str, list[dict], str]) -> dict:
    """Extract all trials of one emotion-video recording in float16."""
    raw_dir, trials, eeg_norm = args
    curated = Path(raw_dir) / trials[0]["recording"]
    onsets = np.array([float(t["onset_s"]) for t in trials], dtype=np.float64)
    trial_ids = [physical_id(t) for t in trials]

    eeg_path, shift = _eeg_source(raw_dir, trials[0])
    candidates = [(eeg_path, shift)] + ([(curated, 0.0)] if eeg_path != curated else [])
    eeg_uv = np.zeros(
        (len(trials), len(CANONICAL_62), WIN_SECONDS, EEG_FS), dtype=np.float32
    )
    valid_eeg = np.zeros((len(trials), WIN_SECONDS), dtype=bool)
    for path, offset in candidates:
        try:
            eeg_uv, valid_eeg = _extract_eeg_128hz(path, onsets + offset)
            break
        except (OSError, ValueError):
            continue

    ecg, eda, resp, valid_body = _extract_body_multirate(curated, onsets)
    valid = np.concatenate([valid_eeg[:, None, :], valid_body], axis=1)  # (N, 4, 15)
    eeg_normed, eeg_scale = normalize_eeg_windows(
        eeg_uv, mode=eeg_norm, return_scale=True
    )
    return {
        "trial_ids": trial_ids,
        "eeg": eeg_normed,
        "eeg_scale": eeg_scale,
        "ecg": ecg.astype(np.float16),
        "eda": eda.astype(np.float16),
        "resp": resp.astype(np.float16),
        "valid": valid,
    }


def load_dataset_multimodal(
    trials: pd.DataFrame,
    raw_dir: Path,
    *,
    eeg_norm: str = "pre_rms",
    workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load aligned 15-s multimodal arrays for all rows of ``trials`` in order."""
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    id_to_pos = {tid: i for i, tid in enumerate(trials["trial_id"])}
    n = len(trials)

    eeg = np.zeros((n, len(CANONICAL_62), WIN_SECONDS, EEG_FS), dtype=np.float16)
    eeg_scale = np.ones((n, len(CANONICAL_62)), dtype=np.float16)
    ecg = np.zeros((n, 2, WIN_SECONDS, ECG_FS), dtype=np.float16)
    eda = np.zeros((n, 2, WIN_SECONDS, SLOW_FS), dtype=np.float16)
    resp = np.zeros((n, 2, WIN_SECONDS, SLOW_FS), dtype=np.float16)
    valid = np.zeros((n, 4, WIN_SECONDS), dtype=bool)

    jobs = [
        (str(raw_dir), grp.to_dict("records"), eeg_norm)
        for _, grp in trials.groupby("recording", sort=True)
    ]
    ctx = multiprocessing.get_context("spawn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for idx, part in enumerate(pool.map(_worker_recording, jobs), 1):
            positions = np.array([id_to_pos[t] for t in part["trial_ids"]], dtype=int)
            eeg[positions] = part["eeg"]
            eeg_scale[positions] = part["eeg_scale"]
            ecg[positions] = part["ecg"]
            eda[positions] = part["eda"]
            resp[positions] = part["resp"]
            valid[positions] = part["valid"]
            if idx % 50 == 0 or idx == len(jobs):
                print(f"Loaded multimodal recordings: {idx}/{len(jobs)}", flush=True)

    return {
        "eeg": eeg,
        "eeg_scale": eeg_scale,
        "ecg": ecg,
        "eda": eda,
        "resp": resp,
        "valid": valid,
    }


def _worker_resting(args: tuple[str, str, str]) -> dict | None:
    """Slice non-overlapping 15-s windows from one resting-state recording."""
    prep_str, cur_str, eeg_norm = args
    prep_path, cur_path = Path(prep_str), Path(cur_str)
    if not (prep_path.is_file() and cur_path.is_file()):
        return None
    try:
        with pyedflib.EdfReader(str(prep_path)) as reader:
            dur_s = float(reader.getFileDuration())
        # Align segment clock if metadata exists
        shift = 0.0
        cur_meta = Path(
            str(cur_path).replace(
                "_task-resting-state_multimodal.bdf", "_metadata.json"
            )
        )
        prep_meta = Path(
            str(prep_path).replace(
                "_task-resting-state_multimodal.bdf", "_metadata.json"
            )
        )
        if cur_meta.is_file() and prep_meta.is_file():
            c_seg = json.loads(cur_meta.read_text())["dataset_export"]["segments"].get(
                "resting-state", {}
            )
            p_seg = json.loads(prep_meta.read_text())["dataset_export"]["segments"].get(
                "resting-state", {}
            )
            co, po = (
                c_seg.get("metadata_time_origin_s"),
                p_seg.get("metadata_time_origin_s"),
            )
            if co is not None and po is not None:
                shift = float(co) - float(po)

        # Pseudo-onsets at 4, 19, 34, ... s so [-3, +12) tiles cleanly inside [1, dur-1]
        onsets = np.arange(4.0, dur_s - 13.0, float(WIN_SECONDS))
        if len(onsets) == 0:
            return None
        eeg_uv, valid_eeg = _extract_eeg_128hz(prep_path, onsets + shift)
        ecg, eda, resp, valid_body = _extract_body_multirate(cur_path, onsets)
        valid = np.concatenate([valid_eeg[:, None, :], valid_body], axis=1)
        keep = valid[:, 0, :].mean(axis=1) >= 0.8
        if not keep.any():
            return None
        return {
            "pid": prep_path.parent.name,
            "eeg": normalize_eeg_windows(eeg_uv[keep], mode=eeg_norm),
            "ecg": ecg[keep].astype(np.float16),
            "eda": eda[keep].astype(np.float16),
            "resp": resp[keep].astype(np.float16),
            "valid": valid[keep],
        }
    except Exception:
        return None


def load_dev_resting_windows(
    raw_dir: Path,
    allowed_people: set[str],
    *,
    eeg_norm: str = "pre_rms",
    workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load strictly dev-only unlabeled resting-state 15-s windows."""
    prep_files = sorted(
        (raw_dir / "PREPROCESSED").glob("P*/P*_task-resting-state_multimodal.bdf")
    )
    jobs = []
    for pf in prep_files:
        pid = pf.parent.name
        if pid not in allowed_people:
            continue
        cf = Path(str(pf).replace("/PREPROCESSED/", "/CURATED/", 1))
        if cf.is_file():
            jobs.append((str(pf), str(cf), eeg_norm))

    if not jobs:
        return {
            "pid": np.array([], dtype=str),
            "eeg": np.zeros(
                (0, len(CANONICAL_62), WIN_SECONDS, EEG_FS), dtype=np.float16
            ),
            "ecg": np.zeros((0, 2, WIN_SECONDS, ECG_FS), dtype=np.float16),
            "eda": np.zeros((0, 2, WIN_SECONDS, SLOW_FS), dtype=np.float16),
            "resp": np.zeros((0, 2, WIN_SECONDS, SLOW_FS), dtype=np.float16),
            "valid": np.zeros((0, 4, WIN_SECONDS), dtype=bool),
        }

    parts = []
    ctx = multiprocessing.get_context("spawn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for res in pool.map(_worker_resting, jobs):
            if res is not None:
                parts.append(res)

    if not parts:
        return load_dev_resting_windows(raw_dir, set(), eeg_norm=eeg_norm, workers=1)

    return {
        "pid": np.concatenate([np.full(len(p["eeg"]), p["pid"]) for p in parts]),
        "eeg": np.concatenate([p["eeg"] for p in parts], axis=0),
        "ecg": np.concatenate([p["ecg"] for p in parts], axis=0),
        "eda": np.concatenate([p["eda"] for p in parts], axis=0),
        "resp": np.concatenate([p["resp"] for p in parts], axis=0),
        "valid": np.concatenate([p["valid"] for p in parts], axis=0),
    }
