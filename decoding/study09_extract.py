"""Study-09 strictly causal multimodal extraction, causal REVE-base, and oculo audit.

Runs ONCE on the HABS H100 backend (~6-10 min). Enforces a **hard causal wall at
t = +5.000 s** for every past predictor (EEG[-3.2, 5.0) s, Body[-3, 5) s,
artefact proxies, and frozen REVE-base embeddings):

- ZERO samples at ``t >= +5.000 s`` enter filtering (forward-only ``sosfilt``),
  resampling, CAR, z-scoring, R-peak detection, conductance smoothing,
  respiratory scaling, or ``REVE-base`` attention!
- Extracts strictly isolated future autonomic target windows at prespecified
  physiological delays: ``[5, 10) s`` (primary), ``[5, 8) s``, ``[6, 9) s``
  (SCR delay), ``[7, 10) s`` (2-s guard gap), and ``[10, 12) s`` (5-s guard gap).
- Extracts frozen ``REVE-base`` representations under both **Protocol A
  (Cold-Start)** and **Protocol B (Causal Unlabeled History + Resting-State)**,
  plus **AC2 (46 interior non-rim channels only)** and **NC4 (time-shuffled
  [0, 5) s)** controls.
- Performs a complete quantitative **Pupil / Gaze Audit** across all 508
  ``emotion-video`` recordings (coverage, binocular correlation, evoked onset
  pupillary reflex alignment test by cohort) and saves compact causal
  oculometry baselines (gated downstream by the pre-registered H4 rule).
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
import pandas as pd
import pyedflib
from scipy.integrate import trapezoid
from scipy.signal import (
    butter,
    find_peaks,
    lfilter,
    resample_poly,
    sosfilt,
    sosfilt_zi,
    welch,
)

from decoding.common import BANDS, REGIONS
from decoding.extract import EEG_UNITS, _eeg_source, _read_body, _windows
from decoding.revision import INTERIOR_CH, LEFT_FRONTAL, RIGHT_FRONTAL, RIM_CH
from decoding.ssl_audit import CANONICAL_62
from habs.signals import eda_to_conductance
from habs.temporal.data import physical_id

REVE_FS = 200
PAST_START_S = (
    -3.2
)  # [-3.2, +5.0) s = 8.2 s = 1640 samples -> exact 9 REVE patches ending at +5.000 s
PAST_END_S = 5.0
PAST_WARMUP_S = 0.5
VID_START_S = -3.0
VID_END_S = 10.0

BAND_EDGES = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
INTERIOR_46_IDX = tuple(i for i, ch in enumerate(CANONICAL_62) if ch not in set(RIM_CH))
INTERIOR_46_NAMES = tuple(CANONICAL_62[i] for i in INTERIOR_46_IDX)
OCULO_BINS = 52  # [-3.0, +10.0) s in 0.25-s bins for alignment reflex audit


# ---------------------------------------------------------------------------
# 1. Strictly causal filtering, resampling, and Body [0:5] / Target [5:10+]
# ---------------------------------------------------------------------------


def _resample_2d(
    x: np.ndarray, src_fs: float, dst_fs: int, target_len: int
) -> np.ndarray:
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


def causal_bandpass_windows(
    sig: np.ndarray,
    fs: float,
    onsets: np.ndarray,
    start_s: float,
    end_s: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 4,
    warmup_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract strictly causal forward-filtered windows on ``[start_s, end_s)``.

    Never accesses any sample at or after ``onset + end_s``.
    Returns ``(windows, finite_mask)``.
    """
    n_tr = len(onsets)
    n_warm = max(0, round(warmup_s * fs))
    n_keep = round((end_s - start_s) * fs)
    n_tot = n_warm + n_keep
    raw_w = np.full((n_tr, n_tot), np.nan, dtype=np.float64)
    for k, onset in enumerate(onsets):
        a = round((onset + start_s) * fs) - n_warm
        b = a + n_tot
        if 0 <= a and b <= len(sig):
            raw_w[k] = sig[a:b]
    fin = np.isfinite(raw_w).all(axis=1)
    out = np.full((n_tr, n_keep), np.nan, dtype=np.float64)
    if not fin.any():
        return out, fin
    hi = min(float(high_hz), 0.45 * fs)
    sos = butter(order, (float(low_hz), hi), btype="bandpass", fs=fs, output="sos")
    zi_unit = sosfilt_zi(sos)
    clean = raw_w[fin]
    # Forward-only causal filter initialized at leading sample of each trial
    filt = np.empty_like(clean)
    for i in range(len(clean)):
        y_i, _ = sosfilt(sos, clean[i], zi=zi_unit * clean[i, 0])
        filt[i] = y_i
    out[fin] = filt[:, n_warm : n_warm + n_keep]
    return out, fin


def _causal_r_peaks_in_window(
    ecg_seg: np.ndarray, fs: float, start_offset_s: float
) -> tuple[np.ndarray, float]:
    """Detect R-peaks strictly inside one isolated ECG segment (causal filter)."""
    if len(ecg_seg) < round(1.5 * fs) or not np.isfinite(ecg_seg).all():
        return np.array([], dtype=float), np.nan
    sos = butter(2, (5.0, min(25.0, 0.45 * fs)), btype="bandpass", fs=fs, output="sos")
    zi = sosfilt_zi(sos)
    y5, _ = sosfilt(sos, ecg_seg.astype(np.float64), zi=zi * float(ecg_seg[0]))
    w_env = max(1, round(0.08 * fs))
    # Strictly causal trailing moving average of |y5|
    env = lfilter(np.ones(w_env) / w_env, [1.0], np.abs(y5))
    # Compute prominence scale strictly on this segment (skip first 0.5 s transient)
    skip = min(len(env) // 4, round(0.5 * fs))
    sd = float(np.std(env[skip:]))
    if not sd > 1e-8:
        return np.array([], dtype=float), np.nan
    peaks, _ = find_peaks(env, distance=round(0.32 * fs), prominence=0.7 * sd)
    times = start_offset_s + peaks / fs
    if len(times) < 2:
        return times, np.nan
    rr = np.diff(times)
    plaus = (rr >= 0.35) & (rr <= 1.50)
    return times, float(plaus.mean())


def _rr_stats_strict(
    beats: np.ndarray, w0: float, w1: float
) -> tuple[float, float, float, float, int]:
    """Compute HR, RR (ms), RMSSD, SDNN, n_rr strictly from beats in [w0, w1)."""
    inside = beats[(beats >= w0) & (beats < w1)]
    if len(inside) < 2:
        return np.nan, np.nan, np.nan, np.nan, 0
    rr = np.diff(inside)
    good = (rr >= 0.35) & (rr <= 1.50)
    if good.sum() < 1 or good.mean() < 0.7:
        return np.nan, np.nan, np.nan, np.nan, 0
    rr_g = rr[good]
    rr_mean = float(rr_g.mean())
    hr = 60.0 / rr_mean
    rmssd = (
        float(np.sqrt(np.mean(np.diff(rr_g) ** 2)) * 1000.0)
        if len(rr_g) >= 2
        else np.nan
    )
    sdnn = float(np.std(rr_g) * 1000.0) if len(rr_g) >= 2 else np.nan
    return hr, rr_mean * 1000.0, rmssd, sdnn, int(len(rr_g))


def extract_causal_body_past(
    ecg: np.ndarray | None,
    ecg_fs: float,
    eda: np.ndarray | None,
    eda_fs: float,
    eda_unit: str | None,
    resp: np.ndarray | None,
    resp_fs: float,
    onsets: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract strictly causal ``Body [-3, 5) s`` features (ZERO samples t >= 5.0 s).

    Returns ``(features_dict, refs_dict)`` where ``refs_dict`` holds per-trial
    causal baseline references (``ecg_hr_ref``, ``ecg_rr_ref``, ``sc_ref``,
    ``sc_last_1s``, ``resp_pre_scale``, ``resp_amp_ref``, ``resp_rate_ref``,
    plus early ``[0, 5)`` means for incremental targets).
    """
    n = len(onsets)
    feats: dict[str, np.ndarray] = {}
    refs: dict[str, np.ndarray] = {}

    def _nan():
        return np.full(n, np.nan, dtype=np.float32)

    # --- 1. ECG strictly on [-5.0, +5.0) s ---
    ecg_keys = (
        "b05_ecg_hr_ref",
        "b05_ecg_hr_e1",
        "b05_ecg_hr_e2",
        "b05_ecg_hr_early",
        "b05_ecg_hr_chg",
        "b05_ecg_hr_slope",
        "b05_ecg_rr_ref",
        "b05_ecg_rr_early",
        "b05_ecg_rr_chg",
        "b05_ecg_rmssd_early",
        "b05_ecg_sdnn_early",
        "b05_ecg_last_rr",
        "b05_ecg_prev_rr",
        "b05_ecg_last_rr_diff",
        "b05_ecg_time_since_r",
        "b05_ecg_qrs_valid_frac",
        *(f"b05_ecg_hr_bin{b}" for b in range(5)),
    )
    for k in ecg_keys:
        feats[k] = _nan()

    if ecg is not None and ecg_fs >= 100:
        w_ecg = _windows(ecg.astype(np.float64), ecg_fs, onsets, -5.0, 10.0)
        for i in range(n):
            seg = w_ecg[i]
            if not np.isfinite(seg).all():
                continue
            beats, qfrac = _causal_r_peaks_in_window(seg, ecg_fs, -5.0)
            # Keep strictly beats < 5.0
            beats = beats[beats < 5.0]
            feats["b05_ecg_qrs_valid_frac"][i] = qfrac
            hr_ref, rr_ref, _, _, _ = _rr_stats_strict(beats, -3.5, 0.0)
            hr_e1, _, _, _, _ = _rr_stats_strict(beats, 0.0, 2.5)
            hr_e2, _, _, _, _ = _rr_stats_strict(beats, 2.5, 5.0)
            hr_ea, rr_ea, rmssd_ea, sdnn_ea, _ = _rr_stats_strict(beats, 0.0, 5.0)
            feats["b05_ecg_hr_ref"][i] = hr_ref
            feats["b05_ecg_hr_e1"][i] = hr_e1
            feats["b05_ecg_hr_e2"][i] = hr_e2
            feats["b05_ecg_hr_early"][i] = hr_ea
            feats["b05_ecg_hr_chg"][i] = hr_ea - hr_ref
            feats["b05_ecg_hr_slope"][i] = hr_e2 - hr_e1
            feats["b05_ecg_rr_ref"][i] = rr_ref
            feats["b05_ecg_rr_early"][i] = rr_ea
            feats["b05_ecg_rr_chg"][i] = rr_ea - rr_ref
            feats["b05_ecg_rmssd_early"][i] = rmssd_ea
            feats["b05_ecg_sdnn_early"][i] = sdnn_ea

            # Strictly causal 1-s backward-held instantaneous HR bins in [0, 5)
            if len(beats) >= 2:
                rr_all = np.diff(beats)
                ok_rr = (rr_all >= 0.35) & (rr_all <= 1.50)
                bt_t = beats[1:][ok_rr]
                bt_hr = 60.0 / rr_all[ok_rr]
                bt_rr = rr_all[ok_rr] * 1000.0
                if len(bt_t) >= 1:
                    feats["b05_ecg_last_rr"][i] = bt_rr[-1]
                    feats["b05_ecg_time_since_r"][i] = 5.0 - float(bt_t[-1])
                    if len(bt_t) >= 2:
                        feats["b05_ecg_prev_rr"][i] = bt_rr[-2]
                        feats["b05_ecg_last_rr_diff"][i] = bt_rr[-1] - bt_rr[-2]
                    for b in range(5):
                        # Only beats strictly <= b + 1.0 (backward hold, zero future!)
                        prior = bt_t <= float(b + 1)
                        in_bin = (bt_t >= float(b)) & prior
                        if in_bin.any():
                            val = float(bt_hr[in_bin].mean())
                        elif prior.any():
                            val = float(bt_hr[prior][-1])
                        else:
                            val = np.nan
                        feats[f"b05_ecg_hr_bin{b}"][i] = val - hr_ref

    refs["ecg_hr_ref"] = feats["b05_ecg_hr_ref"].copy()
    refs["ecg_rr_ref"] = feats["b05_ecg_rr_ref"].copy()
    refs["ecg_hr_early"] = feats["b05_ecg_hr_early"].copy()

    # --- 2. EDA strictly on [-5.0, +5.0) s ---
    eda_keys = (
        "b05_sc_ref",
        "b05_sc_e1",
        "b05_sc_e2",
        "b05_sc_early",
        "b05_sc_chg",
        "b05_sc_slope",
        "b05_sc_last_1s",
        "b05_sc_last_05s",
        "b05_sc_rise_early",
        "b05_sc_deriv_3_5",
        "b05_sc_deriv_max_early",
        "b05_sc_scr_count_early",
        "b05_sc_valid",
        *(f"b05_sc_bin{b}" for b in range(5)),
    )
    for k in eda_keys:
        feats[k] = _nan()
    feats["b05_sc_valid"] = np.zeros(n, dtype=np.float32)
    sc_last_raw = _nan()

    if eda is not None and eda_fs >= 10:
        w_raw = _windows(eda.astype(np.float64), eda_fs, onsets, -5.0, 10.0)
        w_sm = max(1, round(0.25 * eda_fs))
        kern = np.ones(w_sm) / w_sm
        idx_m3 = round(2.0 * eda_fs)  # -3.0 s relative to -5.0 s start
        idx_0 = round(5.0 * eda_fs)
        idx_25 = round(7.5 * eda_fs)
        idx_3 = round(8.0 * eda_fs)
        idx_4 = round(9.0 * eda_fs)
        idx_45 = round(9.5 * eda_fs)
        step05 = max(1, round(0.5 * eda_fs))

        for i in range(n):
            seg = w_raw[i]
            if not np.isfinite(seg).all():
                continue
            cond, _ = eda_to_conductance(seg, eda_unit, max_invalid_fraction=1.0)
            if not np.isfinite(cond[idx_m3:]).all() or cond[idx_m3:].min() <= 0.0:
                continue
            # Strictly causal trailing 0.25-s smoothing
            sm = lfilter(
                kern, [1.0], np.nan_to_num(cond, nan=float(np.nanmedian(cond)))
            )
            ref_v = float(sm[idx_m3:idx_0].mean())
            e1_v = float(sm[idx_0:idx_25].mean())
            e2_v = float(sm[idx_25:].mean())
            ea_v = float(sm[idx_0:].mean())
            l1_v = float(sm[idx_4:].mean())
            l05_v = float(sm[idx_45:].mean())
            feats["b05_sc_valid"][i] = 1.0
            feats["b05_sc_ref"][i] = ref_v
            feats["b05_sc_e1"][i] = e1_v
            feats["b05_sc_e2"][i] = e2_v
            feats["b05_sc_early"][i] = ea_v
            feats["b05_sc_chg"][i] = ea_v - ref_v
            feats["b05_sc_slope"][i] = e2_v - e1_v
            feats["b05_sc_last_1s"][i] = l1_v - ref_v
            feats["b05_sc_last_05s"][i] = l05_v - ref_v
            feats["b05_sc_rise_early"][i] = float(sm[idx_0:].max()) - ref_v
            feats["b05_sc_deriv_3_5"][i] = (float(sm[-1]) - float(sm[idx_3])) / 2.0
            ea_arr = sm[idx_0:]
            if len(ea_arr) > step05:
                feats["b05_sc_deriv_max_early"][i] = float(
                    np.max(ea_arr[step05:] - ea_arr[:-step05]) / 0.5
                )
            pks, _ = find_peaks(
                ea_arr, prominence=0.01, distance=max(1, round(1.0 * eda_fs))
            )
            feats["b05_sc_scr_count_early"][i] = float(len(pks))
            sc_last_raw[i] = l1_v
            for b in range(5):
                a_b = idx_0 + round(b * eda_fs)
                b_b = idx_0 + round((b + 1) * eda_fs)
                feats[f"b05_sc_bin{b}"][i] = float(sm[a_b:b_b].mean()) - ref_v

    refs["sc_ref"] = feats["b05_sc_ref"].copy()
    refs["sc_early"] = feats["b05_sc_early"].copy()
    refs["sc_last_1s_raw"] = sc_last_raw

    # --- 3. Respiration strictly on [-15.0, +5.0) s (ZERO Hilbert across 5 s!) ---
    resp_keys = (
        "b05_resp_amp_ref",
        "b05_resp_amp_e1",
        "b05_resp_amp_e2",
        "b05_resp_amp_early",
        "b05_resp_amp_chg",
        "b05_resp_amp_slope",
        "b05_resp_val_at_5",
        "b05_resp_vel_at_5",
        "b05_resp_acc_at_5",
        "b05_resp_val_at_4",
        "b05_resp_vel_at_4",
        "b05_resp_rate_early",
        "b05_resp_valid",
        *(f"b05_resp_bin{b}" for b in range(5)),
    )
    for k in resp_keys:
        feats[k] = _nan()
    feats["b05_resp_valid"] = np.zeros(n, dtype=np.float32)
    resp_pre_scale = _nan()

    if resp is not None and resp_fs >= 10:
        w_resp = _windows(resp.astype(np.float64), resp_fs, onsets, -15.0, 20.0)
        sos_r = butter(
            2,
            (0.1, min(1.0, 0.45 * resp_fs)),
            btype="bandpass",
            fs=resp_fs,
            output="sos",
        )
        zi_r = sosfilt_zi(sos_r)
        i_m10 = round(5.0 * resp_fs)  # -10 s
        i_m3 = round(12.0 * resp_fs)  # -3 s
        i_0 = round(15.0 * resp_fs)  # 0 s
        i_25 = round(17.5 * resp_fs)  # 2.5 s
        i_4 = round(19.0 * resp_fs)  # 4.0 s
        dt_w = max(1, round(0.25 * resp_fs))

        for i in range(n):
            seg = w_resp[i]
            if not np.isfinite(seg).all():
                continue
            y_r, _ = sosfilt(sos_r, seg, zi=zi_r * float(seg[0]))
            # Strictly pre-onset causal scale [-10, 0) s RMS (zero future!)
            pre_rms = float(np.sqrt(np.mean(y_r[i_m10:i_0] ** 2)))
            if not pre_rms > 1e-8:
                continue
            resp_pre_scale[i] = pre_rms
            yn = np.clip(y_r / pre_rms, -8.0, 8.0)

            amp_ref = float(np.sqrt(np.mean(yn[i_m3:i_0] ** 2)))
            amp_e1 = float(np.sqrt(np.mean(yn[i_0:i_25] ** 2)))
            amp_e2 = float(np.sqrt(np.mean(yn[i_25:] ** 2)))
            amp_ea = float(np.sqrt(np.mean(yn[i_0:] ** 2)))
            feats["b05_resp_valid"][i] = 1.0
            feats["b05_resp_amp_ref"][i] = amp_ref
            feats["b05_resp_amp_e1"][i] = amp_e1
            feats["b05_resp_amp_e2"][i] = amp_e2
            feats["b05_resp_amp_early"][i] = amp_ea
            feats["b05_resp_amp_chg"][i] = amp_ea - amp_ref
            feats["b05_resp_amp_slope"][i] = amp_e2 - amp_e1

            # Oscillatory boundary state (value, velocity, acceleration at t = 4, 5 s)
            v5 = float(yn[-1])
            v475 = float(yn[-1 - dt_w])
            v450 = float(yn[-1 - 2 * dt_w])
            vel5 = (v5 - v475) / (dt_w / resp_fs)
            vel475 = (v475 - v450) / (dt_w / resp_fs)
            feats["b05_resp_val_at_5"][i] = v5
            feats["b05_resp_vel_at_5"][i] = vel5
            feats["b05_resp_acc_at_5"][i] = (vel5 - vel475) / (dt_w / resp_fs)
            feats["b05_resp_val_at_4"][i] = float(yn[i_4])
            feats["b05_resp_vel_at_4"][i] = (float(yn[i_4]) - float(yn[i_4 - dt_w])) / (
                dt_w / resp_fs
            )

            pks, _ = find_peaks(
                yn[i_m10:], distance=round(1.5 * resp_fs), prominence=0.3
            )
            if len(pks) >= 2:
                gaps = np.diff(pks) / resp_fs
                feats["b05_resp_rate_early"][i] = 60.0 / float(gaps.mean())

            for b in range(5):
                a_b = i_0 + round(b * resp_fs)
                b_b = i_0 + round((b + 1) * resp_fs)
                feats[f"b05_resp_bin{b}"][i] = (
                    float(np.sqrt(np.mean(yn[a_b:b_b] ** 2))) - amp_ref
                )

    refs["resp_pre_scale"] = resp_pre_scale
    refs["resp_amp_ref"] = feats["b05_resp_amp_ref"].copy()
    refs["resp_amp_early"] = feats["b05_resp_amp_early"].copy()
    refs["resp_rate_early"] = feats["b05_resp_rate_early"].copy()
    return feats, refs


def extract_future_body_targets(
    ecg: np.ndarray | None,
    ecg_fs: float,
    eda: np.ndarray | None,
    eda_fs: float,
    eda_unit: str | None,
    resp: np.ndarray | None,
    resp_fs: float,
    onsets: np.ndarray,
    refs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract strictly isolated future autonomic targets at prespecified delays.

    Each target window ``[w0, w1)`` uses ONLY beats/samples strictly inside
    ``[w0, w1)`` (with local causal filter warm-up) so zero boundary bleed
    occurs across ``t = 5.0 s``.
    """
    n = len(onsets)

    def _nan():
        return np.full(n, np.nan, dtype=np.float32)

    targets: dict[str, np.ndarray] = {
        # Primary [5, 10) s + incremental [5,10) - [0,5)
        "tgt_ecg_hr_chg_5_10": _nan(),
        "tgt_ecg_rr_chg_5_10": _nan(),
        "tgt_ecg_hr_inc_5_10": _nan(),
        "tgt_ecg_hr_late_raw": _nan(),
        # Prespecified ECG delays
        "tgt_ecg_hr_chg_5_8": _nan(),
        "tgt_ecg_hr_chg_7_10": _nan(),
        "tgt_ecg_hr_chg_10_12": _nan(),
        # Primary EDA [5, 10) s + new post-5s SCR + incremental + delays
        "tgt_eda_sc_chg_5_10": _nan(),
        "tgt_eda_scr_new_5_10": _nan(),
        "tgt_eda_sc_inc_5_10": _nan(),
        "tgt_eda_sc_late_raw": _nan(),
        "tgt_eda_sc_chg_6_9": _nan(),
        "tgt_eda_sc_chg_7_10": _nan(),
        "tgt_eda_sc_chg_10_12": _nan(),
        # Primary Resp [5, 10) s + rate + incremental + delays
        "tgt_resp_amp_chg_5_10": _nan(),
        "tgt_resp_rate_chg_5_10": _nan(),
        "tgt_resp_amp_inc_5_10": _nan(),
        "tgt_resp_amp_late_raw": _nan(),
        "tgt_resp_amp_chg_5_8": _nan(),
        "tgt_resp_amp_chg_7_10": _nan(),
        "tgt_resp_amp_chg_10_12": _nan(),
    }

    # --- 1. ECG future targets (beats strictly inside [w0, w1)) ---
    if ecg is not None and ecg_fs >= 100:
        w_ecg = _windows(
            ecg.astype(np.float64), ecg_fs, onsets, 4.0, 8.0
        )  # [4.0, 12.0)
        for i in range(n):
            seg = w_ecg[i]
            if not np.isfinite(seg).all():
                continue
            beats, _ = _causal_r_peaks_in_window(seg, ecg_fs, 4.0)
            hr_5_10, rr_5_10, _, _, _ = _rr_stats_strict(beats, 5.0, 10.0)
            hr_5_8, _, _, _, _ = _rr_stats_strict(beats, 5.0, 8.0)
            hr_7_10, _, _, _, _ = _rr_stats_strict(beats, 7.0, 10.0)
            hr_10_12, _, _, _, _ = _rr_stats_strict(beats, 10.0, 12.0)

            hr_ref = float(refs["ecg_hr_ref"][i])
            rr_ref = float(refs["ecg_rr_ref"][i])
            hr_ea = float(refs["ecg_hr_early"][i])
            targets["tgt_ecg_hr_late_raw"][i] = hr_5_10
            targets["tgt_ecg_hr_chg_5_10"][i] = hr_5_10 - hr_ref
            targets["tgt_ecg_rr_chg_5_10"][i] = rr_5_10 - rr_ref
            targets["tgt_ecg_hr_inc_5_10"][i] = hr_5_10 - hr_ea
            targets["tgt_ecg_hr_chg_5_8"][i] = hr_5_8 - hr_ref
            targets["tgt_ecg_hr_chg_7_10"][i] = hr_7_10 - hr_ref
            targets["tgt_ecg_hr_chg_10_12"][i] = hr_10_12 - hr_ref

    # --- 2. EDA future targets ---
    if eda is not None and eda_fs >= 10:
        w_eda = _windows(
            eda.astype(np.float64), eda_fs, onsets, 5.0, 7.0
        )  # [5.0, 12.0)
        w_sm = max(1, round(0.25 * eda_fs))
        kern = np.ones(w_sm) / w_sm
        i1 = round(1.0 * eda_fs)
        i2 = round(2.0 * eda_fs)
        i4 = round(4.0 * eda_fs)
        i5 = round(5.0 * eda_fs)
        for i in range(n):
            seg = w_eda[i]
            sc_ref = float(refs["sc_ref"][i])
            if not np.isfinite(sc_ref) or not np.isfinite(seg).all():
                continue
            cond, _ = eda_to_conductance(seg, eda_unit, max_invalid_fraction=1.0)
            if not np.isfinite(cond).all() or cond.min() <= 0.0:
                continue
            sm = lfilter(kern, [1.0], cond)
            sc_5_10 = float(sm[:i5].mean())
            sc_6_9 = float(sm[i1:i4].mean())
            sc_7_10 = float(sm[i2:i5].mean())
            sc_10_12 = float(sm[i5:].mean())
            targets["tgt_eda_sc_late_raw"][i] = sc_5_10
            targets["tgt_eda_sc_chg_5_10"][i] = sc_5_10 - sc_ref
            targets["tgt_eda_sc_inc_5_10"][i] = sc_5_10 - float(refs["sc_early"][i])
            targets["tgt_eda_scr_new_5_10"][i] = float(sm[:i5].max()) - float(
                refs["sc_last_1s_raw"][i]
            )
            targets["tgt_eda_sc_chg_6_9"][i] = sc_6_9 - sc_ref
            targets["tgt_eda_sc_chg_7_10"][i] = sc_7_10 - sc_ref
            targets["tgt_eda_sc_chg_10_12"][i] = sc_10_12 - sc_ref

    # --- 3. Respiration future targets (causal [-15, +12) -> sliced at >= 5 s) ---
    if resp is not None and resp_fs >= 10:
        w_resp = _windows(resp.astype(np.float64), resp_fs, onsets, -15.0, 27.0)
        sos_r = butter(
            2,
            (0.1, min(1.0, 0.45 * resp_fs)),
            btype="bandpass",
            fs=resp_fs,
            output="sos",
        )
        zi_r = sosfilt_zi(sos_r)
        i_5 = round(20.0 * resp_fs)
        i_7 = round(22.0 * resp_fs)
        i_8 = round(23.0 * resp_fs)
        i_10 = round(25.0 * resp_fs)
        i_12 = round(27.0 * resp_fs)
        for i in range(n):
            seg = w_resp[i]
            sc = float(refs["resp_pre_scale"][i])
            if not np.isfinite(sc) or sc <= 1e-8 or not np.isfinite(seg).all():
                continue
            y_r, _ = sosfilt(sos_r, seg, zi=zi_r * float(seg[0]))
            yn = np.clip(y_r / sc, -8.0, 8.0)
            a_5_10 = float(np.sqrt(np.mean(yn[i_5:i_10] ** 2)))
            a_5_8 = float(np.sqrt(np.mean(yn[i_5:i_8] ** 2)))
            a_7_10 = float(np.sqrt(np.mean(yn[i_7:i_10] ** 2)))
            a_10_12 = float(np.sqrt(np.mean(yn[i_10:i_12] ** 2)))
            a_ref = float(refs["resp_amp_ref"][i])
            targets["tgt_resp_amp_late_raw"][i] = a_5_10
            targets["tgt_resp_amp_chg_5_10"][i] = a_5_10 - a_ref
            targets["tgt_resp_amp_inc_5_10"][i] = a_5_10 - float(
                refs["resp_amp_early"][i]
            )
            targets["tgt_resp_amp_chg_5_8"][i] = a_5_8 - a_ref
            targets["tgt_resp_amp_chg_7_10"][i] = a_7_10 - a_ref
            targets["tgt_resp_amp_chg_10_12"][i] = a_10_12 - a_ref
            pks, _ = find_peaks(
                yn[i_5:i_10], distance=round(1.5 * resp_fs), prominence=0.3
            )
            if len(pks) >= 2:
                rate_late = 60.0 / float(np.diff(pks).mean() / resp_fs)
                targets["tgt_resp_rate_chg_5_10"][i] = rate_late - float(
                    refs["resp_rate_early"][i]
                )
    return targets


# ---------------------------------------------------------------------------
# 2. Pupil / Gaze Oculometry Reader & Quantitative Audit (Requirement 11)
# ---------------------------------------------------------------------------


def extract_oculometry_recording(
    curated_path: Path, onsets: np.ndarray
) -> tuple[dict[str, np.ndarray], dict]:
    """Read and audit Oculometry channels via exact ``.record_map`` alignment.

    Follows the exact timeline mapping of ``starting_notebooks/bdf_loader.py``
    without allocating a 1,000-Hz dense DataFrame. Returns per-trial causal
    summaries AND recording-level audit telemetry + 0.25-s evoked pupil curve.
    """
    n = len(onsets)

    def _nan():
        return np.full(n, np.nan, dtype=np.float32)

    feats: dict[str, np.ndarray] = {
        "oc_pupil_ref": _nan(),
        "oc_pupil_early_chg": _nan(),
        "oc_pupil_late_chg": _nan(),
        "oc_pupil_early_slope": _nan(),
        "oc_pupil_early_std": _nan(),
        "oc_dropout_early": _nan(),
        "oc_dropout_late": _nan(),
        "oc_gaze_disp_early": _nan(),
        "oc_gaze_speed_early": _nan(),
        "oc_gaze_disp_late": _nan(),
        "oc_gaze_speed_late": _nan(),
        "oc_binoc_r": _nan(),
        "oc_valid_early": np.zeros(n, dtype=np.float32),
        "oc_valid_late": np.zeros(n, dtype=np.float32),
    }
    audit: dict = {
        "has_record_map": False,
        "has_oculo_stream": False,
        "gaze_rate_hz": None,
        "onset_s": None,
        "n_trials": n,
        "n_valid_early": 0,
        "n_valid_late": 0,
        "binoc_rs": [],
        "evoked_pupil_curves": np.full((n, OCULO_BINS), np.nan, dtype=np.float32),
        "evoked_gaze_speed_curves": np.full((n, OCULO_BINS), np.nan, dtype=np.float32),
    }

    map_path = curated_path.with_suffix(".record_map")
    if not map_path.is_file():
        return feats, audit
    audit["has_record_map"] = True

    try:
        rmap = json.loads(map_path.read_text(encoding="utf-8"))
        groups = [
            (nm, spec)
            for stream in rmap.get("stream_map", {}).values()
            for nm, spec in stream.items()
        ]
        oc_specs = [
            spec
            for _, spec in groups
            if spec.get("correction", {}).get("RecordType", "").lower() == "oculometry"
        ]
        if len(oc_specs) != 1:
            return feats, audit
        spec = oc_specs[0]
        corr = spec["correction"]
        if "channel_ranges" in spec:
            a_c, b_c = spec["channel_ranges"]
            ch_indices = list(range(int(a_c), int(b_c) + 1))
        else:
            ch_indices = list(spec.get("channel_indices", []))
        details = corr.get("Channels", [])
        if len(ch_indices) != len(details):
            return feats, audit

        name_to_idx = {
            str(d.get("name", "")).strip().lower(): idx
            for idx, d in zip(ch_indices, details, strict=True)
        }
        gaze_rate = float(
            corr.get("EffectiveSamplingFrequency", corr.get("SamplingFrequency", 0.0))
        )
        oc_onset = float(corr.get("OnsetSeconds", 0.0))
        if not (gaze_rate > 0 and np.isfinite(oc_onset) and oc_onset >= 0):
            return feats, audit

        with pyedflib.EdfReader(str(curated_path)) as reader:
            sfreq = float(reader.getSampleFrequency(0))
            n_master_bdf = int(reader.getNSamples()[0])
            tot_s = rmap.get("TotalSamples")
            n_master = (
                min(n_master_bdf, tot_s)
                if isinstance(tot_s, int) and tot_s > 0
                else n_master_bdf
            )
            native_count = round(n_master * gaze_rate / sfreq)
            onset_idx = min(max(round(oc_onset * sfreq), 0), n_master - 1)
            filler_count = round(oc_onset * gaze_rate)
            if not (0 < native_count < n_master and filler_count < native_count):
                return feats, audit
            t_gaze = np.linspace(
                onset_idx / sfreq, (n_master - 1) / sfreq, native_count - filler_count
            )
            sample_counts = reader.getNSamples()

            def _read_oc(key: str) -> np.ndarray | None:
                c_idx = name_to_idx.get(key)
                if c_idx is None:
                    return None
                n_read = min(native_count, int(sample_counts[c_idx]))
                arr = reader.readSignal(c_idx, 0, n_read).astype(np.float64)
                if len(arr) < native_count:
                    arr = np.pad(
                        arr, (0, native_count - len(arr)), constant_values=np.nan
                    )
                return arr[filler_count:native_count]

            pl = _read_oc("pupildiameter_left")
            pr = _read_oc("pupildiameter_right")
            gx = _read_oc("gaze_x")
            gy = _read_oc("gaze_y")
    except Exception:
        return feats, audit

    audit["has_oculo_stream"] = True
    audit["gaze_rate_hz"] = round(gaze_rate, 3)
    audit["onset_s"] = round(oc_onset, 4)
    if pl is None and pr is None:
        return feats, audit

    # Physiological pupil validity: finite and strictly in [1.0, 10.0] mm
    pl_ok = (
        np.isfinite(pl) & (pl >= 1.0) & (pl <= 10.0)
        if pl is not None
        else np.zeros(len(t_gaze), dtype=bool)
    )
    pr_ok = (
        np.isfinite(pr) & (pr >= 1.0) & (pr <= 10.0)
        if pr is not None
        else np.zeros(len(t_gaze), dtype=bool)
    )
    pupil = np.full(len(t_gaze), np.nan, dtype=np.float64)
    both = pl_ok & pr_ok
    if pl is not None and pr is not None:
        pupil[both] = 0.5 * (pl[both] + pr[both])
    if pl is not None:
        only_l = pl_ok & (~pr_ok)
        pupil[only_l] = pl[only_l]
    if pr is not None:
        only_r = pr_ok & (~pl_ok)
        pupil[only_r] = pr[only_r]

    for i, ons in enumerate(onsets):
        a_tot = int(np.searchsorted(t_gaze, ons - 3.0))
        b_tot = int(np.searchsorted(t_gaze, ons + 10.0))
        if b_tot - a_tot < 20:
            continue
        t_rel = t_gaze[a_tot:b_tot] - ons
        p_seg = pupil[a_tot:b_tot]

        m_ref = (t_rel >= -3.0) & (t_rel < 0.0)
        m_e1 = (t_rel >= 0.0) & (t_rel < 2.5)
        m_e2 = (t_rel >= 2.5) & (t_rel < 5.0)
        m_ea = (t_rel >= 0.0) & (t_rel < 5.0)
        m_la = (t_rel >= 5.0) & (t_rel < 10.0)

        fin_p = np.isfinite(p_seg)
        drop_ea = 1.0 - float(fin_p[m_ea].mean()) if m_ea.any() else 1.0
        drop_la = 1.0 - float(fin_p[m_la].mean()) if m_la.any() else 1.0
        feats["oc_dropout_early"][i] = drop_ea
        feats["oc_dropout_late"][i] = drop_la

        # Binocular correlation inside [-3, +10) s
        if pl is not None and pr is not None:
            mb = both[a_tot:b_tot]
            if mb.sum() >= 20:
                lv, rv = pl[a_tot:b_tot][mb], pr[a_tot:b_tot][mb]
                if np.std(lv) > 1e-6 and np.std(rv) > 1e-6:
                    r_lr = float(np.corrcoef(lv, rv)[0, 1])
                    feats["oc_binoc_r"][i] = r_lr
                    audit["binoc_rs"].append(r_lr)

        ref_ok = fin_p & m_ref
        ea_ok = fin_p & m_ea
        la_ok = fin_p & m_la
        if ref_ok.sum() >= 10 and drop_ea <= 0.50 and ea_ok.sum() >= 20:
            p_ref = float(p_seg[ref_ok].mean())
            p_ea = float(p_seg[ea_ok].mean())
            feats["oc_pupil_ref"][i] = p_ref
            feats["oc_pupil_early_chg"][i] = p_ea - p_ref
            feats["oc_pupil_early_std"][i] = float(np.std(p_seg[ea_ok]))
            e1_ok, e2_ok = fin_p & m_e1, fin_p & m_e2
            if e1_ok.sum() >= 5 and e2_ok.sum() >= 5:
                feats["oc_pupil_early_slope"][i] = float(
                    p_seg[e2_ok].mean() - p_seg[e1_ok].mean()
                )
            feats["oc_valid_early"][i] = 1.0
            audit["n_valid_early"] += 1

            # 0.25-s evoked pupil curve relative to [-2, 0) s for timing audit
            m_base2 = fin_p & (t_rel >= -2.0) & (t_rel < 0.0)
            base2 = float(p_seg[m_base2].mean()) if m_base2.sum() >= 5 else p_ref
            bins = np.floor((t_rel + 3.0) * 4.0).astype(int)
            for b_k in range(OCULO_BINS):
                mbk = fin_p & (bins == b_k)
                if mbk.sum() >= 2:
                    audit["evoked_pupil_curves"][i, b_k] = (
                        float(p_seg[mbk].mean()) - base2
                    )

            if drop_la <= 0.50 and la_ok.sum() >= 20:
                feats["oc_pupil_late_chg"][i] = float(p_seg[la_ok].mean()) - p_ref
                feats["oc_valid_late"][i] = 1.0
                audit["n_valid_late"] += 1

        if gx is not None and gy is not None:
            x_seg = gx[a_tot:b_tot]
            y_seg = gy[a_tot:b_tot]
            g_ok = (
                np.isfinite(x_seg) & np.isfinite(y_seg) & ((x_seg != 0) | (y_seg != 0))
            )
            for tag, m_w in (("early", m_ea), ("late", m_la)):
                mw = g_ok & m_w
                if mw.sum() >= 20:
                    xs, ys, ts = x_seg[mw], y_seg[mw], t_rel[mw]
                    feats[f"oc_gaze_disp_{tag}"][i] = float(
                        np.sqrt(np.var(xs) + np.var(ys))
                    )
                    dt = np.diff(ts)
                    ok_dt = dt > 1e-4
                    if ok_dt.any():
                        spd = (
                            np.hypot(np.diff(xs)[ok_dt], np.diff(ys)[ok_dt]) / dt[ok_dt]
                        )
                        feats[f"oc_gaze_speed_{tag}"][i] = float(np.median(spd))
            if g_ok.sum() >= 40:
                xs, ys, ts = x_seg[g_ok], y_seg[g_ok], t_rel[g_ok]
                dt = np.diff(ts)
                ok_dt = dt > 1e-4
                if ok_dt.any():
                    spd = np.hypot(np.diff(xs)[ok_dt], np.diff(ys)[ok_dt]) / dt[ok_dt]
                    bins_s = np.floor((ts[1:][ok_dt] + 3.0) * 4.0).astype(int)
                    for b_k in range(OCULO_BINS):
                        ms = bins_s == b_k
                        if ms.sum() >= 2:
                            audit["evoked_gaze_speed_curves"][i, b_k] = float(
                                np.median(spd[ms])
                            )
    return feats, audit


# ---------------------------------------------------------------------------
# 3. Causal Handcrafted EEG [0:5] + Artefact Proxies + Causal 200-Hz EEG
# ---------------------------------------------------------------------------


def _compute_handcrafted_causal_eeg(
    eeg_past_200: np.ndarray, valid_eeg: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute strictly causal handcrafted EEG features from [-3.2, +5.0) s."""
    n = len(eeg_past_200)
    # Trim leading 0.2 s -> exact [-3.0, +5.0) s = 1600 samples at 200 Hz
    x = eeg_past_200[:, :, 40:].astype(np.float64)  # (N, 62, 1600)
    ch_map = {ch: i for i, ch in enumerate(CANONICAL_62)}
    feats: dict[str, np.ndarray] = {}

    def _band_powers(seg: np.ndarray) -> dict[str, np.ndarray]:
        # seg: (N, 62, L) -> returns {band: (N, 62) log10 power}
        nper = min(seg.shape[-1], 200)
        f, p = welch(seg, fs=REVE_FS, nperseg=nper, axis=-1)
        res = {}
        for bname, (lo, hi) in BAND_EDGES.items():
            m = (f >= lo) & (f < hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.log10(np.maximum(trapezoid(p[:, :, m], f[m], axis=-1), 1e-12))
            lp[~valid_eeg] = np.nan
            res[bname] = lp
        return res

    bp_ref = _band_powers(x[:, :, 0:600])  # [-3.0, 0.0)
    bp_e1 = _band_powers(x[:, :, 600:1100])  # [0.0, 2.5)
    bp_e2 = _band_powers(x[:, :, 1100:1600])  # [2.5, 5.0)
    bp_ea = _band_powers(x[:, :, 600:1600])  # [0.0, 5.0)

    # 6 regions x 5 bands: ref levels + changes (e1, e2, early) + slope (e2 - e1)
    for reg, chs in REGIONS.items():
        idxs = [ch_map[c] for c in chs if c in ch_map]
        for bname in BANDS:
            r_v = np.nanmean(bp_ref[bname][:, idxs], axis=1)
            e1_v = np.nanmean(bp_e1[bname][:, idxs], axis=1)
            e2_v = np.nanmean(bp_e2[bname][:, idxs], axis=1)
            ea_v = np.nanmean(bp_ea[bname][:, idxs], axis=1)
            feats[f"e05_ref_{reg}_{bname}"] = r_v.astype(np.float32)
            feats[f"e05_e1chg_{reg}_{bname}"] = (e1_v - r_v).astype(np.float32)
            feats[f"e05_e2chg_{reg}_{bname}"] = (e2_v - r_v).astype(np.float32)
            feats[f"e05_eachg_{reg}_{bname}"] = (ea_v - r_v).astype(np.float32)
            feats[f"e05_slope_{reg}_{bname}"] = (e2_v - e1_v).astype(np.float32)

    # Spatial homogeneity, FAA, anterior-posterior gradient, and 1-s volatility
    lf_idx = [ch_map[c] for c in LEFT_FRONTAL if c in ch_map]
    rf_idx = [ch_map[c] for c in RIGHT_FRONTAL if c in ch_map]
    fr_idx = [ch_map[c] for c in REGIONS["frontal"] if c in ch_map]
    oc_idx = [ch_map[c] for c in REGIONS["occipital"] if c in ch_map]
    int_idx = [ch_map[c] for c in INTERIOR_CH if c in ch_map]
    rim_idx = [ch_map[c] for c in RIM_CH if c in ch_map]

    for bname in BANDS:
        dr = bp_ea[bname] - bp_ref[bname]
        feats[f"e05_sp_std_{bname}"] = (
            np.nanstd(bp_ea[bname], axis=1) - np.nanstd(bp_ref[bname], axis=1)
        ).astype(np.float32)
        feats[f"e05_faa_{bname}"] = (
            np.nanmean(dr[:, rf_idx], axis=1) - np.nanmean(dr[:, lf_idx], axis=1)
        ).astype(np.float32)
        feats[f"e05_ap_grad_{bname}"] = (
            np.nanmean(dr[:, fr_idx], axis=1) - np.nanmean(dr[:, oc_idx], axis=1)
        ).astype(np.float32)
        if bname == "gamma":
            feats["art_prep_int_gamma_early"] = np.nanmean(
                bp_ea["gamma"][:, int_idx], axis=1
            ).astype(np.float32)
            feats["art_prep_rim_gamma_early"] = np.nanmean(
                bp_ea["gamma"][:, rim_idx], axis=1
            ).astype(np.float32)

    # 1-s temporal volatility strictly inside [0, 5) s
    sec_bp = {b: np.zeros((n, 5), dtype=np.float64) for b in BANDS}
    for s_k in range(5):
        bp_s = _band_powers(x[:, :, 600 + s_k * 200 : 600 + (s_k + 1) * 200])
        for bname in BANDS:
            sec_bp[bname][:, s_k] = np.nanmean(bp_s[bname], axis=1)
    for bname in BANDS:
        arr = sec_bp[bname]
        feats[f"e05_tvar_{bname}"] = np.nanstd(arr, axis=1).astype(np.float32)
        feats[f"e05_maxjump_{bname}"] = np.nanmax(
            np.abs(np.diff(arr, axis=1)), axis=1
        ).astype(np.float32)
    return feats


def _worker_recording_study09(args: tuple[str, list[dict]]) -> dict:
    """Process one emotion-video recording with a hard causal wall at t = 5.0 s."""
    raw_dir, trials = args
    curated = Path(raw_dir) / trials[0]["recording"]
    onsets = np.array([float(t["onset_s"]) for t in trials], dtype=np.float64)
    trial_ids = [physical_id(t) for t in trials]
    n_tr = len(trials)
    n_ch = len(CANONICAL_62)
    n_past = round((PAST_END_S - PAST_START_S) * REVE_FS)  # 1640
    n_vid = round((VID_END_S - VID_START_S) * REVE_FS)  # 2600

    # 1. Body past [-3, 5) s (strictly causal) + isolated future targets
    ecg, ecg_fs, resp, resp_fs, eda, eda_fs, eda_unit, _ = _read_body(curated)
    body_past, refs = extract_causal_body_past(
        ecg, ecg_fs, eda, eda_fs, eda_unit, resp, resp_fs, onsets
    )
    body_targets = extract_future_body_targets(
        ecg, ecg_fs, eda, eda_fs, eda_unit, resp, resp_fs, onsets, refs
    )

    # 2. Previous-trial temporal dependence controls (chronological inside recording)
    order = np.argsort(onsets)
    prev_hr = np.full(n_tr, np.nan, dtype=np.float32)
    prev_sc = np.full(n_tr, np.nan, dtype=np.float32)
    prev_resp = np.full(n_tr, np.nan, dtype=np.float32)
    trial_gap = np.full(n_tr, np.nan, dtype=np.float32)
    for rank in range(1, n_tr):
        cur_i, prv_i = int(order[rank]), int(order[rank - 1])
        prev_hr[cur_i] = body_targets["tgt_ecg_hr_late_raw"][prv_i]
        prev_sc[cur_i] = body_targets["tgt_eda_sc_late_raw"][prv_i]
        prev_resp[cur_i] = body_targets["tgt_resp_amp_late_raw"][prv_i]
        trial_gap[cur_i] = float(onsets[cur_i] - onsets[prv_i])
    tc_controls = {
        "prev_ecg_hr_late": prev_hr,
        "prev_eda_sc_late": prev_sc,
        "prev_resp_amp_late": prev_resp,
        "trial_gap_s": trial_gap,
    }

    # 3. Oculometry audit + causal features
    oc_feats, oc_audit = extract_oculometry_recording(curated, onsets)

    # 4. Strictly causal EEG (separate [-3.2, +5.0) s and [-3.0, +10.0) s!)
    eeg_path, shift = _eeg_source(raw_dir, trials[0])
    candidates = [(eeg_path, shift)] + ([(curated, 0.0)] if eeg_path != curated else [])
    eeg_past = np.zeros((n_tr, n_ch, n_past), dtype=np.float32)
    eeg_vid = np.zeros((n_tr, n_ch, n_vid), dtype=np.float32)
    valid_past = np.zeros(n_tr, dtype=bool)
    valid_vid = np.zeros(n_tr, dtype=bool)
    bad_ch_past = np.full(n_tr, n_ch, dtype=np.float32)
    used_path = None

    for path, offset in candidates:
        try:
            adj = onsets + offset
            ch_pres = np.zeros(n_ch, dtype=bool)
            bc_p = np.zeros(n_tr, dtype=np.int32)
            bc_v = np.zeros(n_tr, dtype=np.int32)
            with pyedflib.EdfReader(str(path)) as reader:
                lbl_map = {
                    lbl.upper(): i for i, lbl in enumerate(reader.getSignalLabels())
                }
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

                    # Hard causal wall: slice & filter & resample [-3.2, +5.0) ALONE!
                    wp, fin_p = causal_bandpass_windows(
                        sig,
                        fs,
                        adj,
                        PAST_START_S,
                        PAST_END_S,
                        0.5,
                        45.0,
                        order=4,
                        warmup_s=PAST_WARMUP_S,
                    )
                    rp = _resample_2d(np.nan_to_num(wp), fs, REVE_FS, n_past)
                    bad_p = (
                        (~fin_p)
                        | (np.ptp(rp, axis=-1) > 800.0)
                        | (np.std(rp, axis=-1) < 0.01)
                    )
                    rp[bad_p] = 0.0
                    eeg_past[:, c_i, :] = rp.astype(np.float32)
                    bc_p += bad_p.astype(np.int32)

                    # Full causal stimulus window [-3.0, +10.0) s (for secondary task)
                    wv, fin_v = causal_bandpass_windows(
                        sig,
                        fs,
                        adj,
                        VID_START_S,
                        VID_END_S,
                        0.5,
                        45.0,
                        order=4,
                        warmup_s=PAST_WARMUP_S,
                    )
                    rv = _resample_2d(np.nan_to_num(wv), fs, REVE_FS, n_vid)
                    bad_v = (
                        (~fin_v)
                        | (np.ptp(rv, axis=-1) > 800.0)
                        | (np.std(rv, axis=-1) < 0.01)
                    )
                    rv[bad_v] = 0.0
                    eeg_vid[:, c_i, :] = rv.astype(np.float32)
                    bc_v += bad_v.astype(np.int32)
                    ch_pres[c_i] = True

            if ch_pres.sum() >= 30:
                eeg_past[:, ch_pres, :] -= eeg_past[:, ch_pres, :].mean(
                    axis=1, keepdims=True
                )
                eeg_vid[:, ch_pres, :] -= eeg_vid[:, ch_pres, :].mean(
                    axis=1, keepdims=True
                )
                valid_past = bc_p <= int(0.20 * ch_pres.sum())
                valid_vid = bc_v <= int(0.20 * ch_pres.sum())
                bad_ch_past = bc_p.astype(np.float32)
                used_path = path
                break
        except (OSError, ValueError):
            continue

    eeg_hc = _compute_handcrafted_causal_eeg(eeg_past, valid_past)

    # 5. Strictly causal [0, 5) s pre-ICA CURATED artefact proxies (ocular, EMG)
    art: dict[str, np.ndarray] = {
        "art_cur_fp_ptp_early": np.full(n_tr, np.nan, dtype=np.float32),
        "art_cur_fp_eog_early": np.full(n_tr, np.nan, dtype=np.float32),
        "art_cur_rim_emg30_45_early": np.full(n_tr, np.nan, dtype=np.float32),
        "art_cur_rim_emg52_80_early": np.full(n_tr, np.nan, dtype=np.float32),
        "art_cur_int_emg52_80_early": np.full(n_tr, np.nan, dtype=np.float32),
        "art_ica_diff_rms_early": np.full(n_tr, np.nan, dtype=np.float32),
        "art_bad_ch_count_early": bad_ch_past,
    }
    try:
        art_chs = tuple(dict.fromkeys(("Fp1", "Fp2", *RIM_CH, *INTERIOR_CH)))
        ch_idx_62 = {c: i for i, c in enumerate(CANONICAL_62)}
        fp_ptps, fp_eogs, rim30, rim52, int52, ica_diffs = [], [], [], [], [], []
        with pyedflib.EdfReader(str(curated)) as reader:
            lbl_map = {lbl.upper(): i for i, lbl in enumerate(reader.getSignalLabels())}
            for cname in art_chs:
                idx = lbl_map.get(cname.upper())
                if idx is None:
                    continue
                unit = (
                    reader.getPhysicalDimension(idx).replace("µ", "u").replace("μ", "u")
                )
                fs = float(reader.getSampleFrequency(idx))
                if unit not in EEG_UNITS or fs < 200:
                    continue
                sig = reader.readSignal(idx).astype(np.float64) * EEG_UNITS[unit]
                if not np.isfinite(sig).all():
                    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
                # Broad causal [0, 5) s for EMG up to 80 Hz (zero samples >= 5.0 s!)
                w_emg, fin_e = causal_bandpass_windows(
                    sig, fs, onsets, 0.0, 5.0, 0.5, 85.0, order=4, warmup_s=0.5
                )
                if not fin_e.any():
                    continue
                nper = min(w_emg.shape[1], round(fs))
                f_q, p_q = welch(np.nan_to_num(w_emg), fs=fs, nperseg=nper, axis=-1)

                def _lp(lo: float, hi: float, fq=f_q, pq=p_q, fe=fin_e) -> np.ndarray:
                    m = (fq >= lo) & (fq < hi)
                    v = np.log10(np.maximum(trapezoid(pq[:, m], fq[m], axis=-1), 1e-12))
                    v[~fe] = np.nan
                    return v

                if cname in ("Fp1", "Fp2"):
                    ptp = np.ptp(w_emg, axis=-1)
                    ptp[~fin_e] = np.nan
                    fp_ptps.append(ptp)
                    fp_eogs.append(_lp(0.5, 4.0))
                if cname in RIM_CH:
                    rim30.append(_lp(30.0, 45.0))
                    rim52.append(_lp(52.0, 80.0))
                if cname in INTERIOR_CH:
                    int52.append(_lp(52.0, 80.0))
                if used_path is not None and cname in ch_idx_62:
                    w_bp, fin_b = causal_bandpass_windows(
                        sig, fs, onsets, 0.0, 5.0, 0.5, 45.0, order=4, warmup_s=0.5
                    )
                    r_cur = _resample_2d(np.nan_to_num(w_bp), fs, REVE_FS, 1000)
                    r_prep = eeg_past[:, ch_idx_62[cname], 640:1640]
                    diff_rms = np.sqrt(np.mean((r_cur - r_prep) ** 2, axis=-1))
                    diff_rms[~(fin_b & valid_past)] = np.nan
                    ica_diffs.append(diff_rms)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if fp_ptps:
                art["art_cur_fp_ptp_early"] = np.nanmean(fp_ptps, axis=0).astype(
                    np.float32
                )
                art["art_cur_fp_eog_early"] = np.nanmean(fp_eogs, axis=0).astype(
                    np.float32
                )
            if rim30:
                art["art_cur_rim_emg30_45_early"] = np.nanmean(rim30, axis=0).astype(
                    np.float32
                )
                art["art_cur_rim_emg52_80_early"] = np.nanmean(rim52, axis=0).astype(
                    np.float32
                )
            if int52:
                art["art_cur_int_emg52_80_early"] = np.nanmean(int52, axis=0).astype(
                    np.float32
                )
            if ica_diffs:
                art["art_ica_diff_rms_early"] = np.nanmean(ica_diffs, axis=0).astype(
                    np.float32
                )
    except Exception:
        pass

    # Trial-level channel mean/var on [-3.2, 5.0) s for causal past-history scaling
    tr_ch_mean = eeg_past.mean(axis=-1).astype(np.float32)  # (N, 62)
    tr_ch_var = eeg_past.var(axis=-1).astype(np.float32)  # (N, 62)

    tabular = {**body_past, **body_targets, **tc_controls, **eeg_hc, **art}
    return {
        "trial_ids": trial_ids,
        "tabular": tabular,
        "oculo": oc_feats,
        "oc_audit": oc_audit,
        "eeg_past": np.clip(eeg_past, -500.0, 500.0).astype(np.float16),
        "eeg_vid": np.clip(eeg_vid, -500.0, 500.0).astype(np.float16),
        "valid_past": valid_past,
        "valid_vid": valid_vid,
        "tr_ch_mean": tr_ch_mean,
        "tr_ch_var": tr_ch_var,
    }


def _worker_resting_eeg200(args: tuple[str, int]) -> dict | None:
    """Load up to ``max_windows`` causal 8.2-s 62-ch 200-Hz resting windows."""
    prep_str, max_windows = args
    prep_path = Path(prep_str)
    if not prep_path.is_file():
        return None
    n_ch = len(CANONICAL_62)
    n_samp = round((PAST_END_S - PAST_START_S) * REVE_FS)  # 1640
    try:
        with pyedflib.EdfReader(str(prep_path)) as reader:
            dur_s = float(reader.getFileDuration())
            onsets = np.arange(4.0, dur_s - 6.0, 8.5)
            if len(onsets) == 0:
                return None
            onsets = onsets[:max_windows]
            n_w = len(onsets)
            eeg = np.zeros((n_w, n_ch, n_samp), dtype=np.float32)
            ch_pres = np.zeros(n_ch, dtype=bool)
            bc = np.zeros(n_w, dtype=np.int32)
            lbl_map = {lbl.upper(): i for i, lbl in enumerate(reader.getSignalLabels())}
            for c_i, ch_name in enumerate(CANONICAL_62):
                idx = lbl_map.get(ch_name.upper())
                if idx is None:
                    continue
                unit = (
                    reader.getPhysicalDimension(idx).replace("µ", "u").replace("μ", "u")
                )
                fs = float(reader.getSampleFrequency(idx))
                if unit not in EEG_UNITS or fs < 100:
                    continue
                sig = reader.readSignal(idx).astype(np.float64) * EEG_UNITS[unit]
                if not np.isfinite(sig).all():
                    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
                wp, fin = causal_bandpass_windows(
                    sig,
                    fs,
                    onsets,
                    PAST_START_S,
                    PAST_END_S,
                    0.5,
                    45.0,
                    order=4,
                    warmup_s=PAST_WARMUP_S,
                )
                rp = _resample_2d(np.nan_to_num(wp), fs, REVE_FS, n_samp)
                bad = (
                    (~fin)
                    | (np.ptp(rp, axis=-1) > 800.0)
                    | (np.std(rp, axis=-1) < 0.01)
                )
                rp[bad] = 0.0
                eeg[:, c_i, :] = rp.astype(np.float32)
                bc += bad.astype(np.int32)
                ch_pres[c_i] = True
        if ch_pres.sum() < 30:
            return None
        eeg[:, ch_pres, :] -= eeg[:, ch_pres, :].mean(axis=1, keepdims=True)
        good = bc <= int(0.20 * ch_pres.sum())
        if not good.any():
            return None
        eg = eeg[good]
        return {
            "pid": prep_path.parent.name,
            "eeg": np.clip(eg, -500.0, 500.0).astype(np.float16),
            "ch_mean": eg.mean(axis=(0, 2)).astype(np.float32),
            "ch_var": eg.var(axis=(0, 2)).astype(np.float32),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Strictly Causal Frozen REVE-base Extraction (Cold-Start + Causal History)
# ---------------------------------------------------------------------------


def compute_causal_history_stats(
    trials: pd.DataFrame,
    tr_ch_mean: np.ndarray,
    tr_ch_var: np.ndarray,
    valid_past: np.ndarray,
    rest_stats: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute strictly causal past-only channel ``(mu, sd)`` per trial (Protocol B1).

    For trial ``i`` of participant ``p`` (chronologically sorted), combines
    ``p``'s resting-state stats (if available) with strictly prior valid trials
    ``0..i-1`` of ``p``. Never reads trial ``i``'s future or any trial ``j > i``.
    Falls back to trial ``i``'s own ``[-3.2, 5.0)`` stats when no prior history exists.
    """
    n, n_ch = tr_ch_mean.shape
    hist_mu = np.zeros((n, n_ch), dtype=np.float32)
    hist_sd = np.ones((n, n_ch), dtype=np.float32)

    order_df = pd.DataFrame(
        {
            "pid": trials["participant_id"].astype(str).to_numpy(),
            "day": trials["recorded_day"].to_numpy() if "recorded_day" in trials else 1,
            "eo": trials["event_order"].to_numpy()
            if "event_order" in trials
            else np.arange(n),
            "ons": trials["onset_s"].to_numpy(float)
            if "onset_s" in trials
            else np.arange(n, dtype=float),
            "pos": np.arange(n),
        }
    )
    for pid, grp in order_df.sort_values(
        ["pid", "day", "eo", "ons", "pos"], kind="mergesort"
    ).groupby("pid", sort=False):
        idxs = grp["pos"].to_numpy(dtype=int)
        # Running sum of means and second moments (E[X^2] = var + mu^2)
        cnt = 0.0
        sum_m = np.zeros(n_ch, dtype=np.float64)
        sum_m2 = np.zeros(n_ch, dtype=np.float64)
        if str(pid) in rest_stats:
            rm, rv = rest_stats[str(pid)]
            w_r = 3.0
            sum_m = w_r * rm.astype(np.float64)
            sum_m2 = w_r * (rv.astype(np.float64) + rm.astype(np.float64) ** 2)
            cnt = w_r
        for pos in idxs:
            if cnt > 0.0:
                mu = sum_m / cnt
                var = np.maximum(sum_m2 / cnt - mu * mu, 1e-4)
                hist_mu[pos] = mu.astype(np.float32)
                hist_sd[pos] = np.sqrt(var).astype(np.float32)
            else:
                hist_mu[pos] = tr_ch_mean[pos]
                hist_sd[pos] = np.sqrt(np.maximum(tr_ch_var[pos], 1e-4))
            if valid_past[pos]:
                m_i = tr_ch_mean[pos].astype(np.float64)
                v_i = tr_ch_var[pos].astype(np.float64)
                sum_m += m_i
                sum_m2 += v_i + m_i * m_i
                cnt += 1.0
    return hist_mu, np.maximum(hist_sd, 1e-2)


def _zscore_trial_causal(x: torch.Tensor) -> torch.Tensor:
    """Cold-start per-trial per-channel z-score clipped to [-15, 15]."""
    import torch

    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp(min=1e-2)
    return torch.clamp((x - mu) / sd, -15.0, 15.0)


def extract_reve_study09(
    eeg_past: np.ndarray,
    eeg_vid: np.ndarray,
    hist_mu: np.ndarray,
    hist_sd: np.ndarray,
    rest_windows: dict[str, np.ndarray],
    pids: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 48,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Run strictly causal frozen REVE-base passes on H100."""
    import torch
    from transformers import AutoModel

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions", trust_remote_code=True
    ).to(device)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).to(
        device
    )
    model.eval()
    base_pos_62 = pos_bank(list(CANONICAL_62)).to(device)  # (62, 3)
    base_pos_46 = pos_bank(list(INTERIOR_46_NAMES)).to(device)  # (46, 3)
    n_params = int(sum(p.numel() for p in model.parameters()))

    n = len(eeg_past)
    d = 512
    primary = {
        "reve_cs_pre": np.zeros((n, d), dtype=np.float32),
        "reve_cs_early5": np.zeros((n, d), dtype=np.float32),
        "reve_cs_ctx8_early": np.zeros((n, d), dtype=np.float32),
        "reve_cs_ctx8_pool": np.zeros((n, d), dtype=np.float32),
        "reve_cs_ctx8_diff": np.zeros((n, d), dtype=np.float32),
    }
    extra = {
        "reve_cs_interior_ctx8": np.zeros((n, d), dtype=np.float32),
        "reve_cs_shuf_early5": np.zeros((n, d), dtype=np.float32),
        "reve_hist_ctx8_early": np.zeros((n, d), dtype=np.float32),
        "reve_cs_vid10": np.zeros((n, d), dtype=np.float32),
        "reve_hist_vid10": np.zeros((n, d), dtype=np.float32),
        "reve_rest_trait": np.full((n, d), np.nan, dtype=np.float32),
    }
    rng_shuf = np.random.default_rng(20261124)
    int_idx_t = torch.tensor(INTERIOR_46_IDX, dtype=torch.long, device=device)

    with torch.inference_mode():
        # Pass 1: Causal [-3.2, +5.0) s (Cold-Start, Interior-46, Causal-History)
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            bsz = e - s
            raw8 = torch.from_numpy(eeg_past[s:e]).to(
                device, dtype=torch.float32
            )  # (B, 62, 1640)
            pos62 = base_pos_62.unsqueeze(0).expand(bsz, -1, -1)
            pos46 = base_pos_46.unsqueeze(0).expand(bsz, -1, -1)

            # 1a. Cold-Start [-3.2, +5.0) s -> 9 patches ending at +5.000 s
            x_cs8 = _zscore_trial_causal(raw8)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                toks = model(x_cs8, pos62)  # (B, 62, 9, 512)
                pool_e = model.attention_pooling(toks[:, :, 4:9, :])  # [0.4, 5.0) s
            cm = toks.float().mean(dim=1)  # (B, 9, 512)
            primary["reve_cs_ctx8_early"][s:e] = cm[:, 4:9, :].mean(dim=1).cpu().numpy()
            primary["reve_cs_ctx8_pool"][s:e] = pool_e.float().cpu().numpy()
            primary["reve_cs_ctx8_diff"][s:e] = (
                (cm[:, 6:9, :].mean(dim=1) - cm[:, 4:6, :].mean(dim=1)).cpu().numpy()
            )

            # 1b. Cold-Start isolated pre-stimulus [-3.0, 0.0) s (600 samp, 3 patches)
            x_pre = _zscore_trial_causal(raw8[:, :, 40:640])
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_pre = model(x_pre, pos62)
            primary["reve_cs_pre"][s:e] = tok_pre.float().mean(dim=(1, 2)).cpu().numpy()

            # 1c. Cold-Start isolated pure early [0.0, 5.0) s (1000 samples, 5 patches)
            raw5 = raw8[:, :, 640:1640]
            x_e5 = _zscore_trial_causal(raw5)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_e5 = model(x_e5, pos62)
            primary["reve_cs_early5"][s:e] = (
                tok_e5.float().mean(dim=(1, 2)).cpu().numpy()
            )

            # 1d. NC4: Time-shuffled [0, 5) s (permute the five 1-s segments per trial)
            chunks = raw5.reshape(bsz, 62, 5, 200)
            perm = torch.from_numpy(
                np.stack([rng_shuf.permutation(5) for _ in range(bsz)], axis=0)
            ).to(device)
            p_idx = perm[:, None, :, None].expand(bsz, 62, 5, 200)
            shuf5 = torch.gather(chunks, 2, p_idx).reshape(bsz, 62, 1000)
            x_shuf = _zscore_trial_causal(shuf5)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_sh = model(x_shuf, pos62)
            extra["reve_cs_shuf_early5"][s:e] = (
                tok_sh.float().mean(dim=(1, 2)).cpu().numpy()
            )

            # 1e. AC2: 46 interior channels only (rim & frontopolar excluded!)
            x_int8 = x_cs8.index_select(1, int_idx_t)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_int = model(x_int8, pos46)
            extra["reve_cs_interior_ctx8"][s:e] = (
                tok_int.float().mean(dim=1)[:, 4:9, :].mean(dim=1).cpu().numpy()
            )

            # 1f. Protocol B1: Causal past-history z-score on [-3.2, +5.0) s
            mu_h = torch.from_numpy(hist_mu[s:e]).to(device).unsqueeze(-1)
            sd_h = torch.from_numpy(hist_sd[s:e]).to(device).unsqueeze(-1)
            x_h8 = torch.clamp((raw8 - mu_h) / sd_h.clamp(min=1e-2), -15.0, 15.0)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_h8 = model(x_h8, pos62)
            extra["reve_hist_ctx8_early"][s:e] = (
                tok_h8.float().mean(dim=1)[:, 4:9, :].mean(dim=1).cpu().numpy()
            )

        # Pass 2: Full video [0.0, 10.0) s for Secondary Affect & Trait/State study
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            bsz = e - s
            # eeg_vid is [-3.0, +10.0) s = 2600 samples -> [0.0, 10.0) is [600:2600]
            raw10 = torch.from_numpy(eeg_vid[s:e, :, 600:2600]).to(
                device, dtype=torch.float32
            )
            pos62 = base_pos_62.unsqueeze(0).expand(bsz, -1, -1)

            x_v_cs = _zscore_trial_causal(raw10)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_vcs = model(x_v_cs, pos62)
            extra["reve_cs_vid10"][s:e] = tok_vcs.float().mean(dim=(1, 2)).cpu().numpy()

            mu_h = torch.from_numpy(hist_mu[s:e]).to(device).unsqueeze(-1)
            sd_h = torch.from_numpy(hist_sd[s:e]).to(device).unsqueeze(-1)
            x_v_h = torch.clamp((raw10 - mu_h) / sd_h.clamp(min=1e-2), -15.0, 15.0)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tok_vh = model(x_v_h, pos62)
            extra["reve_hist_vid10"][s:e] = (
                tok_vh.float().mean(dim=(1, 2)).cpu().numpy()
            )

        # Pass 3: Unlabeled resting-state subject trait embeddings (Protocol B0)
        rest_embed_by_pid: dict[str, np.ndarray] = {}
        for pid, arr in rest_windows.items():
            if len(arr) == 0:
                continue
            embs = []
            for s in range(0, len(arr), batch_size):
                x_r = _zscore_trial_causal(
                    torch.from_numpy(arr[s : s + batch_size]).to(
                        device, dtype=torch.float32
                    )
                )
                pos_r = base_pos_62.unsqueeze(0).expand(len(x_r), -1, -1)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    tr = model(x_r, pos_r)
                embs.append(tr.float().mean(dim=(1, 2)).cpu().numpy())
            rest_embed_by_pid[pid] = np.concatenate(embs, axis=0).mean(axis=0)
        for i, pid in enumerate(pids):
            vec = rest_embed_by_pid.get(str(pid))
            if vec is not None:
                extra["reve_rest_trait"][i] = vec

    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    del model, pos_bank
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    info = {
        "n_params": n_params,
        "runtime_s": round(elapsed, 2),
        "peak_vram_gb": round(peak_gb, 2),
        "rest_trait_people": int(len(rest_embed_by_pid)),
    }
    return primary, extra, info


def _save_embed_parquet(
    out_path: Path,
    trial_ids: list[str],
    valid: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict:
    """Save 512-d REVE arrays cleanly in float16 zstd Parquet (< 150 MB)."""
    cols: dict[str, object] = {"trial_id": trial_ids, "reve_valid": valid.astype(bool)}
    for name, mat in arrays.items():
        m16 = np.clip(np.nan_to_num(mat, nan=np.nan), -60000.0, 60000.0).astype(
            np.float16
        )
        for j in range(m16.shape[1]):
            cols[f"{name}_{j}"] = m16[:, j]
    df = pd.DataFrame(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd")
    return {
        "file": out_path.name,
        "mb": round(out_path.stat().st_size / 1e6, 2),
        "cols": int(df.shape[1]),
    }


def summarize_oculometry_audit(
    trials: pd.DataFrame,
    oc_df: pd.DataFrame,
    rec_audits: list[dict],
    evoked_pupil: np.ndarray,
    evoked_gaze: np.ndarray,
) -> dict:
    """Summarize complete 508-recording pupil/gaze coverage, quality, and H4 gate."""
    from scipy.stats import ttest_1samp

    n_recs = len(rec_audits)
    has_map = sum(int(r["has_record_map"]) for r in rec_audits)
    has_oc = sum(int(r["has_oculo_stream"]) for r in rec_audits)
    all_binoc = [r for ra in rec_audits for r in ra["binoc_rs"] if np.isfinite(r)]
    med_binoc = float(np.median(all_binoc)) if all_binoc else None

    valid_ea = oc_df["oc_valid_early"].to_numpy(float) > 0.5
    valid_la = oc_df["oc_valid_late"].to_numpy(float) > 0.5
    cov_ea = float(valid_ea.mean())
    cov_la = float(valid_la.mean())

    # Per-cohort breakdown
    by_cohort: dict[str, dict] = {}
    if "cohort" in trials.columns:
        for coh, grp in trials.groupby("cohort"):
            idx = grp.index.to_numpy()
            br = oc_df.iloc[idx]["oc_binoc_r"].dropna().to_numpy(float)
            c_ev = evoked_pupil[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Reflex window [0.25, 2.5) s -> bins 13..22
                reflex = np.nanmean(c_ev[:, 13:22], axis=1)
                pre_ctrl = np.nanmean(c_ev[:, 0:4], axis=1)
            fin_r = reflex[np.isfinite(reflex)]
            t_st, p_val = (
                ttest_1samp(fin_r, 0.0) if len(fin_r) >= 20 else (np.nan, np.nan)
            )
            by_cohort[str(int(coh))] = {
                "n_trials": int(len(idx)),
                "valid_early_frac": round(float(valid_ea[idx].mean()), 4),
                "valid_late_frac": round(float(valid_la[idx].mean()), 4),
                "median_binoc_r": round(float(np.median(br)), 4) if len(br) else None,
                "reflex_025_25s_mm": round(float(np.mean(fin_r)), 4)
                if len(fin_r)
                else None,
                "reflex_t": round(float(t_st), 2) if np.isfinite(t_st) else None,
                "reflex_p": round(float(p_val), 6) if np.isfinite(p_val) else None,
                "pre_m3_m2s_mm": round(float(np.nanmean(pre_ctrl)), 4)
                if np.isfinite(pre_ctrl).any()
                else None,
            }

    # Grand-average evoked pupil reflex curve (person-averaged across trials)
    pids = trials["participant_id"].astype(str).to_numpy()
    p_curves, p_reflex, p_pre = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pid in np.unique(pids):
            m = pids == pid
            pc = np.nanmean(evoked_pupil[m], axis=0)
            if np.isfinite(pc[13:22]).any():
                p_curves.append(pc)
                p_reflex.append(float(np.nanmean(pc[13:22])))
                p_pre.append(float(np.nanmean(pc[0:4])))
        grand_curve = np.nanmean(p_curves, axis=0).round(4).tolist() if p_curves else []
        grand_gaze = (
            np.nanmean(evoked_gaze, axis=0).round(2).tolist()
            if np.isfinite(evoked_gaze).any()
            else []
        )

    p_ref_arr = np.array([x for x in p_reflex if np.isfinite(x)], dtype=float)
    p_pre_arr = np.array([x for x in p_pre if np.isfinite(x)], dtype=float)
    if len(p_ref_arr) >= 15:
        t_ref, p_ref = ttest_1samp(p_ref_arr, 0.0)
    else:
        t_ref, p_ref = np.nan, np.nan
    t_pre = (
        float(ttest_1samp(p_pre_arr, 0.0).statistic) if len(p_pre_arr) >= 15 else np.nan
    )

    # Evaluate pre-registered H4 usability gate
    max_coh_cov = max((v["valid_early_frac"] for v in by_cohort.values()), default=0.0)
    pass_cov = bool(cov_ea >= 0.50 or max_coh_cov >= 0.60)
    pass_binoc = bool(med_binoc is not None and med_binoc >= 0.50)
    pass_align = bool(
        np.isfinite(t_ref)
        and abs(t_ref) > 4.0
        and p_ref < 1e-4
        and (not np.isfinite(t_pre) or abs(t_ref) > 2.0 * abs(t_pre))
    )
    gate_passed = bool(pass_cov and pass_binoc and pass_align)

    return {
        "n_recordings": n_recs,
        "recordings_with_record_map": has_map,
        "recordings_with_oculo_stream": has_oc,
        "valid_early_fraction": round(cov_ea, 4),
        "valid_late_fraction": round(cov_la, 4),
        "median_binocular_r": round(med_binoc, 4) if med_binoc is not None else None,
        "n_binocular_trials": int(len(all_binoc)),
        "evoked_pupil_reflex_people": int(len(p_ref_arr)),
        "evoked_pupil_reflex_025_25s_mm": round(float(p_ref_arr.mean()), 4)
        if len(p_ref_arr)
        else None,
        "evoked_pupil_reflex_t": round(float(t_ref), 3) if np.isfinite(t_ref) else None,
        "evoked_pupil_reflex_p": float(p_ref) if np.isfinite(p_ref) else None,
        "evoked_pre_m3_m2s_t": round(float(t_pre), 3) if np.isfinite(t_pre) else None,
        "evoked_pupil_curve_025s_bins": grand_curve,
        "evoked_gaze_speed_curve_025s_bins": grand_gaze,
        "by_cohort": by_cohort,
        "h4_usability_gate": {
            "pass_coverage": pass_cov,
            "pass_binocular_r": pass_binoc,
            "pass_temporal_alignment": pass_align,
            "gate_passed": gate_passed,
        },
    }


def main() -> None:
    """Run Study-09 causal extraction + causal REVE-base + oculo audit on H100."""
    import torch

    t_start = time.perf_counter()
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trials = pd.read_csv(
        out_dir / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    id_to_pos = {tid: i for i, tid in enumerate(trials["trial_id"])}
    n = len(trials)
    n_ch = len(CANONICAL_62)
    n_past = round((PAST_END_S - PAST_START_S) * REVE_FS)
    n_vid = round((VID_END_S - VID_START_S) * REVE_FS)

    # 1. Walk 508 emotion-video recordings in parallel
    t_load0 = time.perf_counter()
    jobs = [
        (str(raw_dir), grp.to_dict("records"))
        for _, grp in trials.groupby("recording", sort=True)
    ]
    eeg_past = np.zeros((n, n_ch, n_past), dtype=np.float16)
    eeg_vid = np.zeros((n, n_ch, n_vid), dtype=np.float16)
    valid_past = np.zeros(n, dtype=bool)
    valid_vid = np.zeros(n, dtype=bool)
    tr_ch_mean = np.zeros((n, n_ch), dtype=np.float32)
    tr_ch_var = np.ones((n, n_ch), dtype=np.float32)
    evoked_pupil = np.full((n, OCULO_BINS), np.nan, dtype=np.float32)
    evoked_gaze = np.full((n, OCULO_BINS), np.nan, dtype=np.float32)
    tab_store: dict[str, np.ndarray] = {}
    oc_store: dict[str, np.ndarray] = {}
    rec_audits: list[dict] = []

    workers = min(20, os.cpu_count() or 8)
    ctx = multiprocessing.get_context("spawn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for idx, part in enumerate(pool.map(_worker_recording_study09, jobs), 1):
            pos = np.array([id_to_pos[t] for t in part["trial_ids"]], dtype=int)
            eeg_past[pos] = part["eeg_past"]
            eeg_vid[pos] = part["eeg_vid"]
            valid_past[pos] = part["valid_past"]
            valid_vid[pos] = part["valid_vid"]
            tr_ch_mean[pos] = part["tr_ch_mean"]
            tr_ch_var[pos] = part["tr_ch_var"]
            oa = part["oc_audit"]
            evoked_pupil[pos] = oa.pop("evoked_pupil_curves")
            evoked_gaze[pos] = oa.pop("evoked_gaze_speed_curves")
            rec_audits.append(oa)

            for k, arr in part["tabular"].items():
                if k not in tab_store:
                    tab_store[k] = np.full(n, np.nan, dtype=np.float32)
                tab_store[k][pos] = arr
            for k, arr in part["oculo"].items():
                if k not in oc_store:
                    oc_store[k] = np.full(n, np.nan, dtype=np.float32)
                oc_store[k][pos] = arr
            if idx % 50 == 0 or idx == len(jobs):
                print(
                    f"Study 09 causal extract: {idx}/{len(jobs)} recordings", flush=True
                )

    load_s = round(time.perf_counter() - t_load0, 2)

    # 2. Load resting-state EEG windows (for Protocol B0 trait + B1 causal prior)
    t_rest0 = time.perf_counter()
    audited_pids = set(trials["participant_id"].astype(str))
    rest_files = [
        (str(p), 10)
        for p in sorted(
            (raw_dir / "PREPROCESSED").glob("P*/P*_task-resting-state_multimodal.bdf")
        )
        if p.parent.name in audited_pids
    ]
    rest_windows: dict[str, list[np.ndarray]] = {}
    rest_means: dict[str, list[np.ndarray]] = {}
    rest_vars: dict[str, list[np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for res in pool.map(_worker_resting_eeg200, rest_files):
            if res is None:
                continue
            pid = res["pid"]
            rest_windows.setdefault(pid, []).append(res["eeg"])
            rest_means.setdefault(pid, []).append(res["ch_mean"])
            rest_vars.setdefault(pid, []).append(res["ch_var"])

    rest_win_cat = {
        pid: np.concatenate(vs, axis=0)[:12] for pid, vs in rest_windows.items()
    }
    rest_stats = {
        pid: (np.mean(rest_means[pid], axis=0), np.mean(rest_vars[pid], axis=0))
        for pid in rest_means
    }
    rest_s = round(time.perf_counter() - t_rest0, 2)

    # 3. Compute strictly causal past-only EEG scaling stats (Protocol B1)
    hist_mu, hist_sd = compute_causal_history_stats(
        trials, tr_ch_mean, tr_ch_var, valid_past, rest_stats
    )

    # 4. Run strictly causal frozen REVE-base extraction on H100
    reve_pri, reve_ext, reve_info = extract_reve_study09(
        eeg_past,
        eeg_vid,
        hist_mu,
        hist_sd,
        rest_win_cat,
        trials["participant_id"].astype(str).to_numpy(),
        device,
    )

    # 5. Save private feature parquets (< 150 MB each) + public audit JSON
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    tids = trials["trial_id"].tolist()

    causal_df = pd.DataFrame(
        {
            "trial_id": tids,
            "eeg_valid_past": valid_past,
            "eeg_valid_vid": valid_vid,
            **tab_store,
        }
    )
    causal_pq = feat_dir / "study09_causal.parquet"
    causal_df.to_parquet(causal_pq, compression="zstd")

    oc_df = pd.DataFrame({"trial_id": tids, **oc_store})
    oc_pq = feat_dir / "study09_oculo.parquet"
    oc_df.to_parquet(oc_pq, compression="zstd")

    pri_meta = _save_embed_parquet(
        feat_dir / "study09_reve_primary.parquet", tids, valid_past, reve_pri
    )
    ext_meta = _save_embed_parquet(
        feat_dir / "study09_reve_extra.parquet", tids, valid_past, reve_ext
    )

    oc_audit = summarize_oculometry_audit(
        trials, oc_df, rec_audits, evoked_pupil, evoked_gaze
    )
    cov_summary = {
        "eeg_valid_past": round(float(valid_past.mean()), 4),
        "eeg_valid_vid": round(float(valid_vid.mean()), 4),
        "b05_ecg_hr_early": round(
            float(np.isfinite(tab_store["b05_ecg_hr_early"]).mean()), 4
        ),
        "b05_sc_valid": round(float(np.nanmean(tab_store["b05_sc_valid"])), 4),
        "b05_resp_valid": round(float(np.nanmean(tab_store["b05_resp_valid"])), 4),
        "tgt_ecg_hr_chg_5_10": round(
            float(np.isfinite(tab_store["tgt_ecg_hr_chg_5_10"]).mean()), 4
        ),
        "tgt_ecg_hr_chg_7_10": round(
            float(np.isfinite(tab_store["tgt_ecg_hr_chg_7_10"]).mean()), 4
        ),
        "tgt_eda_sc_chg_5_10": round(
            float(np.isfinite(tab_store["tgt_eda_sc_chg_5_10"]).mean()), 4
        ),
        "tgt_eda_sc_chg_6_9": round(
            float(np.isfinite(tab_store["tgt_eda_sc_chg_6_9"]).mean()), 4
        ),
        "tgt_resp_amp_chg_5_10": round(
            float(np.isfinite(tab_store["tgt_resp_amp_chg_5_10"]).mean()), 4
        ),
        "tgt_resp_amp_chg_7_10": round(
            float(np.isfinite(tab_store["tgt_resp_amp_chg_7_10"]).mean()), 4
        ),
    }

    report = {
        "study": "09_causal_extract_and_oculo_audit",
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "n_trials": n,
        "n_recordings": len(jobs),
        "load_seconds": load_s,
        "rest_load_seconds": rest_s,
        "reve_extract": reve_info,
        "parquets": {
            "study09_causal": {
                "file": causal_pq.name,
                "mb": round(causal_pq.stat().st_size / 1e6, 2),
                "cols": int(causal_df.shape[1]),
            },
            "study09_oculo": {
                "file": oc_pq.name,
                "mb": round(oc_pq.stat().st_size / 1e6, 2),
                "cols": int(oc_df.shape[1]),
            },
            "study09_reve_primary": pri_meta,
            "study09_reve_extra": ext_meta,
        },
        "coverage": cov_summary,
        "oculometry_audit": oc_audit,
        "total_elapsed_seconds": round(time.perf_counter() - t_start, 2),
    }

    pub = out_dir / "public"
    pub.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    (pub / "study09_extract_audit.json").write_text(text)
    print(text[:5000], flush=True)


if __name__ == "__main__":
    main()
