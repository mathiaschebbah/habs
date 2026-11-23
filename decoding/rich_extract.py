"""Enriched multimodal extraction addressing Study-07 revisions.

Runs once on the HABS backend, where the full dataset is mounted. For every
audited trial it extracts:

1. Extra annotation columns from ``aligned_trials.csv.gz``
   (``self_reported_emotion_2``, ``self_reported_emotion_3``,
   ``expected_emotion_label``, ``expected_emotion_source``,
   ``flagged_outliers``).
2. Four physiological phase windows around video onset:
   ``ref`` [-3, 0) s, ``early`` [0, 5) s, ``late`` [5, 10) s,
   ``post`` [10, 12) s, plus full stimulus ``vid`` [0, 10) s for ECG morphology.
3. Beat-by-beat ECG P-QRS-T delineation (template-guided with automatic lead
   polarity alignment and QRS correlation screening): HR, RR, RMSSD, PR peak
   (P-peak to R-peak), PR interval (P-onset to QRS-onset), PR segment (P-offset
   to QRS-onset, AV node conduction), QRS duration, QT, QTc (Bazett and
   Fridericia), RT peak, and normalized T-wave amplitude, averaged per phase.
4. Phase-resolved skin conductance and respiration summaries (level, rise,
   slope, dispersion).
5. Phase-resolved EEG regional log band power, scalp-wide spatial homogeneity
   (inter-channel SD, IQR, normalized Shannon entropy, inter-regional SD),
   anterior-posterior gradient, frontal and temporal hemispheric asymmetries,
   and within-phase temporal power variation (1-s SD, slope, max step).
6. Unprocessed (``CURATED``, common-average referenced across the exact 62
   channels retained in ``PREPROCESSED``, 0.5-30 Hz, NO ICA) heartbeat-evoked
   potentials locked to the identical R peaks as ``decoding.hep_extract``,
   enabling a direct empirical test of whether ICA preprocessing attenuates
   brain-heart stress modulation.
"""

import json
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch

from decoding.common import BANDS, REGIONS
from decoding.extract import (
    EEG_UNITS,
    _channel,
    _clean,
    _eeg_source,
    _r_peaks,
    _read_body,
    _windows,
)
from decoding.hep_extract import NBINS, _band, _locked, _refine
from habs.signals import eda_to_conductance
from habs.temporal.data import physical_id

BAND_EDGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
PHASES = {
    "ref": (-3.0, 0.0),
    "early": (0.0, 5.0),
    "late": (5.0, 10.0),
    "post": (10.0, 12.0),
}
ECG_WINDOWS = {**PHASES, "vid": (0.0, 10.0)}
TVAR_PHASES = {
    "ref": (-3, 0),
    "early": (0, 5),
    "late": (5, 10),
    "vid": (0, 10),
}
EXCLUDED_EEG = {"T9", "T10"}
LEFT_FRONTAL = ("Fp1", "AF3", "AF7", "F1", "F3", "F5", "F7")
RIGHT_FRONTAL = ("Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8")
EXTRA_META = (
    "self_reported_emotion_2",
    "self_reported_emotion_3",
    "expected_emotion_label",
    "expected_emotion_source",
    "flagged_outliers",
)
ECG_KEYS = (
    "pr_pk",
    "pr_on",
    "pr_seg",
    "qrs",
    "qt",
    "qtc_baz",
    "qtc_frid",
    "rt_pk",
    "t_amp",
)


def delineate_ecg(ecg, fs):
    """Delineate P, QRS, and T fiducials beat by beat on a single-lead ECG.

    Automatic polarity calibration aligns the recording so that P and T waves
    are upright (Lead I convention), avoiding inverted-lead failures on deep-S
    complexes. Beats with low QRS correlation to the recording median template
    (r < 0.70) are excluded before extracting P-QRS-T intervals in milliseconds.

    Returns
    -------
    dict
        ``times`` (s), ``rr_s`` (s), per-beat interval arrays in ``ECG_KEYS``,
        ``polarity`` (+1 or -1), and ``good_qrs_frac``.
    """
    empty = {
        "times": np.array([], dtype=float),
        "rr_s": np.array([], dtype=float),
        **{k: np.array([], dtype=float) for k in ECG_KEYS},
        "polarity": np.nan,
        "good_qrs_frac": np.nan,
    }
    if (
        ecg is None
        or fs < 100
        or len(ecg) < round(5 * fs)
        or not np.isfinite(ecg).all()
    ):
        return empty
    y = sosfiltfilt(butter(3, (0.5, 40.0), btype="bandpass", fs=fs, output="sos"), ecg)
    y5 = sosfiltfilt(butter(2, (5.0, 25.0), btype="bandpass", fs=fs, output="sos"), ecg)
    w_env = max(1, round(0.08 * fs))
    env = np.convolve(np.abs(y5), np.ones(w_env) / w_env, mode="same")
    std_env = float(np.std(env))
    if not std_env > 0:
        return empty
    peaks, _ = find_peaks(env, distance=round(0.32 * fs), prominence=0.7 * std_env)
    if len(peaks) < 8:
        return empty
    half = round(0.06 * fs)
    anchors = np.array(
        [
            max(0, p - half)
            + int(np.argmax(np.abs(y[max(0, p - half) : min(len(y), p + half + 1)])))
            for p in peaks
        ]
    )
    keep = [0]
    min_gap = round(0.25 * fs)
    for k in range(1, len(anchors)):
        if anchors[k] - anchors[keep[-1]] >= min_gap:
            keep.append(k)
    anchors = anchors[keep]

    pre_n, post_n = round(0.35 * fs), round(0.55 * fs)
    valid = anchors[(anchors >= pre_n) & (anchors + post_n < len(y))]
    if len(valid) < 8:
        return empty

    tmpl = np.median(
        np.stack([y[a - pre_n : a + post_n] for a in valid], axis=0), axis=0
    )
    iso = float(
        np.median(np.r_[tmpl[: round(0.10 * fs)], tmpl[pre_n + round(0.45 * fs) :]])
    )
    tmpl0 = tmpl - iso
    p_sl = tmpl0[pre_n - round(0.24 * fs) : pre_n - round(0.07 * fs)]
    t_sl = tmpl0[pre_n + round(0.13 * fs) : pre_n + round(0.42 * fs)]
    p_ext = float(p_sl[np.argmax(np.abs(p_sl))]) if len(p_sl) else 0.0
    t_ext = float(t_sl[np.argmax(np.abs(t_sl))]) if len(t_sl) else 0.0
    polarity = -1.0 if (p_ext + t_ext) < 0 else 1.0
    y = y * polarity
    tmpl0 = tmpl0 * polarity

    # Re-anchor to the positive R-peak of the polarity-aligned QRS template
    r_half = round(0.05 * fs)
    r_win = tmpl0[pre_n - r_half : pre_n + r_half]
    if np.max(r_win) >= 0.25 * np.max(np.abs(r_win)):
        r_shift = int(np.argmax(r_win)) - r_half
        rad_r = round(0.025 * fs)
        r_peaks = np.array(
            [
                max(0, a + r_shift - rad_r)
                + int(
                    np.argmax(
                        y[
                            max(0, a + r_shift - rad_r) : min(
                                len(y), a + r_shift + rad_r + 1
                            )
                        ]
                    )
                )
                for a in valid
            ]
        )
    else:
        r_peaks = valid.copy()

    valid_r = r_peaks[(r_peaks >= pre_n) & (r_peaks + post_n < len(y))]
    if len(valid_r) < 8:
        return empty

    qr = round(0.08 * fs)
    qrs_mat = np.stack([y[r - qr : r + qr] for r in valid_r], axis=0)
    qrs_tmpl = np.median(qrs_mat, axis=0)
    qrs_tmpl = qrs_tmpl - qrs_tmpl.mean()
    qrs_c = qrs_mat - qrs_mat.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(qrs_c, axis=1) * np.linalg.norm(qrs_tmpl)
    corr = np.where(denom > 0, (qrs_c @ qrs_tmpl) / denom, 0.0)
    good_qrs = corr >= 0.70
    if good_qrs.sum() < 5:
        return {**empty, "polarity": polarity, "good_qrs_frac": float(good_qrs.mean())}

    tmpl_r = np.median(
        np.stack([y[r - pre_n : r + post_n] for r in valid_r[good_qrs]], axis=0), axis=0
    )
    tmpl_r = tmpl_r - float(np.median(tmpl_r[: round(0.09 * fs)]))

    p_nom = -round(0.24 * fs) + int(
        np.argmax(tmpl_r[pre_n - round(0.24 * fs) : pre_n - round(0.065 * fs)])
    )
    t_seg_tmpl = tmpl_r[pre_n + round(0.13 * fs) : pre_n + round(0.43 * fs)]
    t_sign = 1.0 if np.max(t_seg_tmpl) >= abs(np.min(t_seg_tmpl)) * 0.6 else -1.0
    t_nom = round(0.13 * fs) + int(np.argmax(t_sign * t_seg_tmpl))

    d_tmpl = np.abs(np.gradient(tmpl_r))
    max_dt = float(np.max(d_tmpl[pre_n - round(0.07 * fs) : pre_n + round(0.07 * fs)]))
    q_act = np.where(
        d_tmpl[pre_n - round(0.075 * fs) : pre_n - round(0.01 * fs)] > 0.12 * max_dt
    )[0]
    q_on_nom = -round(0.075 * fs) + int(q_act[0]) if len(q_act) else -round(0.035 * fs)
    s_act = np.where(
        d_tmpl[pre_n + round(0.01 * fs) : pre_n + round(0.085 * fs)] > 0.12 * max_dt
    )[0]
    q_off_nom = round(0.01 * fs) + int(s_act[-1]) if len(s_act) else round(0.045 * fs)

    y_pt = sosfiltfilt(
        butter(3, (0.5, 15.0), btype="bandpass", fs=fs, output="sos"), ecg * polarity
    )
    y_qrs = sosfiltfilt(
        butter(3, (1.0, 35.0), btype="bandpass", fs=fs, output="sos"), ecg * polarity
    )
    dy = np.abs(np.gradient(y_qrs))

    n_b = len(valid_r)
    times = valid_r / fs
    rr_s = np.full(n_b, np.nan)
    diffs = np.diff(times)
    plaus_rr = (diffs >= 0.35) & (diffs <= 1.5)
    rr_s[1:] = np.where(plaus_rr, diffs, np.nan)

    out = {k: np.full(n_b, np.nan) for k in ECG_KEYS}
    p_rad, t_rad, q_rad = round(0.045 * fs), round(0.060 * fs), round(0.025 * fs)
    for i, r in enumerate(valid_r):
        if not good_qrs[i]:
            continue
        base_r = float(np.median(y_qrs[r - round(0.32 * fs) : r - round(0.25 * fs)]))
        r_amp = float(y_qrs[r] - base_r)
        if r_amp <= 0:
            r_amp = float(
                np.max(
                    np.abs(y_qrs[r - round(0.03 * fs) : r + round(0.03 * fs)] - base_r)
                )
            )
        if not r_amp > 0:
            continue

        # P wave: peak, onset, offset
        p_lo = max(r - round(0.26 * fs), r + p_nom - p_rad)
        p_hi = min(r - round(0.065 * fs), r + p_nom + p_rad)
        if p_hi <= p_lo:
            continue
        p_idx = p_lo + int(np.argmax(y_pt[p_lo:p_hi]))
        pb_lo = max(0, p_idx - round(0.085 * fs))
        p_base = float(np.min(y_pt[pb_lo : p_idx + 1]))
        p_thr = p_base + 0.15 * (float(y_pt[p_idx]) - p_base)
        below = np.where(y_pt[pb_lo : p_idx + 1] <= p_thr)[0]
        p_on = pb_lo + int(below[-1]) if len(below) else pb_lo

        pe_hi = min(r - round(0.02 * fs), p_idx + round(0.08 * fs))
        if pe_hi > p_idx:
            p_end_seg = y_pt[p_idx:pe_hi]
            p_end_base = float(np.min(p_end_seg))
            pe_thr = p_end_base + 0.15 * (float(y_pt[p_idx]) - p_end_base)
            p_after = np.where(p_end_seg <= pe_thr)[0]
            p_off = p_idx + int(p_after[0]) if len(p_after) else pe_hi
        else:
            p_off = p_idx

        # QRS onset and offset guided by recording template
        max_dy = float(np.max(dy[r - round(0.06 * fs) : r + round(0.06 * fs)]))
        ql = max(r - round(0.08 * fs), r + q_on_nom - q_rad)
        qh = min(r - round(0.008 * fs), r + q_on_nom + q_rad)
        qa = np.where(dy[ql:qh] > 0.12 * max_dy)[0] if qh > ql else np.array([])
        q_on = ql + int(qa[0]) if len(qa) else r + q_on_nom

        sl = max(r + round(0.008 * fs), r + q_off_nom - q_rad)
        sh = min(r + round(0.09 * fs), r + q_off_nom + q_rad)
        sa = np.where(dy[sl:sh] > 0.12 * max_dy)[0] if sh > sl else np.array([])
        q_off = sl + int(sa[-1]) if len(sa) else r + q_off_nom

        # T wave: peak and offset
        t_lo = max(r + round(0.12 * fs), r + t_nom - t_rad)
        t_hi = min(r + round(0.45 * fs), r + t_nom + t_rad)
        if t_hi <= t_lo:
            continue
        t_idx = t_lo + int(np.argmax(t_sign * y_pt[t_lo:t_hi]))
        te_hi = min(len(y_pt), t_idx + round(0.14 * fs))
        t_sig = t_sign * y_pt[t_idx:te_hi]
        t_base = float(np.min(t_sig))
        t_thr = t_base + 0.15 * (float(t_sig[0]) - t_base)
        tb = np.where(t_sig <= t_thr)[0]
        t_off = t_idx + int(tb[0]) if len(tb) else te_hi - 1

        pr_pk = (r - p_idx) * 1000.0 / fs
        pr_on = (q_on - p_on) * 1000.0 / fs
        pr_seg = (q_on - p_off) * 1000.0 / fs
        qrs = (q_off - q_on) * 1000.0 / fs
        qt = (t_off - q_on) * 1000.0 / fs
        rt_pk = (t_idx - r) * 1000.0 / fs
        t_amp = (float(t_sig[0]) - t_base) / r_amp

        if 75.0 <= pr_pk <= 280.0:
            out["pr_pk"][i] = pr_pk
        if 80.0 <= pr_on <= 280.0:
            out["pr_on"][i] = pr_on
        if 15.0 <= pr_seg <= 180.0:
            out["pr_seg"][i] = pr_seg
        if 45.0 <= qrs <= 160.0:
            out["qrs"][i] = qrs
        if 240.0 <= qt <= 520.0:
            out["qt"][i] = qt
            if np.isfinite(rr_s[i]):
                qtb = qt / np.sqrt(rr_s[i])
                qtf = qt / (rr_s[i] ** (1.0 / 3.0))
                if 260.0 <= qtb <= 560.0:
                    out["qtc_baz"][i] = qtb
                if 260.0 <= qtf <= 560.0:
                    out["qtc_frid"][i] = qtf
        if 140.0 <= rt_pk <= 440.0:
            out["rt_pk"][i] = rt_pk
        if 0.02 <= t_amp <= 2.5:
            out["t_amp"][i] = t_amp

    return {
        "times": times,
        "rr_s": rr_s,
        **out,
        "polarity": polarity,
        "good_qrs_frac": float(good_qrs.mean()),
    }


def _ecg_phase_features(ecg, fs, onsets):
    """Aggregate beat-level ECG P-QRS-T intervals over each physiological phase."""
    n = len(onsets)
    delin = delineate_ecg(ecg, fs)
    cols = {
        "ecg_polarity": np.full(n, delin["polarity"], dtype=float),
        "ecg_good_qrs_frac": np.full(n, delin["good_qrs_frac"], dtype=float),
    }
    times, rr_s = delin["times"], delin["rr_s"]
    for wname, (a, b) in ECG_WINDOWS.items():
        cols[f"ecg_hr_{wname}"] = np.full(n, np.nan)
        cols[f"ecg_rr_{wname}"] = np.full(n, np.nan)
        cols[f"ecg_rmssd_{wname}"] = np.full(n, np.nan)
        cols[f"ecg_nbeats_{wname}"] = np.zeros(n, dtype=float)
        for key in ECG_KEYS:
            cols[f"ecg_{key}_{wname}"] = np.full(n, np.nan)
        if not len(times):
            continue
        for k, onset in enumerate(onsets):
            inside = (times >= onset + a) & (times < onset + b)
            cols[f"ecg_nbeats_{wname}"][k] = float(inside.sum())
            if not inside.any():
                continue
            rr_in = rr_s[inside]
            rr_ok = rr_in[np.isfinite(rr_in)]
            if len(rr_ok) >= 1:
                cols[f"ecg_rr_{wname}"][k] = float(np.mean(rr_ok) * 1000.0)
                cols[f"ecg_hr_{wname}"][k] = float(60.0 / np.mean(rr_ok))
            if len(rr_ok) >= 3:
                cols[f"ecg_rmssd_{wname}"][k] = float(
                    np.sqrt(np.mean(np.diff(rr_ok) ** 2)) * 1000.0
                )
            for key in ECG_KEYS:
                vals = delin[key][inside]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 1:
                    cols[f"ecg_{key}_{wname}"][k] = float(np.mean(vals))
    return cols


def _linear_slope(y, fs):
    """Row-wise linear slope (units per second) for a 2D array (trials, samples)."""
    m = y.shape[1]
    t = np.arange(m, dtype=float) / fs
    t = t - t.mean()
    denom = float(np.sum(t * t))
    if not denom > 0:
        return np.full(len(y), np.nan)
    ok = np.isfinite(y).all(axis=1)
    out = np.full(len(y), np.nan)
    if ok.any():
        yc = y[ok] - y[ok].mean(axis=1, keepdims=True)
        out[ok] = (yc @ t) / denom
    return out


def _eda_phase_features(eda, fs, unit, onsets):
    """Mean, max, rise, slope, and SD of skin conductance per phase."""
    n = len(onsets)
    cols = {
        f"sc_{stat}_{w}": np.full(n, np.nan)
        for w in PHASES
        for stat in ("mean", "max", "rise", "slope", "std")
    }
    if eda is None or fs < 10:
        return cols
    conductance, _ = eda_to_conductance(eda, unit, max_invalid_fraction=1.0)
    # Smooth lightly (0.25 s) to damp sample noise before max/slope
    smooth = max(1, round(0.25 * fs))
    # Downsample to ~10 Hz inside windows for fast, clean stats
    step = max(1, round(fs / 10.0))
    eff_fs = fs / step
    for wname, (a, b) in PHASES.items():
        w = _windows(conductance, fs, onsets, a, b - a)
        ok = np.isfinite(w).all(axis=1)
        if not ok.any():
            continue
        # Moving-average smooth then decimate
        if smooth > 1 and w.shape[1] > smooth:
            csum = np.cumsum(np.pad(w[ok], ((0, 0), (1, 0))), axis=1)
            ws = (csum[:, smooth:] - csum[:, :-smooth]) / smooth
        else:
            ws = w[ok]
        wd = ws[:, ::step]
        cols[f"sc_mean_{wname}"][ok] = wd.mean(axis=1)
        cols[f"sc_max_{wname}"][ok] = wd.max(axis=1)
        cols[f"sc_rise_{wname}"][ok] = wd.max(axis=1) - wd[:, 0]
        cols[f"sc_std_{wname}"][ok] = wd.std(axis=1)
        cols[f"sc_slope_{wname}"][ok] = _linear_slope(wd, eff_fs)
    return cols


def _resp_phase_features(resp, fs, onsets):
    """Respiration Hilbert amplitude, variability, slope, and rate per phase."""
    n = len(onsets)
    cols = {
        f"resp_{stat}_{w}": np.full(n, np.nan)
        for w in PHASES
        for stat in ("amp", "amp_std", "slope", "rate")
    }
    if resp is None or fs < 10 or not np.isfinite(resp).all():
        return cols
    y = sosfiltfilt(butter(2, (0.1, 1.0), btype="bandpass", fs=fs, output="sos"), resp)
    scale = float(np.std(y))
    if not scale > 0:
        return cols
    env = np.abs(hilbert(y / scale))
    tops, _ = find_peaks(y, distance=round(1.5 * fs), prominence=0.3 * scale)
    t_tops = tops / fs
    if len(t_tops) >= 2:
        br_times = t_tops[1:]
        br_vals = 60.0 / np.diff(t_tops)
    else:
        br_times, br_vals = np.array([]), np.array([])
    step = max(1, round(fs / 10.0))
    eff_fs = fs / step
    for wname, (a, b) in PHASES.items():
        w = _windows(env, fs, onsets, a, b - a)
        ok = np.isfinite(w).all(axis=1)
        if ok.any():
            wd = w[ok, ::step]
            cols[f"resp_amp_{wname}"][ok] = wd.mean(axis=1)
            cols[f"resp_amp_std_{wname}"][ok] = wd.std(axis=1)
            cols[f"resp_slope_{wname}"][ok] = _linear_slope(wd, eff_fs)
        if len(br_times) >= 2:
            # Interpolate instantaneous breathing rate at 4 Hz inside [a, b)
            grid_rel = np.linspace(a + 0.125, b - 0.125, max(4, round((b - a) * 4)))
            for k, onset in enumerate(onsets):
                grid = onset + grid_rel
                vals = np.interp(grid, br_times, br_vals, left=np.nan, right=np.nan)
                nearest = np.abs(
                    br_times[
                        np.clip(np.searchsorted(br_times, grid), 0, len(br_times) - 1)
                    ]
                    - grid
                )
                vals[nearest > 8.0] = np.nan
                if np.isfinite(vals).any():
                    cols[f"resp_rate_{wname}"][k] = float(np.nanmean(vals))
    return cols


def _phase_psd(w, fs, nperseg):
    """Band powers (linear and log10) for trials x samples array."""
    f, p = welch(w, fs=fs, nperseg=min(nperseg, w.shape[1]), axis=-1)
    lin = {}
    for band, (lo, hi) in BAND_EDGES.items():
        mask = (f >= lo) & (f < hi)
        lin[band] = trapezoid(p[:, mask], f[mask], axis=-1)
    return lin


def _eeg_phase_and_spatial(path, onsets):
    """4-phase regional power, scalp spatial homogeneity, asymmetries, temporal var."""
    n = len(onsets)
    wanted_region = {
        ch: r for r, chs in REGIONS.items() for ch in chs if ch not in EXCLUDED_EEG
    }
    left_set, right_set = set(LEFT_FRONTAL), set(RIGHT_FRONTAL)
    nperseg_map = {"ref": 1.5, "early": 2.0, "late": 2.0, "post": 1.0}

    # Collect per-channel log & linear power per phase & band, plus 1-s regional bins
    ch_log = {(w, b): [] for w in PHASES for b in BANDS}
    ch_lin = {(w, b): [] for w in PHASES for b in BANDS}
    reg_log = {(w, r, b): [] for w in PHASES for r in REGIONS for b in BANDS}
    lf_log = {(w, b): [] for w in PHASES for b in BANDS}
    rf_log = {(w, b): [] for w in PHASES for b in BANDS}
    # 1-second bins [-3, 10) -> 13 bins for within-phase temporal variation
    reg_1s = {(r, b): [] for r in REGIONS for b in BANDS}

    with pyedflib.EdfReader(str(path)) as reader:
        names = reader.getSignalLabels()
        stop = min(
            [names.index(x) for x in ("ECG", "Resp", "GSR") if x in names] or [64]
        )
        eeg_names = [nm for nm in names[:stop] if nm not in EXCLUDED_EEG]
        for name in eeg_names:
            x, fs, unit = _channel(reader, name)
            if unit not in EEG_UNITS or fs < 100:
                continue
            x = x * EEG_UNITS[unit]
            region = wanted_region.get(name)
            for wname, (a, b) in PHASES.items():
                seg = _clean(_windows(x, fs, onsets, a, b - a))
                lin = _phase_psd(seg, fs, round(nperseg_map[wname] * fs))
                for band in BANDS:
                    lp = lin[band]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lg = np.where(lp > 0, np.log10(lp), np.nan)
                    ch_log[(wname, band)].append(lg)
                    ch_lin[(wname, band)].append(np.where(np.isfinite(lg), lp, np.nan))
                    if region is not None:
                        reg_log[(wname, region, band)].append(lg)
                    if name in left_set:
                        lf_log[(wname, band)].append(lg)
                    if name in right_set:
                        rf_log[(wname, band)].append(lg)
            if region is not None:
                # 1-s bins from -3 to +10 s (13 bins)
                ns = round(fs)
                w1 = _windows(x, fs, onsets, -3.0, ns * 13 / fs)[:, : ns * 13]
                w1 = _clean(w1.reshape(n * 13, ns))
                f, p = welch(w1, fs=fs, nperseg=ns, axis=-1)
                for band, (lo, hi) in BAND_EDGES.items():
                    mask = (f >= lo) & (f < hi)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lg1 = np.log10(trapezoid(p[:, mask], f[mask], axis=-1))
                    reg_1s[(region, band)].append(lg1.reshape(n, 13))

    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for wname in PHASES:
            for band in BANDS:
                # Regional averages
                reg_vals = {}
                for r in REGIONS:
                    parts = reg_log[(wname, r, band)]
                    val = np.nanmean(parts, axis=0) if parts else np.full(n, np.nan)
                    reg_vals[r] = val
                    cols[f"eeg_ep_{wname}_{r}_{band}"] = val
                # Spatial dispersion across 6 regions
                reg_stack = np.stack([reg_vals[r] for r in REGIONS], axis=1)
                cols[f"eeg_reg_std_{wname}_{band}"] = np.nanstd(reg_stack, axis=1)
                # Anterior-posterior gradient & asymmetries
                cols[f"eeg_ap_grad_{wname}_{band}"] = (
                    reg_vals["frontal"] - reg_vals["occipital"]
                )
                cols[f"eeg_temp_asym_{wname}_{band}"] = (
                    reg_vals["right_temporal"] - reg_vals["left_temporal"]
                )
                lf = lf_log[(wname, band)]
                rf = rf_log[(wname, band)]
                if lf and rf:
                    cols[f"eeg_faa_{wname}_{band}"] = np.nanmean(
                        rf, axis=0
                    ) - np.nanmean(lf, axis=0)
                else:
                    cols[f"eeg_faa_{wname}_{band}"] = np.full(n, np.nan)
                # Channel-level spatial homogeneity across all clean scalp derivations
                clg = ch_log[(wname, band)]
                cln = ch_lin[(wname, band)]
                if clg:
                    mat_lg = np.stack(clg, axis=1)  # (trials, n_ch)
                    mat_ln = np.stack(cln, axis=1)
                    cols[f"eeg_sp_std_{wname}_{band}"] = np.nanstd(mat_lg, axis=1)
                    q75 = np.nanpercentile(mat_lg, 75, axis=1)
                    q25 = np.nanpercentile(mat_lg, 25, axis=1)
                    cols[f"eeg_sp_iqr_{wname}_{band}"] = q75 - q25
                    # Normalized Shannon entropy of scalp power distribution
                    valid_n = np.isfinite(mat_ln).sum(axis=1)
                    tot = np.nansum(mat_ln, axis=1, keepdims=True)
                    prob = np.where((tot > 0) & np.isfinite(mat_ln), mat_ln / tot, 0.0)
                    ent = -np.sum(np.where(prob > 0, prob * np.log(prob), 0.0), axis=1)
                    cols[f"eeg_sp_ent_{wname}_{band}"] = np.where(
                        valid_n >= 10, ent / np.log(np.maximum(valid_n, 2)), np.nan
                    )
                else:
                    for stat in ("sp_std", "sp_iqr", "sp_ent"):
                        cols[f"eeg_{stat}_{wname}_{band}"] = np.full(n, np.nan)

        # Within-phase temporal variation of 1-s EEG power
        for band in BANDS:
            # Build (n_regions, trials, 13) array
            r_arr = np.stack(
                [
                    np.nanmean(reg_1s[(r, band)], axis=0)
                    if reg_1s[(r, band)]
                    else np.full((n, 13), np.nan)
                    for r in REGIONS
                ],
                axis=0,
            )
            for tname, (s0, s1) in TVAR_PHASES.items():
                sl = r_arr[:, :, s0 + 3 : s1 + 3]  # (6, trials, L)
                # Temporal SD within phase, averaged across regions
                cols[f"eeg_tvar_{tname}_{band}"] = np.nanmean(
                    np.nanstd(sl, axis=2), axis=0
                )
                if tname != "ref":
                    # Max |delta_t| second-to-second jump inside phase
                    diffs = np.abs(np.diff(sl, axis=2))
                    cols[f"eeg_maxjump_{tname}_{band}"] = np.nanmean(
                        np.nanmax(diffs, axis=2), axis=0
                    )
                    # Mean linear slope (log10 power / s) across the phase
                    length = sl.shape[2]
                    tvec = np.arange(length, dtype=float)
                    tvec -= tvec.mean()
                    denom = float(np.sum(tvec * tvec))
                    # Mean across regions first for a clean scalp-wide slope
                    mean_r = np.nanmean(sl, axis=0)  # (trials, L)
                    ok = np.isfinite(mean_r).all(axis=1)
                    slope = np.full(n, np.nan)
                    if ok.any():
                        mc = mean_r[ok] - mean_r[ok].mean(axis=1, keepdims=True)
                        slope[ok] = (mc @ tvec) / denom
                    cols[f"eeg_tslope_{tname}_{band}"] = slope
    return cols


def _raw_curated_hep(curated_path, beats, onsets):
    """Heartbeat-evoked EEG from unprocessed CURATED data (CAR, 0.5-30 Hz, NO ICA).

    Uses the exact 62-channel montage (excluding T9/T10) and identical R peaks
    as ``decoding.hep_extract`` so any difference from ICA-cleaned HEP isolates
    the effect of ICA preprocessing.
    """
    n = len(onsets)
    wanted_region = {
        ch: r for r, chs in REGIONS.items() for ch in chs if ch not in EXCLUDED_EEG
    }
    cols = {}
    with pyedflib.EdfReader(str(curated_path)) as reader:
        names = reader.getSignalLabels()
        stop = min(
            [names.index(x) for x in ("ECG", "Resp", "GSR") if x in names] or [64]
        )
        eeg_idx = []
        eeg_names = []
        fs_ref = None
        for idx in range(stop):
            nm = names[idx]
            if nm in EXCLUDED_EEG:
                continue
            unit = reader.getPhysicalDimension(idx).replace("µ", "u").replace("μ", "u")
            fs = float(reader.getSampleFrequency(idx))
            if unit not in EEG_UNITS or fs < 100:
                continue
            if fs_ref is None:
                fs_ref = fs
            if abs(fs - fs_ref) > 0.1:
                continue
            eeg_idx.append((idx, EEG_UNITS[unit]))
            eeg_names.append(nm)

        if not eeg_idx or fs_ref is None:
            return cols

        # Read all 62 channels as float32 and apply common-average reference (CAR)
        mat = np.stack(
            [
                (reader.readSignal(idx) * scale).astype(np.float32)
                for idx, scale in eeg_idx
            ],
            axis=0,
        )
    car = np.nanmean(mat, axis=0, dtype=np.float64).astype(np.float32)
    mat -= car[None, :]
    del car

    # Frontopolar peak-to-peak amplitude in raw CURATED (pre-ICA ocular/blink proxy)
    fp_rows = [i for i, nm in enumerate(eeg_names) if nm in ("Fp1", "Fp2")]
    if fp_rows:
        fp_sig = np.nanmean(mat[fp_rows], axis=0).astype(np.float64)
        for tag, (a, b) in (
            ("ref", (-3.0, 0.0)),
            ("0_2", (0.0, 2.0)),
            ("early", (0.0, 5.0)),
            ("late", (5.0, 10.0)),
        ):
            seg = _windows(fp_sig, fs_ref, onsets, a, b - a)
            ok = np.isfinite(seg).all(axis=1)
            ptp = np.full(n, np.nan)
            if ok.any():
                ptp[ok] = np.ptp(seg[ok], axis=1)
            cols[f"raw_fp_ptp_{tag}"] = ptp

    sums_100 = {}
    sums_200 = {}
    counts_100 = {"pre": [], "video": []}
    counts_200 = []
    windows = {"pre": (-3.0, 0.0), "video": (0.0, 10.0)}

    for row_i, nm in enumerate(eeg_names):
        region = wanted_region.get(nm)
        if region is None:
            continue
        x = _band(mat[row_i].astype(np.float64), fs_ref)
        for wname, (a, b) in windows.items():
            res = [_locked(x, fs_ref, beats, t + a, t + b, 100.0) for t in onsets]
            mean = np.array([m for m, _ in res])
            cnt = np.array([c for _, c in res], dtype=float)
            sums_100.setdefault((region, wname), []).append(mean)
            counts_100[wname].append(cnt)
        if region in ("frontal", "central"):
            res2 = [_locked(x, fs_ref, beats, t, t + 10.0, 200.0) for t in onsets]
            sums_200.setdefault(region, []).append(np.array([m for m, _ in res2]))
            counts_200.append(np.array([c for _, c in res2], dtype=float))
    del mat

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for region in REGIONS:
            for wname in windows:
                parts = sums_100.get((region, wname))
                val = (
                    np.nanmean(parts, axis=0) if parts else np.full((n, NBINS), np.nan)
                )
                for k in range(NBINS):
                    cols[f"hep_raw_{wname}_{region}_{k}"] = val[:, k]
        for region in ("frontal", "central"):
            parts = sums_200.get(region)
            val = np.nanmean(parts, axis=0) if parts else np.full((n, NBINS), np.nan)
            for k in range(NBINS):
                cols[f"hep_raw200_video_{region}_{k}"] = val[:, k]
        for wname in windows:
            cols[f"hep_raw_beats_{wname}"] = (
                np.nanmedian(counts_100[wname], axis=0)
                if counts_100[wname]
                else np.zeros(n)
            )
        cols["hep_raw200_beats_video"] = (
            np.nanmedian(counts_200, axis=0) if counts_200 else np.zeros(n)
        )
    return cols


def _recording(arguments):
    """Extract all Study-07 enriched features for one recording."""
    raw_dir, trials = arguments
    curated = Path(raw_dir) / trials[0]["recording"]
    onsets = np.array([float(t["onset_s"]) for t in trials])
    columns = {"trial_id": [physical_id(t) for t in trials]}

    ecg, ecg_fs, resp, resp_fs, eda, eda_fs, eda_unit, _ = _read_body(curated)
    columns.update(_ecg_phase_features(ecg, ecg_fs, onsets))
    columns.update(_eda_phase_features(eda, eda_fs, eda_unit, onsets))
    columns.update(_resp_phase_features(resp, resp_fs, onsets))

    # 4-phase EEG, spatial homogeneity, asymmetries, and temporal variation
    eeg_path, shift = _eeg_source(raw_dir, trials[0])
    try:
        columns.update(_eeg_phase_and_spatial(eeg_path, onsets + shift))
    except (OSError, ValueError) as error:
        print(f"EEG phase read failed for {eeg_path.name}: {error}", flush=True)

    # Raw CURATED HEP locked to identical R peaks as hep_extract.py (NO ICA)
    if ecg is not None and ecg_fs >= 100 and np.isfinite(ecg).all():
        beats = _refine(_r_peaks(ecg, ecg_fs), ecg, ecg_fs)
        try:
            columns.update(_raw_curated_hep(curated, beats, onsets))
        except (OSError, ValueError) as error:
            print(f"CURATED HEP read failed for {curated.name}: {error}", flush=True)

    frame = pd.DataFrame(columns)
    numeric = frame.columns.drop("trial_id")
    frame[numeric] = frame[numeric].astype("float32")
    return frame


def run(trials, raw_dir, *, workers):
    """Apply ``_recording`` across every recording with bounded worker memory."""
    jobs = [
        (str(raw_dir), group.to_dict("records"))
        for _, group in trials.groupby("recording", sort=True)
    ]
    parts = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_recording, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            parts.append(future.result())
            if index % 25 == 0:
                print(f"Recordings: {index}/{len(jobs)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    """Write features/rich.parquet and public/rich.json coverage summary."""
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out = Path(os.environ["HABS_ANALYSIS_OUT"])
    trials = pd.read_csv(
        out / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # 8 workers keeps peak memory safe while loading 62ch CURATED for CAR HEP
    workers = min(8, os.cpu_count() or 4)
    extracted = run(trials, raw_dir, workers=workers)
    keep_meta = [c for c in EXTRA_META if c in trials]
    table = trials[["trial_id", *keep_meta]].merge(
        extracted, on="trial_id", validate="1:1"
    )
    (out / "features").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "features" / "rich.parquet", compression="zstd")

    def _med(col):
        if col not in table:
            return None
        s = table[col].dropna()
        return round(float(s.median()), 2) if len(s) else None

    coverage = {
        "rows": len(table),
        "columns": len(table.columns),
        "workers": workers,
        "coverage": {
            c: round(float(table[c].notna().mean()), 4)
            for c in (
                "ecg_pr_pk_vid",
                "ecg_pr_on_vid",
                "ecg_pr_seg_vid",
                "ecg_qrs_vid",
                "ecg_qt_vid",
                "ecg_qtc_baz_vid",
                "ecg_rt_pk_vid",
                "ecg_t_amp_vid",
                "ecg_pr_pk_early",
                "ecg_pr_pk_late",
                "sc_mean_early",
                "resp_amp_early",
                "eeg_ep_early_frontal_delta",
                "eeg_sp_ent_early_alpha",
                "eeg_faa_early_alpha",
                "eeg_tvar_early_alpha",
                "hep_raw_video_frontal_10",
                "hep_raw_video_central_10",
                "hep_raw200_video_frontal_10",
                "flagged_outliers",
            )
            if c in table
        },
        "medians_vid": {
            c: _med(c)
            for c in (
                "ecg_hr_vid",
                "ecg_pr_pk_vid",
                "ecg_pr_on_vid",
                "ecg_pr_seg_vid",
                "ecg_qrs_vid",
                "ecg_qt_vid",
                "ecg_qtc_baz_vid",
                "ecg_qtc_frid_vid",
                "ecg_rt_pk_vid",
                "ecg_t_amp_vid",
                "ecg_good_qrs_frac",
                "hep_raw_beats_video",
                "hep_raw200_beats_video",
            )
        },
        "ecg_inverted_fraction": round(float((table["ecg_polarity"] < 0).mean()), 4)
        if "ecg_polarity" in table
        else None,
    }
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "rich.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
