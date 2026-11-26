"""Signal calibration, ECG P-QRS-T delineation, and hard causal t = 5.000 s windowing.

Eye-tracking and pupillometry channels are excluded from primary modeling in
accordance with the HABS organizer advisory documenting temporal misalignment
between gaze/pupil streams and the other modalities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyedflib
from scipy.signal import butter, find_peaks, lfilter, sosfilt, sosfilt_zi, sosfiltfilt

CALIBRATION_VOLTS = (-2.14, -1.99755, -1.59528, 0.0167, 1.70074, 2.00199)
CALIBRATION_KOHMS = (1000.0, 500.0, 250.0, 125.0, 50.0, 25.0)

BAND_EDGES: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

PHASES: dict[str, tuple[float, float]] = {
    "ref": (-3.0, 0.0),
    "early": (0.0, 5.0),
    "late": (5.0, 10.0),
    "post": (10.0, 12.0),
}

REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "F1", "F2", "F3", "F4", "Fz"),
    "left_temporal": ("F7", "FT7", "T7", "TP7", "F9", "T9", "P9"),
    "right_temporal": ("F8", "FT8", "T8", "TP8", "F10", "T10", "P10"),
    "central": ("FC1", "FC2", "FCz", "C1", "C2", "C3", "C4", "CP1", "CP2"),
    "parietal": ("P1", "P2", "P3", "P4", "Pz", "P5", "P6", "P7", "P8", "CP3", "CP4"),
    "occipital": ("O1", "O2", "PO3", "PO4", "PO7", "PO8", "POz", "PO5", "PO6"),
}

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


def eda_to_conductance(
    values: np.ndarray, header_unit: str, *, max_invalid_fraction: float = 0.01
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert raw HABS GSR voltage samples to microsiemens without silent clamping.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        Raw GSR channel samples from pyEDFlib.
    header_unit : str
        Physical dimension stored in the BDF header.
    max_invalid_fraction : float, default=0.01
        Maximum tolerated fraction of non-finite or out-of-range samples.

    Returns
    -------
    conductance : numpy.ndarray
        Skin conductance in microsiemens, or NaN for invalid windows.
    quality : dict
        Conversion metadata, unit mismatch flag, and invalid fraction.
    """
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not 0 <= max_invalid_fraction <= 1:
        raise ValueError("Expected a 1D vector and max_invalid_fraction in [0, 1].")
    converted = np.full(raw.shape, np.nan)
    unit = str(header_unit).strip().replace("µ", "u").replace("μ", "u")
    quality: dict[str, Any] = {"header_unit": str(header_unit), "unit_mismatch": False}
    finite = np.isfinite(raw)
    if not raw.size or not finite.any():
        return converted, {**quality, "status": "missing", "invalid_fraction": 1.0}
    if unit in {"V", "mV", "uV"}:
        voltage = raw * {"V": 1.0, "mV": 1e-3, "uV": 1e-6}[unit]
        convention = "header_voltage"
    elif unit == "uS" and np.median(np.abs(raw[finite])) > 1e4:
        voltage = raw * 1e-6
        convention = "documented_microvolts"
        quality["unit_mismatch"] = True
    else:
        return converted, {
            **quality,
            "status": "ambiguous_native_unit",
            "invalid_fraction": 1.0,
        }
    valid = (
        finite & (voltage >= CALIBRATION_VOLTS[0]) & (voltage <= CALIBRATION_VOLTS[-1])
    )
    invalid_fraction = float(1.0 - valid.mean())
    quality.update({"convention": convention, "invalid_fraction": invalid_fraction})
    if invalid_fraction > max_invalid_fraction:
        return converted, {**quality, "status": "outside_calibration"}
    resistance = np.interp(
        voltage, CALIBRATION_VOLTS, CALIBRATION_KOHMS, left=np.nan, right=np.nan
    )
    converted[valid] = 1000.0 / resistance[valid]
    return converted, {**quality, "status": "ok"}


def delineate_ecg(ecg: np.ndarray | None, fs: float) -> dict[str, Any]:
    """Delineate P, QRS, and T fiducials beat by beat with automatic polarity fix.

    In the HABS dataset, 96.6% of Lead I recordings exhibit inverted polarity.
    This routine aligns each recording so P and T waves are upright before
    screening beats against the median QRS template (r >= 0.70).
    """
    empty: dict[str, Any] = {
        "times": np.array([], dtype=float),
        "rr_s": np.array([], dtype=float),
        **{k: np.array([], dtype=float) for k in ECG_KEYS},
        "polarity": np.nan,
        "good_qrs_frac": np.nan,
    }
    if ecg is None or fs < 100 or len(ecg) < round(5 * fs) or not np.isfinite(ecg).all():
        return empty
    y = sosfiltfilt(butter(3, (0.5, 40.0), btype="bandpass", fs=fs, output="sos"), ecg)
    y5 = sosfiltfilt(butter(2, (5.0, 25.0), btype="bandpass", fs=fs, output="sos"), ecg)
    w_env = max(1, round(0.08 * fs))
    env = np.convolve(np.abs(y5), np.ones(w_env) / w_env, mode="same")
    std_env = float(np.std(env))
    if std_env <= 0:
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
    for k in range(1, len(anchors)):
        if anchors[k] - anchors[keep[-1]] >= round(0.25 * fs):
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
    qrs_tmpl -= qrs_tmpl.mean()
    qrs_c = qrs_mat - qrs_mat.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(qrs_c, axis=1) * np.linalg.norm(qrs_tmpl)
    corr = np.where(denom > 0, (qrs_c @ qrs_tmpl) / denom, 0.0)
    good_qrs = corr >= 0.70
    if good_qrs.sum() < 5:
        return {**empty, "polarity": polarity, "good_qrs_frac": float(good_qrs.mean())}

    tmpl_r = np.median(
        np.stack([y[r - pre_n : r + post_n] for r in valid_r[good_qrs]], axis=0), axis=0
    )
    tmpl_r -= float(np.median(tmpl_r[: round(0.09 * fs)]))
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
    rr_s[1:] = np.where((diffs >= 0.35) & (diffs <= 1.5), diffs, np.nan)

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
        if r_amp <= 0:
            continue

        p_lo = max(r - round(0.26 * fs), r + p_nom - p_rad)
        p_hi = min(r - round(0.065 * fs), r + p_nom + p_rad)
        if p_hi <= p_lo:
            continue
        p_idx = p_lo + int(np.argmax(y_pt[p_lo:p_hi]))
        pb_lo = max(0, p_idx - round(0.085 * fs))
        p_base = float(np.min(y_pt[pb_lo : p_idx + 1]))
        below = np.where(
            y_pt[pb_lo : p_idx + 1] <= p_base + 0.15 * (float(y_pt[p_idx]) - p_base)
        )[0]
        p_on = pb_lo + int(below[-1]) if len(below) else pb_lo

        pe_hi = min(r - round(0.02 * fs), p_idx + round(0.08 * fs))
        if pe_hi > p_idx:
            p_end = y_pt[p_idx:pe_hi]
            pe_base = float(np.min(p_end))
            after = np.where(p_end <= pe_base + 0.15 * (float(y_pt[p_idx]) - pe_base))[0]
            p_off = p_idx + int(after[0]) if len(after) else pe_hi
        else:
            p_off = p_idx

        max_dy = float(np.max(dy[r - round(0.06 * fs) : r + round(0.06 * fs)]))
        ql = max(r - round(0.08 * fs), r + q_on_nom - q_rad)
        qh = min(r - round(0.008 * fs), r + q_on_nom + q_rad)
        qa = np.where(dy[ql:qh] > 0.12 * max_dy)[0] if qh > ql else np.array([])
        q_on = ql + int(qa[0]) if len(qa) else r + q_on_nom

        sl = max(r + round(0.008 * fs), r + q_off_nom - q_rad)
        sh = min(r + round(0.09 * fs), r + q_off_nom + q_rad)
        sa = np.where(dy[sl:sh] > 0.12 * max_dy)[0] if sh > sl else np.array([])
        q_off = sl + int(sa[-1]) if len(sa) else r + q_off_nom

        t_lo = max(r + round(0.12 * fs), r + t_nom - t_rad)
        t_hi = min(r + round(0.45 * fs), r + t_nom + t_rad)
        if t_hi <= t_lo:
            continue
        t_idx = t_lo + int(np.argmax(t_sign * y_pt[t_lo:t_hi]))
        te_hi = min(len(y_pt), t_idx + round(0.14 * fs))
        t_sig = t_sign * y_pt[t_idx:te_hi]
        t_base = float(np.min(t_sig))
        tb = np.where(t_sig <= t_base + 0.15 * (float(t_sig[0]) - t_base))[0]
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
    """Extract strictly causal forward-filtered windows on [start_s, end_s).

    Guarantees bit-level invariance to any perturbation at or after
    ``onset + end_s`` (the hard causal wall at t = 5.000 s).
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
    filt = np.empty_like(clean)
    for i in range(len(clean)):
        filt[i], _ = sosfilt(sos, clean[i], zi=zi_unit * clean[i, 0])
    out[fin] = filt[:, n_warm : n_warm + n_keep]
    return out, fin


def causal_r_peaks(
    ecg_seg: np.ndarray, fs: float, start_offset_s: float
) -> tuple[np.ndarray, float]:
    """Detect R-peaks strictly inside one isolated segment via forward filtering."""
    if len(ecg_seg) < round(1.5 * fs) or not np.isfinite(ecg_seg).all():
        return np.array([], dtype=float), np.nan
    sos = butter(2, (5.0, min(25.0, 0.45 * fs)), btype="bandpass", fs=fs, output="sos")
    zi = sosfilt_zi(sos)
    y5, _ = sosfilt(sos, ecg_seg.astype(np.float64), zi=zi * float(ecg_seg[0]))
    w_env = max(1, round(0.08 * fs))
    env = lfilter(np.ones(w_env) / w_env, [1.0], np.abs(y5))
    skip = min(len(env) // 4, round(0.5 * fs))
    sd = float(np.std(env[skip:]))
    if sd <= 1e-8:
        return np.array([], dtype=float), np.nan
    peaks, _ = find_peaks(env, distance=round(0.32 * fs), prominence=0.7 * sd)
    times = start_offset_s + peaks / fs
    if len(times) < 2:
        return times, np.nan
    rr = np.diff(times)
    return times, float(((rr >= 0.35) & (rr <= 1.50)).mean())


def extract_causal_ecg_eda_past(
    ecg: np.ndarray,
    ecg_fs: float,
    eda: np.ndarray,
    eda_fs: float,
    eda_unit: str,
    onsets: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract strictly causal [-3, +5) s ECG and EDA summaries (zero t >= 5.0 s)."""
    n = len(onsets)
    out = {
        "b05_ecg_hr_ref": np.full(n, np.nan),
        "b05_ecg_hr_early": np.full(n, np.nan),
        "b05_ecg_hr_chg": np.full(n, np.nan),
        "b05_sc_ref": np.full(n, np.nan),
        "b05_sc_early": np.full(n, np.nan),
        "b05_sc_chg": np.full(n, np.nan),
    }
    n_ecg = round(10.0 * ecg_fs)
    n_eda = round(10.0 * eda_fs)
    w_sm = max(1, round(0.25 * eda_fs))
    kern = np.ones(w_sm) / w_sm
    idx_m3, idx_0 = round(2.0 * eda_fs), round(5.0 * eda_fs)

    for i, onset in enumerate(onsets):
        a_e = round((onset - 5.0) * ecg_fs)
        if 0 <= a_e and a_e + n_ecg <= len(ecg):
            seg = ecg[a_e : a_e + n_ecg].astype(np.float64)
            beats, _ = causal_r_peaks(seg, ecg_fs, -5.0)
            beats = beats[beats < 5.0]
            for key, w0, w1 in (
                ("b05_ecg_hr_ref", -3.5, 0.0),
                ("b05_ecg_hr_early", 0.0, 5.0),
            ):
                b = beats[(beats >= w0) & (beats < w1)]
                if len(b) >= 2:
                    rr = np.diff(b)
                    good = (rr >= 0.35) & (rr <= 1.50)
                    if good.any() and good.mean() >= 0.7:
                        out[key][i] = 60.0 / float(rr[good].mean())
            out["b05_ecg_hr_chg"][i] = (
                out["b05_ecg_hr_early"][i] - out["b05_ecg_hr_ref"][i]
            )

        a_g = round((onset - 5.0) * eda_fs)
        if 0 <= a_g and a_g + n_eda <= len(eda):
            seg_g = eda[a_g : a_g + n_eda].astype(np.float64)
            cond, _ = eda_to_conductance(seg_g, eda_unit, max_invalid_fraction=1.0)
            if np.isfinite(cond[idx_m3:]).all() and cond[idx_m3:].min() > 0:
                sm = lfilter(
                    kern, [1.0], np.nan_to_num(cond, nan=float(np.nanmedian(cond)))
                )
                out["b05_sc_ref"][i] = float(sm[idx_m3:idx_0].mean())
                out["b05_sc_early"][i] = float(sm[idx_0:].mean())
                out["b05_sc_chg"][i] = (
                    out["b05_sc_early"][i] - out["b05_sc_ref"][i]
                )
    return out


def read_bdf_summary(bdf_path: Path) -> dict[str, Any]:
    """Audit one BDF file: channel inventory, ECG polarity, and GSR calibration."""
    reader = pyedflib.EdfReader(str(bdf_path))
    try:
        labels = reader.getSignalLabels()
        freqs = [float(reader.getSampleFrequency(i)) for i in range(len(labels))]
        duration_s = float(reader.getFileDuration())
        info: dict[str, Any] = {
            "file": bdf_path.name,
            "n_channels": len(labels),
            "duration_s": round(duration_s, 2),
        }
        ecg_idx = next((i for i, lb in enumerate(labels) if "ecg" in lb.lower()), None)
        if ecg_idx is not None:
            ecg = reader.readSignal(ecg_idx)
            delin = delineate_ecg(ecg, freqs[ecg_idx])
            rr = delin["rr_s"][np.isfinite(delin["rr_s"])]
            info["ecg"] = {
                "channel": labels[ecg_idx],
                "fs": freqs[ecg_idx],
                "polarity": float(delin["polarity"]),
                "good_qrs_frac": round(float(delin["good_qrs_frac"]), 4),
                "median_hr_bpm": round(60.0 / float(np.median(rr)), 2)
                if len(rr)
                else None,
            }
        gsr_idx = next(
            (
                i
                for i, lb in enumerate(labels)
                if "gsr" in lb.lower() or "eda" in lb.lower()
            ),
            None,
        )
        if gsr_idx is not None:
            raw_gsr = reader.readSignal(gsr_idx)
            unit = reader.getPhysicalDimension(gsr_idx)
            cond, qual = eda_to_conductance(raw_gsr, unit)
            info["eda"] = {
                "channel": labels[gsr_idx],
                "fs": freqs[gsr_idx],
                "status": qual["status"],
                "unit_mismatch": qual["unit_mismatch"],
                "median_uS": round(float(np.nanmedian(cond)), 4)
                if np.isfinite(cond).any()
                else None,
            }
        return info
    finally:
        reader.close()
