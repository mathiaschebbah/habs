"""Deterministic generation of the four publication figures from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#D95F02",
    "crimson": "#B2182B",
    "slate": "#4D5B6A",
    "gold": "#C98A00",
    "green": "#1B7837",
    "muted": "#8C96A0",
    "bg_band": "#EEF3F7",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((results_dir / name).read_text())


def plot_fig1_physiology_cascade(results_dir: Path, out_path: Path) -> Path:
    """Figure 1: 1-s multimodal stress cascade and beat-by-beat ECG morphology."""
    _style()
    tc = _load(results_dir, "timecourse.json")
    ph = _load(results_dir, "physiology.json")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    # Panel A: 1-s within-person stress response curves (-3..+12 s -> 15 bins)
    curves = tc["curves"]["curves"]["dev"]["stress"]
    t_bins = np.arange(-2.5, 12.5, 1.0)
    series = [
        ("eeg_central_delta", "EEG central delta (<1 s)", PALETTE["navy"], "-"),
        ("hr", "Heart rate (-bpm, 1 s)", PALETTE["crimson"], "-"),
        ("resp_amp", "Respiration amplitude (2-6 s)", PALETTE["gold"], "--"),
        ("sc", "Skin conductance (late rise 4-10 s)", PALETTE["teal"], "-"),
    ]
    ax1.axvspan(0.0, 10.0, color=PALETTE["bg_band"], alpha=0.85, zorder=0)
    ax1.axhline(0.0, color="#666666", lw=0.8, ls=":", zorder=1)
    ax1.axvline(0.0, color="#333333", lw=0.9, ls="--", zorder=1)
    ax1.axvline(10.0, color="#333333", lw=0.8, ls=":", zorder=1)

    for key, label_txt, col, ls in series:
        cur = curves[key]
        d = np.array(cur["d"], dtype=float)
        lo = np.array(cur["low"], dtype=float)
        hi = np.array(cur["high"], dtype=float)
        ax1.plot(t_bins, d, color=col, lw=1.9, ls=ls, marker="o", ms=3.2, label=label_txt)
        ax1.fill_between(t_bins, lo, hi, color=col, alpha=0.14, lw=0)

    ax1.text(0.2, 0.215, "Video onset", fontsize=7.5, color="#333333")
    ax1.text(10.15, 0.215, "Offset", fontsize=7.5, color="#333333")
    ax1.set_xlim(-3.0, 12.2)
    ax1.set_ylim(-0.27, 0.24)
    ax1.set_xlabel("Time relative to stimulus onset (s)")
    ax1.set_ylabel("Within-person Cohen's d (stress vs. non-stress)")
    ax1.set_title("A. 1-s distress timecourse (dev, N = 235)")
    ax1.legend(loc="lower center", ncol=2, handlelength=1.6, columnspacing=1.0, fontsize=7.4)

    # Panel B: Beat-by-beat ECG morphology & RR-adjusted control + Raw vs ICA HEP
    ecg_eff = ph["ecg_morphology"]["ecg_effects"]
    rr_ctrl = ph["ecg_morphology"]["ecg_rr_controlled_and_coupling"]
    hep_var = ph["hep_raw"]["variants"]

    rows = [
        ("RR interval", ecg_eff["rr"]["vid_change"]["stress"], PALETTE["navy"]),
        ("Heart rate", ecg_eff["hr"]["vid_change"]["stress"], PALETTE["crimson"]),
        ("PR segment (AV node)", ecg_eff["pr_seg"]["vid_change"]["stress"], PALETTE["slate"]),
        ("PR interval (onset)", ecg_eff["pr_on"]["vid_change"]["stress"], PALETTE["slate"]),
        ("QRS duration", ecg_eff["qrs"]["vid_change"]["stress"], PALETTE["slate"]),
        ("QTc Bazett (plain beta)", rr_ctrl["qtc_baz"]["plain"], PALETTE["coral"]),
        ("QTc Bazett (RR-adj. beta)", rr_ctrl["qtc_baz"]["rr_controlled"], PALETTE["teal"]),
        ("HEP (ICA preprocessed)", hep_var["ica_preprocessed_100uV"]["contrasts_d"]["stress"], PALETTE["muted"]),
        ("HEP (raw CURATED, no ICA)", hep_var["raw_curated_100uV"]["contrasts_d"]["stress"], PALETTE["muted"]),
    ]

    y_pos = np.arange(len(rows))[::-1]
    ax2.axvline(0.0, color="#333333", lw=0.9, ls="--", zorder=1)
    for y, (_lbl, vals, col) in zip(y_pos, rows, strict=True):
        mean, lo, hi = float(vals[0]), float(vals[1]), float(vals[2])
        ax2.errorbar(
            mean, y, xerr=[[mean - lo], [hi - mean]],
            fmt="o", color=col, ecolor=col, elinewidth=1.6, capsize=3.0, ms=5.5, zorder=3,
        )
        ha = "left" if mean >= 0 else "right"
        ax2.text(
            hi + 0.012 if mean >= 0 else lo - 0.012, y, f"{mean:+.3f}",
            va="center", ha=ha, fontsize=7.5, color="#222222",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlim(-0.22, 0.24)
    ax2.set_xlabel("Standardized within-ID contrast (d or beta; 95% CI)")
    ax2.set_title("B. ECG morphology (96.6% trial rows flipped) & HEP")

    fig.tight_layout(w_pad=2.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig2_causal_brain_heart(results_dir: Path, out_path: Path) -> Path:
    """Figure 2: Strictly causal early-EEG -> future physiology gain and control battery."""
    _style()
    cf = _load(results_dir, "causal_forecasting.json")
    gc = _load(results_dir, "generalization_clinical.json")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )

    # Panel A: Cross-modality comparison + LOCO 5-cohort replication
    pri = cf["audit_forecast"]["module2_primary_forecasting"]["primary_cold_start_3x3"]
    loco_hr = gc["module10c_coral_kshot_loco"]["loco_cross_cohort_dev"]["cardiac_forecast_loco"]

    hr_res = pri["tgt_ecg_hr_chg_5_10"]["paired_vs_m0_linear"]["M2_reve_resid"]["delta_r"]
    hr_jnt = pri["tgt_ecg_hr_chg_5_10"]["paired_vs_m0_linear"]["M2_reve_joint"]["delta_r"]
    hr_gbdt = pri["tgt_ecg_hr_chg_5_10"]["paired_m2_gbdt_vs_m0_gbdt"]["delta_r"]
    hr_loco = loco_hr["paired_M2_minus_M0_raw"]["delta_r"]
    eda_res = pri["tgt_eda_sc_chg_5_10"]["paired_vs_m0_linear"]["M2_reve_resid"]["delta_r"]
    eda_gbdt = pri["tgt_eda_sc_chg_5_10"]["paired_m2_gbdt_vs_m0_gbdt"]["delta_r"]
    rsp_res = pri["tgt_resp_amp_chg_5_10"]["paired_vs_m0_linear"]["M2_reve_resid"]["delta_r"]
    rsp_gbdt = pri["tgt_resp_amp_chg_5_10"]["paired_m2_gbdt_vs_m0_gbdt"]["delta_r"]

    bars = [
        ("HR: M2 joint (3x3)", hr_jnt["mean"], hr_jnt["ci95"], PALETTE["navy"]),
        ("HR: M2 resid (3x3)", hr_res["mean"], hr_res["ci95"], PALETTE["navy"]),
        ("HR: M2 LOCO (5 coh.)", hr_loco["mean"], hr_loco["ci95"], PALETTE["teal"]),
        ("HR: M2 GBDT", hr_gbdt["mean"], hr_gbdt["ci95"], PALETTE["teal"]),
        ("Resp: M2 resid", rsp_res["mean"], rsp_res["ci95"], PALETTE["muted"]),
        ("Resp: M2 GBDT", rsp_gbdt["mean"], rsp_gbdt["ci95"], PALETTE["muted"]),
        ("EDA: M2 resid", eda_res["mean"], eda_res["ci95"], PALETTE["crimson"]),
        ("EDA: M2 GBDT", eda_gbdt["mean"], eda_gbdt["ci95"], PALETTE["crimson"]),
    ]
    y1 = np.arange(len(bars))[::-1]
    ax1.axvline(0.0, color="#333333", lw=0.9, ls="--")
    for y, (_lbl, m, ci, col) in zip(y1, bars, strict=True):
        ax1.barh(y, m, height=0.58, color=col, alpha=0.84, zorder=2)
        ax1.errorbar(
            m, y, xerr=[[m - ci[0]], [ci[1] - m]],
            fmt="none", ecolor="#222222", elinewidth=1.2, capsize=2.8, zorder=3,
        )
        tx = ci[1] + 0.0012 if m >= 0 else ci[0] - 0.0012
        ha = "left" if m >= 0 else "right"
        ax1.text(tx, y, f"{m:+.4f}", va="center", ha=ha, fontsize=7.4)
    ax1.set_yticks(y1)
    ax1.set_yticklabels([b[0] for b in bars])
    ax1.set_xlim(-0.041, 0.026)
    ax1.set_xlabel("Incremental Pearson r (M2 minus M0, Cold-Start)")
    ax1.set_title("A. Early EEG [-3.2, 5) s -> physiology [5, 10) s")

    # Panel B: Cardiac falsification & artefact control battery
    ctrl = cf["controls"]["module3_controls"]["targets"]["tgt_ecg_hr_chg_5_10"]["paired_contrasts"]
    ctrl_rows = [
        ("Own vs. permuted same-video (n=224)", ctrl["NC2_idiosyncratic_real_vs_vid_shuf"]["delta_r"], PALETTE["green"]),
        ("Clean low-artefact trials (AC3, n=207)", ctrl["AC3_clean_subset_M2_vs_M0"]["delta_r"], PALETTE["navy"]),
        ("Primary M2 resid vs. M0 (n=224)", ctrl["M2_real_vs_M0"]["delta_r"], PALETTE["navy"]),
        ("Residualized 9 artefacts (AC2b, n=224)", ctrl["AC2b_art_resid_vs_M0"]["delta_r"], PALETTE["navy"]),
        ("Joint 9-artefact covs (AC1, n=224)", ctrl["AC1_M2_art_vs_M0_art"]["delta_r"], PALETTE["navy"]),
        ("Previous-trial history (TC1, n=224)", ctrl["TC1_M2_tc1_vs_M0_tc1"]["delta_r"], PALETTE["teal"]),
        ("46 interior non-rim EEG (AC2a, n=224)", ctrl["AC2a_interior46_vs_M0"]["delta_r"], PALETTE["teal"]),
        ("Within-video idiosyncratic r (n=198)", ctrl["within_video_idiosyncratic_M2_minus_M0"], PALETTE["teal"]),
        ("Neg.: trial shuffle (NC1, n=224)", ctrl["NC1_within_person_shuf_vs_M0"]["delta_r"], PALETTE["coral"]),
        ("Neg.: pre-stim only (NC3, n=224)", ctrl["NC3_pre_stim_only_vs_M0"]["delta_r"], PALETTE["coral"]),
        ("Neg.: permuted same-video (NC2, n=224)", ctrl["NC2_stimulus_mediated_vs_M0"]["delta_r"], PALETTE["crimson"]),
    ]
    y2 = np.arange(len(ctrl_rows))[::-1]
    ax2.axvline(0.0, color="#333333", lw=0.9, ls="--")
    for y, (_lbl, st, col) in zip(y2, ctrl_rows, strict=True):
        m, ci = float(st["mean"]), st["ci95"]
        ax2.errorbar(
            m, y, xerr=[[m - ci[0]], [ci[1] - m]],
            fmt="o", color=col, ecolor=col, elinewidth=1.5, capsize=2.8, ms=5.0, zorder=3,
        )
        tx = ci[1] + 0.0012 if m >= 0 else ci[0] - 0.0012
        ha = "left" if m >= 0 else "right"
        ax2.text(tx, y, f"{m:+.4f}", va="center", ha=ha, fontsize=7.3)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([r[0] for r in ctrl_rows])
    ax2.set_xlim(-0.028, 0.038)
    ax2.set_xlabel("Paired cardiac delta-r (95% bootstrap CI)")
    ax2.set_title("B. Exploratory cardiac controls & falsification")

    fig.tight_layout(w_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig3_generalization_drift(results_dir: Path, out_path: Path) -> Path:
    """Figure 3: Generalization ladder R1-R4, Day 1 -> Day 2 drift, and k-shot EB-MAP."""
    _style()
    gc = _load(results_dir, "generalization_clinical.json")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.85))

    # Panel A: Generalization ladder R1 -> R4
    m10a = gc["module10a_generalization_ablation"]["generalization_ladder"]
    regimes = [
        ("R1_random_5fold", "R1\nRandom"),
        ("R2_within_person_new_themes", "R2\nNew theme"),
        ("R3_loso_seen_themes", "R3\nNew ID\nseen theme"),
        ("R4_double_unseen_cold_start", "R4\nDouble-\nunseen"),
    ]
    x = np.arange(len(regimes))
    for model_key, lbl, col, mk in (
        ("S07_enriched", "4-phase Ridge ([-3,12) s)", PALETTE["navy"], "o"),
        ("REVE_frozen", "REVE-base EEG ([0,10) s)", PALETTE["coral"], "s"),
    ):
        ys, los, his = [], [], []
        for rkey, _ in regimes:
            st = m10a[model_key][rkey]["stress_ridge_va"]["stress"]
            ys.append(st["auc"])
            los.append(st["auc"] - st["ci95"][0])
            his.append(st["ci95"][1] - st["auc"])
        ax1.errorbar(x, ys, yerr=[los, his], color=col, marker=mk, lw=1.8, capsize=3, label=lbl)
    ax1.axhline(0.5, color="#777777", lw=0.8, ls=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r[1] for r in regimes], fontsize=7.8)
    ax1.set_ylim(0.515, 0.612)
    ax1.set_ylabel("Within-person stress ROC-AUC")
    ax1.set_title("A. Generalization ladder (R1-R4)")
    ax1.legend(loc="lower left")

    # Panel B: Cross-day transfer (Day 1 -> Day 2)
    xday = gc["module10b_longitudinal_habituation"]["cross_day_decoder_transfer_dev"]
    mods = [
        ("S07_enriched", "Multimodal"),
        ("EEG_hc_only", "EEG"),
        ("REVE_frozen", "REVE"),
        ("EDA_only", "EDA"),
        ("Resp_only", "Resp"),
    ]
    xm = np.arange(len(mods))
    w = 0.34
    d1 = [xday[k]["within_day1_3x3"]["auc"] for k, _ in mods]
    d2 = [xday[k]["j1_to_j2_new_people_unseen_themes"]["auc"] for k, _ in mods]
    e1 = [
        [d1[i] - xday[k]["within_day1_3x3"]["ci95"][0] for i, (k, _) in enumerate(mods)],
        [xday[k]["within_day1_3x3"]["ci95"][1] - d1[i] for i, (k, _) in enumerate(mods)],
    ]
    e2 = [
        [d2[i] - xday[k]["j1_to_j2_new_people_unseen_themes"]["ci95"][0] for i, (k, _) in enumerate(mods)],
        [xday[k]["j1_to_j2_new_people_unseen_themes"]["ci95"][1] - d2[i] for i, (k, _) in enumerate(mods)],
    ]
    ax2.bar(xm - w / 2, d1, w, yerr=e1, capsize=2.5, color=PALETTE["navy"], alpha=0.88, label="Within Day 1")
    ax2.bar(xm + w / 2, d2, w, yerr=e2, capsize=2.5, color=PALETTE["coral"], alpha=0.88, label="Day 1 -> Day 2")
    ax2.axhline(0.5, color="#777777", lw=0.8, ls=":")
    ax2.set_xticks(xm)
    ax2.set_xticklabels([m[1] for m in mods], fontsize=8.0)
    ax2.set_ylim(0.47, 0.61)
    ax2.set_ylabel("Within-person stress ROC-AUC")
    ax2.set_title("B. Longitudinal drift (1-week gap)")
    ax2.legend(loc="upper right")

    # Panel C: Chronological k-shot personalization
    kcurve = gc["module10c_coral_kshot_loco"]["kshot_chronological_dev"]["S07_enriched"]["k_curve"]
    ks = [0, 5, 10, 15, 20, 30, 40]
    for method, lbl, col, ls, mk in (
        ("cold_start", "Cold-Start (k=0)", PALETTE["slate"], "--", "none"),
        ("sup_eb_map_shrunk", "EB-MAP shrunk", PALETTE["green"], "-", "o"),
        ("unsup_coral", "Unsup. CORAL", PALETTE["teal"], "-.", "^"),
        ("sup_naive_personal", "Naive personal", PALETTE["crimson"], "-", "s"),
    ):
        ys = [kcurve[str(k)][method]["auc"] for k in ks]
        ax3.plot(ks, ys, color=col, ls=ls, marker=None if mk == "none" else mk, ms=4.5, lw=1.8, label=lbl)
    ax3.axhline(0.5, color="#777777", lw=0.8, ls=":")
    ax3.set_xticks(ks)
    ax3.set_ylim(0.46, 0.60)
    ax3.set_xlabel("Calibration trials per person (k)")
    ax3.set_ylabel("Future-trial ROC-AUC (t >= 40)")
    ax3.set_title("C. Chronological k-shot adaptation")
    ax3.legend(loc="lower right")

    fig.tight_layout(w_pad=1.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig4_coherence_monitoring(results_dir: Path, out_path: Path) -> Path:
    """Figure 4: Varimax 3-factor divergence space and 10 s -> 100 s clinical monitoring."""
    _style()
    gc = _load(results_dir, "generalization_clinical.json")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.15), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # Panel A: Varimax 3-factor loading heatmap
    p10d = gc["module10d_coherence_bridge"]["coherence_all_314"]["pca_within_person_total"]
    vload = p10d["varimax_3f_loadings"]
    var_order = [
        ("d_sc", "Skin cond. (d_sc)"),
        ("d_scr", "Phasic SCR (d_scr)"),
        ("d_delta_cp", "EEG delta (C-P)"),
        ("d_gamma_fr", "EEG gamma (Fr)"),
        ("d_alpha_occ", "EEG alpha (Occ)"),
        ("d_hr", "Heart rate (d_hr)"),
        ("d_resp", "Respiration (d_resp)"),
        ("valence", "Valence rating"),
        ("arousal", "Arousal rating"),
    ]
    mat = np.array([[vload[f][vk] for f in ("F1", "F2", "F3")] for vk, _ in var_order])
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_col = "white" if abs(val) > 0.52 else "#111111"
            ax1.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8.0, color=txt_col, fontweight="bold" if abs(val) >= 0.30 else "normal")
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["F1\nEDA\n(symp.)", "F2\nEEG-Cardio\n(auto.)", "F3\nSubjective-\nCardioResp"], fontsize=8.0)
    ax1.set_yticks(range(len(var_order)))
    ax1.set_yticklabels([lbl for _, lbl in var_order], fontsize=8.2)
    ax1.set_title("A. Within-person Varimax PCA (N = 314)")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("PCA loading", fontsize=8.0)

    # Panel B: Passive monitoring curve 10 s -> 100 s (AUC + Sensitivity @ 90% Spec)
    m10e = gc["module10e_clinical_monitoring"]
    ks = [1, 2, 3, 5, 8, 10]
    secs = [10 * k for k in ks]
    contrasts_cfg = [
        ("stress_strict37_vs_relaxed64", "Strict stress (3/7) vs. relaxed", PALETTE["navy"], "o"),
        ("acute_threat_vs_calm_neutral", "Acute threat vs. calm/neutral", PALETTE["teal"], "s"),
        ("stress_standard_vs_nonstress", "Standard stress vs. all other", PALETTE["coral"], "^"),
    ]
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    for ckey, lbl, col, mk in contrasts_cfg:
        by_k = m10e[ckey]["S07_enriched"]["by_k"]
        aucs = [by_k[str(k)]["auc"]["mean"] for k in ks]
        lo = [aucs[i] - by_k[str(k)]["auc"]["ci95"][0] for i, k in enumerate(ks)]
        hi = [by_k[str(k)]["auc"]["ci95"][1] - aucs[i] for i, k in enumerate(ks)]
        s90 = [100.0 * by_k[str(k)]["sens_at_spec90"]["mean"] for k in ks]
        ax2.errorbar(secs, aucs, yerr=[lo, hi], color=col, marker=mk, lw=2.0, ms=5.0, capsize=2.8, label=f"{lbl} (AUC)")
        ax2_r.plot(secs, s90, color=col, ls="--", lw=1.3, alpha=0.8)

    ax2.set_xticks(secs)
    ax2.set_ylim(0.53, 0.77)
    ax2_r.set_ylim(12.0, 68.0)
    ax2.set_xlabel("Cumulative episode duration (s; K = 1..10 clips)")
    ax2.set_ylabel("Within-person ROC-AUC (solid lines)")
    ax2_r.set_ylabel("Per-person sens. at 90% spec. (%, dashed)", color="#444444")
    ax2.set_title("B. Known-class episode averaging (10 s -> 100 s)")
    ax2.legend(loc="upper left", fontsize=7.6)

    fig.tight_layout(w_pad=2.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Generate all four publication figures deterministically."""
    return [
        plot_fig1_physiology_cascade(results_dir, figures_dir / "fig1_physiology_cascade.png"),
        plot_fig2_causal_brain_heart(results_dir, figures_dir / "fig2_causal_brain_heart.png"),
        plot_fig3_generalization_drift(results_dir, figures_dir / "fig3_generalization_drift.png"),
        plot_fig4_coherence_monitoring(results_dir, figures_dir / "fig4_coherence_monitoring.png"),
    ]
