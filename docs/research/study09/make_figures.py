"""Generate Study-09 French publication figures from verified JSON aggregates.

Run from the repository root:
    .venv-agent/bin/python docs/research/study09/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE = "#246bce"
ORANGE = "#e5632f"
TEAL = "#159a6e"
RED = "#cf3a3a"
PURPLE = "#7350b8"
AMBER = "#d98f0d"
GRAY = "#8b8a84"
DARK = "#1a1a1a"
MUTED = "#555450"
LIGHT = "#ecebe6"

plt.rcParams.update(
    {
        "font.size": 8.4,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": DARK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def fnum(val: float, decimals: int = 3, signed: bool = False) -> str:
    """Format decimal numbers in French style (decimal comma and true minus)."""
    fmt = f"{val:+.{decimals}f}" if signed else f"{val:.{decimals}f}"
    return fmt.replace(".", ",").replace("-", "\u2212")


def fmt_axis(ax: plt.Axes, axis: str = "x", decimals: int = 2) -> None:
    """Apply French decimal comma tick formatting."""
    fn = FuncFormatter(lambda v, _: fnum(v, decimals))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fn)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fn)


def load_json(name: str) -> dict | None:
    """Load a JSON aggregate from ``results/`` if present."""
    p = RES / name
    return json.loads(p.read_text()) if p.is_file() else None


def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure at 300 dpi and close."""
    out = FIG / f"{stem}_fr.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIG 1: Audit des Études 07-08 (normalisation transductive & fuite REVE)
# ---------------------------------------------------------------------------


def make_fig1(af: dict) -> None:
    """Fig 1: Audit of Studies 07-08 (transductive centering + bidirectional leak)."""
    m1 = af["module1_audit"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.1, 1.05]}
    )

    # Panel A: Normalization protocol impact on S07 enriched (dev V&A, dev LogReg, exposed NB)
    norm_audit = m1["normalization_audit_study07_enriched"]
    proto_order = [
        ("cold_start", "Protocole A\nCold-Start\n(zéro hist.)"),
        ("causal_online", "Protocole B1\nCausal ligne\n(passé seul)"),
        ("fold_isolated_session", "Protocole B2\nSession isol.\n(hors-ligne)"),
        ("legacy_global", "Ancien global\n(transductif\nÉ07–08)"),
    ]
    x = np.arange(len(proto_order))
    w = 0.25
    series = [
        ("dev_ridge_va_auc", "S07 Ridge V&A (dev)", BLUE),
        ("dev_logreg_auc", "S07 LogReg (dev)", TEAL),
        ("exposed_test_new_both_va", "S07 V&A (NB exposé*)", ORANGE),
    ]
    for idx_s, (s_key, s_lbl, col) in enumerate(series):
        vals, lo_e, hi_e = [], [], []
        for p_key, _ in proto_order:
            st = norm_audit[p_key][s_key]
            auc = st["auc"]
            ci = st["ci95"]
            vals.append(auc)
            lo_e.append(max(0.0, auc - ci[0]))
            hi_e.append(max(0.0, ci[1] - auc))
        offs = x + (idx_s - 1) * w
        ax1.bar(offs, vals, width=w * 0.90, color=col, alpha=0.88, label=s_lbl)
        ax1.errorbar(
            offs,
            vals,
            yerr=[lo_e, hi_e],
            fmt="none",
            ecolor=DARK,
            elinewidth=1.0,
            capsize=2.4,
        )
        for xi, vi, hi in zip(offs, vals, hi_e):
            ax1.text(
                xi,
                vi + hi + 0.003,
                fnum(vi, 3),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=DARK,
            )

    ax1.axhline(0.5, color=GRAY, ls="--", lw=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in proto_order], fontsize=7.6)
    ax1.set_ylim(0.49, 0.705)
    ax1.set_yticks([0.50, 0.54, 0.58, 0.62, 0.66, 0.70])
    fmt_axis(ax1, "y", 2)
    ax1.set_ylabel("AUC ROC intra-sujet (stress, IC 95 %)")
    ax1.set_xlabel(
        "*NB exposé (sel É08, N = 30) : audit rétrospectif uniquement",
        fontsize=7.1,
        color=MUTED,
    )
    ax1.set_title(
        "A. Impact du protocole de normalisation (S07 enrichi)",
        loc="left",
        fontweight="bold",
    )
    ax1.legend(frameon=False, fontsize=7.1, loc="upper left", ncol=2)

    # Panel B: Bidirectional future-EEG leak in Study 08 REVE vs Strict Causal REVE
    leak = m1["bidirectional_and_stimulus_shortcut_audit"]
    tgts = [
        ("hr_5_10", "ECG (FC)"),
        ("sc_5_10", "EDA (SC)"),
        ("resp_5_10", "Respiration"),
    ]
    xb = np.arange(len(tgts))
    wb = 0.20
    bars_b = [
        (
            "s08_reve_bidirectional_13s",
            "exposed_mixed_s08_style_r",
            "É08 REVE bidir. (mixte thèmes vus*)",
            RED,
        ),
        (
            "s08_reve_bidirectional_13s",
            "dev_nested_unseen_both_r",
            "É08 REVE bidir. (dev bloqué)",
            ORANGE,
        ),
        (
            "s09_reve_causal_ctx8_strict",
            "dev_nested_unseen_both_r",
            "É09 REVE causal [-3,2 ; 5) s",
            BLUE,
        ),
        (
            "s09_reve_causal_early5_strict",
            "dev_nested_unseen_both_r",
            "É09 REVE causal [0 ; 5) s seul",
            TEAL,
        ),
    ]
    for idx_b, (fam_k, reg_k, b_lbl, col) in enumerate(bars_b):
        vals, lo_e, hi_e = [], [], []
        for t_key, _ in tgts:
            r_d = leak[fam_k][t_key][reg_k]
            val = r_d["mean"]
            ci = r_d["ci95"]
            vals.append(val)
            lo_e.append(max(0.0, val - ci[0]))
            hi_e.append(max(0.0, ci[1] - val))
        offs = xb + (idx_b - 1.5) * wb
        ax2.bar(offs, vals, width=wb * 0.90, color=col, alpha=0.88, label=b_lbl)
        ax2.errorbar(
            offs,
            vals,
            yerr=[lo_e, hi_e],
            fmt="none",
            ecolor=DARK,
            elinewidth=0.95,
            capsize=2.2,
        )
        for xi, vi, hi in zip(offs, vals, hi_e):
            ax2.text(
                xi,
                vi + hi + 0.005,
                fnum(vi, 3),
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=DARK,
            )
    ax2.axhline(0.0, color=GRAY, lw=0.8)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([lbl for _, lbl in tgts], fontsize=8.4)
    ax2.set_ylim(0.0, 0.365)
    fmt_axis(ax2, "y", 2)
    ax2.set_ylabel("Corrélation r intra-sujet M1 (EEG → Corps[5:10])")
    ax2.set_title(
        "B. Fuite bidirectionnelle REVE & raccourci thèmes vus",
        loc="left",
        fontweight="bold",
    )
    ax2.legend(frameon=False, fontsize=6.9, loc="upper right")

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig1_audit")


# ---------------------------------------------------------------------------
# FIG 2: Question principale M0/M1/M2 & délais physiologiques préspécifiés
# ---------------------------------------------------------------------------


def make_fig2(af: dict, lora_f: dict | None) -> None:
    """Fig 2: Primary M0/M1/M2 comparison and prespecified physiological delays."""
    m2 = af["module2_primary_forecasting"]
    pri = m2["primary_cold_start_3x3"]
    ext = m2["extended_delays_cold_start_3x3"]
    fdr = m2.get("extended_delays_bh_fdr_q", {})

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.7), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # Panel A: M0 / M1 / M2 across ECG, EDA, Resp (Cold-Start)
    tgts = [
        ("tgt_ecg_hr_chg_5_10", "ECG (FC)"),
        ("tgt_eda_sc_chg_5_10", "EDA (SC)"),
        ("tgt_resp_amp_chg_5_10", "Respiration"),
    ]
    models = [
        ("M1_hc_eeg", "M1 EEG manuel", GRAY),
        ("M1_reve_frozen", "M1 REVE causal", PURPLE),
        ("M0_linear", "M0 Corps linéaire", BLUE),
        ("M2_reve_resid", "M2 REVE résiduel", TEAL),
        ("M0_gbdt", "M0 Corps GBDT", ORANGE),
        ("M2_gbdt_reve", "M2 GBDT + REVE", RED),
    ]
    x = np.arange(len(tgts))
    w = 0.13
    for idx_m, (mk, mlbl, col) in enumerate(models):
        vals, lo_e, hi_e = [], [], []
        for tk, _ in tgts:
            rm = pri[tk]["models"][mk]["r"]
            v = rm["mean"]
            vals.append(v)
            lo_e.append(max(0.0, v - rm["ci95"][0]))
            hi_e.append(max(0.0, rm["ci95"][1] - v))
        offs = x + (idx_m - (len(models) - 1) / 2.0) * w
        ax1.bar(offs, vals, width=w * 0.92, color=col, alpha=0.88, label=mlbl)
        ax1.errorbar(
            offs,
            vals,
            yerr=[lo_e, hi_e],
            fmt="none",
            ecolor=DARK,
            elinewidth=0.9,
            capsize=2.0,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in tgts], fontsize=8.6)
    ax1.set_ylim(0.0, 1.04)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    fmt_axis(ax1, "y", 2)
    ax1.set_ylabel("Corrélation r intra-sujet (dev 3×3 bloqué, IC 95 %)")
    ax1.set_title(
        "A. Prévision M0 / M1 / M2 (Protocole A Cold-Start)",
        loc="left",
        fontweight="bold",
    )
    ax1.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper center")

    # Panel B: Paired M2_reve_resid - M0_linear across prespecified delays & metrics
    delay_specs = [
        ("tgt_ecg_hr_chg_5_8", "ECG FC [5–8 s]", BLUE),
        ("tgt_ecg_hr_chg_5_10", "ECG FC [5–10 s] (pri.)", BLUE),
        ("tgt_ecg_hr_chg_7_10", "ECG FC [7–10 s] (garde 2 s)", BLUE),
        ("tgt_ecg_hr_chg_10_12", "ECG FC [10–12 s] (garde 5 s)", BLUE),
        ("tgt_ecg_rr_chg_5_10", "ECG RR [5–10 s]", BLUE),
        ("tgt_ecg_hr_inc_5_10", "ECG ΔFC pur [5–10 − 0–5]", BLUE),
        ("tgt_eda_sc_chg_5_10", "EDA SC [5–10 s] (pri.)", ORANGE),
        ("tgt_eda_sc_chg_6_9", "EDA SC [6–9 s]", ORANGE),
        ("tgt_eda_sc_chg_7_10", "EDA SC [7–10 s]", ORANGE),
        ("tgt_eda_sc_chg_10_12", "EDA SC [10–12 s]", ORANGE),
        ("tgt_eda_scr_new_5_10", "EDA SCR phasique [5–10 s]", ORANGE),
        ("tgt_resp_amp_chg_5_8", "Resp amp [5–8 s]", TEAL),
        ("tgt_resp_amp_chg_5_10", "Resp amp [5–10 s] (pri.)", TEAL),
        ("tgt_resp_amp_chg_7_10", "Resp amp [7–10 s]", TEAL),
        ("tgt_resp_amp_chg_10_12", "Resp amp [10–12 s]", TEAL),
        ("tgt_resp_rate_chg_5_10", "Resp fréq [5–10 s]", TEAL),
    ]
    rows_b = [s for s in delay_specs if s[0] in ext]
    y_pos = np.arange(len(rows_b))[::-1]
    ax2.axvline(0.0, color=DARK, lw=0.9, ls="--")
    ax2.axvline(0.015, color=GRAY, lw=0.8, ls=":")
    ax2.text(
        0.0156,
        0.4,
        "Seuil pratique\nH1 (+0,015)",
        fontsize=6.5,
        color=MUTED,
        va="bottom",
    )

    for yi, (tk, tlbl, col) in zip(y_pos, rows_b):
        dr = ext[tk]["paired_vs_m0_linear"]["M2_reve_resid"]["delta_r"]
        m_val = dr["mean"]
        lo, hi = dr["ci95"]
        q_val = fdr.get(tk, 1.0)
        star = "*" if (q_val is not None and q_val < 0.05 and lo > 0) else ""
        ax2.errorbar(
            m_val,
            yi,
            xerr=[[max(0.0, m_val - lo)], [max(0.0, hi - m_val)]],
            fmt="o",
            color=col,
            ecolor=col,
            markersize=4.7,
            elinewidth=1.4,
            capsize=2.4,
        )
        ax2.text(
            max(hi, 0.0) + 0.0012,
            yi,
            f"{fnum(m_val, 4, signed=True)}{star}",
            va="center",
            ha="left",
            fontsize=6.7,
            color=DARK if star else MUTED,
            fontweight="bold" if star else "normal",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl, _ in rows_b], fontsize=7.4)
    ax2.set_xlim(-0.026, 0.039)
    fmt_axis(ax2, "x", 3)
    ax2.set_xlabel("Gain apparié Δr (M2 REVE résiduel − M0 Corps, IC 95 %)")
    ax2.set_title(
        "B. Gains résiduels par délai physiologique (* q_FDR < 0,05)",
        loc="left",
        fontweight="bold",
    )

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig2_forecast_delays")


# ---------------------------------------------------------------------------
# FIG 3: Contrôles artefacts, ordre temporel, stimulus vs idiosyncrasie
# ---------------------------------------------------------------------------


def make_fig3(ctrl: dict, sec: dict) -> None:
    """Fig 3: Complete confound, temporal, stimulus-vs-idiosyncratic & negative controls."""
    t_ecg = ctrl["module3_controls"]["targets"]["tgt_ecg_hr_chg_5_10"][
        "paired_contrasts"
    ]
    t_eda = ctrl["module3_controls"]["targets"]["tgt_eda_sc_chg_5_10"][
        "paired_contrasts"
    ]
    t_res = ctrl["module3_controls"]["targets"]["tgt_resp_amp_chg_5_10"][
        "paired_contrasts"
    ]
    oc_ecg = sec["modules4_to_7"]["module6_oculo_audit_and_baselines"]["baselines"][
        "forecast_control_with_oculo"
    ]["tgt_ecg_hr_chg_5_10"]["paired_M2_vs_M0_ctrl_oculo"]["delta_r"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 5.1), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # Panel A: Full ECG HR paired contrast forest plot
    rows = [
        ("M2 réel vs M0 (référence)", t_ecg["M2_real_vs_M0"]["delta_r"], BLUE),
        (
            "AC1 : M2+art. vs M0+art. (9 proxies)",
            t_ecg["AC1_M2_art_vs_M0_art"]["delta_r"],
            TEAL,
        ),
        (
            "AC2a : 46 électrodes intérieures",
            t_ecg["AC2a_interior46_vs_M0"]["delta_r"],
            TEAL,
        ),
        (
            "AC2b : REVE résidualisé sur art.",
            t_ecg["AC2b_art_resid_vs_M0"]["delta_r"],
            TEAL,
        ),
        (
            "AC3 : essais propres (66,8 %)",
            t_ecg["AC3_clean_subset_M2_vs_M0"]["delta_r"],
            TEAL,
        ),
        ("Contrôle oculo (M2+oc. vs M0+oc.)", oc_ecg, TEAL),
        (
            "TC1 : contrôle essai t−1 & ordre",
            t_ecg["TC1_M2_tc1_vs_M0_tc1"]["delta_r"],
            PURPLE,
        ),
        (
            "NC4 : chronologique vs permut. 1 s",
            t_ecg["NC4_early5_real_vs_time_shuf5"]["delta_r"],
            PURPLE,
        ),
        (
            "Intra-vidéo (M2 − M0, 198 vidéos)",
            t_ecg["within_video_idiosyncratic_M2_minus_M0"],
            AMBER,
        ),
        (
            "NC2 idio. : réel − autre spect. vidéo",
            t_ecg["NC2_idiosyncratic_real_vs_vid_shuf"]["delta_r"],
            AMBER,
        ),
        (
            "NC3 : EEG pré-stimulus [−3 ; 0) s seul",
            t_ecg["NC3_pre_stim_only_vs_M0"]["delta_r"],
            GRAY,
        ),
        (
            "NC1 : permutation intra-sujet (nul)",
            t_ecg["NC1_within_person_shuf_vs_M0"]["delta_r"],
            GRAY,
        ),
        (
            "NC2 stim. : autre spect. même vidéo",
            t_ecg["NC2_stimulus_mediated_vs_M0"]["delta_r"],
            RED,
        ),
    ]
    y_pos = np.arange(len(rows))[::-1]
    ax1.axvline(0.0, color=DARK, lw=0.95, ls="--")
    for yi, (lbl, d, col) in zip(y_pos, rows):
        m_val = d["mean"]
        lo, hi = d["ci95"]
        imp = d.get("frac_improved")
        ax1.errorbar(
            m_val,
            yi,
            xerr=[[max(0.0, m_val - lo)], [max(0.0, hi - m_val)]],
            fmt="o",
            color=col,
            ecolor=col,
            markersize=5.2,
            elinewidth=1.5,
            capsize=2.6,
        )
        suffix = f" ({100 * imp:.0f} %)" if imp is not None else ""
        x_txt = hi + 0.0012 if m_val >= -0.005 else hi + 0.0012
        ax1.text(
            x_txt,
            yi,
            f"{fnum(m_val, 4, signed=True)}{suffix}",
            va="center",
            ha="left",
            fontsize=7.0,
            color=DARK if (lo > 0 or hi < 0) else MUTED,
            fontweight="bold" if lo > 0 else "normal",
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=7.8)
    ax1.set_xlim(-0.023, 0.044)
    fmt_axis(ax1, "x", 3)
    ax1.set_xlabel(
        "Effet apparié Δr sur ECG FC [5 ; 10) s (IC 95 %, % sujets améliorés)"
    )
    ax1.set_title(
        "A. Batterie de contrôles sur l'ECG (N = 224 sujets inédits)",
        loc="left",
        fontweight="bold",
    )

    # Panel B: 3-organ comparison on key contrasts
    key_c = [
        ("M2_real_vs_M0", "M2 réel\n− M0"),
        ("AC1_M2_art_vs_M0_art", "AC1\n(+9 art.)"),
        ("NC1_within_person_shuf_vs_M0", "NC1\nperm."),
        ("NC2_stimulus_mediated_vs_M0", "NC2\nstim."),
        ("NC2_idiosyncratic_real_vs_vid_shuf", "NC2\nidio."),
        ("within_video_idiosyncratic_M2_minus_M0", "Intra-\nvidéo"),
    ]
    xb = np.arange(len(key_c))
    wb = 0.25
    organs = [
        ("ECG (FC, N = 224)", t_ecg, BLUE),
        ("EDA (SC, N = 200)", t_eda, ORANGE),
        ("Respiration (N = 228)", t_res, TEAL),
    ]
    for idx_o, (olbl, odict, col) in enumerate(organs):
        vals, lo_e, hi_e = [], [], []
        for ck, _ in key_c:
            entry = odict[ck]
            d = entry["delta_r"] if "delta_r" in entry else entry
            v = d["mean"]
            vals.append(v)
            lo_e.append(max(0.0, v - d["ci95"][0]))
            hi_e.append(max(0.0, d["ci95"][1] - v))
        offs = xb + (idx_o - 1) * wb
        ax2.bar(offs, vals, width=wb * 0.9, color=col, alpha=0.88, label=olbl)
        ax2.errorbar(
            offs,
            vals,
            yerr=[lo_e, hi_e],
            fmt="none",
            ecolor=DARK,
            elinewidth=0.95,
            capsize=2.2,
        )
    ax2.axhline(0.0, color=DARK, lw=0.9, ls="--")
    ax2.set_xticks(xb)
    ax2.set_xticklabels([lbl for _, lbl in key_c], fontsize=7.3)
    ax2.set_ylim(-0.024, 0.034)
    fmt_axis(ax2, "y", 3)
    ax2.set_ylabel("Effet apparié Δr (IC 95 %)")
    ax2.set_title(
        "B. Dissociation organe : ECG vs EDA et Respiration",
        loc="left",
        fontweight="bold",
    )
    ax2.legend(frameon=False, fontsize=7.4, loc="upper left")

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig3_controls")


# ---------------------------------------------------------------------------
# FIG 4: Objectif secondaire LoRA multitâche + ranking & Factorisation Trait/État
# ---------------------------------------------------------------------------


def make_fig4(lora_mt: dict, sec: dict) -> None:
    """Fig 4: Frozen REVE vs LoRA multitask+ranking and Trait/State factorization."""
    m4 = sec["modules4_to_7"]["module4_secondary_affect_and_lora"]
    m5 = sec["modules4_to_7"]["module5_trait_state_factorization"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    # Panel A: Stress AUC (V&A & direct) + affect rho across Cold-Start models
    specs = [
        ("s07_enriched_cold_start", "S07 enrichi\n(0 GPU, multim.)", BLUE),
        ("reve_frozen_linear_cold_start", "REVE gelé\nlinéaire (0 GPU)", TEAL),
        ("reve_mt_frozen_rank", "REVE gelé\nMT + ranking (77 s)", TEAL),
        ("reve_mt_lora_r4", "REVE LoRA r=4\nMT + ranking (569 s)", PURPLE),
        ("reve_mt_lora_rank", "REVE LoRA r=8\nMT + ranking (1665 s)", ORANGE),
        ("reve_mt_lora_norank", "REVE LoRA r=8\nSANS ranking (555 s)", RED),
    ]
    y_pos = np.arange(len(specs))[::-1]
    ax1.axvline(0.50, color=GRAY, ls="--", lw=0.9)
    for yi, (k, lbl, col) in zip(y_pos, specs):
        v = m4[k]
        s_va = v.get("stress_ridge_va", v.get("stress_va", {}))["stress"]
        s_dr = v.get("stress_logreg", v.get("stress_direct", {}))["stress"]
        rho = v["affect"]["mean_rho"]
        # V&A stress AUC (filled circle)
        ax1.errorbar(
            s_va["auc"],
            yi + 0.14,
            xerr=[[s_va["auc"] - s_va["ci95"][0]], [s_va["ci95"][1] - s_va["auc"]]],
            fmt="o",
            color=col,
            ecolor=col,
            markersize=5.6,
            elinewidth=1.5,
            capsize=2.5,
        )
        # Direct stress AUC (open square)
        ax1.errorbar(
            s_dr["auc"],
            yi - 0.14,
            xerr=[[s_dr["auc"] - s_dr["ci95"][0]], [s_dr["ci95"][1] - s_dr["auc"]]],
            fmt="s",
            mfc="white",
            mec=col,
            ecolor=col,
            markersize=4.8,
            elinewidth=1.1,
            capsize=2.2,
            alpha=0.85,
        )
        x_t = max(s_va["ci95"][1], s_dr["ci95"][1]) + 0.003
        ax1.text(
            x_t,
            yi,
            f"V&A {fnum(s_va['auc'], 3)} | ρ̄ = {fnum(rho, 3, signed=True)}",
            va="center",
            ha="left",
            fontsize=7.0,
            color=DARK,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s[1] for s in specs], fontsize=7.6)
    ax1.set_xlim(0.485, 0.655)
    ax1.set_ylim(-0.65, len(specs) + 0.35)
    fmt_axis(ax1, "x", 2)
    ax1.set_xlabel("AUC ROC intra-sujet stress (dev 3×3 Cold-Start, IC 95 %)")
    ax1.set_title(
        "A. REVE gelé vs LoRA multitâche + ranking intra-sujet",
        loc="left",
        fontweight="bold",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=DARK,
            lw=1.2,
            markersize=5.2,
            label="Score V&A (arousal − valence)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=DARK,
            mfc="white",
            lw=1.1,
            markersize=4.8,
            label="Score stress direct",
        ),
    ]
    ax1.legend(handles=handles, frameon=False, fontsize=7.0, loc="upper left", ncol=2)

    # Panel B: Trait / State factorization trade-off (within-person vs pooled AUC)
    fam_pre = m5["cold_start_reve_pre_base"]
    fam_rst = m5["unlabeled_rest_base"]
    modes = [
        ("raw_unfactored", "Brut (non factorisé)", DARK, "o"),
        ("blind_ref_sub", "Soustraction aveugle (réf.)", RED, "^"),
        ("state_only_shrinkage", "État seul (shrinkage EB)", TEAL, "D"),
        ("factored_trait_plus_state", "Factorisé Trait + État", BLUE, "s"),
        ("blind_person_center_causal_b1", "Centrage causal B1", PURPLE, "P"),
        ("trait_only", "Trait seul (16 dim. ANOVA)", GRAY, "X"),
    ]
    for mk, mlbl, col, mrk in modes:
        v_p = fam_pre[mk]
        ax2.scatter(
            v_p["calibration_and_trait"]["pooled_auc"],
            v_p["stress_ridge_va"]["auc"],
            color=col,
            marker=mrk,
            s=58,
            zorder=4,
            label=mlbl,
        )
        if mk not in ("raw_unfactored",):
            v_r = fam_rst[mk]
            ax2.scatter(
                v_r["calibration_and_trait"]["pooled_auc"],
                v_r["stress_ridge_va"]["auc"],
                facecolors="white",
                edgecolors=col,
                marker=mrk,
                s=58,
                linewidths=1.4,
                zorder=4,
            )
            ax2.plot(
                [
                    v_p["calibration_and_trait"]["pooled_auc"],
                    v_r["calibration_and_trait"]["pooled_auc"],
                ],
                [v_p["stress_ridge_va"]["auc"], v_r["stress_ridge_va"]["auc"]],
                color=col,
                lw=0.8,
                ls=":",
                alpha=0.7,
            )
    ax2.axhline(0.50, color=GRAY, ls="--", lw=0.8)
    ax2.axvline(0.50, color=GRAY, ls="--", lw=0.8)
    ax2.set_xlim(0.483, 0.558)
    ax2.set_ylim(0.478, 0.570)
    fmt_axis(ax2, "both", 2)
    ax2.set_xlabel("AUC globale poolée (calibration inter-sujets)")
    ax2.set_ylabel("AUC intra-sujet (stress V&A)")
    ax2.set_title(
        "B. Factorisation Trait / État (plein = pré-stim, vide = repos)",
        loc="left",
        fontweight="bold",
    )
    ax2.legend(frameon=False, fontsize=6.9, loc="lower right")

    # Inset text box with identity geometry R^2 (upper-left empty space)
    geom = m5["geometry_dev"]
    txt = (
        f"Géométrie identité sujet (R²) :\n"
        f"• Brut : {fnum(100 * geom['raw_vid10_identity']['subject_variance_r2'], 1)} % "
        f"(J1→J2 : {fnum(100 * geom['raw_vid10_identity']['cross_day_id']['top1'], 1)} %)\n"
        f"• État EB : {fnum(100 * geom['shrinkage_state_identity']['subject_variance_r2'], 1)} % "
        f"(J1→J2 : {fnum(100 * geom['shrinkage_state_identity']['cross_day_id']['top1'], 1)} %)\n"
        f"• Trait ANOVA : {fnum(100 * geom['anova_trait_identity']['subject_variance_r2'], 1)} % "
        f"(J1→J2 : {fnum(100 * geom['anova_trait_identity']['cross_day_id']['top1'], 1)} %)"
    )
    ax2.text(
        0.02,
        0.97,
        txt,
        transform=ax2.transAxes,
        fontsize=6.7,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": LIGHT,
            "edgecolor": GRAY,
            "alpha": 0.92,
        },
    )

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig4_lora_trait_state")


# ---------------------------------------------------------------------------
# FIG 5: Audit Oculométrie (508 enreg.) & Matrice Complémentarité / Stacking H5
# ---------------------------------------------------------------------------


def make_fig5(ext_audit: dict, sec: dict) -> None:
    """Fig 5: Oculometry audit (coverage, evoked pupil vs gaze-speed) & H5 stacking."""
    oa = ext_audit["oculometry_audit"]
    m7 = sec["modules4_to_7"]["module7_complementarity_and_stacking"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(13.2, 4.2), gridspec_kw={"width_ratios": [0.95, 1.25, 1.05]}
    )

    # Panel A: Coverage & binocular r by cohort
    cohs = ["1", "2", "3", "4", "5"]
    xc = np.arange(len(cohs))
    cov_v = [oa["by_cohort"][c]["valid_early_frac"] for c in cohs]
    bin_r = [oa["by_cohort"][c]["median_binoc_r"] for c in cohs]
    w = 0.36
    ax1.bar(
        xc - w / 2,
        cov_v,
        width=w,
        color=BLUE,
        alpha=0.88,
        label="Essais valides [0 ; 5) s",
    )
    ax1.bar(
        xc + w / 2,
        bin_r,
        width=w,
        color=TEAL,
        alpha=0.88,
        label="Corr. binoculaire r (méd.)",
    )
    for xi, cv, br in zip(xc, cov_v, bin_r):
        ax1.text(
            xi - w / 2,
            cv + 0.012,
            f"{100 * cv:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        ax1.text(
            xi + w / 2, br + 0.012, fnum(br, 2), ha="center", va="bottom", fontsize=6.5
        )
    ax1.axhline(0.50, color=GRAY, ls="--", lw=0.8)
    ax1.set_xticks(xc)
    ax1.set_xticklabels([f"C{c}" for c in cohs])
    ax1.set_ylim(0.0, 1.24)
    fmt_axis(ax1, "y", 2)
    ax1.set_ylabel("Proportion / Corrélation r")
    ax1.set_title(
        "A. Oculométrie par cohorte (500/508 enreg.)", loc="left", fontweight="bold"
    )
    ax1.legend(frameon=False, fontsize=6.8, loc="upper center")

    # Panel B: Evoked 0.25-s curves (Pupil mm vs Gaze Speed px/s)
    p_curve = np.array(oa["evoked_pupil_curve_025s_bins"], dtype=float)
    g_curve = np.array(oa["evoked_gaze_speed_curve_025s_bins"], dtype=float)
    t_bins = -3.0 + (np.arange(len(p_curve)) + 0.5) * 0.25

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["right"].set_color(ORANGE)
    l1 = ax2.plot(
        t_bins, p_curve * 1000.0, color=BLUE, lw=1.8, label="Pupille Δ (μm, axe gauche)"
    )
    l2 = ax2_r.plot(
        t_bins, g_curve, color=ORANGE, lw=1.8, label="Vitesse regard (px/s, axe droit)"
    )
    ax2.axvline(0.0, color=DARK, ls="--", lw=1.0)
    ax2.axhline(0.0, color=BLUE, ls=":", lw=0.7, alpha=0.6)
    ax2_r.annotate(
        "+121 px/s\nà t = 0,00 s",
        xy=(0.125, float(g_curve[12])),
        xytext=(2.3, 1185),
        fontsize=7.0,
        fontweight="bold",
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.0},
    )
    ax2.set_xlim(-3.0, 10.0)
    ax2.set_ylim(-20, 12)
    ax2_r.set_ylim(1020, 1225)
    ax2.set_xlabel("Temps par rapport au début vidéo (s)")
    ax2.set_ylabel("Δ diamètre pupillaire (μm)", color=BLUE)
    ax2_r.set_ylabel("Vitesse du regard (px/s)", color=ORANGE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2_r.tick_params(axis="y", colors=ORANGE)
    ax2.set_title(
        "B. Synchronisation temporelle (tranches 0,25 s)", loc="left", fontweight="bold"
    )
    lns = l1 + l2
    ax2.legend(
        lns,
        [ln.get_label() for ln in lns],
        frameon=False,
        fontsize=6.8,
        loc="lower left",
    )

    # Panel C: Stress OOF correlation heatmap + Stacking gate outcome
    sc = m7["stress_complementarity"]
    m_keys = [
        "S07_enriched_CS",
        "REVE_frozen_linear_CS",
        "reve_mt_frozen_rank",
        "reve_mt_lora_rank",
        "Oculo_simple_CS",
    ]
    m_short = [
        "S07\nenrichi",
        "REVE\nlinéaire",
        "REVE MT\ngelé",
        "REVE MT\nLoRA",
        "Oculo\nsimple",
    ]
    mat = np.array(
        [[sc["pairwise_within_person_corr"][a][b] for b in m_keys] for a in m_keys],
        dtype=float,
    )
    im = ax3.imshow(mat, vmin=-0.1, vmax=1.0, cmap="YlGnBu")
    for i in range(len(m_keys)):
        for j in range(len(m_keys)):
            val = mat[i, j]
            ax3.text(
                j,
                i,
                fnum(val, 2),
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if val > 0.55 else DARK,
            )
    ax3.set_xticks(range(len(m_short)))
    ax3.set_xticklabels(m_short, fontsize=6.9)
    ax3.set_yticks(range(len(m_short)))
    ax3.set_yticklabels(m_short, fontsize=6.9)
    ax3.set_title(
        "C. Complémentarité OOF stress (r intra-sujet)", loc="left", fontweight="bold"
    )
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.8)
    st_s = m7["stress_nested_stacking"]
    ax3.set_xlabel(
        f"Stacking imbriqué : AUC {fnum(st_s['metrics']['stress']['auc'], 3)} "
        f"vs meilleur ({fnum(sc['individual_scores']['S07_enriched_CS']['mean'], 3)}, "
        f"Δ {fnum(st_s['paired_gain_vs_best_single']['gain'], 3, signed=True)})",
        fontsize=7.1,
    )

    fig.tight_layout(w_pad=2.1)
    save_fig(fig, "fig5_oculo_stacking")


def main() -> None:
    """Generate all available Study-09 figures from saved JSON aggregates."""
    ext_audit = load_json("study09_extract_audit.json")
    ctrl = load_json("controls_metrics.json")
    lora_f = load_json("study09_lora_forecast.json")
    lora_mt = load_json("study09_lora_mt.json")
    sec = load_json("secondary_metrics.json")
    af = load_json("audit_forecast_metrics.json")

    if af is not None:
        make_fig1(af)
        make_fig2(af, lora_f)
    else:
        print("Skipping Fig 1 & Fig 2 (audit_forecast_metrics.json not yet present)")

    if ctrl is not None and sec is not None:
        make_fig3(ctrl, sec)
    if lora_mt is not None and sec is not None:
        make_fig4(lora_mt, sec)
    if ext_audit is not None and sec is not None:
        make_fig5(ext_audit, sec)


if __name__ == "__main__":
    main()
