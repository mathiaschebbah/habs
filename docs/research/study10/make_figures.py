"""Generate Study-10 French publication figures from verified JSON aggregates.

Run from the repository root:
    .venv-agent/bin/python docs/research/study10/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
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


def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure at 300 dpi and close."""
    out = FIG / f"{stem}_fr.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIG 1: Generalization Ladder (R1->R4) & R4 Sensor Modality Ablation
# ---------------------------------------------------------------------------


def make_fig1(data: dict) -> None:
    """Generalization ladder R1-R4 and R4 sensor modality ablation."""
    m10a = data["module10a_generalization_ablation"]
    lad = m10a["generalization_ladder"]
    abl = m10a["r4_modality_ablation"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.08, 1.15]}
    )
    fig.subplots_adjust(wspace=0.45)

    # Panel A: R1 -> R4 ladder
    regs = [
        ("R1_random_5fold", "R1 : Aléatoire\n(suj. + thèmes)"),
        ("R3_loso_seen_themes", "R3 : LOSO naïf\n(thèmes vus)"),
        ("R2_within_person_new_themes", "R2 : Intra-sujet\n(thèmes inédits)"),
        ("R4_double_unseen_cold_start", "R4 : Double inédit\n(Cold-Start)"),
    ]
    x = np.arange(len(regs))
    w = 0.25

    s07_wp = [
        lad["S07_enriched"][k]["stress_ridge_va"]["stress"]["auc"] for k, _ in regs
    ]
    s07_ci = [
        lad["S07_enriched"][k]["stress_ridge_va"]["stress"]["ci95"] for k, _ in regs
    ]
    rev_wp = [
        lad["REVE_frozen"][k]["stress_ridge_va"]["stress"]["auc"] for k, _ in regs
    ]
    rev_ci = [
        lad["REVE_frozen"][k]["stress_ridge_va"]["stress"]["ci95"] for k, _ in regs
    ]
    rev_pl = [lad["REVE_frozen"][k]["stress_logreg"]["pooled_auc"] for k, _ in regs]

    s07_err = np.array(
        [[v - c[0], c[1] - v] for v, c in zip(s07_wp, s07_ci, strict=False)]
    ).T
    rev_err = np.array(
        [[v - c[0], c[1] - v] for v, c in zip(rev_wp, rev_ci, strict=False)]
    ).T

    b1 = ax1.bar(
        x - w,
        s07_wp,
        w,
        yerr=s07_err,
        capsize=2.8,
        color=BLUE,
        error_kw={"elinewidth": 1.0, "ecolor": DARK},
        label="S07 enrichi (V&A, AUC intra)",
    )
    b2 = ax1.bar(
        x,
        rev_wp,
        w,
        yerr=rev_err,
        capsize=2.8,
        color=TEAL,
        error_kw={"elinewidth": 1.0, "ecolor": DARK},
        label="REVE-base gelé (V&A, AUC intra)",
    )
    b3 = ax1.bar(
        x + w,
        rev_pl,
        w,
        color=ORANGE,
        alpha=0.85,
        label="REVE-base LogReg (AUC poolée)",
    )

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.0025 if bars is b3 else h + 0.0180,
                fnum(h, 3),
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=DARK,
                fontweight="bold",
            )

    ax1.axhline(0.50, color=GRAY, ls="--", lw=0.9)
    ax1.annotate(
        "Mémorisation sujet :\n+0,041 (AUC poolée)",
        xy=(0 + w, rev_pl[0] + 0.008),
        xytext=(-0.38, 0.629),
        fontsize=6.9,
        color=ORANGE,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.0},
    )
    ax1.annotate(
        "Fuite thèmes vus :\nR3 − R4 = +0,014",
        xy=(1 - w, s07_ci[1][1] + 0.006),
        xytext=(1.58, 0.629),
        fontsize=6.9,
        color=BLUE,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.0},
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in regs], fontsize=7.4)
    ax1.set_ylim(0.495, 0.655)
    ax1.set_ylabel("AUC ROC (dév., N = 208 sujets éligibles)")
    ax1.set_title(
        "A. Échelle de généralisation (R1 → R4)", fontsize=9.2, fontweight="bold", pad=8
    )
    ax1.legend(
        loc="lower left", fontsize=6.9, frameon=True, facecolor="white", framealpha=0.94
    )
    ax1.grid(axis="y", color=LIGHT, lw=0.7)
    fmt_axis(ax1, "y", 2)

    # Panel B: R4 Sensor Modality Ablation (ordered forest plot)
    order = [
        ("All_plus_REVE", "Complet S07 + REVE (960)", "comb"),
        ("All", "Complet S07 (448, réf.)", "ref"),
        ("minus_ECG", "Sans ECG (−ECG, 385)", "abl"),
        ("minus_Resp", "Sans Resp. (−Resp, 433)", "abl"),
        ("REVE_only", "REVE-base seul (512)", "single"),
        ("EDA_only", "EDA seule (15)", "single"),
        ("minus_EDA", "Sans EDA (−EDA, 433)", "abl_sig"),
        ("EEG_hc_only", "EEG manuel seul (355)", "single"),
        ("minus_EEG", "Sans EEG (−EEG, 93)", "abl_sig"),
        ("Resp_only", "Respiration seule (15)", "single"),
        ("ECG_only", "ECG seul (63)", "single"),
    ]
    y = np.arange(len(order))[::-1]
    cmap = {
        "comb": PURPLE,
        "ref": BLUE,
        "abl": TEAL,
        "abl_sig": RED,
        "single": AMBER,
    }
    ref_val = abl["All"]["stress_ridge_va"]["stress"]["auc"]
    ax2.axvline(0.50, color=GRAY, ls="--", lw=0.9)
    ax2.axvline(ref_val, color=BLUE, ls=":", lw=1.0, alpha=0.8)

    for yi, (key, _lbl, grp) in zip(y, order, strict=False):
        st = abl[key]["stress_ridge_va"]["stress"]
        auc = st["auc"]
        lo, hi = st["ci95"]
        pg = abl[key]["paired_vs_All"]
        col = cmap[grp]
        ax2.errorbar(
            auc,
            yi,
            xerr=[[auc - lo], [hi - auc]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.5,
            capsize=3.0,
            markersize=5.5,
        )
        sig = "*" if (pg["ci95"][1] is not None and pg["ci95"][1] < 0) else ""
        if key == "All":
            txt = f"{fnum(auc, 3)} (réf.)"
        else:
            txt = f"{fnum(auc, 3)} ({fnum(pg['gain'], 3, signed=True)}{sig})"
        ax2.text(
            hi + 0.0025,
            yi,
            txt,
            va="center",
            ha="left",
            fontsize=7.0,
            color=RED if sig else DARK,
            fontweight="bold" if (sig or key == "All") else "normal",
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels([lbl for _, lbl, _ in order], fontsize=7.6)
    ax2.set_xlim(0.486, 0.628)
    ax2.set_xlabel("AUC intra-sujet R4 Cold-Start (IC 95 %, Δ apparié vs Complet)")
    ax2.set_title(
        "B. Ablation de modalités en R4 Cold-Start",
        fontsize=9.2,
        fontweight="bold",
        pad=8,
    )
    ax2.grid(axis="x", color=LIGHT, lw=0.7)
    fmt_axis(ax2, "x", 2)
    handles = [
        Patch(facecolor=BLUE, label="Complet (réf.)"),
        Patch(facecolor=RED, label="Ablation signif. (*IC < 0)"),
        Patch(facecolor=TEAL, label="Ablation neutre"),
        Patch(facecolor=AMBER, label="Capteur isolé"),
    ]
    ax2.legend(
        handles=handles,
        loc="upper left",
        fontsize=6.7,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )
    save_fig(fig, "fig1_generalization_ablation")


# ---------------------------------------------------------------------------
# FIG 2: J1->J2 Longitudinal Transfer, k-shot Personalization & LOCO Cardiac
# ---------------------------------------------------------------------------


def make_fig2(data: dict) -> None:
    """Longitudinal J1->J2 transfer, k-shot curve, and LOCO cardiac replication."""
    m10b = data["module10b_longitudinal_habituation"]["cross_day_decoder_transfer_dev"]
    kshot = data["module10c_coral_kshot_loco"]["kshot_chronological_dev"][
        "S07_enriched"
    ]
    loco_fc = data["module10c_coral_kshot_loco"]["loco_cross_cohort_dev"][
        "cardiac_forecast_loco"
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(13.8, 4.4), gridspec_kw={"width_ratios": [1.05, 1.15, 1.0]}
    )
    fig.subplots_adjust(wspace=0.44)

    # Panel A: Cross-day J1 -> J2 transfer by sensor (Sakimura 2025 test)
    mods = [
        ("S07_enriched", "S07\nComplet"),
        ("REVE_frozen", "REVE\nEEG"),
        ("EEG_hc_only", "EEG\nmanuel"),
        ("EDA_only", "EDA\nseule"),
        ("Resp_only", "Resp.\nseule"),
        ("ECG_only", "ECG\nseul"),
    ]
    x = np.arange(len(mods))
    w = 0.34
    j1 = [m10b[k]["within_day1_3x3"]["auc"] for k, _ in mods]
    j1_ci = [m10b[k]["within_day1_3x3"]["ci95"] for k, _ in mods]
    j2 = [m10b[k]["j1_to_j2_new_people_unseen_themes"]["auc"] for k, _ in mods]
    j2_ci = [m10b[k]["j1_to_j2_new_people_unseen_themes"]["ci95"] for k, _ in mods]
    j1_e = np.array([[v - c[0], c[1] - v] for v, c in zip(j1, j1_ci, strict=False)]).T
    j2_e = np.array([[v - c[0], c[1] - v] for v, c in zip(j2, j2_ci, strict=False)]).T

    ax1.bar(
        x - w / 2,
        j1,
        w,
        yerr=j1_e,
        capsize=2.5,
        color=BLUE,
        error_kw={"elinewidth": 1.0, "ecolor": DARK},
        label="Intra-jour J1 (3×3)",
    )
    ax1.bar(
        x + w / 2,
        j2,
        w,
        yerr=j2_e,
        capsize=2.5,
        color=ORANGE,
        error_kw={"elinewidth": 1.0, "ecolor": DARK},
        label="Transfert J1 → J2 (inédits)",
    )
    for i, (k, _) in enumerate(mods):
        g = m10b[k]["paired_j2_newp_minus_j1"]
        sig = g["ci95"][1] is not None and g["ci95"][1] < 0
        if sig:
            ax1.text(
                i,
                max(j1_ci[i][1], j2_ci[i][1]) + 0.005,
                f"{fnum(g['gain'], 3, signed=True)}*",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=RED,
                fontweight="bold",
            )
    ax1.axhline(0.50, color=GRAY, ls="--", lw=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in mods], fontsize=7.4)
    ax1.set_ylim(0.465, 0.618)
    ax1.set_ylabel("AUC intra-sujet (thèmes inédits)")
    ax1.set_title(
        "A. Stabilité J1 → J2 par capteur", fontsize=9.0, fontweight="bold", pad=8
    )
    ax1.legend(loc="lower left", fontsize=6.9, frameon=True, facecolor="white")
    ax1.grid(axis="y", color=LIGHT, lw=0.7)
    fmt_axis(ax1, "y", 2)

    # Panel B: Chronological k-shot personalization curve (k = 0..40)
    ks = [0, 5, 10, 15, 20, 30, 40]
    kc = kshot["k_curve"]
    styles = [
        ("cold_start", "Cold-Start (réf. k=0)", DARK, "--", "o"),
        ("sup_eb_map_shrunk", "Supervisé EB-MAP (rég.)", TEAL, "-", "s"),
        ("unsup_coral", "CORAL non sup.", PURPLE, "-.", "^"),
        ("unsup_b1_center", "Centrage B1 non sup.", BLUE, ":", "d"),
        ("sup_naive_personal", "Supervisé naïf (local)", RED, "-", "X"),
    ]
    for mkey, lbl, col, ls, mk in styles:
        ys = [kc[str(k)][mkey]["auc"] for k in ks]
        ax2.plot(ks, ys, color=col, ls=ls, marker=mk, lw=1.7, markersize=5.2, label=lbl)
        if mkey == "sup_naive_personal":
            lo = [kc[str(k)][mkey]["ci95"][0] for k in ks]
            hi = [kc[str(k)][mkey]["ci95"][1] for k in ks]
            ax2.fill_between(ks, lo, hi, color=RED, alpha=0.10)

    ax2.axhline(0.50, color=GRAY, ls="--", lw=0.8)
    ax2.annotate(
        "Chute naïve : −0,081 à k=5\n(−0,053 encore à k=40)",
        xy=(10, kc["10"]["sup_naive_personal"]["auc"] + 0.002),
        xytext=(15.5, 0.528),
        fontsize=6.9,
        color=RED,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.0},
    )
    ax2.set_xticks(ks)
    ax2.set_ylim(0.455, 0.592)
    ax2.set_xlabel("Nombre d'essais passés k (queue commune [40:], N = 104)")
    ax2.set_ylabel("AUC intra-sujet (S07 enrichi)")
    ax2.set_title(
        "B. Personnalisation chronologique k-shot",
        fontsize=9.0,
        fontweight="bold",
        pad=8,
    )
    ax2.legend(
        loc="lower right",
        fontsize=6.6,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
    )
    ax2.grid(color=LIGHT, lw=0.7)
    fmt_axis(ax2, "y", 2)

    # Panel C: LOCO Cross-Cohort replication of Study 09 ECG M2 - M0
    by_c = loco_fc["by_cohort"]
    rows_c = [
        ("Global LOCO\n(N = 224)", loco_fc["paired_M2_minus_M0_raw"]["delta_r"], True),
        ("Cohorte 1 (N=80)", by_c["cohort_1"]["delta_M2_minus_M0_raw"], False),
        ("Cohorte 2 (N=84)", by_c["cohort_2"]["delta_M2_minus_M0_raw"], False),
        ("Cohorte 3 (N=17)", by_c["cohort_3"]["delta_M2_minus_M0_raw"], False),
        ("Cohorte 4 (N=35)", by_c["cohort_4"]["delta_M2_minus_M0_raw"], False),
        ("Cohorte 5 (N=8)", by_c["cohort_5"]["delta_M2_minus_M0_raw"], False),
    ]
    yc = np.arange(len(rows_c))[::-1]
    ax3.axvline(0.0, color=GRAY, ls="--", lw=0.9)
    for yi, (lbl, dr, is_glob) in zip(yc, rows_c, strict=False):
        m = dr["mean"]
        lo, hi = dr["ci95"]
        pct = dr["frac_improved"] * 100.0
        col = BLUE if is_glob else TEAL
        ax3.errorbar(
            m,
            yi,
            xerr=[[m - lo], [hi - m]],
            fmt="D" if is_glob else "o",
            color=col,
            ecolor=col,
            elinewidth=2.0 if is_glob else 1.4,
            capsize=3.2,
            markersize=6.2 if is_glob else 5.2,
        )
        ax3.text(
            max(hi, m) + 0.0012,
            yi,
            f"{fnum(m, 3, signed=True)} ({pct:.0f}%)",
            va="center",
            ha="left",
            fontsize=7.0,
            color=DARK,
            fontweight="bold" if is_glob else "normal",
        )
    ax3.axhline(4.5, color=LIGHT, lw=1.2)
    ax3.set_yticks(yc)
    ax3.set_yticklabels([r[0] for r in rows_c], fontsize=7.4)
    ax3.set_xlim(-0.014, 0.050)
    ax3.set_xlabel("Gain cardiaque Δr (M2 REVE − M0, IC 95 %)")
    ax3.set_title(
        "C. Réplication LOCO inter-cohortes (5/5)",
        fontsize=9.0,
        fontweight="bold",
        pad=8,
    )
    ax3.grid(axis="x", color=LIGHT, lw=0.7)
    fmt_axis(ax3, "x", 3)

    save_fig(fig, "fig2_longitudinal_personalization")


# ---------------------------------------------------------------------------
# FIG 3: Autonomic-Subjective Space (Cuve), Individual Coherence (Brown) & Within-Clip
# ---------------------------------------------------------------------------


def make_fig3(data: dict) -> None:
    """Cuve 3D factor structure, Brown coherence spread, and Fear vs Disgust dynamics."""
    coh = data["module10d_coherence_bridge"]["coherence_all_314"]
    pca3 = coh["pca_within_person_total"]["varimax_3f_loadings"]
    evr = coh["pca_within_person_total"]["explained_variance_ratio"]
    brown = coh["brown_individual_coherence"]
    wclip = data["module10f_within_clip_emotions"]["by_emotion_vs_neutral"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.4, 4.7), gridspec_kw={"width_ratios": [0.95, 1.15, 1.15]}
    )
    fig.subplots_adjust(wspace=0.56)

    # Panel A: Varimax 3-factor loadings heatmap (Cuve et al. 2023 on N=314)
    vars_order = [
        ("valence", "Valence subj."),
        ("arousal", "Arousal subj."),
        ("d_sc", "Δ EDA (SC)"),
        ("d_scr", "Δ EDA (SCR)"),
        ("d_hr", "Δ ECG (FC)"),
        ("d_resp", "Δ Resp."),
        ("d_delta_cp", "Δ EEG δ CP"),
        ("d_alpha_occ", "Δ EEG α occ."),
        ("d_gamma_fr", "Δ EEG γ fr."),
    ]
    mat = np.array(
        [[pca3["F1"][k], pca3["F2"][k], pca3["F3"][k]] for k, _ in vars_order],
        dtype=float,
    )
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            col = "white" if abs(v) > 0.55 else DARK
            ax1.text(
                j,
                i,
                fnum(v, 2, signed=True),
                ha="center",
                va="center",
                fontsize=7.2,
                color=col,
                fontweight="bold" if abs(v) >= 0.30 else "normal",
            )
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(
        [
            f"F1 : EDA\n({fnum(evr[0] * 100, 1)} %)",
            f"F2 : Neuro\n({fnum(evr[1] * 100, 1)} %)",
            f"F3 : Subj.\n({fnum(evr[2] * 100, 1)} %)",
        ],
        fontsize=7.3,
    )
    ax1.set_yticks(np.arange(len(vars_order)))
    ax1.set_yticklabels([lbl for _, lbl in vars_order], fontsize=7.6)
    ax1.set_title(
        "A. Espace Varimax (N = 314, n = 16 276)",
        fontsize=8.9,
        fontweight="bold",
        pad=8,
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.048, pad=0.035)
    cbar.ax.tick_params(labelsize=6.8)
    cbar.set_label("Charge", fontsize=7.2)

    # Panel B: Brown et al. (2019) individual coherence distribution (IQR + P10-P90 + mean)
    chs = [
        ("neuro_autonomic_composite", "Composite\nneuro-auto", BLUE),
        ("autonomic_composite", "Composite\nautonome", TEAL),
        ("eda_sc_rise", "EDA (+ΔSC)", ORANGE),
        ("delta_cp_inc", "EEG δ (+Δδ)", PURPLE),
        ("heart_decel", "Cœur (−ΔFC)", RED),
        ("alpha_occ_supp", "EEG α (−Δα)", AMBER),
        ("resp_shallow", "Resp. (−ΔAmp)", GRAY),
    ]
    yb = np.arange(len(chs))[::-1]
    ax2.axvline(0.0, color=DARK, ls="--", lw=0.9)
    for yi, (k, _lbl, col) in zip(yb, chs, strict=False):
        st = brown[k]
        p10, p90 = st["p10_p90"]
        q25, q75 = st["iqr"]
        med = st["median"]
        pct_pos = st["frac_positive"] * 100.0
        hi_mod = st["intensity_effect"]["paired_high_minus_mod"]["mean"]
        ax2.plot([p10, p90], [yi, yi], color=col, lw=1.4, alpha=0.55)
        ax2.plot(
            [q25, q75], [yi, yi], color=col, lw=5.5, alpha=0.85, solid_capstyle="round"
        )
        ax2.plot(
            med,
            yi,
            "o",
            color="white",
            markeredgecolor=DARK,
            markersize=5.2,
            markeredgewidth=1.2,
        )
        ax2.text(
            p90 + 0.012,
            yi,
            f"méd. {fnum(med, 2, signed=True)} | {pct_pos:.0f} % > 0\n(int. {fnum(hi_mod, 2, signed=True)})",
            va="center",
            ha="left",
            fontsize=6.5,
            color=DARK,
        )
    ax2.set_yticks(yb)
    ax2.set_yticklabels([lbl for _, lbl, _ in chs], fontsize=7.4)
    ax2.set_xlim(-0.20, 0.52)
    ax2.set_xlabel("Cohérence intra-sujet ρ(détresse, signal) [P10–P90, IQR]")
    ax2.set_title(
        "B. Hétérogénéité individuelle (N = 314)",
        fontsize=8.9,
        fontweight="bold",
        pad=8,
    )
    ax2.grid(axis="x", color=LIGHT, lw=0.7)
    fmt_axis(ax2, "x", 2)

    # Panel C: Within-clip Fear vs Disgust early [0;5) vs late [5;10)
    ch_plot = [
        ("eda_sc", "EDA (SC)"),
        ("ecg_hr", "ECG (FC)"),
        ("resp_amp", "Resp."),
        ("eeg_occ_alpha", "EEG α occ."),
        ("eeg_global_delta", "EEG δ glob."),
        ("eeg_fr_gamma", "EEG γ fr."),
    ]
    yc = np.arange(len(ch_plot))[::-1]
    ax3.axvline(0.0, color=GRAY, ls="--", lw=0.9)
    dy = 0.16
    for yi, (k, _) in zip(yc, ch_plot, strict=False):
        fe = wclip["fear"]["channels"][k]["early_d_vs_neutral"]
        fl = wclip["fear"]["channels"][k]["late_d_vs_neutral"]
        de = wclip["disgust"]["channels"][k]["early_d_vs_neutral"]
        dl = wclip["disgust"]["channels"][k]["late_d_vs_neutral"]

        ax3.errorbar(
            fe[0],
            yi + 1.5 * dy,
            xerr=[[fe[0] - fe[1]], [fe[2] - fe[0]]],
            fmt="o",
            color=BLUE,
            mfc="white",
            ms=4.8,
            elinewidth=1.1,
            capsize=2.0,
            label="Peur [0;5) s" if yi == yc[0] else None,
        )
        ax3.errorbar(
            fl[0],
            yi + 0.5 * dy,
            xerr=[[fl[0] - fl[1]], [fl[2] - fl[0]]],
            fmt="o",
            color=BLUE,
            mfc=BLUE,
            ms=5.0,
            elinewidth=1.4,
            capsize=2.0,
            label="Peur [5;10) s" if yi == yc[0] else None,
        )
        ax3.errorbar(
            de[0],
            yi - 0.5 * dy,
            xerr=[[de[0] - de[1]], [de[2] - de[0]]],
            fmt="s",
            color=ORANGE,
            mfc="white",
            ms=4.8,
            elinewidth=1.1,
            capsize=2.0,
            label="Dégoût [0;5) s" if yi == yc[0] else None,
        )
        ax3.errorbar(
            dl[0],
            yi - 1.5 * dy,
            xerr=[[dl[0] - dl[1]], [dl[2] - dl[0]]],
            fmt="s",
            color=ORANGE,
            mfc=ORANGE,
            ms=5.0,
            elinewidth=1.4,
            capsize=2.0,
            label="Dégoût [5;10) s" if yi == yc[0] else None,
        )
    ax3.set_yticks(yc)
    ax3.set_yticklabels([lbl for _, lbl in ch_plot], fontsize=7.6)
    ax3.set_xlim(-0.38, 0.45)
    ax3.set_ylim(-0.55, 5.75)
    ax3.set_xlabel("Taille d'effet d vs Neutre (IC 95 %, N = 209)")
    ax3.set_title(
        "C. Dynamique intra-clip : Peur vs Dégoût",
        fontsize=8.9,
        fontweight="bold",
        pad=8,
    )
    ax3.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.15),
        ncol=2,
        fontsize=6.8,
        frameon=True,
        facecolor="white",
    )
    ax3.grid(axis="x", color=LIGHT, lw=0.7)
    fmt_axis(ax3, "x", 2)

    save_fig(fig, "fig3_coherence_withinclip")


# ---------------------------------------------------------------------------
# FIG 4: Clinical Multi-Episode Passive Monitoring Aggregation (10 s -> 100 s)
# ---------------------------------------------------------------------------


def make_fig4(data: dict) -> None:
    """Operational passive monitoring curve (K=1..10 episodes = 10-100 s)."""
    m10e = data["module10e_clinical_monitoring"]
    ks = [1, 2, 3, 5, 8, 10]
    secs = [10 * k for k in ks]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.20, 1.0]}
    )
    fig.subplots_adjust(wspace=0.30)

    # Panel A: AUC vs passive window duration (10s -> 100s)
    curves = [
        (
            "stress_strict37_vs_relaxed64",
            "S07_enriched",
            "Multimodal — Stress 3/7 vs Relaxé",
            RED,
            "-",
            "o",
        ),
        (
            "acute_threat_vs_calm_neutral",
            "S07_enriched",
            "Multimodal — Menace vs Calme",
            ORANGE,
            "-",
            "s",
        ),
        (
            "stress_standard_vs_nonstress",
            "S07_enriched",
            "Multimodal — Stress standard",
            BLUE,
            "-",
            "^",
        ),
        (
            "stress_standard_vs_nonstress",
            "REVE_frozen",
            "REVE EEG seul — Stress std.",
            TEAL,
            "--",
            "d",
        ),
        (
            "stress_standard_vs_nonstress",
            "EDA_only",
            "EDA seule — Stress std.",
            PURPLE,
            ":",
            "v",
        ),
        (
            "stress_standard_vs_nonstress",
            "ECG_only",
            "ECG seul — Stress std.",
            GRAY,
            "-.",
            "x",
        ),
    ]
    for cname, mname, lbl, col, ls, mk in curves:
        by_k = m10e[cname][mname]["by_k"]
        ys = [by_k[str(k)]["auc"]["mean"] for k in ks]
        lo = [by_k[str(k)]["auc"]["ci95"][0] for k in ks]
        hi = [by_k[str(k)]["auc"]["ci95"][1] for k in ks]
        ax1.plot(
            secs,
            ys,
            color=col,
            ls=ls,
            marker=mk,
            lw=1.8 if ls == "-" else 1.4,
            ms=5.2,
            label=lbl,
        )
        if ls == "-":
            ax1.fill_between(secs, lo, hi, color=col, alpha=0.09)

    ax1.axhspan(0.73, 0.78, color=TEAL, alpha=0.09)
    ax1.text(
        11,
        0.755,
        "Repère EEG clinique inter-centres (Engemann 2018 : 0,73–0,78)",
        fontsize=6.8,
        color=TEAL,
        fontweight="bold",
        va="center",
    )
    ax1.axhline(0.50, color=GRAY, ls="--", lw=0.8)

    for cname, mname, col, dy in (
        ("stress_strict37_vs_relaxed64", "S07_enriched", RED, 0.008),
        ("acute_threat_vs_calm_neutral", "S07_enriched", ORANGE, -0.006),
        ("stress_standard_vs_nonstress", "S07_enriched", BLUE, -0.014),
        ("stress_standard_vs_nonstress", "REVE_frozen", TEAL, 0.003),
    ):
        v100 = m10e[cname][mname]["by_k"]["10"]["auc"]["mean"]
        ax1.text(
            102.5,
            v100 + dy,
            fnum(v100, 3),
            fontsize=7.0,
            color=col,
            fontweight="bold",
            va="center",
        )

    ax1.set_xticks(secs)
    ax1.set_xticklabels(
        [f"{s} s\n(K={k})" for s, k in zip(secs, ks, strict=False)], fontsize=7.2
    )
    ax1.set_xlim(6, 114)
    ax1.set_ylim(0.49, 0.79)
    ax1.set_xlabel("Durée cumulée d'observation passive (Cold-Start R4)")
    ax1.set_ylabel("AUC intra-sujet (IC 95 % bootstrap sujets)")
    ax1.set_title(
        "A. Gain de l'agrégation multi-épisodes (10 s → 100 s)",
        fontsize=9.1,
        fontweight="bold",
        pad=8,
    )
    ax1.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.13),
        fontsize=6.4,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )
    ax1.grid(color=LIGHT, lw=0.7)
    fmt_axis(ax1, "y", 2)

    # Panel B: Sensitivity @ 80% and 90% Specificity at K=1, K=5, K=10 (S07_enriched)
    ctrs_b = [
        ("stress_standard_vs_nonstress", "Stress standard\nvs Non-stress"),
        ("acute_threat_vs_calm_neutral", "Menace aiguë\nvs Calme/Neutre"),
        ("stress_strict37_vs_relaxed64", "Stress 3/7\nvs Relaxé 6/4"),
    ]
    xb = np.arange(len(ctrs_b))
    w = 0.25
    k_cols = [
        ("1", "10 s (K=1)", "#8bb4e8"),
        ("5", "50 s (K=5)", BLUE),
        ("10", "100 s (K=10)", "#143d7a"),
    ]

    for idx_k, (k_s, _klbl, col) in enumerate(k_cols):
        s80 = [
            m10e[c]["S07_enriched"]["by_k"][k_s]["sens_at_spec80"]["mean"] * 100
            for c, _ in ctrs_b
        ]
        s90 = [
            m10e[c]["S07_enriched"]["by_k"][k_s]["sens_at_spec90"]["mean"] * 100
            for c, _ in ctrs_b
        ]
        pos = xb + (idx_k - 1) * w
        ax2.bar(pos, s80, w, color=col, alpha=0.38, edgecolor=col, lw=1.0)
        ax2.bar(pos, s90, w, color=col, edgecolor=DARK, lw=0.6)
        for p_x, v80, v90 in zip(pos, s80, s90, strict=False):
            ax2.text(
                p_x,
                v80 + 1.1,
                f"{v80:.0f} %",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=MUTED,
            )
            ax2.text(
                p_x,
                v90 - 4.5 if v90 > 24 else v90 + 1.0,
                f"{v90:.0f} %",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="white" if v90 > 24 else DARK,
                fontweight="bold",
            )

    handles_b = [
        Patch(
            facecolor="#8bb4e8", edgecolor=DARK, lw=0.6, label="10 s (K=1) — Spec 90 %"
        ),
        Patch(facecolor=BLUE, edgecolor=DARK, lw=0.6, label="50 s (K=5) — Spec 90 %"),
        Patch(
            facecolor="#143d7a",
            edgecolor=DARK,
            lw=0.6,
            label="100 s (K=10) — Spec 90 %",
        ),
        Patch(
            facecolor=BLUE,
            alpha=0.38,
            edgecolor=BLUE,
            lw=1.0,
            label="Partie claire : Spec 80 %",
        ),
    ]
    ax2.set_xticks(xb)
    ax2.set_xticklabels([lbl for _, lbl in ctrs_b], fontsize=7.6)
    ax2.set_ylim(0, 74)
    ax2.set_ylabel("Sensibilité (%) à spécificité fixée")
    ax2.set_title(
        "B. Sensibilité opérationnelle (Multimodal Cold-Start)",
        fontsize=9.1,
        fontweight="bold",
        pad=8,
    )
    ax2.legend(
        handles=handles_b,
        loc="upper left",
        fontsize=6.6,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
    )
    ax2.grid(axis="y", color=LIGHT, lw=0.7)

    save_fig(fig, "fig4_clinical_monitoring")


def main() -> None:
    """Generate all 4 Study-10 French publication figures."""
    data = json.loads((RES / "study10_metrics.json").read_text())
    make_fig1(data)
    make_fig2(data)
    make_fig3(data)
    make_fig4(data)


if __name__ == "__main__":
    main()
