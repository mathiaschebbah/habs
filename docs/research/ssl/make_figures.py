"""Generate Study-08 bilingual (EN/FR) publication figures from saved JSON aggregates.

Run from the repository root:
    .venv-agent/bin/python docs/research/ssl/make_figures.py        # English
    .venv-agent/bin/python docs/research/ssl/make_figures.py fr     # French (_fr.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FR = "fr" in sys.argv[1:]
SUFFIX = "_fr" if FR else ""

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEAL = "#1baf7a"
RED = "#d94343"
PURPLE = "#7c5cbf"
AMBER = "#e39c12"
GRAY = "#8f8e88"
INK = "#111111"
MUTED = "#555450"
LIGHT = "#ecebe6"

plt.rcParams.update(
    {
        "font.size": 8.2,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def tr(en: str, fr: str) -> str:
    """Return French or English string based on CLI flag."""
    return fr if FR else en


def fnum(val: float, decimals: int = 3, signed: bool = False) -> str:
    """Format decimal number with comma in French mode and dot in English."""
    fmt = f"{val:+.{decimals}f}" if signed else f"{val:.{decimals}f}"
    if FR:
        fmt = fmt.replace(".", ",").replace("-", "\u2212")
    return fmt


def apply_decimal_formatter(ax: plt.Axes, axis: str = "x", decimals: int = 2) -> None:
    """Format tick labels with French decimal commas when in FR mode."""
    if not FR:
        return
    fn = FuncFormatter(lambda v, _: fnum(v, decimals))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fn)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fn)


def load_json(name: str) -> dict | None:
    """Load an aggregate JSON file from ``results/`` if present."""
    p = RES / name
    return json.loads(p.read_text()) if p.is_file() else None


def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure to ``figures/<stem>[_fr].png`` and close."""
    out = FIG / f"{stem}{SUFFIX}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def make_fig1(fm_s07: dict, prim: dict, modwin: dict | None, fuse: dict) -> None:
    """Fig 1: Complete 4-family benchmark (dev vs sealed test) + paired gains."""
    s07 = fm_s07["study07_baselines"]
    fms = fm_s07["foundation_models"]
    pm = prim["primary_models"]
    ji = fuse["jepa_integration"]
    pg = ji["paired_gains_vs_study07"]

    rows = []

    def add_row(
        label: str,
        fam: str,
        dev_auc: float,
        dev_ci: list[float],
        nb_auc: float,
        nb_ci: list[float],
        gain: float | None,
        gain_ci: list[float] | None,
        ref_tag: str = "full",
    ) -> None:
        rows.append(
            {
                "label": label,
                "fam": fam,
                "dev": dev_auc,
                "dev_ci": dev_ci,
                "nb": nb_auc,
                "nb_ci": nb_ci,
                "gain": gain,
                "gain_ci": gain_ci,
                "ref_tag": ref_tag,
            }
        )

    # 1. Study 07 linear baselines
    for key, lbl in (
        (
            "sequence_plus_enriched",
            tr("S07 Sequence + Enriched (1068)", "É07 Séquence + Enrichi (1068)"),
        ),
        (
            "enriched_full",
            tr("S07 Enriched Full (448, ref)", "É07 Enrichi Complet (448, réf.)"),
        ),
        (
            "enriched_eeg_only",
            tr("S07 Enriched EEG-only (355)", "É07 Enrichi EEG seul (355)"),
        ),
        ("phase4_body_only", tr("S07 Body-only (40)", "É07 Corps seul (40)")),
    ):
        m = s07[key]
        g = m["test_new_both"].get("gain_ridge_va_vs_ref")
        add_row(
            lbl,
            "s07",
            m["dev"]["stress_ridge_va"]["stress"]["auc"],
            m["dev"]["stress_ridge_va"]["stress"]["ci95"],
            m["test_new_both"]["stress_ridge_va"]["stress"]["auc"],
            m["test_new_both"]["stress_ridge_va"]["stress"]["ci95"],
            0.0 if key == "enriched_full" else (g["gain"] if g else None),
            [0.0, 0.0] if key == "enriched_full" else (g["ci95"] if g else None),
        )

    # 2. Pretrained EEG Foundation Models (raw_pc best mode + fusion)
    for key, lbl in (
        ("labram", tr("LaBraM 5.8M (raw_pc)", "LaBraM 5,8M (raw_pc)")),
        ("reve", tr("REVE-base 69.2M (raw_pc)", "REVE-base 69,2M (raw_pc)")),
        ("eegpt", tr("EEGPT 25.3M (raw_pc)", "EEGPT 25,3M (raw_pc)")),
        ("cbramod", tr("CBraMod 5.1M (raw_pc)", "CBraMod 5,1M (raw_pc)")),
        ("sjepa", tr("Signal-JEPA 3.5M (delta_pc)", "Signal-JEPA 3,5M (delta_pc)")),
    ):
        mode = "probe_delta_pc" if key == "sjepa" else "probe_raw_pc"
        m = fms[key][mode]
        g = m["test_new_both"]["gain_ridge_va_vs_ref"]
        add_row(
            lbl,
            "fm",
            m["dev"]["stress_ridge_va"]["stress"]["auc"],
            m["dev"]["stress_ridge_va"]["stress"]["ci95"],
            m["test_new_both"]["stress_ridge_va"]["stress"]["auc"],
            m["test_new_both"]["stress_ridge_va"]["stress"]["ci95"],
            g["gain"],
            g["ci95"],
            ref_tag="eeg",
        )

    # Best FM fusion with Study 07
    lab_f = fms["labram"]["fusion_plus_study07_enriched"]
    g_lf = lab_f["test_new_both"]["gain_ridge_va_vs_ref"]
    add_row(
        tr("LaBraM + S07 Enriched", "LaBraM + É07 Enrichi"),
        "fuse",
        lab_f["dev"]["stress_ridge_va"]["stress"]["auc"],
        lab_f["dev"]["stress_ridge_va"]["stress"]["ci95"],
        lab_f["test_new_both"]["stress_ridge_va"]["stress"]["auc"],
        lab_f["test_new_both"]["stress_ridge_va"]["stress"]["ci95"],
        g_lf["gain"],
        g_lf["ci95"],
    )

    # 3. End-to-End Supervised Neural (3-seed ensembles)
    for key, head, pg_col, lbl in (
        (
            "eegnet",
            "stress_direct",
            "eegnet_ens_stress",
            tr("EEGNet 2.2k (3-seed dir.)", "EEGNet 2,2k (3 seeds dir.)"),
        ),
        (
            "eeg_conformer",
            "stress_direct",
            "eeg_conformer_ens_stress",
            tr("EEG Conformer 237k (dir.)", "EEG Conformer 237k (dir.)"),
        ),
        (
            "multimodal_conformer_scratch",
            "stress_va",
            "multimodal_conformer_scratch_ens_stress_va",
            tr("Multimodal Scratch 0.90M", "Multimodal Scratch 0,90M"),
        ),
    ):
        ens = pm[key]["seed_ensemble"]
        g = pg[pg_col]["test_new_both"]["vs_study07_full"]
        add_row(
            lbl,
            "sup",
            ens["dev"][head]["stress"]["auc"],
            ens["dev"][head]["stress"]["ci95"],
            ens["test_new_both"][head]["stress"]["auc"],
            ens["test_new_both"][head]["stress"]["ci95"],
            g["gain"],
            g["ci95"],
        )

    # 4. HABS-Native Multimodal PhysioJEPA
    j_frz = pm["physio_jepa"]["frozen_seed_ensemble"]
    g_jf = pg["physio_jepa_frozen_ens_stress_va"]["test_new_both"]["vs_study07_full"]
    add_row(
        tr("PhysioJEPA Frozen 15s (3-seed)", "PhysioJEPA Gelé 15s (3 seeds)"),
        "jepa",
        j_frz["dev"]["stress_va"]["stress"]["auc"],
        j_frz["dev"]["stress_va"]["stress"]["ci95"],
        j_frz["test_new_both"]["stress_va"]["stress"]["auc"],
        j_frz["test_new_both"]["stress_va"]["stress"]["ci95"],
        g_jf["gain"],
        g_jf["ci95"],
    )
    j_ft = pm["physio_jepa"]["finetuned_seed_ensemble"]
    g_jft = pg["physio_jepa_ft_ens_stress_va"]["test_new_both"]["vs_study07_full"]
    add_row(
        tr("PhysioJEPA Finetuned 15s (3-seed)", "PhysioJEPA Fine-tuné 15s (3 seeds)"),
        "jepa",
        j_ft["dev"]["stress_va"]["stress"]["auc"],
        j_ft["dev"]["stress_va"]["stress"]["ci95"],
        j_ft["test_new_both"]["stress_va"]["stress"]["auc"],
        j_ft["test_new_both"]["stress_va"]["stress"]["ci95"],
        g_jft["gain"],
        g_jft["ci95"],
    )
    if modwin is not None and "video_0_10s" in modwin["ablations"]["window"]:
        w10 = modwin["ablations"]["window"]["video_0_10s"]["frozen_seed0"]
        add_row(
            tr("PhysioJEPA Frozen [0,10)s (abl.)", "PhysioJEPA Gelé [0,10)s (abl.)"),
            "jepa",
            w10["dev"]["stress_va"]["stress"]["auc"],
            w10["dev"]["stress_va"]["stress"]["ci95"],
            w10["test_new_both"]["stress_va"]["stress"]["auc"],
            w10["test_new_both"]["stress_va"]["stress"]["ci95"],
            None,
            None,
        )
    j_fuse = ji["physio_jepa_plus_study07_enriched"]
    g_jfu = j_fuse["test_new_both"]["gain_ridge_va_vs_ref"]
    add_row(
        tr("PhysioJEPA + S07 Enriched", "PhysioJEPA + É07 Enrichi"),
        "fuse",
        j_fuse["dev"]["stress_ridge_va"]["stress"]["auc"],
        j_fuse["dev"]["stress_ridge_va"]["stress"]["ci95"],
        j_fuse["test_new_both"]["stress_ridge_va"]["stress"]["auc"],
        j_fuse["test_new_both"]["stress_ridge_va"]["stress"]["ci95"],
        g_jfu["gain"],
        g_jfu["ci95"],
    )

    fam_colors = {
        "s07": BLUE,
        "fm": PURPLE,
        "sup": AMBER,
        "jepa": TEAL,
        "fuse": ORANGE,
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.2, 6.4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    y = np.arange(len(rows))[::-1]

    # Reference lines
    ref_nb = s07["enriched_full"]["test_new_both"]["stress_ridge_va"]["stress"]["auc"]
    ax1.axvline(0.50, color=GRAY, ls=":", lw=1.0, zorder=1)
    ax1.axvline(ref_nb, color=BLUE, ls="--", lw=1.0, alpha=0.55, zorder=1)
    ax2.axvline(0.0, color=INK, ls="--", lw=1.0, alpha=0.65, zorder=1)

    for yi, r in zip(y, rows, strict=False):
        c = fam_colors[r["fam"]]
        # Dev (hollow circle, slight upward offset)
        ax1.errorbar(
            r["dev"],
            yi + 0.17,
            xerr=[[r["dev"] - r["dev_ci"][0]], [r["dev_ci"][1] - r["dev"]]],
            fmt="o",
            color=c,
            mfc="white",
            mec=c,
            ms=4.8,
            lw=1.1,
            capsize=2.0,
            zorder=3,
        )
        # Sealed test_new_both (filled square, slight downward offset)
        ax1.errorbar(
            r["nb"],
            yi - 0.17,
            xerr=[[r["nb"] - r["nb_ci"][0]], [r["nb_ci"][1] - r["nb"]]],
            fmt="s",
            color=c,
            mfc=c,
            mec=INK,
            mew=0.45,
            ms=5.2,
            lw=1.35,
            capsize=2.3,
            zorder=4,
        )
        ax1.text(
            r["nb_ci"][1] + 0.004,
            yi - 0.19,
            fnum(r["nb"], 3),
            va="center",
            fontsize=6.9,
            color=INK,
        )

        # Paired gain panel
        if r["gain"] is not None and r["gain_ci"] is not None:
            marker = "D" if r["ref_tag"] == "eeg" else "s"
            ax2.errorbar(
                r["gain"],
                yi,
                xerr=[[r["gain"] - r["gain_ci"][0]], [r["gain_ci"][1] - r["gain"]]],
                fmt=marker,
                color=c,
                mfc=c,
                mec=INK,
                mew=0.45,
                ms=5.0,
                lw=1.25,
                capsize=2.3,
                zorder=4,
            )
            ax2.text(
                r["gain_ci"][1] + 0.005,
                yi,
                fnum(r["gain"], 3, signed=True),
                va="center",
                fontsize=6.9,
                color=MUTED,
                bbox={"fc": "white", "ec": "none", "pad": 0.4, "alpha": 0.85},
            )

    # Subtle group separators
    for sep in (len(rows) - 4.5, len(rows) - 10.5, len(rows) - 13.5):
        ax1.axhline(sep, color=LIGHT, lw=1.1, zorder=0)
        ax2.axhline(sep, color=LIGHT, lw=1.1, zorder=0)

    ax1.set_yticks(y)
    ax1.set_yticklabels([r["label"] for r in rows], fontsize=7.8, color=INK)
    ax1.set_xlim(0.425, 0.725)
    ax2.set_xlim(-0.185, 0.115)
    ax1.set_xlabel(
        tr(
            "Within-person Acute Stress ROC AUC (95% person-bootstrap CI)",
            "AUC ROC intra-sujet — Stress aigu (IC 95 % bootstrap sujets)",
        )
    )
    ax2.set_xlabel(
        tr(
            "Paired AUC gain on sealed test_new_both (95% CI)",
            "Gain apparié d'AUC sur test scellé P&T (IC 95 %)",
        )
    )
    ax1.set_title(
        tr(
            "A. Four-family generalization (dev 3×3 CV vs sealed new people × new themes)",
            "A. Généralisation des 4 familles (dév. 3×3 CV vs test scellé P&T)",
        ),
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )
    ax2.set_title(
        tr(
            "B. Paired gain vs S07 (■ vs Full, ◆ vs EEG-only)",
            "B. Gain apparié vs É07 (■ vs Complet, ◆ vs EEG seul)",
        ),
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )
    apply_decimal_formatter(ax1, "x", 2)
    apply_decimal_formatter(ax2, "x", 2)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=MUTED,
            mfc="white",
            ms=5,
            lw=1,
            label=tr("Dev 3×3 CV (n=208)", "Dév. 3×3 CV (n=208)"),
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=MUTED,
            mfc=MUTED,
            ms=5,
            lw=1,
            label=tr("Sealed test_new_both (n=30)", "Test scellé P&T (n=30)"),
        ),
        Line2D(
            [0], [0], color=BLUE, lw=3, label=tr("Study 07 linear", "Étude 07 linéaire")
        ),
        Line2D(
            [0],
            [0],
            color=PURPLE,
            lw=3,
            label=tr("Pretrained EEG FM", "FM EEG préentraîné"),
        ),
        Line2D(
            [0],
            [0],
            color=AMBER,
            lw=3,
            label=tr("Supervised neural", "Neuronal supervisé"),
        ),
        Line2D([0], [0], color=TEAL, lw=3, label="PhysioJEPA (SSL)"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            lw=3,
            label=tr("FM/JEPA + S07 fusion", "Fusion FM/JEPA + É07"),
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.6,
        bbox_to_anchor=(0.5, -0.045),
    )
    fig.suptitle(
        tr(
            "Study 08 Benchmark: Self-Supervised & Neural Models Do Not Beat Study 07 Linear Baselines on Double-Unseen Stress",
            "Benchmark Étude 08 : les modèles SSL et neuronaux ne surpassent pas les baselines linéaires É07 sur le stress (P&T inédits)",
        ),
        fontsize=9.8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig1_benchmark_comparison")


def make_fig2(fm_s07: dict, prim: dict, fuse: dict, strat: dict | None) -> None:
    """Fig 2: Double dissociation, Identity Shortcut, PE confound & Temporal Shuffle."""
    s07_dg = fm_s07["study07_baselines"]["study07_handcrafted_diagnostics"]
    fms = fm_s07["foundation_models"]
    j_dg = prim["primary_models"]["physio_jepa"]["diagnostics"]
    j_cm = fuse["jepa_integration"]["physio_jepa_cross_modal_eeg_to_body"]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    axA, axB, axC, axD = axes.flat

    # --- Panel A: Cross-modal brain -> body forecasting (EEG [0,5) -> Body [5,10)) ---
    cm_models = [
        (
            tr("S07 EEG\n(0 h)", "É07 EEG\n(0 h)"),
            s07_dg["cross_modal_eeg_early_to_late_body"],
        ),
        (tr("PhysioJEPA\n(53 h)", "PhysioJEPA\n(53 h)"), j_cm),
        (
            tr("Signal-JEPA\n(Lee2019)", "Signal-JEPA\n(Lee2019)"),
            fms["sjepa"]["diagnostics"]["cross_modal_eeg_early_to_late_body"],
        ),
        (
            tr("EEGPT\n(mixed)", "EEGPT\n(mixte)"),
            fms["eegpt"]["diagnostics"]["cross_modal_eeg_early_to_late_body"],
        ),
        (
            tr("LaBraM\n(2.5k h)", "LaBraM\n(2,5k h)"),
            fms["labram"]["diagnostics"]["cross_modal_eeg_early_to_late_body"],
        ),
        (
            tr("CBraMod\n(9k h)", "CBraMod\n(9k h)"),
            fms["cbramod"]["diagnostics"]["cross_modal_eeg_early_to_late_body"],
        ),
        (
            tr("REVE-base\n(60k h)", "REVE-base\n(60k h)"),
            fms["reve"]["diagnostics"]["cross_modal_eeg_early_to_late_body"],
        ),
    ]
    x = np.arange(len(cm_models))
    w = 0.25
    for idx, (tkey, col, lbl) in enumerate(
        (
            ("hr_late_chg", RED, tr("Heart rate (HR)", "Fréq. cardiaque (FC)")),
            ("sc_late_chg", ORANGE, tr("Skin cond. (SC)", "Cond. cutanée (SC)")),
            ("resp_amp_late_chg", TEAL, tr("Respiration amp.", "Ampl. respiratoire")),
        )
    ):
        vals = [m[tkey]["within_person_r"] for _, m in cm_models]
        los = [m[tkey]["within_person_r"] - m[tkey]["ci95"][0] for _, m in cm_models]
        his = [m[tkey]["ci95"][1] - m[tkey]["within_person_r"] for _, m in cm_models]
        axA.bar(
            x + (idx - 1) * w,
            vals,
            width=w,
            color=col,
            alpha=0.88,
            label=lbl,
            yerr=[los, his],
            error_kw={"lw": 0.9, "capsize": 1.8, "ecolor": INK},
        )
    axA.axhline(0.0, color=MUTED, lw=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels([n for n, _ in cm_models], fontsize=7.0)
    axA.set_ylabel(
        tr(
            "Within-person Pearson r (unseen people)",
            "r de Pearson intra-sujet (sujets inédits)",
        )
    )
    axA.set_title(
        tr(
            "A. Cross-modal forecasting: EEG [0,5)s → Body [5,10)s",
            "A. Prévision inter-modale : EEG [0,5)s → Corps [5,10)s",
        ),
        loc="left",
        fontsize=8.6,
        fontweight="bold",
    )
    axA.legend(frameon=False, fontsize=7.1, loc="upper left")
    apply_decimal_formatter(axA, "y", 2)

    # --- Panel B: Identity Shortcut (R^2_subject & Cross-Day Top-1 ID, raw vs delta) ---
    id_models = [
        (tr("S07", "É07"), s07_dg),
        ("CBraMod", fms["cbramod"]["diagnostics"]),
        ("LaBraM", fms["labram"]["diagnostics"]),
        ("EEGPT", fms["eegpt"]["diagnostics"]),
        ("REVE", fms["reve"]["diagnostics"]),
        ("PhysioJEPA", j_dg),
    ]
    if strat is not None and "jepa_fixed_raw_norm" in strat["ablations"]["strategy"]:
        f_dg = strat["ablations"]["strategy"]["jepa_fixed_raw_norm"].get("diagnostics")
        if f_dg:
            id_models.append((tr("JEPA\n(fixed)", "JEPA\n(fixe)"), f_dg))

    xb = np.arange(len(id_models))
    wb = 0.21
    r2_raw = [
        m["subject_identity_raw_vid"]["subject_variance_r2"] for _, m in id_models
    ]
    r2_del = [m["subject_identity_delta"]["subject_variance_r2"] for _, m in id_models]
    cd_raw = [
        m["subject_identity_raw_vid"]["cross_day_id"]["top1"] for _, m in id_models
    ]
    cd_del = [m["subject_identity_delta"]["cross_day_id"]["top1"] for _, m in id_models]

    axB.bar(
        xb - 1.5 * wb,
        r2_raw,
        width=wb,
        color=BLUE,
        label=tr("$R^2_{\\mathrm{subj}}$ raw", "$R^2_{\\mathrm{sujet}}$ brut"),
    )
    axB.bar(
        xb - 0.5 * wb,
        r2_del,
        width=wb,
        color=BLUE,
        alpha=0.38,
        label=tr(
            "$R^2_{\\mathrm{subj}}$ Δ (vid−ref)", "$R^2_{\\mathrm{sujet}}$ Δ (vid−réf)"
        ),
    )
    axB.bar(
        xb + 0.5 * wb,
        cd_raw,
        width=wb,
        color=PURPLE,
        label=tr("Cross-day ID raw", "ID inter-jours brut"),
    )
    axB.bar(
        xb + 1.5 * wb,
        cd_del,
        width=wb,
        color=PURPLE,
        alpha=0.38,
        label=tr("Cross-day ID Δ", "ID inter-jours Δ"),
    )
    axB.axhline(
        1 / 48, color=GRAY, ls=":", lw=1.0, label=tr("Chance (2.1%)", "Hasard (2,1 %)")
    )
    axB.set_xticks(xb)
    axB.set_xticklabels([n for n, _ in id_models], fontsize=7.2)
    axB.set_ylim(0, 0.68)
    axB.set_ylabel(
        tr(
            "Variance fraction / Top-1 accuracy (n=48)",
            "Part de variance / Précision Top-1 (n=48)",
        )
    )
    axB.set_title(
        tr(
            "B. Identity Shortcut: static trait fingerprint vs baseline Δ",
            "B. Raccourci identité : empreinte statique vs Δ référence",
        ),
        loc="left",
        fontsize=8.6,
        fontweight="bold",
    )
    axB.legend(frameon=False, fontsize=6.7, ncol=2, loc="upper left")
    apply_decimal_formatter(axB, "y", 2)

    # --- Panel C: Positional-Encoding Confound vs True Window-Isolated Dynamics ---
    dyn_entries = [
        (tr("S07\nhandcrafted", "É07\nmanuel"), s07_dg["temporal_dynamics"], False),
        (
            tr("EEGPT\n(isolated)", "EEGPT\n(isolé)"),
            fms["eegpt"]["diagnostics"]["temporal_dynamics"],
            False,
        ),
        (
            tr("PhysioJEPA\n(iso 3s)", "PhysioJEPA\n(isolé 3s)"),
            j_dg["temporal_dynamics_iso_3s"],
            False,
        ),
        (
            tr("PhysioJEPA\n(1-pass PE)", "PhysioJEPA\n(1-passe PE)"),
            j_dg["temporal_dynamics_single_pass"],
            True,
        ),
        (
            tr("REVE/CBra\n(1-pass PE)", "REVE/CBra\n(1-passe PE)"),
            fms["reve"]["diagnostics"]["temporal_dynamics"],
            True,
        ),
    ]
    xc = np.arange(len(dyn_entries))
    wc = 0.25
    for idx, (dk, col, lbl) in enumerate(
        (
            ("pre_vs_video", BLUE, tr("Pre vs Video", "Pré vs Vidéo")),
            ("early_vs_late", ORANGE, tr("Early vs Late", "Précoce vs Tardif")),
            ("arrow_of_time", TEAL, tr("Arrow of time", "Flèche du temps")),
        )
    ):
        vals = [e[1][dk]["within_person_auc"] for e in dyn_entries]
        hatch = [("//" if e[2] else None) for e in dyn_entries]
        bars = axC.bar(
            xc + (idx - 1) * wc, vals, width=wc, color=col, alpha=0.88, label=lbl
        )
        for b, h in zip(bars, hatch, strict=False):
            if h:
                b.set_hatch(h)
                b.set_alpha(0.45)
                b.set_edgecolor(INK)
                b.set_linewidth(0.5)
    axC.axhline(0.50, color=GRAY, ls=":", lw=1.0)
    axC.axvline(2.5, color=RED, ls="--", lw=1.0, alpha=0.7)
    axC.text(
        3.5,
        1.03,
        tr("PE shortcut (hatched)", "Raccourci PE (hachuré)"),
        ha="center",
        fontsize=7.0,
        color=RED,
        fontweight="bold",
    )
    axC.set_xticks(xc)
    axC.set_xticklabels([e[0] for e in dyn_entries], fontsize=7.0)
    axC.set_ylim(0.45, 1.15)
    axC.set_ylabel(
        tr(
            "Within-person AUC (unseen people, n=86)",
            "AUC intra-sujet (sujets inédits, n=86)",
        )
    )
    axC.set_title(
        tr(
            "C. Directed temporal dynamics: PE-free vs single-pass shortcut",
            "C. Dynamiques temporelles orientées : sans PE vs raccourci 1-passe",
        ),
        loc="left",
        fontsize=8.6,
        fontweight="bold",
    )
    axC.legend(frameon=False, fontsize=6.9, ncol=3, loc="upper left")
    apply_decimal_formatter(axC, "y", 2)

    # --- Panel D: Temporal-Shuffle Collapse & Affective Rho vs Scratch ---
    shuf = j_dg["temporal_shuffle_test"]
    j_s0 = prim["primary_models"]["physio_jepa"]["frozen_seed0"]
    sc_ens = prim["primary_models"]["multimodal_conformer_scratch"]["seed_ensemble"]
    jf_ens = prim["primary_models"]["physio_jepa"]["frozen_seed_ensemble"]

    cats = [
        tr("Stress NB\n(ordered)", "Stress P&T\n(ordonné)"),
        tr("Stress NB\n(shuffled)", "Stress P&T\n(mélangé)"),
        tr("Stress NP\n(ordered)", "Stress Pers.\n(ordonné)"),
        tr("Stress NP\n(shuffled)", "Stress Pers.\n(mélangé)"),
    ]
    aucs = [
        j_s0["test_new_both"]["stress_va"]["stress"]["auc"],
        shuf["test_new_both"]["stress_va_shuffled"]["auc"],
        j_s0["test_new_person"]["stress_va"]["stress"]["auc"],
        shuf["test_new_person"]["stress_va_shuffled"]["auc"],
    ]
    cis = [
        j_s0["test_new_both"]["stress_va"]["stress"]["ci95"],
        shuf["test_new_both"]["stress_va_shuffled"]["ci95"],
        j_s0["test_new_person"]["stress_va"]["stress"]["ci95"],
        shuf["test_new_person"]["stress_va_shuffled"]["ci95"],
    ]
    xd = np.arange(len(cats))
    cols_d = [TEAL, RED, TEAL, RED]
    axD.bar(
        xd,
        aucs,
        width=0.52,
        color=cols_d,
        alpha=0.85,
        yerr=[
            [a - c[0] for a, c in zip(aucs, cis, strict=False)],
            [c[1] - a for a, c in zip(aucs, cis, strict=False)],
        ],
        error_kw={"lw": 1.1, "capsize": 2.5, "ecolor": INK},
    )
    for xi, a, c in zip(xd, aucs, cis, strict=False):
        axD.text(
            xi, c[1] + 0.008, fnum(a, 3), ha="center", fontsize=7.3, fontweight="bold"
        )
    axD.axhline(0.50, color=INK, ls="--", lw=0.9, alpha=0.65)
    axD.set_xticks(xd)
    axD.set_xticklabels(cats, fontsize=7.3)
    axD.set_ylim(0.41, 0.655)
    axD.set_ylabel(
        tr("PhysioJEPA within-person Stress AUC", "AUC Stress intra-sujet (PhysioJEPA)")
    )
    axD.set_title(
        tr(
            "D. Temporal-shuffle ablation collapses PhysioJEPA to chance",
            "D. La permutation temporelle ramène PhysioJEPA au hasard pur",
        ),
        loc="left",
        fontsize=8.6,
        fontweight="bold",
    )
    # Inset annotation comparing affective rho vs matched scratch
    note = tr(
        f"Matched-backbone affect ρ (3-seed ens.):\n"
        f"Scratch: dev {fnum(sc_ens['dev']['affect']['mean_rho'], 3)}, NB {fnum(sc_ens['test_new_both']['affect']['mean_rho'], 3)}\n"
        f"PhysioJEPA: dev {fnum(jf_ens['dev']['affect']['mean_rho'], 3)}, NB {fnum(jf_ens['test_new_both']['affect']['mean_rho'], 3)}",
        f"ρ affect à backbone identique (ens. 3 seeds) :\n"
        f"Scratch : dév. {fnum(sc_ens['dev']['affect']['mean_rho'], 3)}, P&T {fnum(sc_ens['test_new_both']['affect']['mean_rho'], 3)}\n"
        f"PhysioJEPA : dév. {fnum(jf_ens['dev']['affect']['mean_rho'], 3)}, P&T {fnum(jf_ens['test_new_both']['affect']['mean_rho'], 3)}",
    )
    axD.text(
        0.97,
        0.95,
        note,
        transform=axD.transAxes,
        ha="right",
        va="top",
        fontsize=6.9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#f6f6f2", "ec": GRAY, "lw": 0.6},
    )
    apply_decimal_formatter(axD, "y", 2)

    fig.suptitle(
        tr(
            "Representation Diagnostics: Double Dissociation, Identity Shortcut, PE Confound & True Temporal Reliance",
            "Diagnostics des représentations : double dissociation, raccourci identité, biais PE et dynamique temporelle réelle",
        ),
        fontsize=9.8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig2_double_dissociation_geometry")


def make_fig3(prim: dict, modwin: dict, strat: dict) -> None:
    """Fig 3: Four-panel PhysioJEPA systematic ablations (modality, window, strategy, scale)."""
    j0 = prim["primary_models"]["physio_jepa"]["frozen_seed0"]
    sc0 = prim["primary_models"]["multimodal_conformer_scratch"]["seed0"]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    axA, axB, axC, axD = axes.flat

    def _bar_dev_nb(
        ax: plt.Axes, entries: list[tuple[str, dict]], title: str, ylabel: bool = True
    ) -> None:
        xs = np.arange(len(entries))
        w = 0.34
        dev_v = [e[1]["dev"]["stress_va"]["stress"]["auc"] for e in entries]
        dev_lo = [
            e[1]["dev"]["stress_va"]["stress"]["auc"]
            - e[1]["dev"]["stress_va"]["stress"]["ci95"][0]
            for e in entries
        ]
        dev_hi = [
            e[1]["dev"]["stress_va"]["stress"]["ci95"][1]
            - e[1]["dev"]["stress_va"]["stress"]["auc"]
            for e in entries
        ]
        nb_v = [e[1]["test_new_both"]["stress_va"]["stress"]["auc"] for e in entries]
        nb_lo = [
            e[1]["test_new_both"]["stress_va"]["stress"]["auc"]
            - e[1]["test_new_both"]["stress_va"]["stress"]["ci95"][0]
            for e in entries
        ]
        nb_hi = [
            e[1]["test_new_both"]["stress_va"]["stress"]["ci95"][1]
            - e[1]["test_new_both"]["stress_va"]["stress"]["auc"]
            for e in entries
        ]

        ax.bar(
            xs - w / 2,
            dev_v,
            width=w,
            color=TEAL,
            alpha=0.9,
            label=tr("Dev 3×3 CV (n=208)", "Dév. 3×3 CV (n=208)"),
            yerr=[dev_lo, dev_hi],
            error_kw={"lw": 0.9, "capsize": 2.0, "ecolor": INK},
        )
        ax.bar(
            xs + w / 2,
            nb_v,
            width=w,
            color=BLUE,
            alpha=0.82,
            label=tr("Sealed test P&T (n=30)", "Test scellé P&T (n=30)"),
            yerr=[nb_lo, nb_hi],
            error_kw={"lw": 0.9, "capsize": 2.0, "ecolor": INK},
        )
        for xi, dv in zip(xs, dev_v, strict=False):
            ax.text(
                xi - w / 2,
                0.456,
                fnum(dv, 3),
                ha="center",
                fontsize=6.5,
                color="white",
                fontweight="bold",
                rotation=90,
                va="bottom",
            )
        ax.axhline(0.50, color=GRAY, ls=":", lw=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([e[0] for e in entries], fontsize=7.1)
        ax.set_ylim(0.45, 0.665)
        if ylabel:
            ax.set_ylabel(
                tr(
                    "Within-person Stress AUC (95% CI)",
                    "AUC Stress intra-sujet (IC 95 %)",
                )
            )
        ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold")
        apply_decimal_formatter(ax, "y", 2)

    # Panel A: Modality
    ma = modwin["ablations"]["modality"]
    mod_entries = [
        (tr("All 4\n(ref)", "4 mod.\n(réf.)"), j0),
        ("EEG+EDA", ma["eeg_eda"]["frozen_seed0"]),
        (tr("Body\nonly", "Corps\nseul"), ma["body_only"]["frozen_seed0"]),
        (tr("EEG\nonly", "EEG\nseul"), ma["eeg_only"]["frozen_seed0"]),
        ("EEG+Resp", ma["eeg_resp"]["frozen_seed0"]),
        ("EEG+ECG", ma["eeg_ecg"]["frozen_seed0"]),
    ]
    _bar_dev_nb(
        axA,
        mod_entries,
        tr(
            "A. Modality ablation (frozen PhysioJEPA, seed 0)",
            "A. Ablation par modalité (PhysioJEPA gelé, seed 0)",
        ),
    )
    axA.legend(frameon=False, fontsize=7.0, loc="upper right")

    # Panel B: Temporal Window
    wa = modwin["ablations"]["window"]
    win_entries = [
        (tr("[0, 5)s\nearly", "[0, 5)s\nprécoce"), wa["early_0_5s"]["frozen_seed0"]),
        (tr("[0, 10)s\nvideo*", "[0, 10)s\nvidéo*"), wa["video_0_10s"]["frozen_seed0"]),
        (
            tr("[-3, 10)s\npre+vid (Δ)", "[-3, 10)s\npré+vid (Δ)"),
            wa["pre_video_m3_10s"]["frozen_seed0"],
        ),
        (tr("[-3, 12)s\nfull 15s (Δ)", "[-3, 12)s\n15s (Δ)"), j0),
    ]
    _bar_dev_nb(
        axB,
        win_entries,
        tr(
            "B. Temporal window ablation (no ref-sub noise* vs Δ)",
            "B. Ablation fenêtre temporelle (sans bruit réf.* vs Δ)",
        ),
        ylabel=False,
    )

    # Panel C: Strategy ablation
    sa = strat["ablations"]["strategy"]
    strat_entries = [
        (tr("JEPA\nCrossMod", "JEPA\nCrossMod"), j0),
        (tr("JEPA\nFuture", "JEPA\nFutur"), sa["jepa_future"]["frozen_seed0"]),
        (tr("JEPA\nMasked", "JEPA\nMasqué"), sa["jepa_masked"]["frozen_seed0"]),
        (tr("MAE\nRecon", "MAE\nRecon"), sa["mae_recon"]["frozen_seed0"]),
        (
            tr("JEPA\nFixedNorm", "JEPA\nNormFixe"),
            sa["jepa_fixed_raw_norm"]["frozen_seed0"],
        ),
        (tr("JEPA\n+Rest", "JEPA\n+Repos"), sa["jepa_plus_rest"]["frozen_seed0"]),
        (tr("Scratch\n(no SSL)", "Scratch\n(sans SSL)"), sc0),
    ]
    _bar_dev_nb(
        axC,
        strat_entries,
        tr(
            "C. Pretraining strategy & resting-state ablation (+9.5 h)",
            "C. Ablation stratégie SSL et données de repos (+9,5 h)",
        ),
    )

    # Panel D: Data-size scaling curve (10%, 25%, 50%, 100%)
    ds = strat["ablations"]["data_scale"]
    fracs = [10, 25, 50, 100]
    j_dev_auc = [
        ds["pct_10"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["stress_va"][
            "stress"
        ]["auc"],
        ds["pct_25"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["stress_va"][
            "stress"
        ]["auc"],
        ds["pct_50"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["stress_va"][
            "stress"
        ]["auc"],
        j0["dev"]["stress_va"]["stress"]["auc"],
    ]
    j_dev_ci = [
        ds["pct_10"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["stress_va"][
            "stress"
        ]["ci95"],
        ds["pct_25"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["stress_va"][
            "stress"
        ]["ci95"],
        ds["pct_50"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["stress_va"][
            "stress"
        ]["ci95"],
        j0["dev"]["stress_va"]["stress"]["ci95"],
    ]
    s_dev_auc = [
        ds["pct_10"]["multimodal_scratch"]["seed0"]["dev"]["stress_va"]["stress"][
            "auc"
        ],
        ds["pct_25"]["multimodal_scratch"]["seed0"]["dev"]["stress_va"]["stress"][
            "auc"
        ],
        ds["pct_50"]["multimodal_scratch"]["seed0"]["dev"]["stress_va"]["stress"][
            "auc"
        ],
        sc0["dev"]["stress_va"]["stress"]["auc"],
    ]
    s_dev_ci = [
        ds["pct_10"]["multimodal_scratch"]["seed0"]["dev"]["stress_va"]["stress"][
            "ci95"
        ],
        ds["pct_25"]["multimodal_scratch"]["seed0"]["dev"]["stress_va"]["stress"][
            "ci95"
        ],
        ds["pct_50"]["multimodal_scratch"]["seed0"]["dev"]["stress_va"]["stress"][
            "ci95"
        ],
        sc0["dev"]["stress_va"]["stress"]["ci95"],
    ]
    j_rho = [
        ds["pct_10"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["affect"]["mean_rho"],
        ds["pct_25"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["affect"]["mean_rho"],
        ds["pct_50"]["physio_jepa_frozen"]["frozen_seed0"]["dev"]["affect"]["mean_rho"],
        j0["dev"]["affect"]["mean_rho"],
    ]
    s_rho = [
        ds["pct_10"]["multimodal_scratch"]["seed0"]["dev"]["affect"]["mean_rho"],
        ds["pct_25"]["multimodal_scratch"]["seed0"]["dev"]["affect"]["mean_rho"],
        ds["pct_50"]["multimodal_scratch"]["seed0"]["dev"]["affect"]["mean_rho"],
        sc0["dev"]["affect"]["mean_rho"],
    ]

    axD.errorbar(
        fracs,
        j_dev_auc,
        yerr=[
            [a - c[0] for a, c in zip(j_dev_auc, j_dev_ci, strict=False)],
            [c[1] - a for a, c in zip(j_dev_auc, j_dev_ci, strict=False)],
        ],
        fmt="-o",
        color=TEAL,
        lw=2.0,
        ms=5.5,
        capsize=3.0,
        label=tr("PhysioJEPA Frozen (AUC)", "PhysioJEPA Gelé (AUC)"),
    )
    axD.errorbar(
        fracs,
        s_dev_auc,
        yerr=[
            [a - c[0] for a, c in zip(s_dev_auc, s_dev_ci, strict=False)],
            [c[1] - a for a, c in zip(s_dev_auc, s_dev_ci, strict=False)],
        ],
        fmt="--s",
        color=AMBER,
        lw=1.8,
        ms=5.2,
        capsize=3.0,
        label=tr("Multimodal Scratch (AUC)", "Multimodal Scratch (AUC)"),
    )
    axD.axhline(0.50, color=GRAY, ls=":", lw=1.0)
    axD.set_xticks(fracs)
    axD.set_xticklabels(
        ["10%\n(23)", "25%\n(57)", "50%\n(114)", "100%\n(228)"], fontsize=7.4
    )
    axD.set_xlabel(
        tr("Dev participants used (% / n)", "Participants dév. utilisés (% / n)")
    )
    axD.set_ylabel(tr("Dev 3×3 CV Stress AUC", "AUC Stress dév. 3×3 CV"))
    axD.set_ylim(0.485, 0.59)
    axD.set_title(
        tr(
            "D. Data-size scaling: PhysioJEPA vs matched Scratch",
            "D. Courbe d'échelle : PhysioJEPA vs contrôle Scratch",
        ),
        loc="left",
        fontsize=8.6,
        fontweight="bold",
    )
    axD2 = axD.twinx()
    axD2.spines["top"].set_visible(False)
    axD2.plot(
        fracs,
        j_rho,
        ":^",
        color=BLUE,
        lw=1.4,
        ms=4.5,
        label=tr("PhysioJEPA mean ρ", "PhysioJEPA ρ moyen"),
    )
    axD2.plot(
        fracs,
        s_rho,
        ":v",
        color=RED,
        lw=1.4,
        ms=4.5,
        label=tr("Scratch mean ρ", "Scratch ρ moyen"),
    )
    axD2.set_ylabel(tr("Dev V&A mean Spearman ρ", "ρ moyen V&A (dév.)"), color=BLUE)
    axD2.set_ylim(0.0, 0.095)
    h1, l1 = axD.get_legend_handles_labels()
    h2, l2 = axD2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.7, loc="upper left")
    apply_decimal_formatter(axD, "y", 2)
    apply_decimal_formatter(axD2, "y", 2)

    fig.suptitle(
        tr(
            "PhysioJEPA Systematic Ablations: Modality, Window, Pretraining Strategy & Participant Scaling",
            "Ablations systématiques de PhysioJEPA : modalités, fenêtres, stratégies SSL et échelle de données",
        ),
        fontsize=9.8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig3_jepa_ablations")


def make_fig4(fm_s07: dict, fm_ext: dict, prim: dict, fuse: dict) -> None:
    """Fig 4: Compute / Parameter / Pretraining-Scale Pareto Frontier."""
    s07 = fm_s07["study07_baselines"]
    fms = fm_s07["foundation_models"]
    pm = prim["primary_models"]
    j_cm = fuse["jepa_integration"]["physio_jepa_cross_modal_eeg_to_body"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # --- Panel A: Parameter scale (log) vs Sealed Test AUC (bubble = H100 GPU-min) ---
    pts = [
        (
            tr("S07 Enriched\n(0 GPU)", "É07 Enrichi\n(0 GPU)"),
            448,
            s07["enriched_full"]["test_new_both"]["stress_ridge_va"]["stress"]["auc"],
            0.1,
            BLUE,
            (8, 5),
        ),
        (
            tr("S07 Seq+Enr\n(0 GPU)", "É07 Séq+Enr\n(0 GPU)"),
            1068,
            s07["sequence_plus_enriched"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
            0.1,
            BLUE,
            (8, -12),
        ),
        (
            tr("EEGNet\n(2.2k)", "EEGNet\n(2,2k)"),
            2219,
            pm["eegnet"]["seed_ensemble"]["test_new_both"]["stress_direct"]["stress"][
                "auc"
            ],
            pm["eegnet"]["total_gpu_seconds"] / 60,
            AMBER,
            (7, 5),
        ),
        (
            "EEGConformer\n(237k)",
            236987,
            pm["eeg_conformer"]["seed_ensemble"]["test_new_both"]["stress_direct"][
                "stress"
            ]["auc"],
            pm["eeg_conformer"]["total_gpu_seconds"] / 60,
            AMBER,
            (-12, 10),
        ),
        (
            tr("Scratch\n(0.90M)", "Scratch\n(0,90M)"),
            904832,
            pm["multimodal_conformer_scratch"]["seed_ensemble"]["test_new_both"][
                "stress_va"
            ]["stress"]["auc"],
            pm["multimodal_conformer_scratch"]["total_gpu_seconds"] / 60,
            AMBER,
            (-30, -17),
        ),
        (
            tr("PhysioJEPA\n(1.53M)", "PhysioJEPA\n(1,53M)"),
            1526083,
            pm["physio_jepa"]["finetuned_seed_ensemble"]["test_new_both"]["stress_va"][
                "stress"
            ]["auc"],
            pm["physio_jepa"]["total_gpu_seconds"] / 60,
            TEAL,
            (8, -4),
        ),
        (
            tr("Signal-JEPA\n(3.5M)", "Signal-JEPA\n(3,5M)"),
            fm_ext["models"]["sjepa"]["n_params"],
            fms["sjepa"]["probe_delta_pc"]["test_new_both"]["stress_ridge_va"][
                "stress"
            ]["auc"],
            fm_ext["models"]["sjepa"]["runtime_s"] / 60,
            PURPLE,
            (-26, 8),
        ),
        (
            tr("CBraMod\n(5.1M)", "CBraMod\n(5,1M)"),
            fm_ext["models"]["cbramod"]["n_params"],
            fms["cbramod"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"][
                "stress"
            ]["auc"],
            fm_ext["models"]["cbramod"]["runtime_s"] / 60,
            PURPLE,
            (-34, -15),
        ),
        (
            tr("LaBraM\n(5.8M)", "LaBraM\n(5,8M)"),
            fm_ext["models"]["labram"]["n_params"],
            fms["labram"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
            fm_ext["models"]["labram"]["runtime_s"] / 60,
            PURPLE,
            (6, 4),
        ),
        (
            tr("EEGPT\n(25.3M)", "EEGPT\n(25,3M)"),
            fm_ext["models"]["eegpt"]["n_params"],
            fms["eegpt"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
            fm_ext["models"]["eegpt"]["runtime_s"] / 60,
            PURPLE,
            (-15, -16),
        ),
        (
            tr("REVE-base\n(69.2M)", "REVE-base\n(69,2M)"),
            fm_ext["models"]["reve"]["n_params"],
            fms["reve"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
            fm_ext["models"]["reve"]["runtime_s"] / 60,
            PURPLE,
            (-46, 8),
        ),
    ]
    ax1.axhline(0.50, color=GRAY, ls=":", lw=1.0)
    ax1.axhline(
        s07["enriched_full"]["test_new_both"]["stress_ridge_va"]["stress"]["auc"],
        color=BLUE,
        ls="--",
        lw=1.0,
        alpha=0.5,
    )
    for lbl, npar, auc, gmin, col, ofs in pts:
        sz = 42 + 12 * np.sqrt(max(gmin, 0.1))
        ax1.scatter(
            npar,
            auc,
            s=sz,
            color=col,
            edgecolors=INK,
            linewidth=0.7,
            alpha=0.9,
            zorder=4,
        )
        ax1.annotate(
            lbl,
            (npar, auc),
            xytext=ofs,
            textcoords="offset points",
            fontsize=6.8,
            color=INK,
        )
    ax1.set_xscale("log")
    ax1.set_xlim(250, 1.8e8)
    ax1.set_ylim(0.475, 0.650)
    ax1.set_xlabel(
        tr(
            "Model parameters (log scale; size ∝ H100 GPU-min)",
            "Paramètres du modèle (éch. log ; taille ∝ min GPU H100)",
        )
    )
    ax1.set_ylabel(
        tr("Sealed test_new_both Stress ROC AUC", "AUC Stress — test scellé P&T")
    )
    ax1.set_title(
        tr(
            "A. Complexity vs subjective stress generalization",
            "A. Complexité vs généralisation du stress subjectif",
        ),
        loc="left",
        fontsize=8.8,
        fontweight="bold",
    )
    apply_decimal_formatter(ax1, "y", 2)

    # --- Panel B: Pretraining EEG hours vs Brain->Body mean r AND Subjective Stress AUC ---
    def _mean_bb(cm: dict) -> float:
        return float(
            np.mean(
                [
                    cm["hr_late_chg"]["within_person_r"],
                    cm["sc_late_chg"]["within_person_r"],
                    cm["resp_amp_late_chg"]["within_person_r"],
                ]
            )
        )

    scale_pts = [
        (
            tr("S07\n(0 h)", "É07\n(0 h)"),
            1.0,
            _mean_bb(
                s07["study07_handcrafted_diagnostics"][
                    "cross_modal_eeg_early_to_late_body"
                ]
            ),
            s07["enriched_eeg_only"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
        ),
        (
            tr("PhysioJEPA\n(53 h)", "PhysioJEPA\n(53 h)"),
            53.0,
            _mean_bb(j_cm),
            pm["physio_jepa"]["frozen_seed_ensemble"]["test_new_both"]["stress_va"][
                "stress"
            ]["auc"],
        ),
        (
            tr("LaBraM\n(2.5k h)", "LaBraM\n(2,5k h)"),
            2500.0,
            _mean_bb(
                fms["labram"]["diagnostics"]["cross_modal_eeg_early_to_late_body"]
            ),
            fms["labram"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
        ),
        (
            tr("CBraMod\n(9k h)", "CBraMod\n(9k h)"),
            9000.0,
            _mean_bb(
                fms["cbramod"]["diagnostics"]["cross_modal_eeg_early_to_late_body"]
            ),
            fms["cbramod"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"][
                "stress"
            ]["auc"],
        ),
        (
            tr("REVE-base\n(60k h)", "REVE-base\n(60k h)"),
            60000.0,
            _mean_bb(fms["reve"]["diagnostics"]["cross_modal_eeg_early_to_late_body"]),
            fms["reve"]["probe_raw_pc"]["test_new_both"]["stress_ridge_va"]["stress"][
                "auc"
            ],
        ),
    ]
    hrs = [p[1] for p in scale_pts]
    bb = [p[2] for p in scale_pts]
    st = [p[3] for p in scale_pts]

    l1 = ax2.plot(
        hrs,
        bb,
        "-o",
        color=TEAL,
        lw=2.2,
        ms=6.2,
        label=tr(
            "Brain→Body forecast mean r (HR/SC/Resp)",
            "Prévision Cerveau→Corps r moyen (FC/SC/Resp)",
        ),
    )
    bb_offsets = [(0, 8), (0, 8), (-4, 9), (26, -15), (-4, 9)]
    for (nm, h, b, _), ofs in zip(scale_pts, bb_offsets, strict=False):
        ax2.annotate(
            nm,
            (h, b),
            xytext=ofs,
            textcoords="offset points",
            ha="center",
            fontsize=6.7,
            color=TEAL,
            fontweight="bold",
        )
    ax2.set_xscale("log")
    ax2.set_xlim(0.5, 1.5e5)
    ax2.set_ylim(0.06, 0.24)
    ax2.set_xlabel(
        tr(
            "Unlabeled EEG pretraining corpus (hours, log scale; S07 shown at 1 h)",
            "Corpus de préentraînement EEG (heures, éch. log ; É07 à 1 h)",
        )
    )
    ax2.set_ylabel(
        tr(
            "Mean Brain→Body forecasting Pearson r",
            "r de Pearson moyen (Cerveau→Corps)",
        ),
        color=TEAL,
    )

    ax2r = ax2.twinx()
    ax2r.spines["top"].set_visible(False)
    l2 = ax2r.plot(
        hrs,
        st,
        "--s",
        color=RED,
        lw=1.9,
        ms=5.5,
        label=tr(
            "Subjective stress AUC (sealed test P&T)",
            "AUC stress subjectif (test scellé P&T)",
        ),
    )
    ax2r.axhline(0.50, color=GRAY, ls=":", lw=0.9)
    ax2r.set_ylim(0.48, 0.65)
    ax2r.set_ylabel(
        tr(
            "Subjective stress ROC AUC (test_new_both)",
            "AUC ROC stress subjectif (test P&T)",
        ),
        color=RED,
    )

    ax2.set_title(
        tr(
            "B. Pretraining scale helps physiology, not subjective stress",
            "B. L'échelle SSL aide la physiologie, pas le stress subjectif",
        ),
        loc="left",
        fontsize=8.8,
        fontweight="bold",
    )
    lines = l1 + l2
    ax2.legend(
        lines,
        [ln.get_label() for ln in lines],
        frameon=False,
        fontsize=7.0,
        loc="upper left",
    )
    apply_decimal_formatter(ax2, "y", 2)
    apply_decimal_formatter(ax2r, "y", 2)

    fig.suptitle(
        tr(
            "Compute & Scale Trade-Off: Strong Scaling on Physiological Forecasting vs Flat Ceiling on Subjective Appraisal",
            "Coût & passage à l'échelle : fort gain sur la prévision physiologique vs plafond plat sur le stress subjectif",
        ),
        fontsize=9.7,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig4_compute_pareto")


def main() -> None:
    """Generate all available Study-08 figures."""
    fm_s07 = load_json("fm_and_study07.json")
    fm_ext = load_json("fm_extract.json")
    prim = load_json("ssl_jepa_primary.json")
    modwin = load_json("ssl_jepa_abl_modwin.json")
    strat = load_json("ssl_jepa_abl_stratscale.json")
    fuse = load_json("jepa_fusion.json")

    if fm_s07 and prim and fuse:
        make_fig1(fm_s07, prim, modwin, fuse)
        make_fig2(fm_s07, prim, fuse, strat)
    if prim and modwin and strat:
        make_fig3(prim, modwin, strat)
    if fm_s07 and fm_ext and prim and fuse:
        make_fig4(fm_s07, fm_ext, prim, fuse)


if __name__ == "__main__":
    main()
