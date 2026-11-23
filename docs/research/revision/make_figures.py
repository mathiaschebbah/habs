"""Generate all Study-07 bilingual publication figures from saved JSON aggregates.

Run from the repository root:
    python docs/research/revision/make_figures.py        # English figures
    python docs/research/revision/make_figures.py fr     # French figures (_fr.png)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
GRAY = "#9c9b95"
INK = "#111111"
MUTED = "#555450"
LIGHT = "#ecebe6"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    }
)

FR_MAP = {
    # Emotions
    "anger": "Colère",
    "fear": "Peur",
    "disgust": "Dégoût",
    "sadness": "Tristesse",
    "surprise": "Surprise",
    "curiosity": "Curiosité",
    "guilt": "Culpabilité",
    "neutral": "Neutre",
    "excitement": "Excitation",
    "relaxation": "Relaxation",
    "sensuality": "Sensualité",
    "joy": "Joie",
    "core_threat": "Menace (P+C+D)",
    "all_negative": "Toutes négatives",
    # Signals
    "heart": "Cœur (FC)",
    "skin": "Peau (SC)",
    "breathing": "Respiration",
    "delta": "EEG delta",
    "alpha": "EEG alpha",
    # Regions & bands
    "frontal": "Frontal",
    "central": "Central",
    "left_temporal": "Temp. G",
    "right_temporal": "Temp. D",
    "parietal": "Pariétal",
    "occipital": "Occipital",
    "Delta": "Delta",
    "Theta": "Thêta",
    "Alpha": "Alpha",
    "Beta": "Bêta",
    "Gamma": "Gamma",
}

EN_MAP = {
    "anger": "Anger",
    "fear": "Fear",
    "disgust": "Disgust",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "curiosity": "Curiosity",
    "guilt": "Guilt",
    "neutral": "Neutral",
    "excitement": "Excitement",
    "relaxation": "Relaxation",
    "sensuality": "Sensuality",
    "joy": "Joy",
    "core_threat": "Core threat (F+A+D)",
    "all_negative": "All negative",
    "heart": "Heart rate",
    "skin": "Skin cond.",
    "breathing": "Breathing",
    "delta": "EEG delta",
    "alpha": "EEG alpha",
    "frontal": "Frontal",
    "central": "Central",
    "left_temporal": "L. temp.",
    "right_temporal": "R. temp.",
    "parietal": "Parietal",
    "occipital": "Occipital",
    "Delta": "Delta",
    "Theta": "Theta",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Gamma": "Gamma",
}


def tr(en, fr):
    """Return French or English string."""
    return fr if FR else en


def lbl(key):
    """Translate a canonical key."""
    return (FR_MAP if FR else EN_MAP).get(key, key)


def _comma(fig, axes=None, *, x=False, y=True):
    """Format numeric axes with French decimal comma and true minus sign."""
    if not FR:
        return
    fmt = FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    for ax in fig.axes if axes is None else axes:
        if x:
            ax.xaxis.set_major_formatter(fmt)
        if y:
            ax.yaxis.set_major_formatter(fmt)


def _load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def fig1_criterion():
    """Criterion sanity check (rank-1 emotions) and threshold purification curve."""
    data = _load("criterion")
    if not data:
        return
    cr = data["criterion"]
    sw = cr["threshold_sweep"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.05, 1.25]}
    )

    # Panel A: Rank-ordered self_reported_emotion_1 inside V <= 4 & A >= 6
    ax = axes[0]
    pct = sw["v_le4_a_ge6"]["emotion_1"]["pct_ranked"]
    items = list(pct.items())
    emos = [k for k, _ in items]
    vals = [100.0 * v for _, v in items]
    y = np.arange(len(emos))[::-1]
    colors = []
    for e in emos:
        if e in ("anger", "fear", "disgust"):
            colors.append(RED)
        elif e in ("sadness", "guilt"):
            colors.append(ORANGE)
        elif e == "surprise":
            colors.append(AMBER)
        else:
            colors.append(GRAY)
    ax.barh(y, vals, color=colors, height=0.72, edgecolor="none")
    for yi, v in zip(y, vals, strict=True):
        if v >= 0.5:
            txt = f"{v:.1f}%".replace(".", ",") if FR else f"{v:.1f}%"
            ax.text(v + 0.5, yi, txt, va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(y, [lbl(e) for e in emos], fontsize=7)
    ax.set_xlim(0, 32.5)
    ax.grid(True, axis="x", color=LIGHT, lw=0.6)
    ax.set_xlabel(
        tr("Share of included stress trials (%)", "Part des essais stress inclus (%)")
    )
    ax.set_title(
        tr(
            "A. First emotion in V≤4 ∩ A≥6 (n = 3,938)\n89.2% negative (97.2% in ranks 1–3)",
            "A. Émotion n°1 dans V≤4 ∩ A≥6 (n = 3 938)\n89,2 % négatives (97,2 % aux rangs 1–3)",
        ),
        fontsize=7.5,
        color=INK,
    )
    # Mini legend
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=RED, label=tr("Acute threat (72.5%)", "Menace aiguë (72,5 %)")),
        Patch(
            facecolor=ORANGE, label=tr("Sadness/guilt (16.7%)", "Trist./culp. (16,7 %)")
        ),
        Patch(facecolor=AMBER, label=tr("Surprise (5.5%)", "Surprise (5,5 %)")),
        Patch(facecolor=GRAY, label=tr("Incoherent (5.3%)", "Incohérentes (5,3 %)")),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.0, loc="lower right")

    # Panel B: Threshold purification sweep (purity + physiological d divergence)
    ax2 = axes[1]
    order = [
        ("v_lt5_a_gt5", "<5/>5"),
        ("v_le4.5_a_ge5.5", "4.5/5.5"),
        ("v_le4_a_ge6", "4/6"),
        ("v_le3.5_a_ge6.5", "3.5/6.5"),
        ("v_le3_a_ge7", "3/7"),
        ("v_le2.5_a_ge7.5", "2.5/7.5"),
        ("v_le2_a_ge8", "2/8"),
    ]
    xs = np.arange(len(order))
    xlabels = [t.replace(".", ",") if FR else t for _, t in order]
    sc_d = [sw[k]["effect_d"]["skin"][0] for k, _ in order]
    dl_d = [sw[k]["effect_d"]["delta"][0] for k, _ in order]
    hr_d = [sw[k]["effect_d"]["heart"][0] for k, _ in order]
    neg_p = [100.0 * sw[k]["emotion_1"]["negative_purity"] for k, _ in order]
    inc_p = [100.0 * sw[k]["emotion_1"]["incoherent_rate"] for k, _ in order]

    ax2.axhline(0, color=MUTED, lw=0.7)
    ax2.grid(True, axis="y", color=LIGHT, lw=0.6)
    l1 = ax2.plot(
        xs,
        sc_d,
        color=ORANGE,
        lw=1.9,
        marker="o",
        ms=3.5,
        label=tr("Skin cond. (d)", "Peau (d)"),
    )
    l2 = ax2.plot(
        xs,
        dl_d,
        color=PURPLE,
        lw=1.9,
        marker="s",
        ms=3.5,
        label=tr("EEG delta (d)", "EEG delta (d)"),
    )
    l3 = ax2.plot(
        xs,
        hr_d,
        color=BLUE,
        lw=1.9,
        marker="^",
        ms=3.5,
        label=tr("Heart rate (d)", "Fréq. card. (d)"),
    )
    ax2.set_ylabel(tr("Within-person effect size d", "Taille d'effet intra-personne d"))
    ax2.set_ylim(-0.20, 0.44)
    ax2.set_xticks(xs, xlabels, fontsize=6.6)
    ax2.set_xlabel(
        tr("Threshold (valence ≤ / arousal ≥)", "Seuil (valence ≤ / arousal ≥)")
    )
    ax2.axvline(2, color=MUTED, ls=":", lw=0.9)
    ax2.axvline(4, color=TEAL, ls="--", lw=0.9)

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["right"].set_color(MUTED)
    l4 = ax2_r.plot(
        xs,
        neg_p,
        color=TEAL,
        lw=1.5,
        ls="--",
        marker="d",
        ms=3,
        label=tr("Negative purity (%)", "Pureté négative (%)"),
    )
    l5 = ax2_r.plot(
        xs,
        inc_p,
        color=GRAY,
        lw=1.3,
        ls=":",
        marker="x",
        ms=3.5,
        label=tr("Incoherent (%)", "Incohérentes (%)"),
    )
    ax2_r.set_ylabel(
        tr("Rank-1 emotion share (%)", "Part émotion n°1 (%)"), color=MUTED
    )
    ax2_r.set_ylim(0, 108)
    ax2.set_title(
        tr(
            "B. Purification: purity & SC/delta rise,\ncardiac slowing weakens at extreme arousal",
            "B. Purification : pureté & peau/delta montent,\ndécélération atténuée à arousal extrême",
        ),
        fontsize=7.3,
        color=INK,
    )
    lines = l1 + l2 + l3 + l4 + l5
    ax2.legend(
        lines,
        [ln.get_label() for ln in lines],
        frameon=False,
        fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    _comma(fig, [ax], x=True, y=False)
    _comma(fig, [ax2], x=False, y=True)
    fig.tight_layout()
    fig.savefig(FIG / f"fig1_criterion{SUFFIX}.png", bbox_inches="tight")
    plt.close(fig)


def fig2_emotions():
    """Per-emotion physiological signatures vs neutral & direct emotion classification."""
    data = _load("emotion_models")
    if not data:
        return
    em = data["emotion_models"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.15, 1.05]}
    )

    # Panel A: Heatmap of Cohen's d vs neutral for raw self_reported_emotion_1
    ax = axes[0]
    emos = [
        "fear",
        "disgust",
        "anger",
        "sadness",
        "surprise",
        "excitement",
        "joy",
        "relaxation",
    ]
    sigs = ["heart", "skin", "breathing", "delta", "alpha"]
    mat = np.array(
        [[em["signatures"][e]["vs_neutral_d"][s][0] for s in sigs] for e in emos]
    )
    sig_mask = np.array(
        [
            [
                (em["signatures"][e]["vs_neutral_d"][s][1] > 0)
                or (em["signatures"][e]["vs_neutral_d"][s][2] < 0)
                for s in sigs
            ]
            for e in emos
        ]
    )
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.26, vmax=0.26, aspect="auto")
    for i in range(len(emos)):
        for j in range(len(sigs)):
            v = mat[i, j]
            txt = f"{v:+.2f}" + ("*" if sig_mask[i, j] else "")
            if FR:
                txt = txt.replace(".", ",").replace("-", "\u2212")
            col = "white" if abs(v) > 0.16 else INK
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=6.4,
                color=col,
                fontweight="bold" if sig_mask[i, j] else "normal",
            )
    short_sigs = {
        "heart": tr("Heart", "Cœur"),
        "skin": tr("Skin", "Peau"),
        "breathing": tr("Resp.", "Resp."),
        "delta": "Delta",
        "alpha": "Alpha",
    }
    ax.set_xticks(range(len(sigs)), [short_sigs[s] for s in sigs], fontsize=6.8)
    ax.set_yticks(range(len(emos)), [lbl(e) for e in emos], fontsize=7.0)
    ax.set_title(
        tr(
            "A. Physiology by raw emotion #1 (d vs neutral)\n* 95% CI excludes 0 (cohorts 2–5, n = 17,142)",
            "A. Physiologie par émotion n°1 brute (d vs neutre)\n* IC 95 % hors de 0 (cohortes 2–5, n = 17 142)",
        ),
        fontsize=7.2,
        color=INK,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)

    # Panel B: Balanced binary classifiers (within-person OvR & same-video AUC)
    ax2 = axes[1]
    targets = [
        "relaxation",
        "core_threat",
        "disgust",
        "joy",
        "all_negative",
        "fear",
        "sadness",
        "anger",
    ]
    y = np.arange(len(targets))[::-1]
    ax2.axvline(0.5, color=MUTED, lw=0.8)
    ax2.grid(True, axis="x", color=LIGHT, lw=0.6)
    bc = em["binary_classifiers"]
    for off, key, col, name in (
        (
            0.15,
            "within_person_ovr",
            BLUE,
            tr("Across videos (same person)", "Entre vidéos (même pers.)"),
        ),
        (
            -0.15,
            "same_video_ovr",
            ORANGE,
            tr("Across people (same video)", "Entre pers. (même vidéo)"),
        ),
    ):
        pts, lo, hi = [], [], []
        for t in targets:
            if key == "within_person_ovr":
                d = bc[t][key]
                pts.append(d["auc"])
                lo.append(d["ci95"][0])
                hi.append(d["ci95"][1])
            else:
                d = bc[t][key]
                pts.append(d["auc_weighted"])
                lo.append(d["ci95_weighted"][0])
                hi.append(d["ci95_weighted"][1])
        pts, lo, hi = np.array(pts), np.array(lo), np.array(hi)
        ax2.errorbar(
            pts,
            y + off,
            xerr=[pts - lo, hi - pts],
            fmt="o",
            ms=3.8,
            color=col,
            ecolor=col,
            elinewidth=1.2,
            label=name,
        )
    ax2.set_yticks(y, [lbl(t) for t in targets], fontsize=6.8)
    ax2.set_xlim(0.455, 0.585)
    ax2.set_xlabel(
        tr(
            "ROC AUC (balanced logistic, unseen people/themes)",
            "AUC ROC (logistique équilibrée)",
        )
    )
    ax2.set_title(
        tr(
            "B. Direct binary classifiers on raw emotion #1\n(5-class balanced acc.: 22.3% vs 20% chance)",
            "B. Classifieurs binaires sur l'émotion n°1\n(multiclasse 5 classes : 22,3 % vs 20 %)",
        ),
        fontsize=7.2,
        color=INK,
    )
    ax2.legend(
        frameon=False,
        fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        columnspacing=0.8,
    )
    _comma(fig, [ax2], x=True, y=False)
    fig.tight_layout()
    fig.savefig(FIG / f"fig2_emotions{SUFFIX}.png", bbox_inches="tight")
    plt.close(fig)


def fig3_ecg_eeg_rich():
    """ECG P-QRS-T morphology (with RR control) & new EEG spatial/temporal features."""
    data = _load("ecg_morphology")
    if not data:
        return
    em = data["ecg_morphology"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.85), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # Panel A: ECG P-QRS-T stress effects (raw & RR-controlled)
    ax = axes[0]
    items = [
        ("rr", tr("RR interval (+5.1 ms)", "Intervalle RR (+5,1 ms)"), False),
        ("hr", tr("Heart rate (−0.53 bpm)", "Fréq. card. (−0,53 bpm)"), False),
        ("pr_on", tr("PR interval (AV node)", "Intervalle PR (NAV)"), True),
        ("pr_pk", tr("PR peak (P→R)", "Pic PR (P→R)"), True),
        ("pr_seg", tr("PR segment (pure NAV)", "Segment PR (NAV pur)"), True),
        ("qrs", tr("QRS duration", "Durée QRS"), True),
        ("rt_pk", tr("RT peak (repol.)", "Pic RT (repol.)"), True),
        ("qt", tr("QT interval", "Intervalle QT"), True),
        ("qtc_baz", tr("QTc Bazett", "QTc Bazett"), True),
        ("t_amp", tr("T/R amplitude ratio", "Ratio d'ampl. T/R"), True),
    ]
    y = np.arange(len(items))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(True, axis="x", color=LIGHT, lw=0.6)
    eff = em["ecg_effects"]
    ctrl = em["ecg_rr_controlled_and_coupling"]
    for idx_i, (k, _name, has_ctrl) in enumerate(items):
        yi = y[idx_i]
        d = eff[k]["vid_change"]["stress"]
        off = 0.14 if has_ctrl else 0.0
        ax.errorbar(
            d[0],
            yi + off,
            xerr=[[d[0] - d[1]], [d[2] - d[0]]],
            fmt="o",
            ms=3.8,
            color=BLUE,
            ecolor=BLUE,
            elinewidth=1.2,
            label=tr("Stress vs other (d)", "Stress vs autres (d)")
            if idx_i == 0
            else None,
        )
        if has_ctrl:
            c = ctrl[k]["rr_controlled"]
            ax.errorbar(
                c[0],
                yi - 0.14,
                xerr=[[c[0] - c[1]], [c[2] - c[0]]],
                fmt="s",
                ms=3.3,
                color=ORANGE,
                ecolor=ORANGE,
                elinewidth=1.1,
                label=tr("Adjusted for ΔRR (β)", "Ajusté sur ΔRR (β)")
                if idx_i == 2
                else None,
            )
    ax.set_yticks(y, [x[1] for x in items], fontsize=6.6)
    ax.set_xlim(-0.175, 0.185)
    ax.set_xlabel(tr("Within-person effect (95% CI)", "Effet intra-personne (IC 95 %)"))
    ax.set_title(
        tr(
            "A. ECG P-QRS-T (n = 314, lead-aligned)\nPR null; QTc vanishes after ΔRR control",
            "A. Ondes ECG P-QRS-T (n = 314)\nPR nul ; QTc nul après contrôle ΔRR",
        ),
        fontsize=7.1,
        color=INK,
    )
    ax.legend(
        frameon=False,
        fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=2,
        columnspacing=0.8,
    )

    # Panel B: New EEG temporal variation & spatial homogeneity features
    ax2 = axes[1]
    sp = em["eeg_spatial_homogeneity_and_asymmetry"]
    tv = em["eeg_temporal_variation"]
    eeg_items = [
        (
            tr("Delta max 1-s jump (early)", "Saut max 1 s delta (précoce)"),
            tv["delta"]["maxjump_early"]["stress"],
            PURPLE,
        ),
        (
            tr("Delta temporal SD (early)", "Écart-type temp. delta (préc.)"),
            tv["delta"]["tvar_early"]["stress"],
            PURPLE,
        ),
        (
            tr("Alpha temporal SD (early)", "Écart-type temp. alpha (préc.)"),
            tv["alpha"]["tvar_early"]["stress"],
            PURPLE,
        ),
        (
            tr("Beta temporal SD (early)", "Écart-type temp. bêta (préc.)"),
            tv["beta"]["tvar_early"]["stress"],
            PURPLE,
        ),
        (
            tr("Alpha temporal slope (early)", "Pente temp. alpha (précoce)"),
            tv["alpha"]["tslope_early"]["stress"],
            PURPLE,
        ),
        (
            tr("AP gradient delta (late)", "Gradient AP delta (tardif)"),
            sp["ap_grad"]["delta"]["late"]["stress"],
            TEAL,
        ),
        (
            tr("AP gradient alpha (late)", "Gradient AP alpha (tardif)"),
            sp["ap_grad"]["alpha"]["late"]["stress"],
            TEAL,
        ),
        (
            tr("Temporal beta asym. (late)", "Asym. temp. bêta (tardif)"),
            sp["temp_asym"]["beta"]["late"]["stress"],
            TEAL,
        ),
        (
            tr("Spatial IQR theta (early)", "IQR spatial thêta (précoce)"),
            sp["sp_iqr"]["theta"]["early"]["stress"],
            RED,
        ),
        (
            tr("Spatial entropy alpha (post)", "Entropie spat. alpha (post)"),
            sp["sp_ent"]["alpha"]["post"]["stress"],
            RED,
        ),
    ]
    y2 = np.arange(len(eeg_items))[::-1]
    ax2.axvline(0, color=MUTED, lw=0.8)
    ax2.grid(True, axis="x", color=LIGHT, lw=0.6)
    for idx_i, (_name, d, col) in enumerate(eeg_items):
        yi = y2[idx_i]
        ax2.errorbar(
            d[0],
            yi,
            xerr=[[d[0] - d[1]], [d[2] - d[0]]],
            fmt="o",
            ms=3.8,
            color=col,
            ecolor=col,
            elinewidth=1.2,
        )
    ax2.set_yticks(y2, [x[0] for x in eeg_items], fontsize=6.5)
    ax2.set_xlim(-0.14, 0.215)
    ax2.set_xlabel(
        tr(
            "Cohen's d (stress vs other, 95% CI)",
            "d de Cohen (stress vs autres, IC 95 %)",
        )
    )
    ax2.set_title(
        tr(
            "B. EEG temporal dynamics & spatial homogeneity\nPurple: temp. var · Teal: gradients · Red: homog.",
            "B. Dynamique temporelle & homogénéité EEG\nViolet : var. temp. · Vert : grad. · Rouge : homog.",
        ),
        fontsize=7.1,
        color=INK,
    )
    _comma(fig, [ax, ax2], x=True, y=False)
    fig.tight_layout()
    fig.savefig(FIG / f"fig3_ecg_eeg_rich{SUFFIX}.png", bbox_inches="tight")
    plt.close(fig)


def fig4_hep_raw_vs_ica():
    """Direct empirical comparison of ICA-cleaned vs RAW CURATED HEP and ICA-diff."""
    data = _load("hep_raw")
    if not data:
        return
    hr = data["hep_raw"]
    curves = hr["curves"]
    t = -200 + 50 * np.arange(16) + 25

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharex=True, sharey=True)
    panels = [
        (
            "ica",
            tr("A. PREPROCESSED (ICA-cleaned)", "A. PRÉTRAITÉ (nettoyé par ICA)"),
            tr(
                "d = +0.039 [−0.002, +0.078]\nvs relaxed: +0.072*",
                "d = +0,039 [−0,002 ; +0,078]\nvs relaxé : +0,072*",
            ),
            BLUE,
        ),
        (
            "raw",
            tr("B. CURATED (raw CAR, NO ICA)", "B. CURATED (brut CAR, SANS ICA)"),
            tr(
                "d = +0.038 [−0.001, +0.077]\nvs relaxed: +0.076*",
                "d = +0,038 [−0,001 ; +0,077]\nvs relaxé : +0,076*",
            ),
            ORANGE,
        ),
        (
            "icadiff",
            tr(
                "C. ICA-removed (CURATED − ICA)",
                "C. Signal retiré par ICA (Brut − ICA)",
            ),
            tr(
                "d = +0.012 [−0.028, +0.053]\nECG field corr.: −0.004",
                "d = +0,012 [−0,028 ; +0,053]\ncorr. champ ECG : −0,004",
            ),
            MUTED,
        ),
    ]
    for ax, (prefix, title, annot, col) in zip(axes, panels, strict=True):
        ax.axvspan(250, 600, color=LIGHT, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.7)
        c_f = curves[f"{prefix}_frontal"]
        c_c = curves[f"{prefix}_central"]
        ax.fill_between(t, c_f["low"], c_f["high"], color=col, alpha=0.18, lw=0)
        ax.plot(t, c_f["d"], color=col, lw=1.8, label=tr("Frontal", "Frontal"))
        ax.plot(
            t,
            c_c["d"],
            color=col,
            lw=1.2,
            ls="--",
            alpha=0.85,
            label=tr("Central", "Central"),
        )
        ax.set_xticks((0, 200, 400, 600))
        ax.set_ylim(-0.085, 0.115)
        ax.set_title(title, fontsize=7.2, color=INK)
        ax.text(
            0.04,
            0.94,
            annot,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.1,
            color=INK,
            bbox=dict(
                boxstyle="round,pad=0.22", facecolor="white", edgecolor=LIGHT, alpha=0.9
            ),
        )
    axes[0].legend(frameon=False, fontsize=6.0, loc="lower left")
    axes[1].set_xlabel(
        tr(
            "ms after R peak (grey: 250–600 ms HEP window)",
            "ms après pic R (gris : fenêtre HEP 250–600 ms)",
        )
    )
    axes[0].set_ylabel(
        tr("Effect size d\n(stress vs other)", "Taille d'effet d\n(stress vs autres)")
    )
    _comma(fig, axes, x=False, y=True)
    fig.tight_layout()
    fig.savefig(FIG / f"fig4_hep_raw_vs_ica{SUFFIX}.png", bbox_inches="tight")
    plt.close(fig)


def fig5_fixed_video():
    """Non-adjacent fixed-video contrasts, 2x2 consensus x state matrix, & video skew."""
    data = _load("fixed_video")
    if not data:
        return
    fv = data["fixed_video"]

    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]}
    )

    # Panel A: Non-adjacent fixed-video AUC (stress vs all vs stress vs oklm)
    ax = axes[0]
    c_defs = [
        ("all_stress_vs_rest", tr("All\n(diluted)", "Tous\n(dilué)")),
        ("polarizing_20_80", tr("Polariz.\n20–80%", "Polaris.\n20–80 %")),
        ("stress_vs_oklm_5", tr("vs Calm\n(5 / 5)", "vs Calme\n(5 / 5)")),
        ("stress_vs_oklm_strict", tr("vs Calm+\n(5.5/4.5)", "vs Calme+\n(5,5/4,5)")),
    ]
    xs = np.arange(len(c_defs))
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.grid(True, axis="y", color=LIGHT, lw=0.6)
    for off, skey, col, name in (
        (-0.18, "model", BLUE, tr("Model", "Modèle")),
        (0.0, "heart", RED, tr("Heart", "Cœur")),
        (0.18, "delta", PURPLE, tr("Delta", "Delta")),
    ):
        pts, lo, hi = [], [], []
        for ck, _ in c_defs:
            st = fv["fixed_video_contrasts"][ck][skey]
            pts.append(st["auc_weighted"])
            lo.append(st["ci95_weighted"][0])
            hi.append(st["ci95_weighted"][1])
        pts, lo, hi = np.array(pts), np.array(lo), np.array(hi)
        ax.errorbar(
            xs + off,
            pts,
            yerr=[pts - lo, hi - pts],
            fmt="o",
            ms=3.6,
            color=col,
            ecolor=col,
            elinewidth=1.1,
            label=name,
        )
    ax.set_xticks(xs, [x[1] for x in c_defs], fontsize=6.1)
    ax.set_ylim(0.47, 0.635)
    ax.set_ylabel(
        tr("Same-video AUC (95% video CI)", "AUC à vidéo fixe (IC 95 % vidéos)")
    )
    ax.set_title(
        tr(
            "A. Same-video separation\njumps when excluding borderline",
            "A. Séparation à vidéo fixe\nsaute hors essais limites",
        ),
        fontsize=7.2,
        color=INK,
    )
    ax.legend(
        frameon=False,
        fontsize=5.8,
        loc="upper left",
        ncol=3,
        columnspacing=0.6,
        handletextpad=0.2,
    )

    # Panel B: 2x2 Stimulus Consensus x Viewer State for Heart Rate (bpm)
    ax2 = axes[1]
    mat = fv["matrix_2x2"]
    strata = [
        ("low_consensus_le10", tr("Low\n(≤10%)", "Bas\n(≤10 %)")),
        ("mid_polarizing_20_55", tr("Polariz.\n(20–55%)", "Polaris.\n(20–55 %)")),
        ("high_consensus_ge50", tr("High\n(≥50%)", "Haut\n(≥50 %)")),
    ]
    x2 = np.arange(len(strata))
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.grid(True, axis="y", color=LIGHT, lw=0.6)
    for off, vstate, col, name in (
        (-0.18, "oklm_strict", TEAL, tr("Very calm (oklm)", "Très calme (oklm)")),
        (0.0, "borderline", AMBER, tr("Borderline", "Intermédiaire")),
        (0.18, "stressed", RED, tr("Stressed", "Stressé")),
    ):
        pts, lo, hi = [], [], []
        for sk, _ in strata:
            c = mat[sk][vstate]["heart"]
            pts.append(c[0])
            lo.append(c[1])
            hi.append(c[2])
        pts, lo, hi = np.array(pts), np.array(lo), np.array(hi)
        ax2.errorbar(
            x2 + off,
            pts,
            yerr=[pts - lo, hi - pts],
            fmt="o",
            ms=4.0,
            color=col,
            ecolor=col,
            elinewidth=1.2,
            label=name,
        )
    ax2.set_xticks(x2, [x[1] for x in strata], fontsize=6.4)
    ax2.set_xlabel(
        tr("Video crowd stress consensus", "Consensus de stress de la vidéo")
    )
    ax2.set_ylabel(tr("Heart rate change (bpm vs ref)", "Variation FC (bpm vs réf.)"))
    ax2.set_ylim(-5.5, 3.2)
    ax2.set_title(
        tr(
            "B. 2×2 Consensus × Appraisal\nCalm viewers don't decelerate",
            "B. Matrice 2×2 Consensus × Vécu\nLes calmes ne décélèrent pas",
        ),
        fontsize=7.2,
        color=INK,
    )
    ax2.legend(
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT,
        framealpha=0.92,
        fontsize=5.6,
        loc="upper left",
    )

    # Panel C: Per-video stress-rate distribution (252 videos)
    ax3 = axes[2]
    vd = fv["video_distribution"]
    counts = vd["rate_histogram_counts"]
    centers = np.linspace(0.05, 0.95, 10) * 100
    bar_cols = [GRAY if c < 20 else (BLUE if c <= 80 else RED) for c in centers]
    ax3.bar(centers, counts, width=8.5, color=bar_cols, edgecolor="none")
    ax3.grid(True, axis="y", color=LIGHT, lw=0.6)
    ax3.set_xlabel(tr("Video stress rate (%)", "Taux de stress (%)"))
    ax3.set_ylabel(tr("Number of videos", "Nombre de vidéos"))
    ax3.set_title(
        tr(
            "C. 252 videos\n112 polarizing (20–80%)",
            "C. 252 vidéos\n112 polarisantes (20–80 %)",
        ),
        fontsize=7.2,
        color=INK,
    )
    _comma(fig, [ax, ax2], x=False, y=True)
    fig.tight_layout()
    fig.savefig(FIG / f"fig5_fixed_video{SUFFIX}.png", bbox_inches="tight")
    plt.close(fig)


def fig6_sensitivity_and_models():
    """EEG sensitivity topography (6 regions x 5 bands) & sequence model benchmark."""
    sn_data = _load("sensitivity")
    sm_data = _load("sequence_models")
    if not sn_data:
        return
    sn = sn_data["sensitivity"]

    if sm_data:
        fig, axes = plt.subplots(
            1, 2, figsize=(7.0, 2.85), gridspec_kw={"width_ratios": [1.0, 1.25]}
        )
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(3.8, 2.85))
        axes = [ax0]

    # Panel A: 6 regions x 5 bands beta_own on valence (negative = rises when person feels worse)
    ax = axes[0]
    regions = [
        "frontal",
        "central",
        "left_temporal",
        "right_temporal",
        "parietal",
        "occipital",
    ]
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    mat = np.array(
        [
            [
                -sn["regional_coupling"][r][b]["full_valence"]["beta_own"][0]
                for b in bands
            ]
            for r in regions
        ]
    )
    sig = np.array(
        [
            [
                sn["regional_coupling"][r][b]["full_valence"]["beta_own"][2] < 0
                or sn["regional_coupling"][r][b]["full_valence"]["beta_own"][1] > 0
                for b in bands
            ]
            for r in regions
        ]
    )
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.045, vmax=0.045, aspect="auto")
    for i in range(len(regions)):
        for j in range(len(bands)):
            v = mat[i, j]
            txt = f"{v:+.03f}"[0] + f"{abs(v):.3f}"[1:] + ("*" if sig[i, j] else "")
            if FR:
                txt = txt.replace(".", ",").replace("-", "\u2212")
            col = "white" if abs(v) > 0.028 else INK
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=6.0,
                color=col,
                fontweight="bold" if sig[i, j] else "normal",
            )
    ax.set_xticks(range(len(bands)), [lbl(b.capitalize()) for b in bands], fontsize=6.8)
    ax.set_yticks(range(len(regions)), [lbl(r) for r in regions], fontsize=6.8)
    ax.set_title(
        tr(
            "A. EEG sensitivity ($-\\beta_{\\mathrm{own}}$ valence)\nFast frontal β/γ & scalp δ; α/θ null",
            "A. Sensibilité EEG ($-\\beta_{\\mathrm{propre}}$ valence)\nβ/γ frontal rapide & δ diffus ; α/θ nuls",
        ),
        fontsize=7.2,
        color=INK,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6.2)
    if FR:
        _comma(fig, [cb.ax], x=False, y=True)

    # Panel B: Model architecture & feature representation benchmark
    if sm_data:
        ax2 = axes[1]
        models = sm_data["sequence_models"]["models"]
        spec_order = [
            (
                "baseline_1s_logreg",
                tr("1-s bins (linear baseline)", "Bins 1 s (référence linéaire)"),
            ),
            (
                "phase4_logreg",
                tr("4 phases (Ref/Early/Late/Post)", "4 phases (Réf/Préc/Tard/Post)"),
            ),
            ("phase4_deltas_logreg", tr("4 phases + Δ(t→t+1)", "4 phases + Δ(t→t+1)")),
            (
                "ablation_plus_ecg_pqrst",
                tr("+ ECG P-QRS-T (PR/QT/T)", "+ ECG P-QRS-T (PR/QT/T)"),
            ),
            (
                "ablation_plus_eeg_spatial",
                tr("+ EEG spatial homogeneity", "+ EEG homogénéité spatiale"),
            ),
            (
                "ablation_plus_eeg_tvar",
                tr("+ EEG temporal variation", "+ EEG variation temporelle"),
            ),
            (
                "enriched_full_logreg",
                tr("Full enriched (linear)", "Complet enrichi (linéaire)"),
            ),
            (
                "enriched_full_ridge_va",
                tr("Full enriched (Ridge V&A)", "Complet enrichi (Ridge V&A)"),
            ),
            (
                "enriched_full_gbdt",
                tr("Full enriched (GBDT)", "Complet enrichi (GBDT)"),
            ),
            (
                "sequence_multiscale_conv_logreg",
                tr("Multi-scale 1D conv seq.", "Séquence conv. 1D multi-éch."),
            ),
            (
                "sequence_plus_enriched_logreg",
                tr("Sequence + enriched (lin.)", "Séquence + enrichi (lin.)"),
            ),
            (
                "sequence_plus_enriched_mlp",
                tr("Sequence + enriched (MLP)", "Séquence + enrichi (MLP)"),
            ),
        ]
        y2 = np.arange(len(spec_order))[::-1]
        ax2.axvline(0.5, color=MUTED, lw=0.8)
        ax2.grid(True, axis="x", color=LIGHT, lw=0.6)
        for off, regime, col, name in (
            (
                0.15,
                "dev",
                BLUE,
                tr("Dev CV (new person & theme)", "CV dév. (pers. & thème nouv.)"),
            ),
            (
                -0.15,
                "test_new_both",
                ORANGE,
                tr("Sealed test (new both)", "Test scellé (pers. & thème)"),
            ),
        ):
            pts, lo, hi = [], [], []
            for mk, _ in spec_order:
                d = models[mk][regime]["within_person"]
                pts.append(d["auc"])
                lo.append(d["ci95"][0])
                hi.append(d["ci95"][1])
            pts, lo, hi = np.array(pts), np.array(lo), np.array(hi)
            ax2.errorbar(
                pts,
                y2 + off,
                xerr=[pts - lo, hi - pts],
                fmt="o",
                ms=3.4,
                color=col,
                ecolor=col,
                elinewidth=1.1,
                label=name,
            )
        ax2.set_yticks(y2, [x[1] for x in spec_order], fontsize=6.1)
        ax2.set_xticks([0.48, 0.50, 0.54, 0.58, 0.62])
        ax2.set_xlabel(tr("Within-person AUC (95% CI)", "AUC intra-personne (IC 95 %)"))
        ax2.set_title(
            tr(
                "B. Feature & temporal-architecture benchmark\n(fresh person × theme split)",
                "B. Comparaison des features et architectures\n(nouveau découpage personne × thème)",
            ),
            fontsize=7.2,
            color=INK,
        )
        ax2.legend(
            frameon=False,
            fontsize=5.8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            columnspacing=0.8,
        )
        if FR:
            ax2.set_xticklabels(["0,48", "0,50", "0,54", "0,58", "0,62"])
        else:
            ax2.set_xticklabels(["0.48", "0.50", "0.54", "0.58", "0.62"])

    fig.tight_layout()
    fig.savefig(FIG / f"fig6_sensitivity_models{SUFFIX}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_criterion()
    fig2_emotions()
    fig3_ecg_eeg_rich()
    fig4_hep_raw_vs_ica()
    fig5_fixed_video()
    fig6_sensitivity_and_models()
    print("Generated figures in", FIG, "suffix=", repr(SUFFIX))
