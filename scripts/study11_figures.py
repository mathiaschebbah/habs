"""Draw the publication figures from the versioned Study 11 aggregate JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "study11"
FIG = ROOT / "paper" / "figures"
COLORS = ["#184f76", "#a33c30", "#9c681d", "#007466"]
REGIMES = ["R1_random_pool", "R2_seen_id_new_themes", "R3_new_id_seen_themes", "R4_double_unseen"]


def read(name):
    return json.loads((DATA / f"{name}.json").read_text())


def finish(fig, name, fr):
    fig.savefig(
        FIG / f"study11_{name}{'_fr' if fr else ''}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def timeline(controls, fr):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.7, 3.1), gridspec_kw={"width_ratios": [1.55, 1]})
    data = read("finite_curves")["audited_decoding"]["signals"]
    x = np.arange(-2.5, 12, 1)
    ax.axvspan(0, 10, color="#eef3f4")
    ax.axhline(0, color="#666", lw=0.7, ls=":")
    labels = (
        ["EEG delta", "HR", "Respiration", "EDA"]
        if not fr
        else ["EEG delta", "FC", "Respiration", "EDA"]
    )
    for i, (k, label) in enumerate(
        zip(["eeg_central_delta", "hr", "resp_amp", "sc"], labels, strict=True)
    ):
        d = data[k]
        ax.fill_between(x, d["low"], d["high"], color=COLORS[i], alpha=0.09)
        ax.plot(x, d["d"], "o-", color=COLORS[i], lw=1.5, ms=2.4, label=label)
    ax.set_xlim(-3, 12)
    ax.set_xlabel("Temps depuis le début du clip (s)" if fr else "Time from clip onset (s)")
    ax.set_ylabel("Contraste intra-ID (d)" if fr else "Within-ID contrast (d)")
    ax.set_title("A   Réponses par seconde" if fr else "A   Responses by second", loc="left")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.23), frameon=False)
    baselines = controls["priority3_order_and_confound_controls"]["baselines"]
    keys = [
        "ref_S07_no_post",
        "p3a_index_linear",
        "p3b_cohort_x_index_poly",
        "p3d_prev_trial_s07_nonfirst",
    ]
    names = (
        ["Signaux du clip", "Indice de l’essai", "Cohorte × indice", "Essai précédent"]
        if fr
        else ["Current clip signals", "Trial index", "Cohort × index", "Previous trial"]
    )
    bx.axvline(0.5, color="#666", ls=":", lw=0.9)
    for i, k in enumerate(keys):
        m = baselines[k]["stress_va"]["overall"]
        val = m["within_id_auc"]
        lo, hi = m["ci95"]
        bx.errorbar(
            val,
            3 - i,
            xerr=[[val - lo], [hi - val]],
            fmt="o",
            ms=4,
            capsize=3,
            color=COLORS[0] if i == 0 else COLORS[3],
        )
    bx.set_yticks(range(4), names[::-1])
    bx.set_xlim(0.45, 0.60)
    bx.set_ylim(-0.55, 3.55)
    bx.set_xlabel("AUC intra-ID" if fr else "Within-ID AUC")
    bx.set_title("B   Temps et physiologie" if fr else "B   Timing and physiology", loc="left")
    fig.subplots_adjust(wspace=0.58, bottom=0.25)
    finish(fig, "time", fr)


def ladder(fr):
    ridge = read("ladder_ridge")["models"]["S07_with_post"]
    reve = read("ladder_reve")["models"]["REVE_frozen_vid10"]
    models = [ridge["stress_va"], reve["stress_va"], reve["stress_logreg"]]
    sr = read("same_ridge")["models"]["S07_with_post"]
    se = read("same_reve")["models"]["REVE_frozen_vid10"]
    restricted = [sr["stress_va"], se["stress_va"], se["stress_logreg"]]
    names = (
        ["Ridge multimodal", "REVE + Ridge", "REVE + logistique"]
        if fr
        else ["Multimodal Ridge", "REVE + Ridge", "REVE + logistic"]
    )
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (model, name) in enumerate(zip(models, names, strict=True)):
        data = model["matched_n_train"]
        vals = [data["regimes"][r]["within_id_auc"] for r in REGIMES]
        ax.plot(
            np.arange(4), vals, marker=["o", "^", "s"][i], lw=1.6, ms=4, label=name, color=COLORS[i]
        )
        for j, result in enumerate((model, restricted[i])):
            con = result["matched_n_train"]["contrasts"]["R4_minus_R1"]
            val = con["diff"]
            lo, hi = con["ci95"]
            bx.errorbar(
                val,
                2 - i + (0.12 if j == 0 else -0.12),
                xerr=[[val - lo], [hi - val]],
                fmt="o" if j == 0 else "s",
                ms=4,
                capsize=2,
                color=COLORS[i],
                markerfacecolor=COLORS[i] if j == 0 else "white",
            )
    ax.set_xticks(np.arange(4), ["R1", "R2", "R3", "R4"])
    ax.set_ylabel("AUC intra-ID" if fr else "Within-ID AUC")
    ax.set_title(
        "A   Budget d’entraînement égal" if fr else "A   Equal training budget", loc="left"
    )
    ax.legend(fontsize=7.6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    bx.axvline(0, color="#555", ls=":", lw=0.9)
    bx.set_ylim(-0.5, 2.5)
    bx.set_yticks(range(3), names[::-1])
    bx.set_xlabel("Δ AUC (R4 − R1)")
    bx.set_title(
        "B   Écarts appariés et IC à 95 %" if fr else "B   Paired contrasts and 95% CIs", loc="left"
    )
    fig.subplots_adjust(wspace=0.60, bottom=0.26)
    finish(fig, "ladder", fr)


def cardiac(fr):
    data = read("raw_scores")["paired"]
    keys = [
        "strong_release-body",
        "strong_hc_after-body",
        "strong_net_before-body",
        "strong_net_after-body",
        "nonlinear_eeg_gain",
    ]
    labels = (
        [
            "REVE, EEG archivé",
            "Bandes après régr. ECG",
            "REVE brut",
            "REVE après régr. ECG",
            "Arbres + REVE après régr.",
        ]
        if fr
        else [
            "REVE, archived EEG",
            "Bands after ECG regression",
            "Raw REVE",
            "REVE after ECG regression",
            "Trees + REVE after regression",
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.65), sharey=True)
    for ax, measure, lab in zip(
        axes,
        ["delta_r", "reduction_rmse"],
        [
            "Gain de corrélation" if fr else "Correlation gain",
            "Réduction de RMSE (bpm)" if fr else "RMSE reduction (bpm)",
        ],
        strict=True,
    ):
        for i, k in enumerate(keys):
            st = data[k][measure]
            v = st["mean"]
            lo, hi = st["ci95"]
            ax.errorbar(
                v,
                len(keys) - 1 - i,
                xerr=[[v - lo], [hi - v]],
                fmt="o",
                ms=4,
                capsize=3,
                color=COLORS[3] if i == 3 else COLORS[0],
            )
        ax.axvline(0, color="#555", ls=":", lw=0.9)
        ax.set_xlabel(lab)
    axes[0].set_yticks(range(len(keys)), labels[::-1])
    axes[0].set_ylim(-0.5, len(keys) - 0.5)
    fig.subplots_adjust(left=0.24, wspace=0.22, bottom=0.26)
    finish(fig, "cardiac", fr)


def main():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    for fr in (False, True):
        timeline(read("controls"), fr)
        ladder(fr)
        cardiac(fr)


if __name__ == "__main__":
    main()
