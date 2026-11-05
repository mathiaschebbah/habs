"""Build publication figures and tables from the aggregate HABS results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "eda": "EDA",
    "raw": "Voltage GSR",
    "eeg": "EEG",
    "ecg": "ECG",
    "all": "EDA + EEG + ECG",
    "video": "Moyenne physiologique par vidéo",
    "missing": "Disponibilité des capteurs",
}
COLORS = {"eda": "#D55E00", "eeg": "#0072B2", "ecg": "#009E73", "all": "#0072B2"}


def save_figure(figure, destination):
    """Save a vector figure and a readable preview."""
    figure.tight_layout()
    figure.savefig(
        destination.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def quality_figure(quality, destination):
    """Show how complementary sensors retain repeated-session participants."""
    cohorts = sorted(quality["cohorts"], key=int)
    x = np.arange(len(cohorts))
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for position, modality in enumerate(("eda", "ecg", "eeg")):
        values = [
            quality["cohorts"][c][modality]["two_days_20_trials"] for c in cohorts
        ]
        bars = ax.bar(
            x + (position - 1) * 0.24,
            values,
            width=0.23,
            color=COLORS[modality],
            label=LABELS[modality],
        )
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(x, [f"Cohorte {c}" for c in cohorts])
    ax.set_ylabel("Identifiants avec deux séances\nd'au moins 20 essais exploitables")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    save_figure(fig, destination)


def gain_figure(case, destination):
    """Compare physiological corrections to the same calibrated reference."""
    modes = ["eda", "raw", "eeg", "ecg", "all", "video", "missing"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for y, mode in enumerate(modes):
        result = case["effects_vs_affine"][mode]["centered_mse"]
        mean, (low, high) = result["mean"], result["ci95"]
        ax.plot([low, high], [y, y], color=COLORS.get(mode, "#777777"), linewidth=2)
        ax.scatter([mean], [y], color=COLORS.get(mode, "#777777"), s=32, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.1, color="#777777", linestyle=":", linewidth=0.8)
    ax.set_yticks(range(len(modes)), [LABELS[m] for m in modes])
    ax.invert_yaxis()
    ax.set_xlabel("Réduction de l'erreur centrée par rapport à la référence (points²)")
    ax.set_title(
        f"Cohorte 2 · {case['test_people']} identifiants réservés au test", fontsize=10
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    save_figure(fig, destination)


def replication_figure(study, destination):
    """Keep cohort-specific estimates separate to avoid assuming unique identities."""
    cases = [
        (cohort, case)
        for cohort, case in sorted(study["cohorts"].items())
        if case["status"] == "complete"
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for y, (_cohort, case) in enumerate(cases):
        for offset, result, color, label in (
            (
                -0.12,
                case["effects_vs_affine"]["all"]["centered_mse"],
                "#0072B2",
                "Référence calibrée",
            ),
            (
                0.12,
                case["individual_signal_controls"]["video"]["centered_mse"],
                "#009E73",
                "Contrôle par vidéo",
            ),
        ):
            low, high = result["ci95"]
            ax.plot([low, high], [y + offset, y + offset], color=color, linewidth=2)
            ax.scatter(
                [result["mean"]],
                [y + offset],
                color=color,
                s=28,
                label=label if y == 0 else None,
            )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(
        range(len(cases)),
        [f"Cohorte {c} (test : {s['test_people']})" for c, s in cases],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Réduction de l'erreur centrée avec les trois modalités (points²)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_figure(fig, destination)


def tables(alignment, quality, study, destination):
    """Write the numerical tables directly from the saved results."""
    rows = []
    for cohort in sorted(quality["cohorts"], key=int):
        case = study["cohorts"].get(cohort, {})
        values = [
            cohort,
            str(alignment["cohorts"][cohort]["people_with_notes"]),
            str(quality["cohorts"][cohort]["any"]["two_days_20_trials"]),
            str(case.get("source_only_people", "n.a.")),
            str(case.get("development_people", "n.a.")),
            str(case.get("test_people", "n.a.")),
            str(case.get("test_trials", "n.a.")),
        ]
        rows.append(" & ".join(values) + r" \\")
    table1 = (
        r"""\begin{table}[ht]
\centering\small
\caption{Effectifs par cohorte. Un individu peut avoir plusieurs identifiants
entre cohortes. Réf. désigne les identifiants utilisés seulement pour estimer
la référence par vidéo. Dév. et Test sont les effectifs après support des vidéos.}
\label{tab:population}
\begin{tabular}{lrrrrrr}\toprule
Cohorte & ID annotés & Deux séances & Réf. & Dév. & Test & Essais \\
\midrule
"""
        + "\n".join(rows)
        + "\n"
        + r"\bottomrule\end{tabular}\end{table}"
        + "\n"
    )
    case = study["cohorts"]["2"]
    rows = []
    for mode, label in [
        ("video_only", "Vidéo"),
        ("level", "Vidéo + niveau personnel"),
        ("affine", "Référence calibrée"),
        *[(m, LABELS[m]) for m in LABELS],
    ]:
        scores = case["models"][mode]
        values = [label] + [
            "n.a." if scores[m] is None else f"{scores[m]:.3f}"
            for m in ("centered_mse", "mae", "spearman")
        ]
        if scores["spearman"] is not None:
            values[-1] += f" ({scores['rank_people']})"
        rows.append(" & ".join(values) + r" \\")
    table2 = (
        r"""\begin{table}[ht]
\centering\small
\caption{Résultats de la cohorte 2. Chaque personne a le même poids.
L'erreur centrée est en points au carré ; la MAE est en points et conserve
les erreurs de niveau. Le nombre de corrélations définies est entre parenthèses.}
\label{tab:models}
\begin{tabular}{lrrr}\toprule
Modèle & Erreur centrée & MAE & Spearman \\\midrule
"""
        + "\n".join(rows)
        + "\n"
        + r"\bottomrule\end{tabular}\end{table}"
        + "\n"
    )
    destination.write_text(table1 + "\n" + table2)


def build(result_dir, figure_dir):
    """Generate available figures, requiring real final results for model plots."""
    result_dir, figure_dir = Path(result_dir), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    quality_path = result_dir / "multimodal_quality.json"
    if not quality_path.exists():
        quality_path = result_dir / "quality.json"
    quality = json.loads(quality_path.read_text())
    quality_figure(quality, figure_dir / "quality")
    path = result_dir / "study.json"
    if not path.exists():
        path = result_dir / "study/results.json"
    if path.exists():
        study = json.loads(path.read_text())
        if study["stage"] != "final" or "provenance" not in study:
            raise ValueError(
                "Model figures require a final, provenance-linked HABS run"
            )
        gain_figure(study["cohorts"]["2"], figure_dir / "individual_gain")
        replication_figure(study, figure_dir / "cohorts")
        alignment = json.loads((result_dir / "alignment.json").read_text())
        tables(alignment, quality, study, figure_dir.parent / "tables.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("figure_dir", type=Path)
    arguments = parser.parse_args()
    build(arguments.result_dir, arguments.figure_dir)
