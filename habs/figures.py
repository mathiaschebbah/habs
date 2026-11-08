"""Export the seven prespecified scientific figure categories as aggregate plots."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from habs.analysis import MARKERS
from habs.data import target_values

LABELS = {
    "eeg_frontal_alpha": "Frontal relative alpha",
    "ecg_hr": "Median heart rate",
    "eda_log_level": "Log skin conductance",
}
COLORS = ("#276b9a", "#b96032", "#577947")


def _save(figure, directory, name):
    path = directory / (name + ".png")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return str(path)


def _style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "savefig.facecolor": "white",
        }
    )


def export_figures(features, result, structure, directory):
    """Write empirical distributions, comparisons and conditional profile figures.

    Parameters
    ----------
    features : pandas.DataFrame
        Rated aligned trials from the three analysed cohorts.
    result : dict
        Frozen reserved evaluation and descriptive aggregate statistics.
    structure : dict or None
        In-memory PCA coordinates; published projections contain binned values only.
    directory : str or pathlib.Path
        Publication-safe figure output directory.

    Returns
    -------
    list of str
        Paths of the seven exported PNG files.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _style()
    paths = []
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), sharex=True)
    for ax, cohort in zip(axes, (1, 2, 4), strict=True):
        for day, color in zip((1, 2), COLORS[:2], strict=True):
            rows = features.loc[
                (features["cohort"] == cohort) & (features["recorded_day"] == day)
            ]
            ax.hist(
                target_values(rows),
                bins=np.linspace(0, 1, 21),
                histtype="step",
                linewidth=1.6,
                color=color,
                label=f"J{day}, n={len(rows):,}",
            )
        ax.set(
            title=f"Cohort {cohort}", xlabel="Aversive activation proxy S", xlim=(0, 1)
        )
        ax.legend(fontsize=8, frameon=False)
    axes[0].set(ylabel="Aligned rated trials")
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure1_proxy_distribution"))

    contrast = result["primary"]["comparisons"]["M1_minus_M4"]
    reference = np.asarray(contrast["anonymous_reference_errors"])
    multimodal = np.asarray(contrast["anonymous_candidate_errors"])
    figure, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.scatter(reference, multimodal, s=35, color=COLORS[0], alpha=0.8)
    limit = max(0.05, float(max(reference.max(), multimodal.max()) * 1.08))
    ax.plot([0, limit], [0, limit], color="#888888", linewidth=1, linestyle="--")
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="M1 individual MAE",
        ylabel="M4 individual MAE",
        title="Paired later-session prediction",
    )
    lo, hi = contrast["ci95"]
    ax.text(
        0.04,
        0.96,
        f"{contrast['people']} reserved identifiers\n"
        f"Mean gain {contrast['delta_mae']:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure2_paired_prediction"))

    specificity = result["descriptions"]["specificity"]
    figure, ax = plt.subplots(figsize=(7.4, 3.3))
    for position, name in enumerate(MARKERS):
        item = specificity["markers"][name]
        if item["mean"] is None:
            ax.text(
                0, position, f" insufficient support, n={item['people']}", fontsize=9
            )
            continue
        mean = item["mean"]
        lo, hi = item["ci95"]
        ax.errorbar(
            mean,
            position,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=COLORS[position],
            capsize=4,
        )
        ax.text(
            1.01,
            position,
            f"n={item['people']}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color="#999999", linewidth=1)
    ax.set(
        yticks=range(3),
        yticklabels=[LABELS[n] for n in MARKERS],
        xlabel="Negative minus positive response, development SD units",
        title="High-arousal responses matched within person",
    )
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure3_specificity"))

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), sharex=True)
    for ax, name, color in zip(axes, MARKERS, COLORS, strict=True):
        points = [
            p
            for p in result["descriptions"]["dose_response"][name]["points"]
            if p["mean"] is not None
        ]
        if points:
            x = [p["score_center"] for p in points]
            mean = np.asarray([p["mean"] for p in points])
            intervals = np.asarray([p["ci95"] for p in points])
            ax.errorbar(
                x,
                mean,
                yerr=[mean - intervals[:, 0], intervals[:, 1] - mean],
                fmt="o-",
                color=color,
                capsize=3,
                linewidth=1.3,
            )
        ax.axhline(0, color="#aaaaaa", linewidth=0.8)
        ax.set(title=LABELS[name], xlabel="Proxy S bin", xlim=(-0.04, 1))
    axes[0].set(ylabel="J1- and stimulus-centered physiology (SD)")
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure4_dose_response"))

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharex=True, sharey=True)
    if structure is not None:
        xy = structure["development_coordinates"]
        cluster = structure["development_cluster"]
        count = len(structure["centroids"])
        axes[0].hexbin(
            xy[:, 0],
            xy[:, 1],
            C=cluster,
            gridsize=18,
            mincnt=5,
            reduce_C_function=lambda a: Counter(a).most_common(1)[0][0],
            cmap="tab10",
            vmin=-0.5,
            vmax=count - 0.5,
        )
        for index, center in enumerate(structure["centroids"]):
            axes[0].text(
                center[0],
                center[1],
                chr(65 + index),
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "#555555", "pad": 3},
                fontsize=11,
            )
        future_xy = structure["test_coordinates"]
        if len(future_xy):
            image = axes[1].hexbin(
                future_xy[:, 0],
                future_xy[:, 1],
                C=structure["test_score"],
                gridsize=18,
                mincnt=5,
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            figure.colorbar(image, ax=axes[1], label="Mean proxy S")
        variance = structure["explained_variance"]
        axes[0].set(ylabel=f"PC2, {100 * variance[1]:.1f}% dev variance")
        for ax in axes:
            ax.set(xlabel=f"PC1, {100 * variance[0]:.1f}% dev variance")
    else:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Insufficient complete-case support",
                transform=ax.transAxes,
                ha="center",
            )
    axes[0].set_title("Development J1: modal cluster per bin")
    axes[1].set_title("Reserved J2: mean response per bin")
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure5_profile_structure"))

    figure, ax = plt.subplots(figsize=(6.9, 3.6))
    stability = result["descriptions"]["profiles"].get("stability", {})
    if stability.get("status") == "evaluated":
        ax.hist(
            stability["identity_shuffle_means"],
            bins=25,
            color="#c6d5df",
            edgecolor="white",
        )
        ax.axvline(
            stability["matched_similarity"],
            color=COLORS[1],
            linewidth=2,
            label="Matched identity",
        )
        ax.axvline(
            stability["between_person_similarity"],
            color="#555555",
            linestyle="--",
            label="All different identities",
        )
        ax.set(
            title=f"J1/J2 profile similarity, n={stability['people']}",
            xlabel="Mean Hellinger similarity",
            ylabel="Identity shuffles",
        )
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient two-session complete-case support",
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_title("Individual profile stability")
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure6_profile_stability"))

    names = [
        name
        for name in (
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "missingness",
            "no_EDA",
            "history_only",
            "nonlinear",
            "M5",
            "continuous_profile",
        )
        if name in result["primary"]["models"]
    ]
    labels = {
        "M0": "Stimulus and context",
        "M1": "M1 personal calibration",
        "M2": "M2 EEG",
        "M3": "M3 ECG + EDA",
        "M4": "M4 multimodal",
        "missingness": "Sensor-availability control",
        "no_EDA": "EEG + ECG",
        "history_only": "J1 physiology only",
        "nonlinear": "Physiological splines",
        "M5": "M5 cluster history",
        "continuous_profile": "Continuous PCA history",
    }
    figure, ax = plt.subplots(figsize=(8.3, 5.1))
    for position, name in enumerate(names):
        errors = np.asarray(
            result["primary"]["models"][name]["anonymous_person_errors"]
        )
        rng = np.random.default_rng(20261108)
        boot = errors[rng.integers(0, len(errors), size=(5000, len(errors)))].mean(
            axis=1
        )
        mean = float(errors.mean())
        lo, hi = np.quantile(boot, [0.025, 0.975])
        color = COLORS[0] if name == "M4" else "#657884"
        ax.barh(position, mean, height=0.62, color=color, alpha=0.85)
        ax.errorbar(
            mean,
            position,
            xerr=[[mean - lo], [hi - mean]],
            fmt="none",
            capsize=3,
            color="#222222",
            linewidth=1,
        )
    zero = result["primary"]["models"]["constant_zero"]["macro_mae"]
    ax.axvline(
        zero, color="#b96032", linestyle="--", linewidth=1.2, label="Constant zero"
    )
    ax.set(
        yticks=range(len(names)),
        yticklabels=[labels[name] for name in names],
        xlabel="Participant-macro MAE, lower is better",
        title="Frozen later-session ablation",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure7_modality_ablation"))
    return paths
