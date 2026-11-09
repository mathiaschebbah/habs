"""Plot measured support, simulated precision and audited carryover estimates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from habs.figures import COLORS, _save, _style


def export_figures(result, precision, feasibility, directory):
    """Export five figures while clearly separating simulation from observations.

    Parameters
    ----------
    result : dict
        Frozen empirical carryover and grouped prediction outputs.
    precision : dict
        Pre-effect simulated design-precision results.
    feasibility : dict
        Read-only actual transition support before final restrictions.
    directory : str or pathlib.Path
        Publication-safe PNG directory.

    Returns
    -------
    list of str
        Saved figure paths, without any individual identifiers.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _style()
    paths = []
    structure = result["structure"]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    labels = ["All valid pairs", "Primary J1", "Primary J2"]
    counts = [
        feasibility["eligible_transitions"],
        structure["by_day"]["1"],
        structure["by_day"]["2"],
    ]
    axes[0].bar(labels, counts, color=["#999999", COLORS[0], COLORS[1]])
    for i, n in enumerate(counts):
        axes[0].text(i, n + 80, f"{n:,}", ha="center", fontsize=9)
    axes[0].set(
        ylabel="Observed video transitions",
        ylim=(0, max(counts) * 1.18),
        title=f"Paired support: {structure['people']} identifiers",
    )
    gap = feasibility["gap_quantiles_seconds"]
    positions = [gap[k] for k in ("min", "q10", "q25", "median", "q75", "q90", "max")]
    axes[1].plot(positions, np.zeros(7), "o-", color=COLORS[0], markersize=5)
    for i, (x, label) in enumerate(
        zip(
            positions, ("min", "10%", "25%", "median", "75%", "90%", "max"), strict=True
        )
    ):
        axes[1].text(
            x,
            0.05 if i % 2 else -0.09,
            label,
            ha="center",
            va="center",
            fontsize=8,
            rotation=45,
        )
    axes[1].set(
        xlabel="Actual gap after previous video (seconds)",
        ylim=(-0.2, 0.2),
        yticks=[],
        title="All valid-pair gap quantiles",
    )
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure1_transition_support"))

    scenarios = [
        s
        for s in precision["scenarios"]
        if s["scenario"] == "random_effect"
        and (s["true_correlation"] == 0.6 or s["true_random_sd"] == 0)
    ]
    figure, ax = plt.subplots(figsize=(7.5, 3.7))
    for i, s in enumerate(scenarios):
        lo, mid, hi = s["estimated_covariance_quantiles"]
        ax.errorbar(
            i, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", color=COLORS[0], capsize=4
        )
        ax.scatter(i, s["true_covariance"], marker="x", color=COLORS[1], s=60, zorder=4)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set(
        xticks=list(range(len(scenarios))),
        xticklabels=[f"SD {s['true_random_sd']:.2f}" for s in scenarios],
        xlabel="Simulated between-person variation, true cross-day correlation 0.6",
        ylabel="Cross-day covariance",
        title="SIMULATED design precision, not empirical effects",
    )
    ax.plot([], [], "o", color=COLORS[0], label="Median and 95% simulation dispersion")
    ax.plot([], [], "x", color=COLORS[1], label="Known generating covariance")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure2_simulated_precision"))

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    errors = np.asarray(result["anonymous_delta_standard_errors"])
    for ax, key, title in zip(
        axes,
        ("anonymous_delta_estimates", "anonymous_delta_adjusted_estimates"),
        (
            "Observed adjusted-for-context slopes",
            "Also adjusted for J1 scale ratio and counts",
        ),
        strict=True,
    ):
        values = np.asarray(result[key])
        ax.errorbar(
            values[0],
            values[1],
            xerr=errors[0],
            yerr=errors[1],
            fmt="none",
            ecolor="#cccccc",
            alpha=0.45,
            linewidth=0.6,
        )
        ax.scatter(values[0], values[1], s=22, color=COLORS[0], alpha=0.8)
        ax.axhline(0, color="#aaaaaa", linewidth=0.7)
        ax.axvline(0, color="#aaaaaa", linewidth=0.7)
        ax.set(
            xlabel="J1 subjective minus EEG history slope",
            ylabel="J2 subjective minus EEG history slope",
            title=title,
        )
        ax.text(
            0.03,
            0.97,
            f"n={structure['people']}; bars: sampling ±1 SE",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure3_personal_contrasts"))

    component = result["variance_components"]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    profile = np.asarray(component["rho_profile"])
    axes[0].plot(profile[:, 0], profile[:, 1], color=COLORS[0], linewidth=1.6)
    axes[0].axhline(
        3.841458820694124,
        color=COLORS[1],
        linestyle="--",
        label="95% profile reference",
    )
    axes[0].set(
        xlabel="Random-effect correlation J1–J2",
        ylabel="Twice restricted log-likelihood loss",
        xlim=(-1, 1),
        title="Correlation identifiability",
    )
    axes[0].legend(frameon=False, fontsize=8)
    if component["correlation_weakly_identified"]:
        axes[0].text(
            0.03,
            0.97,
            "Weakly identified correlation",
            transform=axes[0].transAxes,
            va="top",
            fontsize=9,
        )
    lo, hi = component["covariance_ci95"]
    point = component["covariance"]
    axes[1].plot([lo, hi], [0, 0], color=COLORS[0], linewidth=2)
    axes[1].scatter([point], [0], color=COLORS[0], s=45)
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Cross-day random-effect covariance",
        yticks=[],
        ylim=(-0.5, 0.5),
        title="Model-based 95% bootstrap interval",
    )
    axes[1].text(
        0.03,
        0.95,
        f"Covariance {point:+.4f}\n95% [{lo:+.4f}, {hi:+.4f}]\n"
        f"Null bootstrap p={component['null_bootstrap_p_two_sided']:.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure4_reproducibility_uncertainty"))

    prediction = result["prediction"]
    ref = np.asarray(prediction["models"]["ablation"]["anonymous_person_errors"])
    full = np.asarray(prediction["models"]["full"]["anonymous_person_errors"])
    gains = np.sort(ref - full)
    figure, axes = plt.subplots(
        1, 2, figsize=(9.5, 3.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    axes[0].bar(
        list(range(len(gains))),
        gains,
        color=[COLORS[0] if v > 0 else COLORS[1] for v in gains],
        width=0.9,
    )
    axes[0].axhline(0, color="#888888", linewidth=0.8)
    axes[0].set(
        xlabel="Anonymous identifiers, ordered by gain",
        ylabel="MAE gain from personal history slope",
        title="Out-of-person, one-step J2 prediction",
    )
    for index, (key, _label) in enumerate(
        (
            ("full_vs_ablation", "Exact slope ablation"),
            ("full_vs_refitted_restriction", "Refitted restriction"),
        )
    ):
        item = prediction[key]
        low, high = item["ci95"]
        axes[1].plot([low, high], [index, index], color=COLORS[index], linewidth=2)
        axes[1].scatter([item["gain"]], [index], color=COLORS[index], s=40)
        axes[1].text(
            item["gain"], index + 0.13, f"{item['gain']:+.4f}", ha="center", fontsize=8
        )
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    axes[1].set(
        yticks=[0, 1],
        yticklabels=["Exact ablation", "Refitted restriction"],
        ylim=(-0.4, 1.5),
        xlabel="Equal-person MAE gain (positive is better)",
        title="Paired participant 95% intervals",
    )
    figure.tight_layout()
    paths.append(_save(figure, directory, "figure5_personal_prediction_gain"))
    return paths
