"""Draw the study-06 figures from public aggregates in ``results/``.

Run from the repository root after the analyses have been saved.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"
SIGNALS = {
    "heart": "Heart rate",
    "skin": "Skin conductance",
    "breathing": "Breathing depth",
    "delta": "EEG delta",
    "alpha": "EEG alpha",
}
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


def _load(name):
    return json.loads((HERE / "results" / f"{name}.json").read_text())


def coupling():
    """Shared versus own coefficients of each response on valence and arousal."""
    data = _load("main")["coupling"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    y = np.arange(len(SIGNALS))[::-1]
    for ax, rating in zip(axes, ("valence", "arousal"), strict=True):
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(True, axis="x", color="#e6e5e0", lw=0.6)
        for offset, part, color, name in (
            (0.14, "beta_shared", BLUE, "What other viewers felt (the video)"),
            (-0.14, "beta_own", ORANGE, "What this person felt beyond them"),
        ):
            values = np.array([data[s][rating][part] for s in SIGNALS])
            ax.errorbar(
                values[:, 0],
                y + offset,
                xerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]],
                fmt="o",
                ms=4,
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=0,
                label=name,
            )
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_title(f"Reported {rating}", fontsize=7.5, color=INK)
        ax.set_xlabel("Within-person coefficient (SD per SD)")
    axes[0].set_yticks(y, list(SIGNALS.values()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "figures" / "coupling.png")


def room():
    """Within-person AUC versus same-video AUC for each score."""
    data = _load("main")["room"]
    names = {"model": "All signals (model)", **SIGNALS}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    y = np.arange(len(names))[::-1]
    ax.axvline(0.5, color=MUTED, lw=0.8)
    ax.grid(True, axis="x", color="#e6e5e0", lw=0.6)
    for offset, key, color, label in (
        (0.14, "within_person", BLUE, "Across videos, same person"),
        (-0.14, "same_video", ORANGE, "Across people, same video"),
    ):
        values = np.array([[data[n][key]["auc"], *data[n][key]["ci95"]] for n in names])
        ax.errorbar(
            values[:, 0],
            y + offset,
            xerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]],
            fmt="o",
            ms=4,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            label=label,
        )
    ax.set_yticks(y, list(names.values()))
    ax.set_xlabel("AUC, stress reaction vs other")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, loc="lower center", ncol=1)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(HERE / "figures" / "room.png")


def hep():
    """Heartbeat-evoked EEG and ECG effect size of stress, per 50 ms bin."""
    path = HERE / "results" / "hep.json"
    if not path.is_file():
        return
    curves = json.loads(path.read_text())["hep"]["curves"]
    t = -200 + 50 * np.arange(16) + 25
    panels = (
        ("hep_video_frontal", "EEG, frontal", BLUE),
        ("hep_video_central", "EEG, central", BLUE),
        ("ecg_video", "ECG", GRAY),
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9), sharex=True, sharey=True)
    for ax, (key, title, color) in zip(axes, panels, strict=True):
        c = curves[key]
        ax.axvspan(250, 600, color="#f0efec", zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.fill_between(t, c["low"], c["high"], color=color, alpha=0.2, lw=0)
        ax.plot(t, c["d"], color=color, lw=2)
        ax.set_title(f"{title}, locked to R peaks", fontsize=7.5)
    axes[1].set_xlabel("ms after the R peak (grey: tested window)")
    axes[0].set_ylabel("Effect size d\n(stress vs other)")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "hep.png")


if __name__ == "__main__":
    coupling()
    room()
    hep()
