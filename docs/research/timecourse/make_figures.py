"""Draw the study-05 figures from public aggregates in ``results/``.

Run from the repository root after the six analyses have been saved.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GRAY, INK, MUTED = "#e87ba4", "#a3a29c", "#0b0b0b", "#52514e"
SECONDS = np.arange(15) - 3 + 0.5  # bin centres
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


def _video(ax):
    ax.axvspan(0, 10, color="#f0efec", zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.6)


def curves():
    """Within-person effect size per second, development band and test line."""
    data = _load("curves")["curves"]
    panels = {
        "hr": "Heart rate",
        "sc": "Skin conductance",
        "resp_amp": "Breathing amplitude",
        "eeg_frontal_delta": "Frontal delta power",
    }
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0), sharex=True)
    for ax, (signal, title) in zip(axes, panels.items(), strict=True):
        _video(ax)
        dev = data["dev"]["stress"][signal]
        test = data["test_people"]["stress"][signal]
        ax.fill_between(SECONDS, dev["low"], dev["high"], color=BLUE, alpha=0.2, lw=0)
        ax.plot(
            SECONDS, dev["d"], color=BLUE, lw=2, label=f"Development ({dev['people']})"
        )
        ax.plot(
            SECONDS,
            test["d"],
            color=ORANGE,
            lw=1.4,
            ls="--",
            label=f"Test people ({test['people']})",
        )
        ax.set_title(title, fontsize=7.5, color=INK)
    fig.supxlabel("Seconds from video onset (grey: video playing)", fontsize=7.5)
    axes[0].set_ylabel("Effect size d\n(stress vs other)")
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "curves.png")


def latency():
    """Plot detection AUC from the first t seconds, development and test."""
    dev = _load("latency_dev")["latency_dev"]
    test = _load("latency_test")["latency_test"]
    families = {
        "all": ("All signals", BLUE),
        "eeg": ("EEG", AQUA),
        "body": ("Heart, skin, breathing", ORANGE),
    }
    t = np.arange(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for family, (label, color) in families.items():
        auc = [dev[family][str(s)]["auc"] for s in t]
        low = [dev[family][str(s)]["ci95"][0] for s in t]
        high = [dev[family][str(s)]["ci95"][1] for s in t]
        axes[0].fill_between(t, low, high, color=color, alpha=0.18, lw=0)
        axes[0].plot(t, auc, color=color, lw=2, marker="o", ms=3, label=label)
        regime = [test[family][str(s)]["test_new_video"] for s in t]
        axes[1].fill_between(
            t,
            [r["ci95"][0] for r in regime],
            [r["ci95"][1] for r in regime],
            color=color,
            alpha=0.18,
            lw=0,
        )
        axes[1].plot(t, [r["auc"] for r in regime], color=color, lw=2, marker="o", ms=3)
    axes[0].set_title("Development cross-fit (new person, new theme)", fontsize=7.5)
    axes[1].set_title("Sealed test (known person, new theme)", fontsize=7.5)
    for ax in axes:
        ax.axhline(0.5, color=MUTED, lw=0.8)
        ax.grid(True, axis="y", color="#e6e5e0", lw=0.6)
        ax.set_xlabel("Seconds of video used")
        ax.set_xticks(t)
    axes[0].set_ylabel("Within-person AUC")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "latency.png")


def emotions():
    """Heart-rate change of stress trials by reported emotion (development)."""
    data = _load("emotions")["emotions"]["dev"]["hr"]
    colors = {"fear": BLUE, "disgust": AQUA, "sadness": YELLOW, "anger": MAGENTA}
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    _video(ax)
    for emotion, color in colors.items():
        item = data[emotion]
        ax.plot(
            SECONDS,
            item["curve"],
            color=color,
            lw=1.8,
            label=f"{emotion} ({item['people']})",
        )
    ax.set_xlabel("Seconds from video onset")
    ax.set_ylabel("Heart rate vs the person's\nnon-stress trials (bpm)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "emotions.png")


def hrv():
    """Agreement of ultra-short windows with the same 60-second window."""
    data = _load("hrv")
    seconds = (5, 10, 20, 30)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.grid(True, axis="y", color="#e6e5e0", lw=0.6)
    for measure, label, color in (
        ("hr", "Heart rate", BLUE),
        ("rmssd", "RMSSD", ORANGE),
    ):
        ax.plot(
            seconds,
            [data[f"{measure}_{s}"]["pearson"] for s in seconds],
            color=color,
            lw=2,
            marker="o",
            ms=4,
            label=f"{label}, all windows",
        )
        ax.plot(
            seconds,
            [data[f"{measure}_{s}"]["within_recording_r"] for s in seconds],
            color=color,
            lw=1.4,
            ls="--",
            marker="o",
            ms=3,
            label=f"{label}, within recording",
        )
    ax.set_xticks(seconds)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("Correlation with 60 s")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "hrv.png")


if __name__ == "__main__":
    curves()
    emotions()
    hrv()
    if (HERE / "results" / "latency_dev.json").is_file():
        latency()
