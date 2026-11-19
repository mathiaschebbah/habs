"""Draw the stress-study figures from public aggregates.

Inputs: ``experiments.tsv``, ``results/sealed_test.json`` and
``results/correlates.json`` in this folder. Run from the repository root.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, GRAY, INK, MUTED = "#e34948", "#a3a29c", "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("bgr", [BLUE, "#f0efec", RED])
REGIONS = (
    "frontal",
    "left_temporal",
    "right_temporal",
    "central",
    "parietal",
    "occipital",
)
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
CONTRASTS = {
    "stress": "Stress vs all other trials",
    "vs_excitement": "Stress vs excitement\n(same high arousal)",
    "vs_calm_negative": "Stress vs calm negative\n(same negative valence)",
}
BODY = {
    "hrv_hr_delta": "Heart rate change vs 5 s before",
    "hrv_rr_mean": "Mean RR interval",
    "hrv_rmssd": "RMSSD (10 s)",
    "hrv_sdnn": "SDNN (10 s)",
    "ecg_hr": "Heart rate",
    "scr_rise": "Skin conductance rise, 1-6 s",
    "scr_slope_max": "Steepest conductance rise",
    "scr_count": "Number of skin responses",
    "eda_log_level": "Skin conductance level",
    "breath_rate": "Breathing rate",
    "breath_rate_delta": "Breathing rate change",
    "breath_depth": "Breathing depth",
    "breath_cv": "Breathing irregularity",
    "breath_pause": "Longest breathing pause",
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


def progress(log, short):
    """Development score of every experiment, with the running best."""
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    searched = log["status"].isin(["keep", "discard"])
    kept = log["status"] == "keep"
    ax.scatter(
        log.index[searched],
        log.loc[searched, "score"],
        s=12,
        color=GRAY,
        label="Discarded",
        zorder=2,
    )
    best = log["score"].where(kept).cummax().ffill()
    ax.step(
        log.index,
        best,
        where="post",
        color=BLUE,
        linewidth=2,
        label="Best so far",
        zorder=3,
    )
    ax.scatter(
        log.index[kept],
        log.loc[kept, "score"],
        s=28,
        color=BLUE,
        edgecolor="#fcfcfb",
        linewidth=1,
        label="Kept",
        zorder=4,
    )
    ax.scatter(
        log.index[~searched],
        log.loc[~searched, "score"],
        s=18,
        facecolor="none",
        edgecolor=ORANGE,
        linewidth=1.2,
        label="Ablation",
        zorder=2,
    )
    for i, row in log[kept].iterrows():
        ax.annotate(
            short.get(row["commit"], ""),
            (i, row["score"]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=6.5,
            color=MUTED,
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Experiment (in the order we ran them)")
    ax.set_ylabel("Development score\n(within-person AUC)")
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.2))
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "progress.png")


def sealed(results, models):
    """Sealed-test AUC per contrast, for the two regimes with new people."""
    rows = [
        ("score", "Stress vs all other"),
        ("vs_excitement", "Stress vs excitement"),
        ("vs_calm_negative", "Stress vs calm negative"),
        ("vs_relaxed", "Stress vs relaxed"),
    ]
    regimes = {
        "test_new_both": "New person, new theme (primary)",
        "test_new_person": "New person, seen theme",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True, sharex=True)
    step = 0.8 / len(models)
    for ax, (regime, title) in zip(axes, regimes.items(), strict=True):
        ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.6)
        first = results[f"candidates/{next(iter(models))}.py"][regime]
        for k, (ref, (label, color)) in enumerate(models.items()):
            m = results[f"candidates/{ref}.py"][regime]
            for j, (key, _) in enumerate(rows):
                y = len(rows) - 1 - j + 0.4 - (k + 0.5) * step
                ax.scatter(
                    m[key],
                    y,
                    s=22,
                    color=color,
                    zorder=3,
                    label=label if j == 0 else None,
                )
                if key == "score":
                    ax.plot(m["ci95"], [y, y], color=color, linewidth=2)
        ax.axvline(0.5, color=MUTED, linewidth=0.8)
        counts = [first["people"]] + [first[f"{key}_people"] for key, _ in rows[1:]]
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xlabel("Within-person AUC (chance = 0.5)")
        for j, n in enumerate(counts):
            ax.annotate(
                f"n={n}",
                (1, len(rows) - 1 - j + 0.42),
                xycoords=("axes fraction", "data"),
                ha="right",
                fontsize=6,
                color=MUTED,
            )
    axes[0].set_yticks(range(len(rows)), [r[1] for r in reversed(rows)])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(HERE / "figures" / "sealed_test.png")


def eeg_maps(correlates):
    """Regional band-power effect sizes for three contrasts."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4), sharey=True)
    limit = max(
        abs(correlates[c][f"bp_vid_{r}_{b}"]["d"])
        for c in CONTRASTS
        for r in REGIONS
        for b in BANDS
    )
    for ax, (contrast, title) in zip(axes, CONTRASTS.items(), strict=True):
        grid = np.array(
            [
                [correlates[contrast][f"bp_vid_{r}_{b}"]["d"] for b in BANDS]
                for r in REGIONS
            ]
        )
        image = ax.imshow(grid, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        for i, r in enumerate(REGIONS):
            for j, b in enumerate(BANDS):
                low, high = correlates[contrast][f"bp_vid_{r}_{b}"]["ci95"]
                if low > 0 or high < 0:  # interval excludes zero
                    ax.text(j, i, "•", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(BANDS)), BANDS, rotation=45)
        ax.set_title(title, fontsize=7.5, color=INK)
    axes[0].set_yticks(range(len(REGIONS)), [r.replace("_", " ") for r in REGIONS])
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Within-person effect size d", fontsize=7)
    fig.savefig(HERE / "figures" / "correlates_eeg.png", bbox_inches="tight")


def body(correlates):
    """Heart, skin and breathing effect sizes for three contrasts."""
    names = [n for n in BODY if n in correlates["stress"]]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.0), sharey=True, sharex=True)
    for ax, (contrast, title) in zip(axes, CONTRASTS.items(), strict=True):
        ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.6)
        for k, name in enumerate(names):
            item = correlates[contrast][name]
            y = len(names) - 1 - k
            significant = item["ci95"][0] > 0 or item["ci95"][1] < 0
            color = (RED if item["d"] > 0 else BLUE) if significant else GRAY
            ax.plot(item["ci95"], [y, y], color=color, linewidth=1.6)
            ax.scatter(item["d"], y, s=16, color=color, zorder=3)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xlabel("Effect size d")
    axes[0].set_yticks(range(len(names)), [BODY[n] for n in reversed(names)])
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "correlates_body.png")


def main(short, models):
    """Write the four figures."""
    log = pd.read_csv(HERE / "experiments.tsv", sep="\t", dtype={"commit": str})
    (HERE / "figures").mkdir(exist_ok=True)
    progress(log, short)
    results = json.loads((HERE / "results" / "sealed_test.json").read_text())
    sealed(results["models"], models)
    correlates = json.loads((HERE / "results" / "correlates.json").read_text())
    eeg_maps(correlates)
    body(correlates)


if __name__ == "__main__":
    main(
        short={
            "a780870": "baseline",
            "c4d5630": "arousal minus valence",
            "68486f1": "person-centred features",
            "3b4b57f": "",
        },
        models={
            "a780870": ("Starting model", GRAY),
            "3b4b57f": ("Final model", BLUE),
            "39471d3": ("Final, EEG only", AQUA),
            "c6a1840": ("Final, body only", ORANGE),
        },
    )
