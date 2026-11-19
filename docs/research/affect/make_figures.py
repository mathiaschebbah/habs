"""Draw the paper figures from the run log and the sealed-test aggregates.

Inputs are public aggregates only: ``docs/research/affect/experiments.tsv`` and
``results/sealed_test.json``. Run from the repository root.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"
MODELS = {
    "ed5ec4d": ("Baseline", GRAY),
    "b800330": ("Final", BLUE),
    "511f5f4": ("Final, EEG only", AQUA),
    "0dbb2d7": ("Final, peripheral only", ORANGE),
    "098e253": ("Session position only", YELLOW),
}
REGIMES = {
    "test_new_both": "New person, new theme\n(primary)",
    "test_new_person": "New person, seen theme",
    "test_new_video": "Seen person, new theme",
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
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
    }
)


def progress(log):
    """Development score of every experiment, with the running best."""
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    x = range(len(log))
    searched = log["status"].isin(["keep", "discard"])
    ax.scatter(
        [i for i, s in zip(x, searched, strict=True) if s],
        log.loc[searched, "score"],
        s=12,
        color=GRAY,
        label="Discarded",
        zorder=2,
    )
    kept = log["status"] == "keep"
    best = log["score"].where(kept).cummax().ffill()
    ax.step(
        x, best, where="post", color=BLUE, linewidth=2, label="Running best", zorder=3
    )
    ax.scatter(
        log.index[kept],
        log.loc[kept, "score"],
        s=28,
        color=BLUE,
        zorder=4,
        edgecolor="#fcfcfb",
        linewidth=1,
        label="Kept",
    )
    other = ~searched
    ax.scatter(
        log.index[other],
        log.loc[other, "score"],
        s=18,
        facecolor="none",
        edgecolor=ORANGE,
        linewidth=1.2,
        label="Ablation / reference",
        zorder=2,
    )
    offsets = {"baseline": (6, -11), "regional power": (-4, 9)}
    for i, row in log[kept].iterrows():
        dx, dy = offsets.get(row["short"], (3, 6))
        ax.annotate(
            row["short"],
            (i, row["score"]),
            xytext=(dx, dy),
            ha="left" if dx > 0 else "right",
            textcoords="offset points",
            fontsize=6.5,
            color=MUTED,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Experiment (chronological)")
    ax.set_ylabel("Development score\n(within-person ρ)")
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "progress.png")


def sealed(results):
    """Sealed-test score per regime and model, with 95% person bootstrap intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4), sharey=True, sharex=True)
    names = list(MODELS)
    for ax, (regime, title) in zip(axes, REGIMES.items(), strict=True):
        for k, ref in enumerate(names):
            m = results[f"candidates/{ref}.py"][regime]
            label, color = MODELS[ref]
            y = len(names) - 1 - k
            ax.plot(m["ci95"], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.scatter(m["score"], y, s=30, color=color, edgecolor="#fcfcfb", zorder=3)
            ax.annotate(
                f"{m['score']:.3f}",
                (m["score"], y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=MUTED,
            )
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xlabel("Score (within-person ρ)")
    axes[0].set_yticks(range(len(names)), [MODELS[r][0] for r in reversed(names)])
    axes[0].set_ylim(-0.5, len(names) - 0.2)
    axes[0].grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "sealed_test.png")


def modalities(log, results):
    """Valence and arousal rho by modality, development and sealed test."""
    refs = ["b800330", "511f5f4", "0dbb2d7"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), sharex=True, sharey=True)
    dev = log.set_index("commit")
    for ax, target in zip(axes, ("valence", "arousal"), strict=True):
        for k, ref in enumerate(refs):
            label, color = MODELS[ref]
            y = len(refs) - 1 - k
            test = results[f"candidates/{ref}.py"]["test_new_both"][f"{target}_rho"]
            ax.scatter(
                dev.loc[ref, f"{target}_rho"],
                y + 0.12,
                s=26,
                marker="o",
                facecolor="none",
                edgecolor=color,
                linewidth=1.4,
            )
            ax.scatter(test, y - 0.12, s=26, marker="o", color=color)
        ax.set_title(target.capitalize(), fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel("Within-person ρ")
    axes[0].set_yticks(range(len(refs)), [MODELS[r][0] for r in reversed(refs)])
    axes[0].set_ylim(-0.6, len(refs) - 0.4)
    proxies = [
        plt.scatter(
            [], [], s=26, facecolor="none", edgecolor=MUTED, label="Development"
        ),
        plt.scatter(
            [], [], s=26, color=MUTED, label="Sealed test (new person, new theme)"
        ),
    ]
    fig.legend(handles=proxies, frameon=False, ncol=2, loc="upper center", fontsize=6.5)
    fig.subplots_adjust(top=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "figures" / "modalities.png")


def main():
    """Write the three figures."""
    log = pd.read_csv(
        Path("docs/research/affect/experiments.tsv"), sep="\t", dtype={"commit": str}
    )
    short = {
        "ed5ec4d": "baseline",
        "abc371d": "person-centred ratings",
        "82a4253": "regional power",
        "4b39398": "+ relative power",
        "0f030c6": "+ channel ridge",
        "b800330": "",
    }
    log["short"] = log["commit"].map(short).fillna("")
    results = json.loads((HERE / "results" / "sealed_test.json").read_text())["models"]
    (HERE / "figures").mkdir(exist_ok=True)
    progress(log)
    sealed(results)
    modalities(log, results)


if __name__ == "__main__":
    main()
