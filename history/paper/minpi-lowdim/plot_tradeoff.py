"""Compare the all-width single-peer rule directly with each timed baseline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
CASES = PAPER.parents[1] / "research" / "peer_crossover" / "evidence" / "paired-cases.csv"
COMPARATORS = [
    ("full", "Full", "#006B70", "-"),
    ("F_single", "Width-matched F", "#485469", "--"),
    ("random_single", "Width-matched random", "#B15A24", ":"),
]
SCORE_EPS = 1e-12  # The comparison protocol's tolerance for numerical ties.
MAX_LOSS = 0.10


def draw_panel(ax, cases):
    for key, label, color, style in COMPARATORS:
        qualifies = (
            (cases.k_single < cases.p)
            & (cases.total_ms1_single < cases[f"total_ms1_{key}"])
            & (cases.total_ms2_single < cases[f"total_ms2_{key}"])
        )
        losses = (cases.loc[qualifies, f"score_{key}"] - cases.loc[qualifies, "score_single"])
        thresholds = np.sort(np.maximum(0, losses.to_numpy() - SCORE_EPS))
        cutoffs = np.unique(np.r_[0, thresholds[thresholds <= MAX_LOSS], MAX_LOSS])
        counts = np.searchsorted(thresholds, cutoffs, side="right")
        rates = 100 * counts / len(cases)
        ax.step(cutoffs, rates, where="post", color=color, linestyle=style,
                linewidth=1.7, label=label, zorder=3)
        ax.plot(0, rates[0], marker="o", markersize=3.3, color=color,
                markeredgewidth=.3, markeredgecolor="white", clip_on=False, zorder=4)
        ax.annotate(f"{counts[-1]}/{len(cases)}", xy=(MAX_LOSS, rates[-1]),
                    xytext=(3, 0), textcoords="offset points", color=color,
                    va="center", ha="left", fontsize=8, clip_on=False)
    ax.set_xlim(0, MAX_LOSS)
    ax.set_ylim(-5, 105)
    ax.set_xticks([0, .02, .04, .06, .08, .10],
                  ["0", ".02", ".04", ".06", ".08", ".10"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", linewidth=.45, color="#D9DDE0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#83898D")
    ax.tick_params(length=2.5, width=.6, color="#83898D", pad=3)


def main():
    cases = pd.read_csv(CASES, keep_default_na=False)
    panels = [cases.loc[cases.p <= 32], cases.loc[cases.p > 32]]
    if [len(panel) for panel in panels] != [56, 331]:
        raise ValueError("Unexpected configuration counts in the benchmark.")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "axes.labelcolor": "#242A2D",
        "text.color": "#242A2D", "axes.linewidth": .6, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(160 / 25.4, 69 / 25.4), sharey=True)
    fig.subplots_adjust(left=.102, right=.908, bottom=.315, top=.855, wspace=.33)
    titles = ["(a)  32 or fewer features", "(b)  More than 32 features"]
    for ax, panel, title in zip(axes, panels, titles):
        draw_panel(ax, panel)
        ax.set_title(title, loc="left", pad=7, fontweight="bold")
    axes[0].set_ylabel("Qualifying configurations (%)")
    fig.text(.5, .16, "Allowed loss of Single-peer relative to each comparator", ha="center", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False,
               bbox_to_anchor=(.5, .018), handlelength=2.4, columnspacing=1.7)
    output = PAPER / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "score-runtime.pdf", metadata={"CreationDate": None})
    fig.savefig(output / "score-runtime.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
