"""Plot paired time savings on every configuration using the low-dimensional rule."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
DATA = PAPER.parents[1] / "research/lowdim/evidence/all-cases.csv"
TEAL = "#006B70"
INK = "#26363E"


def time_savings(frame, stage, comparator):
    ratios = np.column_stack([
        frame[f"{stage}{process}_new"] / frame[f"{stage}{process}_{comparator}"]
        for process in (1, 2)
    ])
    return 100 * (1 - ratios.max(axis=1))


def main():
    cases = pd.read_csv(DATA, keep_default_na=False)
    low = cases[cases.p <= 32]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": "#7C888B",
    })
    fig, panels = plt.subplots(1, 2, figsize=(160 / 25.4, 60 / 25.4), sharey=True)
    fig.subplots_adjust(left=.186, right=.975, bottom=.26, top=.80, wspace=.15)
    stages = [("select_ms", "(a)  Feature selection"), ("ms", "(b)  Selection to prediction")]
    for ax, (stage, title) in zip(panels, stages):
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=7)
        ax.set_xlim(-24, 104)
        ax.set_ylim(-.51, 1.55)
        ax.set_xticks([-20, 0, 25, 50, 75, 100])
        ax.set_yticks([1, 0], ["Nested-peer rule", "Width-matched F"])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=3, pad=3)
        ax.set_axisbelow(True)
        ax.axvline(0, color="#657073", lw=.85, ls=(0, (3, 2)))
        ax.grid(axis="y", color="#D6DEDD", lw=.6)
        for row, comparator in [(1, "previous"), (0, "F")]:
            savings = np.sort(time_savings(low, stage, comparator))
            # Spread near-coincident observations vertically without changing their x values.
            offset = ((np.arange(len(savings)) * .61803399) % 1 - .5) * .38
            ax.scatter(savings, row + offset, s=11, color=TEAL, alpha=.56,
                       edgecolors="none", zorder=2)
            median = np.median(savings)
            ax.plot([median, median], [row-.22, row+.22], color=INK, lw=1.2, zorder=3)
            ax.text(median, row+.31, f"{median:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=TEAL)
    fig.supxlabel("Time saved by the single-peer rule relative to each comparator (%)", y=.06, fontsize=8.5)
    out = PAPER / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "lowdim-runtime.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "lowdim-runtime.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
