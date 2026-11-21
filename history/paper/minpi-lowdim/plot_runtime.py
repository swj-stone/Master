"""Plot paired time savings for single-peer selection on inputs of at most 32 columns."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
DATA = PAPER.parents[1] / "research/fast_single/evidence-extension/all-405.csv"
TEAL = "#006B70"
INK = "#26363E"


def time_savings(frame, stage, comparator):
    ratios = np.column_stack([
        frame[f"{stage}{process}_fast"] / frame[f"{stage}{process}_{comparator}"]
        for process in (1, 2)
    ])
    return 100 * (1 - ratios.max(axis=1))


def main():
    cases = pd.read_csv(DATA, keep_default_na=False, float_precision="round_trip")
    low = cases[cases.p <= 32]
    if (len(cases) != 405 or len(low) != 74 or not cases.case_id.is_unique
            or not low.k_F_fast.eq(low.k_fast).all() or not low.k_random_fast.eq(low.k_fast).all()):
        raise ValueError("Expected 74 low-dimensional configurations with width-matched controls")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": "#7C888B",
    })
    fig, panels = plt.subplots(1, 2, figsize=(160 / 25.4, 60 / 25.4), sharey=True)
    fig.subplots_adjust(left=.186, right=.975, bottom=.26, top=.80, wspace=.15)
    stages = [("select_ms", "(a)  Feature selection"), ("total_ms", "(b)  Selection to prediction")]
    for ax, (stage, title) in zip(panels, stages):
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=7)
        ax.set_ylim(-.51, 1.55)
        if stage == "select_ms":
            ax.set_xscale("symlog", linthresh=100)
            ax.set_xticks([-8000, -1000, -100, 0, 100], ["−8000", "−1000", "−100", "0", "100"])
            ax.set_xlim(-12500, 160)
        else:
            ax.set_xticks([-30, -15, 0, 15, 30])
            ax.set_xlim(-40, 43)
        ax.set_yticks([1, 0], ["Width-matched F", "Width-matched\nrandom"])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=3, pad=3)
        ax.set_axisbelow(True)
        ax.axvline(0, color="#657073", lw=.85, ls=(0, (3, 2)))
        ax.grid(axis="y", color="#D6DEDD", lw=.6)
        for row, comparator in [(1, "F_fast"), (0, "random_fast")]:
            savings = np.sort(time_savings(low, stage, comparator))
            if savings[0] < ax.get_xlim()[0] or savings[-1] > ax.get_xlim()[1]:
                raise ValueError(f"Time savings outside the plotted range for {stage}, {comparator}")
            # Spread near-coincident observations vertically without changing their x values.
            offset = ((np.arange(len(savings)) * .61803399) % 1 - .5) * .38
            ax.scatter(savings, row + offset, s=11, color=TEAL, alpha=.56,
                       edgecolors="none", zorder=2)
            median = np.median(savings)
            ax.plot([median, median], [row-.22, row+.22], color=INK, lw=1.2, zorder=3)
            ax.text(median, row+.31, f"{median:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=TEAL)
    fig.supxlabel("Time saved by sampled Single-peer relative to each comparator (%)", y=.06, fontsize=8.5)
    out = PAPER / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "lowdim-runtime.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "lowdim-runtime.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
