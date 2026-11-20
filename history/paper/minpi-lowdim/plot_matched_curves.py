"""Compare the three low-dimensional selectors against the same full model."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_tradeoff import CASES, MAX_LOSS, PAPER, SCORE_EPS


METHODS = [
    ("new", "Single-peer rule", "#006B70", "-", 2.5),
    ("F", "Width-matched F", "#515A82", (0, (4, 2.5)), 1.45),
    ("random", "Width-matched random", "#AD572C", (0, (1.1, 1.65)), 1.8),
]


def counts_by_loss(cases, method, require_faster):
    eligible = cases[f"k_{method}"] < cases.p
    if require_faster:
        for process in (1, 2):
            eligible &= cases[f"ms{process}_{method}"] < cases[f"ms{process}_full"]
    losses = cases.loc[eligible, "score_full"] - cases.loc[eligible, f"score_{method}"]
    thresholds = np.sort(np.maximum(0, losses.to_numpy() - SCORE_EPS))
    levels = np.unique(np.r_[0, thresholds[thresholds <= MAX_LOSS], MAX_LOSS])
    return levels, np.searchsorted(thresholds, levels, side="right")


def main():
    archive = pd.read_csv(CASES, keep_default_na=False)
    cases = archive.loc[archive.p <= 32]
    if len(cases) != 56 or cases.source.nunique() != 10:
        raise ValueError("Expected the 56 single-peer configurations from ten sources")
    if not (cases.k_new.eq(cases.k_F) & cases.k_new.eq(cases.k_random)).all():
        raise ValueError("Mismatched feature counts")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "text.color": "#213139", "axes.labelcolor": "#213139",
        "xtick.color": "#394950", "ytick.color": "#394950",
    })
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(160 / 25.4, 72 / 25.4))
    fig.subplots_adjust(left=.107, right=.978, top=.78, bottom=.23, wspace=.16)
    for index, (ax, title) in enumerate(zip(axes, [
        "(a)  Within score tolerance", "(b)  Also faster than Full",
    ])):
        for key, label, color, style, width in METHODS:
            levels, counts = counts_by_loss(cases, key, require_faster=(index == 1))
            ax.step(levels, 100 * counts / len(cases), where="post", label=label,
                    color=color, linestyle=style, linewidth=width, solid_capstyle="butt",
                    dash_capstyle="butt", zorder=3)
        ax.set(xlim=(0, MAX_LOSS), ylim=(-2, 100), yticks=[0, 25, 50, 75, 100])
        ax.set_xticks([0, .02, .04, .06, .08, .10], ["0", ".02", ".04", ".06", ".08", ".10"])
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=.52, color="#DFE3E5")
        ax.axvline(.02, color="#A6ADB0", linewidth=.68, linestyle=(0, (2, 2)), zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#839094")
        ax.tick_params(length=2.5, width=.6, pad=3)
    axes[0].set_ylabel("Qualifying configurations (%)", fontsize=8.4)
    fig.supxlabel("Allowed loss in test score relative to the full model", y=.052, fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.52, .995), frameon=False, handlelength=2.5, columnspacing=1.8)
    output = PAPER / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "matched-curves.pdf", metadata={"CreationDate": None})
    fig.savefig(output / "matched-curves.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
