"""Compare matched-width selectors against Full on either side of the 32-column cutoff."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_tradeoff import CASES, MAX_LOSS, PAPER, SCORE_EPS


BASELINES = [
    ("F", "Width-matched F", "#515A82", (0, (4, 2.5)), 1.45),
    ("random", "Width-matched random", "#AD572C", (0, (1.1, 1.65)), 1.8),
]


def faster_cases(cases, method):
    return ((cases[f"ms1_{method}"] < cases.ms1_full)
            & (cases[f"ms2_{method}"] < cases.ms2_full))


def counts_by_loss(losses, bounds):
    low, high = bounds
    thresholds = np.sort(losses.to_numpy() - SCORE_EPS)
    inside = (thresholds >= low) & (thresholds <= high)
    levels = np.unique(np.r_[low, thresholds[inside], high])
    return levels, np.searchsorted(thresholds, levels, side="right")


def main():
    archive = pd.read_csv(CASES, keep_default_na=False)
    low = archive.loc[archive.p <= 32]
    wide = archive.loc[archive.p > 32]
    if (len(low), len(wide)) != (56, 331):
        raise ValueError("Expected 56 low-dimensional and 331 wider configurations")
    if not (archive.k_new.eq(archive.k_F) & archive.k_new.eq(archive.k_random)).all():
        raise ValueError("Mismatched feature counts")
    methods = ["new", *(baseline[0] for baseline in BASELINES)]
    fast_losses = {
        key: (archive.score_full - archive[f"score_{key}"]).loc[faster_cases(archive, key)]
        for key in methods
    }
    observed = pd.concat(fast_losses.values())
    full_range = (np.floor(min(0, observed.min()) * 4) / 4,
                  np.ceil(max(MAX_LOSS, observed.max()) * 4) / 4)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "text.color": "#213139", "axes.labelcolor": "#213139",
        "xtick.color": "#394950", "ytick.color": "#394950",
    })
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(160 / 25.4, 116 / 25.4))
    fig.subplots_adjust(left=.105, right=.976, top=.850, bottom=.108, wspace=.18, hspace=.69)
    panels = [(low, "≤32 features", "Single-peer rule"),
              (wide, ">32 features", "Nested-peer rule")]
    for col, (cases, title, rule) in enumerate(panels):
        speed_ax, joint_ax = axes[:, col]
        fig.text(speed_ax.get_position().x0, .982, f"{title} ({len(cases)} configurations)",
                 va="top", fontsize=9.5, fontweight="bold")
        for key, label, color, style, width in [("new", rule, "#006B70", "-", 2.5), *BASELINES]:
            losses = fast_losses[key]
            losses = losses.loc[losses.index.isin(cases.index)]
            reduced = cases.loc[losses.index, f"k_{key}"] < cases.loc[losses.index, "p"]
            for ax, values, bounds in (
                (speed_ax, losses, full_range),
                (joint_ax, losses.loc[reduced], (0, MAX_LOSS)),
            ):
                levels, counts = counts_by_loss(values, bounds)
                ax.step(levels, 100 * counts / len(cases), where="post", label=label,
                        color=color, linestyle=style, linewidth=width, solid_capstyle="butt",
                        dash_capstyle="butt", zorder=3)
        speed_ax.set_xlim(full_range)
        speed_ax.axvspan(0, MAX_LOSS, color="#E3ECEE", zorder=0)
        speed_ax.xaxis.set_major_locator(plt.MultipleLocator(.5))
        joint_ax.set_xlim(0, MAX_LOSS)
        joint_ax.set_xticks([0, .02, .04, .06, .08, .10], ["0", ".02", ".04", ".06", ".08", ".10"])
        speed_ax.set_title(("(a)", "(b)")[col] + "  Faster than Full", loc="left", fontsize=8.6, pad=5)
        joint_ax.set_title(("(c)", "(d)")[col] + "  All three criteria", loc="left", fontsize=8.6, pad=5)
        joint_ax.legend(loc="upper right", fontsize=7.1, handlelength=2.2, borderaxespad=.24,
                        frameon=False)
        joint_ax.axvline(.02, color="#A6ADB0", linewidth=.68, linestyle=(0, (2, 2)), zorder=1)
    for ax in axes.flat:
        ax.set(ylim=(-2, 101), yticks=[0, 25, 50, 75, 100])
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=.52, color="#DFE3E5")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#839094")
        ax.tick_params(length=2.5, width=.6, pad=3)
    fig.supylabel("Qualifying configurations (%)", x=.013, fontsize=8.4)
    fig.text(.54, axes[0, 0].get_position().y0-.075, "Test-score loss threshold (Full minus selector)",
             ha="center", fontsize=8.4)
    fig.text(.54, .023, "Allowed loss in test score relative to Full",
             ha="center", fontsize=8.4)
    output = PAPER / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "matched-curves.pdf", metadata={"CreationDate": None})
    fig.savefig(output / "matched-curves.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
