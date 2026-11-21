"""Plot the empirical distribution of retained width for the two peer rules."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RULES = (
    ("single", "Single-peer", "#006B70", "-"),
    ("nested", "Nested-peer", "#70519C", (0, (4, 2))),
)


def draw_widths(cases):
    """Return a two-panel figure using one observation per benchmark configuration."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.0, 3.0))
        fig.subplots_adjust(left=.096, right=.972, top=.853, bottom=.213, wspace=.16)
        panels = ((cases.p <= 32, "≤32 input features"), (cases.p > 32, ">32 input features"))
        for ax, (mask, title) in zip(axes, panels):
            group = cases.loc[mask]
            for rule, label, color, style in RULES:
                widths = group[f"k_{rule}"]
                if group.empty or not widths.between(1, group.p).all():
                    raise ValueError(f"Invalid retained widths for {title}, {rule}.")
                pct = 100 * widths / group.p
                values, counts = np.unique(pct, return_counts=True)
                fraction = 100 * np.cumsum(counts) / len(group)
                ax.step(np.r_[0., values, 101.], np.r_[0., fraction, 100.],
                        where="post", color=color, ls=style, lw=1.85,
                        label=f"{label}  {pct.median():.1f}%")
            ax.set_title(f"{title}  ($n$ = {len(group)})", loc="left", fontsize=10.3)
            ax.set(xlim=(-.5, 101), ylim=(-1, 102), xticks=[0, 25, 50, 75, 100],
                   yticks=[0, 25, 50, 75, 100])
            ax.tick_params(length=3, width=.6, colors="#334853")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#87959A")
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#D9E0E2", lw=.6)
            legend_corner = "upper left" if ax is axes[0] else "lower right"
            ax.legend(title="Median retained", loc=legend_corner, frameon=False,
                      title_fontsize=8.2, fontsize=8.1, alignment="left", handlelength=2.2)
        axes[0].set_ylabel("Cumulative share of configurations (%)")
        fig.supxlabel("Input features retained, K/p (%)", y=.035, fontsize=9.3)
    return fig


def main():
    cases = pd.read_csv(HERE / "evidence" / "paired-cases.csv", keep_default_na=False)
    if cases.case_id.nunique() != len(cases) or len(cases) != 387 or (cases.p <= 32).sum() != 56:
        raise ValueError("Unexpected crossover register.")
    fig = draw_widths(cases)
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(folder / "retained-widths.pdf", metadata={"CreationDate": None})
    fig.savefig(folder / "retained-widths.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
