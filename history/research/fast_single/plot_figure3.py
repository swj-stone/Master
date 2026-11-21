"""Draw Figure 3 for the sampled Single-peer rule on the historical and paper panels."""
from pathlib import Path

from research.peer_crossover.plot_curves import METHOD_STYLES, ZOOM, plot_panel
from research.peer_crossover.summarize import compare

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OVERVIEW_RANGE = (-0.25, 0.50)
ARMS = [("fast", "Sampled Single-peer", "single"),
        ("F_fast", "Width-matched F", "F"),
        ("random_fast", "Width-matched random", "random")]


def draw(data, expected, output_stem):
    if not data.case_id.is_unique or not data.k_F_fast.eq(data.k_fast).all() or not data.k_random_fast.eq(data.k_fast).all():
        raise ValueError("Expected unique configurations and controls matched at the candidate width")
    low = data.p.le(32)
    if (low.sum(), (~low).sum()) != expected:
        raise ValueError("Unexpected configuration counts")
    arms = {key: compare(data, key, "full") for key, _, _ in ARMS}

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.35), sharey=True)
    fig.subplots_adjust(left=.105, right=.975, top=.789, bottom=.118, wspace=.15, hspace=.74)
    fig.text(.105, .98, "Sampled Single-peer and width-matched baselines",
             fontsize=14, weight="bold", va="top")
    for col, (mask, label) in enumerate([(low, "≤32 features"), (~low, ">32 features")]):
        upper, lower = axes[:, col]
        curves = [(name, *METHOD_STYLES[style], arms[key].loc[mask]) for key, name, style in ARMS]
        fig.text(upper.get_position().x0, .859, f"{label}  ({int(mask.sum())} configurations)",
                 fontsize=10.6, weight="bold")
        for ax, reduced, letter in [(upper, False, "ab"[col]), (lower, True, "cd"[col])]:
            plot_panel(ax, curves, ZOOM if reduced else OVERVIEW_RANGE, reduced)
            ax.set_title(f"({letter})  " + ("Fewer columns and faster than Full" if reduced else "Faster than Full"),
                         loc="left", fontsize=9.5, pad=5)
            ax.set_xlabel("Allowed test-score loss relative to Full", fontsize=9)
        upper.axvline(.02, color="#87999C", linestyle=(0, (2, 2)), linewidth=1, zorder=0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, loc="upper center",
               bbox_to_anchor=(.54, .945), frameon=False, handlelength=2.8, columnspacing=1.8)
    fig.supylabel("Share of all configurations in each group (%)", fontsize=9.6, x=.016, y=.45)
    fig.text(.105, .018, "Loss = Full score − subset score. Shading marks the 0–0.10 range enlarged below. "
             "The dashed guide marks 0.02.", fontsize=8, color="#485D66")
    output_stem.parent.mkdir(exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    if sum(expected) == 405:
        fig.savefig(output_stem.with_suffix(".png"), dpi=190, facecolor="white")
    plt.close(fig)


def render(source, counts, output_stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "text.color": "#192F38", "axes.labelcolor": "#192F38"})
    draw(pd.read_csv(source, keep_default_na=False, float_precision="round_trip"), counts, output_stem)


if __name__ == "__main__":
    for source, counts, stem in [("evidence-extension/all-405.csv", (74, 331), "figure-3-sampled-single"),
                                 ("evidence/paired-cases.csv", (56, 331), "figure-3-paper-panel")]:
        render(HERE / source, counts, OUT / stem)
