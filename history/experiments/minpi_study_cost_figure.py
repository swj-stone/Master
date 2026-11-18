"""Draw the paper's budget, arithmetic and precision comparisons from archived trials."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from experiments.minpi_study import OUTPUT, ROOT


INK = "#222729"
SLATE = "#526068"
TEAL = "#14695F"
RUST = "#9C5138"
RULE = "#D5DADB"
BITS = [0, 8, 20, 40]


def heading(fig, x, y, letter, title, detail):
    fig.text(x, y, letter, fontsize=9.3, weight="bold", color=TEAL, va="top")
    fig.text(x + .038, y, title, fontsize=9.0, weight="bold", color=INK, va="top")
    if detail:
        fig.text(x, y - .058, detail, fontsize=7.2, color=SLATE, va="top")


def budget_panel(ax):
    trials = pd.read_csv(OUTPUT / "frontier/trials.csv")
    keys = ["p", "w", "abstain", "seed", "fraction"]
    runs = trials.groupby(keys + ["method"]).elapsed_ms
    assert runs.size().eq(5).all()
    times = runs.median().unstack()
    ratios = (times.cert / times.full).groupby(["p", "fraction"])
    assert ratios.size().eq(12).all()
    quantiles = ratios.quantile([.25, .5, .75]).unstack()
    widths = [128, 4096, 16384]
    fractions = [.02, .1, .5]
    for row, width in enumerate(widths):
        for col, fraction in enumerate(fractions):
            low, median, high = quantiles.loc[(width, fraction)]
            # Parity is white; reciprocal departures use the same colour intensity.
            intensity = min(abs(np.log2(median)) * 1.1, .65)
            hue = np.array(to_rgb(TEAL if median < 1 else RUST))
            fill = (1 - intensity) * np.ones(3) + intensity * hue
            ax.add_patch(Rectangle((col + .015, row + .02), .97, .96,
                                   facecolor=fill, edgecolor=RULE, linewidth=.45))
            ax.text(col + .5, row + .39, f"{median:.2f}", va="center", ha="center",
                    fontsize=9.0, weight="bold")
            ax.text(col + .5, row + .72, f"{low:.2f}–{high:.2f}", va="center",
                    ha="center", fontsize=6.8)
    ax.set(xticks=np.arange(3) + .5, xticklabels=["2%", "10%", "50%"],
           yticks=np.arange(3) + .5, yticklabels=[f"{p:,}" for p in widths],
           xlim=(0, 3), ylim=(3, 0))
    ax.xaxis.tick_top()
    ax.text(-.04, 1.036, "Width", ha="right", va="bottom", transform=ax.transAxes,
            fontsize=7.3, color=SLATE)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.spines[:].set_visible(False)


def runtime_panel(ax):
    trials = pd.read_csv(OUTPUT / "word-design/trials.csv")
    trials = trials[trials.source.ne("madelon")]
    groups = trials.groupby(["source", "seed", "native", "method"]).select_ms
    assert groups.size().eq(5).all()
    times = groups.median().unstack("method")
    styles = [(True, -.14, SLATE, "s"), (False, .14, TEAL, "o")]
    controls = ["full", "anchor", "float"]
    for row, control in enumerate(controls):
        for native, offset, color, marker in styles:
            panels = times.xs(native, level="native")
            ratios = (panels.dispatch / panels[control]).sort_values().to_numpy()
            assert len(ratios) == 21
            y = row + offset
            jitter = np.resize([0, -.044, .044], len(ratios))
            ax.scatter(ratios, y + jitter, s=6.5, marker=marker, color=color,
                       linewidths=0, alpha=.50, zorder=2)
            ax.plot(np.median(ratios), y, marker=marker, markersize=4.5, color=color,
                    markeredgecolor="white", markeredgewidth=.55, zorder=3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlim=(.30, 2.46), xticks=[.5, 1, 2], xticklabels=["0.5", "1", "2"],
           ylim=(2.42, -.43), yticks=range(3),
           yticklabels=["Exhaustive", "Anchor", "Float"])
    ax.axvline(1, color=SLATE, lw=.7, ls=(0, (2, 2)), zorder=0)
    ax.xaxis.grid(True, color=RULE, lw=.45)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.set_xlabel("Algorithm / control time", labelpad=1.3)


def precision_panels(measured_ax, constructed_ax):
    counts = pd.read_csv(OUTPUT / "zero-bounds/counts.csv")
    counts = counts[counts.policy.eq("zero_point")]
    targets = pd.read_csv(OUTPUT / "precision-design/targets.csv")
    keys = ["panel", "seed", "game", "h"]
    data = counts.merge(targets[keys + ["gap_exact"]], on=keys, validate="many_to_one")
    assert len(data) == len(counts) and data.oracle_match.all()
    constructed = data.panel.str.startswith("constructed")
    measured = data[~constructed].assign(tied=data.loc[~constructed, "gap_exact"].eq("0"))
    stress = data[constructed & data.game.eq("default") & data.h.eq(1)]
    settings = [(False, TEAL, "o", "-", "Gap"), (True, RUST, "D", "--", "Tie")]
    for tied, color, marker, line, label in settings:
        group = measured[measured.tied.eq(tied)].groupby("bits").exact_columns
        denominator = group.size().loc[BITS]
        assert denominator.nunique() == 1 and denominator.iloc[0] == (19 if tied else 83)
        numerator = group.agg(lambda cols: cols.gt(0).sum()).loc[BITS]
        percent = 100 * numerator / denominator
        stress_values = stress[stress.panel.eq("constructed_tied_max" if tied else
                                               "constructed_unique_max")].set_index("bits")
        columns = stress_values.exact_columns.loc[BITS]
        assert stress_values.p.eq(4096).all() and len(columns) == 4
        for ax, series in [(measured_ax, percent), (constructed_ax, columns)]:
            ax.plot(BITS, series, color=color, ls=line, lw=1.25, marker=marker,
                    ms=4, mfc="white" if tied else color, mec=color, mew=.9, zorder=3)
        if not tied:
            constructed_ax.annotate(f"{columns.iloc[0]:,}", (0, columns.iloc[0]),
                                    xytext=(7, -1), textcoords="offset points", va="center",
                                    fontsize=7.2, color=color)
        # At 40 bits both rates are near zero. Label their last observations
        # above the lines and link the labels to the corresponding markers.
        measured_ax.annotate(f"{label}  {numerator.iloc[-1]}/{denominator.iloc[-1]}",
                             xy=(40, percent.iloc[-1]), xytext=(30, 31 if tied else 17),
                             arrowprops=dict(arrowstyle="-", color=color, lw=.55, shrinkB=3),
                             ha="center", va="center", color=color, fontsize=7.2)
        constructed_ax.annotate(f"{label}  {columns.iloc[-1]:,}", (29.4, columns.iloc[-1]),
                                xytext=(0, 3.6), textcoords="offset points", ha="center",
                                color=color, fontsize=7.2)
    measured_ax.set(ylim=(-6, 101), yticks=[0, 25, 50, 75, 100])
    constructed_ax.set(ylim=(-230, 4340), yticks=[0, 1024, 2048, 3072, 4096],
                       yticklabels=["0", "1,024", "2,048", "3,072", "4,096"])
    for ax in [measured_ax, constructed_ax]:
        ax.set(xlim=(-1.6, 41.8), xticks=BITS, xlabel="Fractional bits")
        ax.yaxis.grid(True, color=RULE, linewidth=.55)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.set_axisbelow(True)


def draw():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SLATE,
                         "axes.linewidth": .55, "axes.labelsize": 7.6, "axes.labelpad": 2,
                         "xtick.color": SLATE, "ytick.color": SLATE, "xtick.labelsize": 7.3,
                         "ytick.labelsize": 7.3, "xtick.major.width": .55,
                         "xtick.major.size": 2.5})
    # The ICOMP template has a 5.5-inch text block. Use its final print size.
    fig = plt.figure(figsize=(5.5, 3.42))
    heading(fig, .006, .985, "a", "Budget and pruning cost", "Floating pruning / exhaustive scoring time")
    heading(fig, .514, .985, "b", "Training-to-support cost", None)
    for x, marker, color, label in [(.520, "s", SLATE, "Machine-word path"),
                                    (.753, "o", TEAL, "Dyadic path")]:
        fig.add_artist(plt.Line2D([x], [.915], marker=marker, markersize=3.9, color=color,
                                  linestyle="none", transform=fig.transFigure))
        fig.text(x + .013, .915, label, fontsize=7.2, color=SLATE, va="center")
    heading(fig, .006, .477, "c", "Fallback on measured targets", "Targets with full-precision column scores (%)")
    heading(fig, .514, .477, "d", "Constructed budget boundary", "Columns requiring full-precision scores")
    budget_panel(fig.add_axes([.10, .578, .365, .245]))
    runtime_panel(fig.add_axes([.660, .605, .323, .245]))
    precision_panels(fig.add_axes([.085, .102, .375, .263]),
                     fig.add_axes([.634, .102, .349, .263]))
    fig.savefig(ROOT / "paper/minpi/figures/cost-frontier.pdf",
                metadata={"Title": "Budget, arithmetic and exact fallback for pivotal minPI",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    draw()
