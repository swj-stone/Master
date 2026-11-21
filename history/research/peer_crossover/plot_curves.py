"""Plot both dimensional strata, both rules and all four fixed routing policies."""
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from research.peer_crossover.summarize import EPS, compare
from research.peer_crossover.plot_widths import draw_widths


HERE = Path(__file__).resolve().parent
DATA = HERE / "evidence"
OUT = HERE / "figures"
RULES = ("single", "nested")
NAMES = {"single": "Single-peer", "nested": "Nested-peer"}
TEAL, PURPLE = "#006B70", "#70519C"
INK, MUTED, GRID = "#192F38", "#485D66", "#D8E2E4"
ZOOM = (0., .10)
METHOD_STYLES = {"single": (TEAL, "-"), "nested": (PURPLE, "-"),
                 "F": ("#476789", (0, (4.3, 2))), "random": ("#AE572C", (0, (1.1, 1.7)))}
ROUTES = list(product(RULES, repeat=2))
ROUTE_STYLES = [(TEAL, "-"), ("#A45B19", (0, (5, 2))),
                ("#42678D", (0, (3, 1.6, 1, 1.6))), (PURPLE, (0, (1, 1.65)))]


def observations(data, arms):
    return {arm: compare(data, arm, "full") for arm in arms}


def series(rule, results):
    keys = (rule, "F", "random")
    names = (NAMES[rule], "Width-matched F", "Width-matched random")
    arms = (rule, "F_" + rule, "random_" + rule)
    return [(label, *METHOD_STYLES[key], results[arm]) for label, key, arm in zip(names, keys, arms)]


def eligible_losses(comparison, reduced):
    valid = comparison.fast & comparison.cut if reduced else comparison.fast
    return np.sort(-comparison.loc[valid, "delta"].to_numpy() - EPS)


def plot_panel(ax, curves, bounds, reduced):
    low, high = bounds
    for label, color, style, results in curves:
        losses = eligible_losses(results, reduced)
        levels = np.unique(np.r_[low, losses[(losses > low) & (losses < high)], high])
        counts = np.searchsorted(losses, levels, side="right")
        ax.step(levels, 100 * counts / len(results), where="post", label=label, color=color,
                linestyle=style, linewidth=2, solid_capstyle="butt", dash_capstyle="butt")
    ax.set(xlim=bounds, ylim=(-2, 102), yticks=[0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, linewidth=.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CAEB2")
    ax.tick_params(width=.6, length=3, pad=4)
    if reduced:
        ax.set_xticks([0, .02, .04, .06, .08, .1], ["0", "0.02", "0.04", "0.06", "0.08", "0.10"])
        ax.axvline(.02, color="#87999C", linestyle=(0, (2, 2)), linewidth=1, zorder=0)
    else:
        ax.xaxis.set_major_locator(MultipleLocator(.25))
        ax.axvspan(*ZOOM, color="#DFECEC", zorder=0)
        ax.axvline(0, color="#8CA2A5", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def legend(fig, styles, ncol, y):
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=2.5, label=label)
               for label, color, style, _ in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, y), ncol=ncol,
               frameon=False, handlelength=2.5, columnspacing=2.1)


def two_columns(title, columns, full_limits, legend_styles):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.55), sharey=True)
    fig.subplots_adjust(left=.089, right=.976, bottom=.135, top=.752, hspace=.69, wspace=.13)
    fig.text(.089, .972, title, va="top", fontsize=15, fontweight="bold")
    legend(fig, legend_styles, 3, .914)
    for col, (heading, curves) in enumerate(columns):
        upper, lower = axes[:, col]
        fig.text(upper.get_position().x0, .821, heading, fontsize=11, fontweight="bold", va="bottom")
        plot_panel(upper, curves, full_limits, False)
        plot_panel(lower, curves, ZOOM, True)
        upper.set_title("Faster than Full", loc="left", fontsize=10, pad=5)
        lower.set_title("Fewer columns and faster than Full", loc="left", fontsize=10, pad=5)
    for ax in axes[0]:
        ax.set_xlabel("Test-score loss threshold")
    for ax in axes[1]:
        ax.set_xlabel("Allowed test-score loss")
    fig.supylabel("Share of all configurations in panel (%)", x=.011, y=.445, fontsize=10)
    fig.text(.089, .019, "Full is the full-feature model. Faster means lower selection-through-prediction time in both runs.\n"
             "Loss = Full score − subset score. The shading covers 0 to 0.10. The dotted line marks 0.02.",
             color=MUTED, fontsize=8.5, linespacing=1.35)
    return fig


def routing_pair(low_curve, high_curve, is_low):
    return pd.concat([low_curve.loc[is_low], high_curve.loc[~is_low]]).sort_index()


def policy_figure(curves, full_limits):
    fig, axes = plt.subplots(1, 2, figsize=(9., 4.25), sharey=True)
    fig.subplots_adjust(left=.088, right=.974, bottom=.205, top=.692, wspace=.14)
    fig.text(.088, .97, "Four routing policies across all 387 configurations",
             va="top", fontsize=14.5, fontweight="bold")
    legend(fig, curves, 2, .881)
    plot_panel(axes[0], curves, full_limits, False)
    plot_panel(axes[1], curves, ZOOM, True)
    axes[0].set_title("Faster than Full", loc="left", fontsize=11)
    axes[1].set_title("Fewer columns and faster than Full", loc="left", fontsize=11)
    axes[0].set_xlabel("Test-score loss threshold")
    axes[1].set_xlabel("Allowed test-score loss")
    axes[0].set_ylabel("Share of 387 configurations (%)")
    fig.text(.088, .037, "Legend order: rule at ≤32 features / rule at >32 features.     Dotted guide: loss of 0.02.",
             color=MUTED, fontsize=9)
    return fig


def verify(results, data, routes):
    cells = pd.read_csv(DATA / "width-groups.csv").set_index(["width_group", "arm"])
    policies = pd.read_csv(DATA / "routes.csv").set_index(["low", "wide"])
    rows = [(results[arm].loc[data.width_group == group], ref) for (group, arm), ref in cells.iterrows()]
    rows.extend((curves, policies.loc[route]) for route, (_, _, _, curves) in zip(ROUTES, routes))
    for group, ref in rows:
        assert len(group) == ref.n
        losses = eligible_losses(group, True)
        assert len(losses) == ref.cut_faster
        for threshold, field in [(0., "joint_0"), (.02, "joint_002"), (.05, "joint_005"), (.1, "joint_010")]:
            assert np.searchsorted(losses, threshold, side="right") == ref[field]


def save_preview(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=175, facecolor="white")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "pdf.fonttype": 42,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    data = pd.read_csv(DATA / "paired-cases.csv", keep_default_na=False, float_precision="round_trip")
    is_low = data.p <= 32
    assert (is_low.sum(), (~is_low).sum(), data.case_id.nunique()) == (56, 331, 387)
    results = observations(data, [name for rule in RULES for name in (rule, "F_"+rule, "random_"+rule)])
    observed = np.concatenate([eligible_losses(c, False) for c in results.values()])
    bounds = (np.floor(observed.min()*4)/4, np.ceil(observed.max()*4)/4)
    policy_curves = [(f"{NAMES[a]} / {NAMES[b]}", *style, routing_pair(results[a], results[b], is_low))
                     for (a, b), style in zip(ROUTES, ROUTE_STYLES)]
    verify(results, data, policy_curves)
    OUT.mkdir(exist_ok=True)
    with PdfPages(OUT / "crossover-curves.pdf", metadata={"Title": "Peer-rule crossover curves",
                                                         "CreationDate": None}) as pdf:
        for rule in RULES:
            curves = series(rule, results)
            columns = []
            for heading, mask in (("≤32 features  ·  56 configurations", is_low),
                                  (">32 features  ·  331 configurations", ~is_low)):
                columns.append((heading, [(name, color, style, c.loc[mask]) for name, color, style, c in curves]))
            fig = two_columns(f"{NAMES[rule]} rule and width-matched baselines", columns, bounds, curves)
            pdf.savefig(fig)
            save_preview(fig, rule + "-curves")
            plt.close(fig)
        fig = policy_figure(policy_curves, bounds)
        pdf.savefig(fig)
        save_preview(fig, "routing-curves")
        plt.close(fig)

        # The same fixed policy determines K for both matched control curves.
        for low_rule in RULES:
            columns = []
            for wide_rule in RULES:
                route = {k: routing_pair(results[low], results[wide], is_low)
                         for k, low, wide in (("rule", low_rule, wide_rule),
                                              ("F", "F_"+low_rule, "F_"+wide_rule),
                                              ("random", "random_"+low_rule, "random_"+wide_rule))}
                styles = [("Peer rule", *METHOD_STYLES[low_rule], route["rule"]),
                          ("Width-matched F", *METHOD_STYLES["F"], route["F"]),
                          ("Width-matched random", *METHOD_STYLES["random"], route["random"])]
                columns.append((f">32 features: {NAMES[wide_rule]}", styles))
            fig = two_columns(f"All 387 configurations  ·  ≤32 features: {NAMES[low_rule]}",
                              columns, bounds, columns[0][1])
            pdf.savefig(fig)
            plt.close(fig)
        fig = draw_widths(data)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Four rule/width cells and four policies checked against the reference counts. Figures in {OUT}.")


if __name__ == "__main__":
    main()
