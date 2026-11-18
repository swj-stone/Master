"""Build the fixed-width subset comparisons from the recorded test scores."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
HISTORY = PAPER.parents[1]
EVIDENCE = HISTORY / "research" / "lowdim" / "peer_evidence"
TEAL = "#006B70"
GRAY = "#7D8790"
INK = "#27353B"
TOL = 1e-12


def draw_outcomes(ax, cases):
    groups = np.arange(2)
    width = 0.23
    gaps = cases["score_new"].to_numpy()[:, None] - cases[
        ["random32_median", "score_empty"]
    ].to_numpy()
    outcomes = [
        ("Higher", gaps > TOL, TEAL, ""),
        ("Tied", np.abs(gaps) <= TOL, "#C2C9CC", ""),
        ("Lower", gaps < -TOL, "#B15A24", "///"),
    ]
    for pos, (label, mask, color, hatch) in enumerate(outcomes):
        counts = mask.sum(axis=0)
        bars = ax.bar(groups + (pos - 1) * width, counts, width * 0.86,
                      label=label, color=color, hatch=hatch, edgecolor="white",
                      linewidth=0.5, zorder=2)
        ax.bar_label(bars, labels=[str(n) for n in counts], padding=2, fontsize=8)
    ax.set_ylabel("Configurations")
    ax.set_xticks(groups, ["Median of 32\nrandom subsets", "Empty-peer\nranking"])
    ax.set_ylim(0, len(cases) + 1)
    ax.set_xlim(-0.47, 1.47)
    ax.set_yticks([0, 10, 20, 30])
    ax.grid(axis="y")
    ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0, 1.025),
              borderaxespad=0, handlelength=1, handletextpad=.35, columnspacing=.85)


def draw_abalone(ax, cases):
    folds = cases.loc[cases.dataset.eq("abalone") & cases.cohort.eq("original")]
    subsets = pd.read_csv(EVIDENCE / "abalone-subset-scores.csv", float_precision="round_trip")
    assert set(subsets.fold) == set(folds.fold)
    for fold, reps in folds.groupby("fold"):
        rep = reps.iloc[0]
        assert reps.score_new.nunique() == 1 and reps.replaced_empty.eq(0).all()
        results = subsets.loc[subsets.fold.eq(fold)]
        scores = results.test_r2.to_numpy()
        selected = results.minpi_selected.eq(1).to_numpy()
        assert results.columns_zero_based.nunique() == len(scores) == float(rep.exact_n)
        assert selected.sum() == 1
        minpi_score = scores[selected].item()
        np.testing.assert_allclose(minpi_score, rep.score_new, rtol=0, atol=TOL)
        rank = 1 + np.count_nonzero(scores > minpi_score + TOL)
        assert rank == 1 + float(rep.exact_better)

        # Offsets spread neighbouring dots within each row without changing R².
        order = np.argsort(scores)
        bins = np.floor(scores[order] / .005).astype(int)
        offsets = np.zeros(len(scores))
        for bin_id in np.unique(bins):
            idx = order[bins == bin_id]
            offsets[idx] = .05 * (np.arange(len(idx)) - (len(idx) - 1) / 2)
        ax.scatter(scores, fold + offsets, s=10, color=GRAY, linewidths=0,
                   label="All subsets" if fold == 1 else None, zorder=2)
        ax.scatter(minpi_score, fold + offsets[selected].item(), s=35, marker="D",
                   linewidths=.65, edgecolors="white", color=TEAL,
                   label="Single-peer rule" if fold == 1 else None, zorder=3)
        ax.text(.587, fold, f"#{rank}", ha="right", va="center",
                fontsize=8, color=TEAL, fontweight="bold")

    ax.set_xlabel("Test $R^2$")
    ax.set_yticks(sorted(folds.fold.unique()), [f"Split {f}" for f in sorted(folds.fold.unique())])
    ax.set_xlim(.277, .591)
    ax.set_xticks([.30, .35, .40, .45, .50, .55])
    ax.set_ylim(3.55, .45)
    ax.grid(axis="x")
    ax.legend(ncols=2, loc="lower left", bbox_to_anchor=(0, 1.025),
              borderaxespad=0, handlelength=1, handletextpad=.4, columnspacing=1)


def main():
    cases = pd.read_csv(EVIDENCE / "cases.csv", keep_default_na=False)
    assert len(cases) == 31 and cases.k.lt(cases.p).all()
    assert not cases.duplicated(["cohort", "dataset", "fold", "seed"]).any()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
        "xtick.color": INK, "ytick.color": INK, "grid.color": "#DFE4E5",
        "grid.linewidth": .55, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": .6, "legend.frameon": False, "pdf.fonttype": 42,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(160 / 25.4, 66 / 25.4))
    fig.subplots_adjust(left=.091, right=.983, top=.745, bottom=.20, wspace=.43)
    draw_outcomes(left, cases)
    draw_abalone(right, cases)
    for ax, title in [(left, "(a)  Fixed-width score comparisons"),
                      (right, "(b)  Abalone, all 45 subsets")]:
        ax.text(0, 1.31, title, transform=ax.transAxes, ha="left", va="center",
                fontsize=8.6, fontweight="bold")
        ax.tick_params(length=3, width=.6)
        ax.set_axisbelow(True)
    figures = PAPER / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "subset-quality.pdf", metadata={"CreationDate": None})
    fig.savefig(figures / "subset-quality.png", dpi=280)
    plt.close(fig)


if __name__ == "__main__":
    main()
