"""
Friedman + Nemenyi post-hoc + critical-difference (CD) diagram for the §6
main table (reviewer #7).  Produces a Demsar-style CD diagram (PDF), average
ranks, a pairwise "significantly different" matrix, and Kendall's W effect size.

Input: any tidy CSV with columns ``method``, ``dataset`` and a value column
(e.g. ``results/tables/<run_id>/mean_metrics.csv`` with ``hv``, or
``pareto_metrics_mean.csv`` with ``igd_plus``).  Seeds, if present, are averaged.

Run:
    python experiments/make_cd_diagram.py <metrics.csv> --metric hv --higher
    python experiments/make_cd_diagram.py <pareto_metrics_mean.csv> --metric igd_plus --lower
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402

# Nemenyi critical values q_{0.05} (studentized range / sqrt(2)) by #methods k
_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
}


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi CD = q_alpha * sqrt(k(k+1)/(6N))."""
    if alpha != 0.05:
        raise ValueError("only alpha=0.05 tabulated")
    return _Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def average_ranks(pivot: pd.DataFrame, higher_is_better: bool) -> pd.Series:
    """Average rank per method across datasets (rank 1 = best)."""
    # rank within each dataset (row); ascending=False so higher value -> rank 1
    ranks = pivot.rank(axis=1, ascending=not higher_is_better, method="average")
    return ranks.mean(axis=0).sort_values()


def draw_cd_diagram(avg_rank: pd.Series, cd: float, out_pdf: Path, title: str) -> None:
    methods = list(avg_rank.index)
    ranks = avg_rank.values
    k = len(methods)
    lo, hi = 1, k
    fig_h = 1.2 + 0.32 * k
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.5
    # axis line + ticks
    ax.plot([lo, hi], [top, top], "k-", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [top, top + 0.12], "k-", lw=1.0)
        ax.text(r, top + 0.22, str(r), ha="center", va="bottom", fontsize=9)
    ax.text((lo + hi) / 2, top + 0.5, f"{title}  (CD = {cd:.2f})",
            ha="center", va="bottom", fontsize=10)

    # method labels: best ranks (left) on the left side, worst on the right
    half = (k + 1) // 2
    level = top - 0.35
    step = 0.32
    for i, (name, r) in enumerate(zip(methods, ranks)):
        if i < half:                       # left block
            y = level - i * step
            ax.plot([r, r], [top, y], "k-", lw=0.8)
            ax.plot([r, lo - 0.45], [y, y], "k-", lw=0.8)
            ax.text(lo - 0.5, y, f"{name} ({r:.2f})", ha="right", va="center", fontsize=9)
        else:                              # right block
            y = level - (i - half) * step
            ax.plot([r, r], [top, y], "k-", lw=0.8)
            ax.plot([r, hi + 0.45], [y, y], "k-", lw=0.8)
            ax.text(hi + 0.5, y, f"({r:.2f}) {name}", ha="left", va="center", fontsize=9)

    # cliques: groups of consecutive methods within CD of each other
    order = list(ranks)
    bar_y = top - 0.12
    drawn: list[tuple[float, float]] = []
    for i in range(k):
        j = i
        while j + 1 < k and order[j + 1] - order[i] <= cd:
            j += 1
        if j > i:
            seg = (order[i], order[j])
            if seg not in drawn:
                ax.plot([seg[0] - 0.03, seg[1] + 0.03], [bar_y, bar_y], "k-", lw=3.0)
                bar_y -= 0.10
                drawn.append(seg)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--metric", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--higher", action="store_true", help="higher value is better (HV, AUC)")
    g.add_argument("--lower", action="store_true", help="lower value is better (IGD+, spacing)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    higher = args.higher

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f"column '{args.metric}' not in {args.csv} ({list(df.columns)})")
    agg = df.groupby(["method", "dataset"])[args.metric].mean().reset_index()
    pivot = (
        agg.pivot(index="dataset", columns="method", values=args.metric)
        .dropna(axis=1, how="all")      # drop fully-NaN methods (e.g. non-NSGA-II for HV)
        .dropna(axis=0)                  # drop datasets missing any method (regression sets for IGD+)
    )
    k, n = pivot.shape[1], pivot.shape[0]
    if k < 2 or n < 3:
        raise SystemExit(f"need >=2 methods and >=3 datasets; got k={k}, N={n}")

    stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    kendall_w = float(stat / (n * (k - 1)))           # Friedman effect size
    cd = critical_difference(k, n)
    avg_rank = average_ranks(pivot, higher)

    out_dir = args.csv.parent
    title = f"{args.metric} ({'higher' if higher else 'lower'} better)"
    out_pdf = args.out or (out_dir / f"cd_diagram_{args.metric}.pdf")
    draw_cd_diagram(avg_rank, cd, out_pdf, title)

    # pairwise: significant if |avg-rank diff| > CD
    methods = list(avg_rank.index)
    sig = pd.DataFrame("", index=methods, columns=methods)
    for a in methods:
        for b in methods:
            sig.loc[a, b] = "" if a == b else (
                "sig" if abs(avg_rank[a] - avg_rank[b]) > cd else "ns"
            )
    avg_rank.rename("avg_rank").to_csv(out_dir / f"avg_ranks_{args.metric}.csv")
    sig.to_csv(out_dir / f"nemenyi_pairs_{args.metric}.csv")

    print(f"Friedman chi2={stat:.3f}, p={p:.4g}  (k={k}, N={n})")
    print(f"Kendall's W (effect size) = {kendall_w:.3f}")
    print(f"Nemenyi CD (alpha=0.05) = {cd:.3f}")
    print("Average ranks (1=best):")
    for m, r in avg_rank.items():
        print(f"  {m:16s} {r:.3f}")
    print(f"\nSaved: {out_pdf.name}, avg_ranks_{args.metric}.csv, nemenyi_pairs_{args.metric}.csv")


if __name__ == "__main__":
    main()
