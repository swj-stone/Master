"""Explain the selection cost and fit/predict savings on each low-dimensional source."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from research.block_selection.benchmark import ARMS, HERE, KEY


def main(run, out):
    paired = pd.read_csv(out / "paired-cases.csv")
    low = paired.loc[paired.p <= 32].copy()
    stages = []
    for suite in ("archive", "external", "fresh"):
        for process in (1, 2):
            timings = pd.read_csv(run / f"{suite}-pass{process}" / "times.csv")
            timings["downstream_ms"] = timings.total_ms - timings.select_ms
            stages.append(timings)
    trials = pd.concat(stages, ignore_index=True)
    counts = trials.groupby(KEY + ["method"]).size()
    assert counts.eq(12).all() and len(counts) == len(paired) * len(ARMS)
    means = trials.groupby(KEY + ["method"])[["select_ms", "downstream_ms", "total_ms"]].mean()
    means = means.unstack("method")
    diff = pd.DataFrame({
        "extra_selection_ms": means["select_ms"]["block"] - means["select_ms"]["full"],
        "downstream_saved_ms": means["downstream_ms"]["full"] - means["downstream_ms"]["block"],
        "net_saved_ms": means["total_ms"]["full"] - means["total_ms"]["block"],
    })
    np.testing.assert_allclose(diff.downstream_saved_ms - diff.extra_selection_ms, diff.net_saved_ms,
                               atol=1e-10, rtol=1e-10)
    low = low.join(diff, on=KEY, validate="one_to_one")
    rows = []
    for (task, dataset, metric), group in low.groupby(["task", "dataset", "metric"]):
        row = dict(task=task, dataset=dataset, metric=metric, n=len(group),
                   p=group.p.iloc[0], k_min=group.k_block.min(), k_max=group.k_block.max(),
                   k_cap_min=group.k_current.min(), k_cap_max=group.k_current.max(),
                   faster_full=int((group.ratio_full < 1).sum()),
                   score_gap_min=group.gap_full.min(), score_gap_max=group.gap_full.max(),
                   ratio_min=group.ratio_full.min(), ratio_max=group.ratio_full.max(),
                   ratio_median=group.ratio_full.median())
        row.update(group[diff.columns].mean().to_dict())
        # Only compare folds with the same documented feature encoding.
        if group.cohort.isin(["fresh", "external"]).all():
            supports = [set(str(s).split()) for s in group.cols_block]
            row["cross_fold_jaccard"] = np.mean([
                len(a & b) / len(a | b) for a, b in combinations(supports, 2)
            ])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "costs-by-source.csv", index=False)

    absolute = []
    for scope, subset in (("low", low), ("external", low[low.cohort.eq("external")]),
                          ("fresh", low[low.cohort.eq("fresh")])):
        for arm in ARMS:
            row = dict(scope=scope, method=arm, n=len(subset))
            for field in ("select_ms", "total_ms"):
                # Each process contributes its own six-rotation median.
                both = (subset[f"{field}1_{arm}"] + subset[f"{field}2_{arm}"]) / 2
                row[field] = both.median()
            absolute.append(row)
    pd.DataFrame(absolute).to_csv(out / "absolute-ms.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    main(args.run, args.out)
