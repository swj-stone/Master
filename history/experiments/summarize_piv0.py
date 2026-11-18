"""Summarize the 30 matched PIv0 and direct-Gain jobs from a completed five-seed run."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.thesis_prepared_catalog import FORMAL_DATASETS, FORMAL_SEEDS


METHODS = ("PIv0", "direct_gain")
INDEX = ["dataset", "seed"]
COUNTS = ["width", "negative", "retained", "fits", "candidates", "checked"]
MEASURES = ["validation_utility", "test_utility", "wall_s", "voting_s"]


def summarize(run: Path):
    arms = pd.read_csv(run / "summary.csv", encoding="utf-8-sig")
    expected = pd.MultiIndex.from_product(
        [FORMAL_DATASETS, FORMAL_SEEDS, METHODS], names=[*INDEX, "method"]
    )
    arms = arms.set_index([*INDEX, "method"])
    if (not arms.index.is_unique or not expected.difference(arms.index).empty
            or not arms.index.difference(expected).empty):
        raise ValueError("A complete report requires exactly the 60 distinct planned arms.")
    arms = arms.reindex(expected)
    if not arms["status"].eq("completed").all():
        raise ValueError("All 60 arms must have completed before publishing this report.")

    piv0 = arms.xs("PIv0", level="method").copy()
    gain = arms.xs("direct_gain", level="method").copy()
    # A direct-Gain job has no prefilter. Its negative count is normally blank.
    if not gain["negative"].fillna(0).eq(0).all():
        raise ValueError("A direct-Gain job lists a nonempty negative group.")
    gain["negative"] = 0
    for arm in (piv0, gain):
        values = arm[COUNTS + MEASURES].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not arm["wall_s"].gt(0).all():
            raise ValueError("Every count, runtime and utility must be recorded.")
        counts = arm[COUNTS]
        if (counts.lt(0) | counts.mod(1).ne(0)).any().any():
            raise ValueError("Feature, candidate and fit counts must be nonnegative integers.")
        candidates = arm["width"] - arm["negative"]
        if not (arm["checked"].eq(candidates) & arm["candidates"].eq(candidates)
                & arm["width"].between(1, 100) & arm["retained"].between(1, candidates)).all():
            raise ValueError("A feature count or completed prefix scan is inconsistent.")
    if not piv0[["metric", "width"]].equals(gain[["metric", "width"]]):
        raise ValueError("Matched methods disagree on input width or utility.")
    if piv0.groupby(level="dataset")["metric"].nunique().ne(1).any():
        raise ValueError("A dataset changed utility between seeds.")

    pairs = piv0[["metric", "width"]].copy()
    for name, arm in (("piv0", piv0), ("gain", gain)):
        for field in ("retained", "fits", "validation_utility", "test_utility", "wall_s", "origin"):
            pairs[f"{name}_{field}"] = arm[field]
    pairs["piv0_negative"] = piv0["negative"].astype(int)
    pairs["piv0_voting_s"] = piv0["voting_s"]
    pairs["paired_test_delta"] = pairs["piv0_test_utility"] - pairs["gain_test_utility"]
    pairs["seconds_saved"] = pairs["gain_wall_s"] - pairs["piv0_wall_s"]
    pairs["time_saved_pct"] = 100 * pairs["seconds_saved"] / pairs["gain_wall_s"]

    groups = pairs.groupby(level="dataset", sort=False)
    dataset = groups.agg(
        metric=("metric", "first"), n_pairs=("metric", "size"),
        width_min=("width", "min"), width_max=("width", "max"),
    )
    for field in ("piv0_test_utility", "gain_test_utility", "paired_test_delta", "piv0_wall_s", "gain_wall_s"):
        dataset[f"{field}_mean"] = groups[field].mean()
        dataset[f"{field}_sd"] = groups[field].std(ddof=1)
    for field in ("piv0_negative", "piv0_retained", "gain_retained", "piv0_fits", "gain_fits",
                  "piv0_voting_s", "seconds_saved"):
        dataset[f"{field}_mean"] = groups[field].mean()
    delta = pairs["paired_test_delta"]
    dataset["piv0_score_higher"] = delta.gt(1e-12).groupby(level="dataset").sum()
    dataset["scores_tied"] = delta.abs().le(1e-12).groupby(level="dataset").sum()
    dataset["piv0_score_lower"] = delta.lt(-1e-12).groupby(level="dataset").sum()
    # Ratio of total elapsed times weights each second equally.
    dataset["time_saved_pct"] = 100 * groups["seconds_saved"].sum() / groups["gain_wall_s"].sum()

    totals = pairs[["piv0_fits", "gain_fits", "piv0_wall_s", "gain_wall_s", "seconds_saved"]].sum().to_dict()
    totals["n_pairs"] = len(pairs)
    totals["time_saved_pct"] = 100 * totals["seconds_saved"] / totals["gain_wall_s"]
    reports = {
        "paired_comparison.csv": pairs.reset_index(),
        "dataset_comparison.csv": dataset.reset_index(),
        "cost_totals.csv": pd.DataFrame([totals]),
    }
    for name, frame in reports.items():
        path = run / name
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Five-seed output directory containing summary.csv")
    args = parser.parse_args()
    reports = summarize(args.run)
    for name, frame in reports.items():
        print(f"{args.run / name} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
