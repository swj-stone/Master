"""Aggregate precision-study timings within target and tabulate exact refinements."""
import argparse
from pathlib import Path

import pandas as pd


def report(folder):
    targets = pd.read_csv(folder / "targets.csv", dtype={"float_omitted": str, "float_added": str})
    counts, trials = (pd.read_csv(folder / (name + ".csv")) for name in ("counts", "trials"))
    key = ["panel", "seed", "h", "game"]
    precision_key = [*key, "bits"]
    assert not targets.duplicated(key).any()
    assert not counts.duplicated(precision_key).any()
    assert not trials.duplicated([*precision_key, "repeat"]).any()
    assert counts.oracle_match.all() and (trials.elapsed_ms > 0).all()
    assert (counts.exact_columns <= counts.refined).all() and (counts.refined <= counts.p).all()
    assert len(counts) == 4 * len(targets)
    assert trials.groupby(precision_key).size().eq(3).all() and len(trials) == 3 * len(counts)
    for row in targets.itertuples():
        ids = list(map(int, str(row.support).split()))
        assert len(ids) == row.h and ids == sorted(set(ids)) and 0 <= ids[0] <= ids[-1] < row.p

    durations = trials.groupby(precision_key).elapsed_ms.median().unstack("bits")
    timing_ratios = durations.div(durations[40], axis=0).stack().rename("ratio_to_40")
    counts = counts.merge(timing_ratios, on=precision_key, validate="one_to_one")
    counts["panel_type"] = counts.panel.str.startswith("constructed").map(
        {True: "constructed", False: "measured"})
    summary = counts.groupby(["panel_type", "bits"]).agg(
        targets=("p", "size"),
        targets_with_fallback=("exact_columns", lambda x: (x > 0).sum()),
        columns_with_fallback=("exact_columns", "sum"),
        largest_fallback=("exact_columns", "max"),
        median_time_ratio_to_40=("ratio_to_40", "median"),
    )
    summary.to_csv(folder / "precision-summary.csv")
    durations.to_csv(folder / "target-times-ms.csv")
    observed = targets[~targets.panel.str.startswith("constructed")]
    observed.groupby("game").agg(
        targets=("p", "size"),
        changed_from_default=("swaps_vs_default", lambda x: (x > 0).sum()),
        maximum_replacements=("swaps_vs_default", "max"),
    ).to_csv(folder / "peer-set-summary.csv")
    targets[targets.float_swaps > 0].drop(columns="support").to_csv(
        folder / "float-discrepancies.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    report(parser.parse_args().input)
