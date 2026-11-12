"""Compare frozen PIv1 and direct Gain across the six official datasets and five seeds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.thesis_prepared_catalog import FORMAL_DATASETS, FORMAL_SEEDS


METHODS = {"PIv1": "piv1", "direct_gain": "gain"}
COHORTS = {42: "development_42", 123: "replication_123", 456: "new_three", 789: "new_three", 1024: "new_three"}
TIE = 1e-14  # Match the frozen PIv1 pilot comparison.


def paired_arms(run):
    arms = pd.read_csv(run / "summary.csv", encoding="utf-8-sig")
    keys = ["dataset", "seed", "method"]
    expected = pd.MultiIndex.from_product([FORMAL_DATASETS, FORMAL_SEEDS, METHODS], names=keys)
    arms = arms.set_index(keys)
    if (not arms.index.is_unique or len(arms) != len(expected)
            or not expected.isin(arms.index).all() or not arms["status"].eq("completed").all()):
        raise ValueError("The five-seed comparison needs 60 distinct completed scheduled arms.")
    arms = arms.reindex(expected)
    counts = arms[["width", "retained", "fits", "candidates", "checked"]]
    numeric = [*counts, "validation_utility", "test_utility", "wall_s", "voting_s"]
    if (not np.isfinite(arms[numeric].to_numpy(dtype=float)).all()
            or not (counts.ge(1) & counts.mod(1).eq(0)).all().all()
            or not arms["width"].le(100).all()
            or not arms["retained"].le(arms["width"]).all()
            or not arms["checked"].eq(arms["candidates"]).all()
            or not arms["fits"].eq(arms["checked"] + 1).all()
            or not arms["wall_s"].gt(0).all()
            or arms[["metric", "origin"]].isna().any().any()):
        raise ValueError("Missing or inconsistent utility, runtime, fit count, or subset count.")
    gain = arms.xs("direct_gain", level="method")
    piv1 = arms.xs("PIv1", level="method")
    if (not gain[["width", "metric"]].equals(piv1[["width", "metric"]])
            or gain.groupby(level="dataset")["metric"].nunique().ne(1).any()
            or not gain["checked"].eq(gain["width"]).all()):
        raise ValueError("The two methods must use matched input widths and metrics and complete the Gain scan.")

    pairs = piv1[["width", "metric"]].copy()
    pairs["cohort"] = pairs.index.get_level_values("seed").map(COHORTS)
    fields = ["origin", "retained", "fits", "candidates", "validation_utility", "test_utility", "wall_s", "voting_s"]
    for method, prefix in METHODS.items():
        pairs = pairs.join(arms.xs(method, level="method")[fields].add_prefix(prefix + "_"))
    pairs["paired_test_delta"] = pairs["piv1_test_utility"] - pairs["gain_test_utility"]
    pairs["seconds_saved"] = pairs["gain_wall_s"] - pairs["piv1_wall_s"]
    pairs["time_saved_pct"] = 100 * pairs["seconds_saved"] / pairs["gain_wall_s"]
    pairs["wins"] = pairs["paired_test_delta"] > TIE
    pairs["ties"] = pairs["paired_test_delta"].abs() <= TIE
    pairs["losses"] = pairs["paired_test_delta"] < -TIE
    pairs["faster"] = pairs["seconds_saved"] > 0
    return pairs.reset_index()


def dataset_summary(rows, keys):
    groups = rows.groupby(keys, sort=False)
    result = groups.agg(
        metric=("metric", "first"), n_pairs=("seed", "size"), width_min=("width", "min"), width_max=("width", "max"),
        wins=("wins", "sum"), ties=("ties", "sum"), losses=("losses", "sum"), faster=("faster", "sum"),
    )
    for field in ("piv1_test_utility", "gain_test_utility", "paired_test_delta", "piv1_wall_s", "gain_wall_s"):
        result[field + "_mean"] = groups[field].mean()
        result[field + "_sd"] = groups[field].std(ddof=1)
    for field in ("piv1_retained", "gain_retained", "piv1_fits", "gain_fits", "piv1_voting_s", "seconds_saved"):
        result[field + "_mean"] = groups[field].mean()
    result["time_saved_pct"] = 100 * groups["seconds_saved"].sum() / groups["gain_wall_s"].sum()
    return result.reset_index()


def summarize(run):
    pairs = paired_arms(run)
    # Cross-dataset totals combine resources and counts; each utility is reported within its own dataset.
    costs = []
    for cohort, rows in [("all", pairs), *pairs.groupby("cohort", sort=False)]:
        count_fields = ["wins", "ties", "losses", "faster", "piv1_fits", "gain_fits"]
        total = {field: int(rows[field].sum()) for field in count_fields}
        total.update({field: rows[field].sum() for field in ["piv1_wall_s", "gain_wall_s", "seconds_saved"]})
        costs.append(dict(cohort=cohort, n_pairs=len(rows), **total,
                          time_saved_pct=100 * total["seconds_saved"] / total["gain_wall_s"]))
    reports = {"paired_comparison.csv": pairs, "dataset_comparison.csv": dataset_summary(pairs, ["dataset"]),
               "dataset_by_cohort.csv": dataset_summary(pairs, ["dataset", "cohort"]),
               "cost_totals.csv": pd.DataFrame(costs)}
    for name, frame in reports.items():
        path = run / name
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, encoding="utf-8-sig", index=False)
        tmp.replace(path)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed PIv1 benchmark directory containing summary.csv")
    directory = parser.parse_args().run
    for filename, report in summarize(directory).items():
        print(f"{directory / filename} ({len(report)} rows)")
