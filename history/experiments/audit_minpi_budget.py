"""Audit strict compression, measured speed and held-out scores across minPI runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]


def read_trials(run):
    trials = pd.read_csv(run/"trials.csv")
    columns = [*KEY, "method"]
    assert not trials.duplicated([*columns, "repeat"]).any()
    assert trials.groupby(columns).repeat.size().eq(7).all()
    assert trials.groupby(columns).score.nunique().eq(1).all()
    cases = trials.groupby(columns, as_index=False).agg(
        width=("width", "first"), p=("p", "first"), score=("score", "first"),
        total_ms=("total_ms", "median"), select_ms=("select_ms", "median"))
    full = cases.query("method == 'full'")[[*KEY, "score", "total_ms"]]
    cases = cases.merge(full.rename(columns={"score": "full_score", "total_ms": "full_ms"}),
                        on=KEY, validate="many_to_one")
    cases["gap"] = cases.score - cases.full_score
    cases["ratio"] = cases.total_ms / cases.full_ms
    cases["reduced"] = cases.width < cases.p
    cases["quality"] = cases.gap >= -.02
    cases["reduced_quality"] = cases.reduced & cases.quality
    cases["faster"] = cases.ratio < 1.
    cases["joint"] = cases.reduced_quality & cases.faster
    for margin in (5, 10):
        cases[f"joint_{margin}pct"] = cases.reduced_quality & (cases.ratio <= 1. - margin / 100)
    cases["joint_no_loss"] = cases.joint & (cases.gap >= 0)
    assert not cases.loc[~cases.reduced, "joint"].any()
    original = pd.read_csv(run/"cases.csv")[[*columns, "joint", "joint_5pct", "joint_10pct"]]
    check = cases.merge(original, on=columns, validate="one_to_one", suffixes=("", "_saved"))
    for name in ("joint", "joint_5pct", "joint_10pct"):
        np.testing.assert_array_equal(check[name], check[name + "_saved"])
    return cases


def main(args):
    first = pd.concat([read_trials(args.five), read_trials(args.deploy)], ignore_index=True)
    first["round"] = "first"
    moment = read_trials(args.moment)
    moment["round"] = "moment"
    cases = pd.concat([first, moment], ignore_index=True)
    for block in (first, moment):
        assert not block.duplicated([*KEY, "method"]).any()
        assert block.groupby("method").size().eq(69).all()
        assert block.dataset.nunique() == 13 and len(block[KEY[:2]].drop_duplicates()) == 29
    before = first[first.method == "full"].set_index(KEY)
    after = moment[moment.method == "full"].set_index(KEY)
    pd.testing.assert_series_equal(before.score.sort_index(), after.score.sort_index())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cases.to_csv(out/"cases.csv", index=False)
    metrics = {
        "cases": ("fold", "size"), **{name: (name, "sum") for name in (
            "reduced", "quality", "reduced_quality", "faster", "joint", "joint_5pct",
            "joint_10pct", "joint_no_loss")},
        "median_ratio": ("ratio", "median"), "worst_gap": ("gap", "min")}
    for extra, name in (([], "summary"), (["dataset"], "by-source"), (["seed"], "by-seed")):
        cases.groupby(["round", "method", *extra]).agg(**metrics).to_csv(out/f"{name}.csv")
    verified = [pd.read_csv(path/"fresh-process-audit.csv") for path in
                (args.five, args.deploy, args.moment)]
    assert [len(v) for v in verified] == [90, 48, 138]
    old_deploy = pd.read_csv(args.deploy/"consensus-rematch.csv")
    old_five = pd.read_csv(args.five/"consensus-rematch.csv")
    print("Archived strict joint:", int(old_five.joint_historical.sum() + old_deploy.historical_joint.sum()))
    print(cases.groupby(["round", "method"]).agg(**metrics).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("five", "deploy", "moment", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    main(parser.parse_args())
