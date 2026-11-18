"""Recount candidate-versus-baseline comparisons directly from individual full-pipeline runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]


def run(out):
    trials = pd.read_csv(out/"trials.csv")
    keys = KEY + ["method"]
    assert not trials.duplicated(keys+["repeat"]).any()
    counts = trials.groupby(keys).size()
    assert counts.nunique() == 1 and counts.iat[0] in (5,7)
    five = trials.groupby(keys).agg(k=("width","first"),p=("p","first"),score=("score","first"),
                                     ms=("total_ms","median"),select_ms=("select_ms","median"))
    assert trials.groupby(keys)[["width","p","score"]].nunique().eq(1).all().all()
    pair = pd.read_csv(out/"baseline-comparisons.csv").set_index(KEY+["candidate","baseline"])
    result = []
    candidates = sorted(set(trials.method).intersection({"budget","ridge","capped"}))
    for candidate in candidates:
        a = five.xs(candidate,level="method")
        for same in ("F_"+candidate,"random_"+candidate):
            pd.testing.assert_series_equal(a.k,five.xs(same,level="method").k)
        for name, baselines in {
            "matched": ("full","F_"+candidate,"random_"+candidate),
            "independent": ("full","F50","random50","signal_auto"),
            "all": ("full","F50","random50","signal_auto","F_"+candidate,"random_"+candidate),
        }.items():
            accepted = pd.DataFrame(True,index=a.index,columns=(
                "relative_joint", "relative_joint_width_10pct","exact_quality_joint","both_windows"))
            for arm in baselines:
                b = five.xs(arm,level="method")
                eligible = (a.k<a.p) & (a.ms<b.ms)
                quality = a.score - b.score >= -.02
                single = pd.DataFrame(dict(
                    relative_joint=eligible & quality & (a.k<=b.k),
                    relative_joint_width_10pct=eligible & quality & (10*a.k<=11*b.k),
                    exact_quality_joint=eligible & (a.score>=b.score) & (a.k<=b.k)))
                saved = pair.xs((candidate,arm),level=("candidate","baseline")).sort_index()
                for field in single:
                    np.testing.assert_array_equal(single[field],saved[field])
                single["both_windows"] = single.relative_joint.copy()
                n = counts.iat[0]
                for sample in (trials.repeat < n//2, trials.repeat >= n-n//2):
                    times = trials[sample].groupby(keys).total_ms.median().unstack()
                    single["both_windows"] &= times[candidate]<times[arm]
                accepted &= single
            result.append(accepted.reset_index().assign(candidate=candidate,group=name))
    cases = pd.concat(result,ignore_index=True)
    cases.to_csv(out/"common-baselines.csv",index=False)
    summary = cases.groupby(["candidate","group"]).agg(
        configs=("relative_joint","size"), joint=("relative_joint","sum"), margin=("relative_joint_width_10pct","sum"),
        exact_quality=("exact_quality_joint","sum"), robust=("both_windows","sum"))
    summary.to_csv(out/"common-baselines-summary.csv")
    # Restore unique headings in each single-metric export made by the timed runner.
    all_methods = cases[cases.group=="all"]
    for field in ["relative_joint","relative_joint_width_10pct","exact_quality_joint","also_full_joint"]:
        output = all_methods[["candidate",*KEY]].copy()
        output[field] = all_methods["relative_joint" if field=="also_full_joint" else field]
        output.to_csv(out/(field+"-all-baselines.csv"),index=False)
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",type=Path,nargs="+")
    for path in parser.parse_args().directories:
        run(path)
