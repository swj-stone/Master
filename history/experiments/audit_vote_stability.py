"""Recount joint successes and paired ranking gains from raw signal-vote runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]
ARMS = ("signal_auto", "no_peer", "random_signal", "F_signal", "stable_auto", "full")


def run(directory):
    raw = pd.read_csv(directory/"trials.csv")
    keys = KEY + ["method"]
    assert not raw.duplicated(keys+["repeat"]).any()
    counts = raw.groupby(keys).size()
    assert counts.nunique() == 1 and counts.iat[0] in (5,7)
    assert set(raw.method) == set(ARMS)
    assert raw.groupby(keys)[["score","width","p"]].nunique().eq(1).all().all()
    config = raw.groupby(keys).agg(score=("score","first"), width=("width","first"),
                                   p=("p","first"), select_ms=("select_ms","median"),
                                   total_ms=("total_ms","median"))
    pivot = config.width.unstack()
    for arm in ("no_peer","F_signal","random_signal"):
        pd.testing.assert_series_equal(pivot[arm],pivot.signal_auto,check_names=False)
    full = config.xs("full",level="method")[["score","total_ms"]].add_suffix("_full")
    config = config.join(full)
    config["gap"] = config.score-config.score_full
    config["ratio"] = config.total_ms/config.total_ms_full
    config["reduced"] = config.width < config.p
    config["quality"] = config.gap >= -.02
    config["joint"] = config.reduced & config.quality & (config.ratio<1.)
    config["joint_no_loss"] = config.reduced & (config.gap>=0.) & (config.ratio<1.)
    n = counts.iat[0]
    # Disjoint early/late repeat groups; omit the central repeat to test consistency.
    window = n//2
    for label, samples in (("early",raw.repeat<window),("late",raw.repeat>=n-window)):
        t = raw.loc[samples].groupby(keys).total_ms.median()
        ref = t.xs("full",level="method").rename("ref")
        q = t.to_frame().join(ref)
        config[label] = q.total_ms < q.ref
    config["both_windows"] = config.joint & config.early & config.late
    config["kept_fraction"] = config.width/config.p
    archive = pd.read_csv(directory/"cases.csv").set_index(keys).sort_index()
    pd.testing.assert_series_equal(config.joint,archive.joint)
    config.to_csv(directory/"raw-recount.csv")
    votes = config.xs("signal_auto",level="method").copy()
    for arm in ("random_signal","F_signal","no_peer","stable_auto"):
        control = config.xs(arm,level="method")
        delta = votes.score-control.score
        votes["delta_"+arm] = delta
        votes["win_"+arm] = delta.gt(1e-12)
        votes["loss_"+arm] = delta.lt(-1e-12)
        votes["tie_"+arm] = np.abs(delta) <= 1e-12
    draws = pd.read_csv(directory/"random-sensitivity.csv").set_index(KEY).sort_index()
    assert draws.index.equals(votes.index) and draws.draws.eq(20).all()
    votes["above_random_median"] = draws.median_delta.gt(1e-12)
    votes["below_random_median"] = draws.median_delta.lt(-1e-12)
    votes["joint_above_random_median"] = votes.joint & votes.above_random_median
    votes["wins_random_draws"] = draws.wins
    votes["ties_random_draws"] = draws.ties
    votes.to_csv(directory/"vote-attribution.csv")
    for levels, suffix in ((["dataset"],"by-source"),(None,"overall")):
        stats = dict(configs=("score","size"), within=("quality","sum"), joint=("joint","sum"),
                     no_loss=("joint_no_loss","sum"), both_windows=("both_windows","sum"),
                     joint_above_random_median=("joint_above_random_median","sum"),
                     median_width_fraction=("kept_fraction","median"), median_time_ratio=("ratio","median"))
        for col in votes.columns:
            if col.startswith(("win_","loss_","tie_","above_","below_","wins_random","ties_random")):
                stats[col]=(col,"sum")
            if col.startswith("delta_"):
                stats["median_"+col]=(col,"median")
        grouped = votes.groupby(level=levels) if levels else votes.groupby(lambda _:"all")
        grouped.agg(**stats).to_csv(directory/("vote-attribution-"+suffix+".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",type=Path,nargs="+")
    for directory in parser.parse_args().directories:
        run(directory)
