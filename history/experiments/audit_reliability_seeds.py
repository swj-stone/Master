"""Independently reconstruct paired reliability-seed results from both trial files."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]
ARMS = ("reliability_vote", "fast_vote", "full", "F_50", "chi_50", "random_50",
        "F_match", "pooled_match", "random_match")
EPS = 1e-12


def reconstruct(path):
    raw = pd.read_csv(path / "trials.csv")
    assert len(raw) == 24 * 9 * 9 and set(raw.method) == set(ARMS)
    assert not raw.duplicated(KEY + ["method", "repeat"]).any()
    groups = raw.groupby(KEY + ["method"])
    assert (groups.size() == 9).all()
    for col in ("width", "p", "score"):
        assert (groups[col].nunique() == 1).all()
    assert (raw[["select_ms", "refit_ms", "total_ms"]] >= 0).all().all()
    np.testing.assert_allclose(raw.total_ms, raw.select_ms + raw.refit_ms, atol=1e-10, rtol=0)
    cases = groups[["width", "p", "score", "select_ms", "refit_ms", "total_ms"]].median()
    own = cases.xs("reliability_vote", level="method").copy()
    full = cases.xs("full", level="method")
    old = cases.xs("fast_vote", level="method")
    own["full_score"] = full.score
    own["gap"] = own.score - full.score
    own["time_ratio"] = own.total_ms / full.total_ms
    own["quality"] = own.gap >= -.02 - EPS
    own["joint"] = (own.width < own.p) & own.quality & (own.time_ratio < 1)
    own["joint10"] = own.joint & (own.time_ratio <= .9)
    own["old_score"] = old.score
    own["old_width"] = old.width
    own["old_joint"] = ((old.width < old.p) & (old.score - full.score >= -.02 - EPS)
                        & (old.total_ms < full.total_ms))
    for method in ARMS[1:]:
        base = cases.xs(method, level="method")
        delta = own.score - base.score
        own[method + "_score_delta"] = delta
        own[method + "_score_win"] = delta > EPS
        own[method + "_score_tie"] = delta.abs() <= EPS
        own[method + "_score_loss"] = delta < -EPS
        own[method + "_time_ratio"] = own.total_ms / base.total_ms
        own[method + "_width_ratio"] = own.width / base.width
        own[method + "_joint"] = (own.joint & (own.width <= base.width)
                                   & (delta >= -.02 - EPS) & (own.total_ms < base.total_ms))
    own["half_all"] = own.F_50_joint & own.chi_50_joint & own.random_50_joint
    own["same_supervised"] = own.F_match_joint & own.pooled_match_joint
    own["same_all"] = own.same_supervised & own.random_match_joint
    saved = pd.read_csv(path / "comparisons.csv").set_index(KEY).sort_index()
    for col in ("joint", "old_joint") + tuple(arm + "_joint" for arm in ARMS[1:]):
        np.testing.assert_array_equal(own.sort_index()[col], saved[col])
    for col in ("score", "width", "gap") + tuple(arm + "_score_delta" for arm in ARMS[1:]):
        np.testing.assert_allclose(own.sort_index()[col], saved[col], atol=EPS, rtol=0)
    return own.sort_index(), cases.sort_index()


def audit(first, second, output):
    a, ac = reconstruct(first)
    b, bc = reconstruct(second)
    assert a.index.equals(b.index)
    for col in ("width", "p", "score"):
        np.testing.assert_array_equal(ac[col], bc[col])
    aa = pd.read_csv(first / "fresh-process-audit.csv").set_index(KEY + ["method"]).sort_index()
    ba = pd.read_csv(second / "fresh-process-audit.csv").set_index(KEY + ["method"]).sort_index()
    assert len(aa) == len(ba) == 216
    for col in ("width", "score", "support_sha256", "prediction_sha256"):
        np.testing.assert_array_equal(aa[col], ba[col])
    output.mkdir(parents=True, exist_ok=False)
    flags = ("joint", "joint10", "old_joint", "half_all", "same_supervised", "same_all") + tuple(
        method + "_joint" for method in ARMS[1:])
    joined = a.add_prefix("pass1_").join(b.add_prefix("pass2_"))
    for flag in flags:
        joined[flag + "_both"] = a[flag] & b[flag]
    joined.to_csv(output / "paired-configurations.csv")
    summaries = []
    for label in ["ALL", *sorted(a.index.get_level_values("dataset").unique())]:
        mask = np.ones(len(a), dtype=bool) if label == "ALL" else a.index.get_level_values("dataset") == label
        x, y = a.loc[mask], b.loc[mask]
        row = dict(dataset=label, configurations=len(x), width_min=int(x.width.min()),
                   width_max=int(x.width.max()), score_mean=x.score.mean(), old_score_mean=x.old_score.mean(),
                   old_score_delta_mean=x.fast_vote_score_delta.mean(),
                   old_score_delta_min=x.fast_vote_score_delta.min(), old_score_delta_max=x.fast_vote_score_delta.max(),
                   old_width_ratio_median=x.fast_vote_width_ratio.median())
        for flag in flags:
            row[flag + "_pass1"] = int(x[flag].sum())
            row[flag + "_pass2"] = int(y[flag].sum())
            row[flag + "_both"] = int((x[flag] & y[flag]).sum())
        for pass_id, frame in ((1, x), (2, y)):
            for metric in ("select_ms", "total_ms", "time_ratio") + tuple(
                arm + "_time_ratio" for arm in ARMS[1:]):
                row[metric + f"_median_pass{pass_id}"] = frame[metric].median()
        for method in ARMS[1:]:
            for kind in ("win", "tie", "loss"):
                row[method + "_score_" + kind] = int(x[method + "_score_" + kind].sum())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    ranges = a.groupby(["dataset", "fold"]).agg(
        seeds=("score", "size"), score_min=("score", "min"), score_max=("score", "max"),
        old_score_min=("old_score", "min"), old_score_max=("old_score", "max"),
        width_min=("width", "min"), width_max=("width", "max"),
        old_width_min=("old_width", "min"), old_width_max=("old_width", "max"))
    assert (ranges.seeds == 3).all() and len(ranges) == 8
    ranges["score_range"] = ranges.score_max - ranges.score_min
    ranges["old_score_range"] = ranges.old_score_max - ranges.old_score_min
    ranges["range_change"] = ranges.score_range - ranges.old_score_range
    ranges.to_csv(output / "per-fold-seed-ranges.csv")
    print(pd.DataFrame(summaries)[["dataset", "configurations", "joint_both", "fast_vote_joint_both",
                                   "half_all_both", "same_supervised_both", "same_all_both"]].to_string(index=False))
    print(ranges[["score_range", "old_score_range", "range_change"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.first, args.second, args.output)
