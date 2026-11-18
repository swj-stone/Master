"""Recompute revised minPI comparisons from two independent raw timing passes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]
SHARED = ["old_vote", "F50", "chi50", "pooled50", "random50", "full"]
ROUNDING_TOLERANCE = 1e-12


def read_pass(directory):
    state = json.loads((directory / "protocol.json").read_text(encoding="utf8"))
    raw = pd.read_csv(directory / "trials.csv")
    keys = KEY + ["method"]
    groups = raw.groupby(keys)
    assert not raw.duplicated(keys + ["repeat"]).any()
    assert groups[["width", "p", "score"]].nunique().eq(1).all().all()
    assert groups.size().eq(state["repeats"]).all()
    assert raw.groupby(keys).position.nunique().eq(state["repeats"]).all()
    assert set(raw.repeat) == set(range(state["repeats"]))
    assert raw.total_ms.gt(0).all() and raw.select_ms.ge(0).all()
    np.testing.assert_allclose(raw.select_ms + raw.refit_ms, raw.total_ms, rtol=1e-12, atol=1e-10)
    frame = groups.agg(width=("width", "first"), p=("p", "first"), score=("score", "first"),
                       time=("total_ms", "median"), select=("select_ms", "median"))
    for name, mask in (("early", raw.repeat < state["repeats"]//2),
                       ("late", raw.repeat >= state["repeats"]-state["repeats"]//2)):
        frame[name] = raw[mask].groupby(keys).total_ms.median()
    saved = pd.read_csv(directory / "cases.csv").set_index(keys).sort_index()
    pd.testing.assert_index_equal(frame.index, saved.index)
    for field, prior in (("width", "width"), ("p", "p"), ("score", "score"), ("time", "total_ms")):
        np.testing.assert_allclose(frame[field], saved[prior], rtol=0, atol=1e-11)
    audit_path = directory / "fresh-process-audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path).set_index(keys).sort_index()
        pd.testing.assert_index_equal(frame.index, audit.index)
        np.testing.assert_allclose(frame.score, audit.score, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(frame.width, audit.width)
    return state, frame, pd.read_csv(directory / "panels.csv").set_index(KEY).sort_index()


def compare(first, second, metadata, candidates):
    output = []
    for name in candidates:
        vote_a, vote_b = first.xs(name, level="method"), second.xs(name, level="method")
        baseline_groups = {"full": ["full"],
                           "half": ["full", "F50", "chi50", "pooled50", "random50"],
                           "matched_supervised": ["full", "F_" + name, "pooled_" + name],
                           "matched_all": ["full", "F_" + name, "pooled_" + name, "random_" + name]}
        for ref in SHARED[:-1] + ["F_" + name, "pooled_" + name, "random_" + name]:
            baseline_groups[ref] = ["full", ref]
        for group, refs in baseline_groups.items():
            row = metadata[["suite", "stage", "task"]].copy()
            row["candidate"], row["group"] = name, group
            row["score"], row["width"], row["p"] = vote_a.score, vote_a.width, vote_a.p
            row["keep"] = vote_a.width/vote_a.p
            row["time"] = np.maximum(vote_a.time, vote_b.time)
            row["select"] = np.maximum(vote_a["select"], vote_b["select"])
            row["worst_gap"] = np.inf
            row["worst_ratio"] = 0.
            row["worst_width_ratio"] = 0.
            row["windows"] = True
            for ref in refs:
                base_a, base_b = first.xs(ref, level="method"), second.xs(ref, level="method")
                row["worst_gap"] = np.minimum(row.worst_gap, vote_a.score-base_a.score)
                row["worst_ratio"] = np.maximum(row.worst_ratio,
                                                np.maximum(vote_a.time/base_a.time, vote_b.time/base_b.time))
                row["worst_width_ratio"] = np.maximum(row.worst_width_ratio, vote_a.width/base_a.width)
                row["windows"] &= ((vote_a.early<base_a.early) & (vote_a.late<base_a.late) &
                                   (vote_b.early<base_b.early) & (vote_b.late<base_b.late))
            reduction = vote_a.width < vote_a.p
            safe = reduction & (row.worst_gap >= -.02 - ROUNDING_TOLERANCE)
            row["joint"] = safe & (row.worst_ratio < 1) & (row.worst_width_ratio <= 1)
            row["joint_5pct"] = safe & (row.worst_ratio <= .95) & (row.worst_width_ratio <= 1)
            row["joint_10pct"] = safe & (row.worst_ratio <= .90) & (row.worst_width_ratio <= 1)
            row["joint_10pct_width"] = safe & (row.worst_ratio < 1) & (row.worst_width_ratio <= 1.1)
            row["joint_no_loss"] = row.joint & (row.worst_gap >= -ROUNDING_TOLERANCE)
            strict_joint = reduction & (row.worst_gap >= -.02) & (row.worst_ratio < 1) & (row.worst_width_ratio <= 1)
            row["rounding_changed_joint"] = row.joint != strict_joint
            row["both_windows"] = row.joint & row.windows
            row["direct_joint"] = False
            row["direct_joint_10pct_width"] = False
            if group not in ("half", "matched_supervised", "matched_all"):
                ref = refs[-1]
                base_a, base_b = first.xs(ref, level="method"), second.xs(ref, level="method")
                row["direct_score_delta"] = vote_a.score-base_a.score
                row["direct_time_ratio"] = np.maximum(vote_a.time/base_a.time, vote_b.time/base_b.time)
                row["direct_width_ratio"] = vote_a.width/base_a.width
                direct = reduction & (row.direct_score_delta >= -.02 - ROUNDING_TOLERANCE) & (row.direct_time_ratio < 1)
                row["direct_joint"] = direct & (row.direct_width_ratio <= 1)
                row["direct_joint_10pct_width"] = direct & (row.direct_width_ratio <= 1.1)
            output.append(row.reset_index())
    return pd.concat(output, ignore_index=True)


def run(args):
    states, frames, metadata = zip(*(read_pass(directory) for directory in args.runs))
    assert {state["independent_pass"] for state in states} == {1, 2}
    fields = set(states[0]) - {"independent_pass"}
    assert all(states[0][field] == states[1][field] for field in fields)
    pd.testing.assert_index_equal(frames[0].index, frames[1].index)
    for field in ("width", "p", "score"):
        np.testing.assert_array_equal(frames[0][field], frames[1][field])
    meta_fields = metadata[0].columns.drop("common_prepare_ms")
    pd.testing.assert_frame_equal(metadata[0][meta_fields], metadata[1][meta_fields])
    for source in args.runs[0].glob("*.npz"):
        with np.load(source) as left, np.load(args.runs[1]/source.name) as right:
            assert set(left.files) == set(right.files)
            for field in left:
                np.testing.assert_array_equal(left[field], right[field], err_msg=source.name+" "+field)
    candidates = [spec.split("=", 1)[0] for spec in states[0]["candidates"]]
    rows = compare(*frames, metadata[0], candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "cases.csv", index=False)
    aggregations = dict(configs=("fold", "size"), joint=("joint", "sum"),
                        joint_5pct=("joint_5pct", "sum"), joint_10pct=("joint_10pct", "sum"),
                        joint_10pct_width=("joint_10pct_width", "sum"),
                        joint_no_loss=("joint_no_loss", "sum"), both_windows=("both_windows", "sum"),
                        direct_joint=("direct_joint", "sum"),
                        direct_joint_10pct_width=("direct_joint_10pct_width", "sum"),
                        rounding_changed_joint=("rounding_changed_joint", "sum"),
                        mean_score=("score", "mean"), median_keep=("keep", "median"),
                        median_time_ms=("time", "median"), median_select_ms=("select", "median"),
                        worst_score_gap=("worst_gap", "min"), median_worst_ratio=("worst_ratio", "median"),
                        median_direct_score_delta=("direct_score_delta", "median"),
                        median_direct_time_ratio=("direct_time_ratio", "median"),
                        median_direct_width_ratio=("direct_width_ratio", "median"))
    rows.groupby(["candidate", "suite", "stage", "group"]).agg(**aggregations).to_csv(args.output/"summary.csv")
    rows.groupby(["candidate", "suite", "group"]).agg(**aggregations).to_csv(args.output/"summary-all.csv")
    rows.groupby(["candidate", "dataset", "stage", "group"]).agg(**aggregations).to_csv(args.output/"by-source.csv")
    random_paths = [directory/"random-draws.csv" for directory in args.runs]
    if all(path.exists() for path in random_paths):
        draws, replay = (pd.read_csv(path) for path in random_paths)
        pd.testing.assert_frame_equal(draws, replay, rtol=0, atol=1e-12)
        keys = ["candidate", *KEY]
        assert not draws.duplicated(keys+["draw"]).any()
        assert draws.groupby(keys).size().eq(20).all()
        actual = frames[0][["score", "width"]].rename_axis(index={"method": "candidate"}).reorder_levels(keys)
        matched = draws.merge(actual.rename(columns={"score": "actual_score", "width": "actual_width"}),
                              left_on=keys, right_index=True, validate="many_to_one")
        np.testing.assert_allclose(matched.vote_score, matched.actual_score, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(matched.width, matched.actual_width)
        np.testing.assert_array_equal(matched.win, matched.vote_score > matched.score+1e-12)
        random = draws.groupby(keys).agg(score=("vote_score", "first"), width=("width", "first"),
                                        random_wins=("win", "sum"), random_median=("score", "median"),
                                        random_best=("score", "max"))
        random = random.join(metadata[0][["suite", "stage"]], on=KEY).reset_index()
        random["above_median"] = random.score > random.random_median+1e-12
        random["above_all20"] = random.random_wins == 20
        random["median_gap"] = random.score-random.random_median
        random.to_csv(args.output/"random-cases.csv", index=False)
        random.groupby(["candidate", "suite", "stage"]).agg(
            configs=("fold", "size"), above_median=("above_median", "sum"),
            above_all20=("above_all20", "sum"), median_gap=("median_gap", "median"),
            total_random_wins=("random_wins", "sum")
        ).to_csv(args.output/"random-summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
