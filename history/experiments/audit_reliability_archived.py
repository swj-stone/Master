"""Recompute two-pass archived-panel decisions with exact balanced accuracy."""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_refined_document import read_pass
from experiments.document_vote_archived_panels import cases
from experiments.stable_minpi_stress import file_hash, write_json


KEY = ["dataset", "fold", "seed"]
EPSILON = 1e-12
MARGIN = Fraction(1, 50)


def run(sources, output):
    states, frames, metadata = zip(*(read_pass(source) for source in sources))
    assert {state["independent_pass"] for state in states} == {1, 2}
    assert {k: v for k, v in states[0].items() if k != "independent_pass"} == {
        k: v for k, v in states[1].items() if k != "independent_pass"}
    pd.testing.assert_frame_equal(metadata[0], metadata[1])
    for field in ("score", "width", "p"):
        pd.testing.assert_series_equal(frames[0][field], frames[1][field])
    for source in sources:
        assert (source / "fresh-process-audit.csv").is_file()
    for path in sources[0].glob("*.npz"):
        with np.load(path) as left, np.load(sources[1] / path.name) as right:
            assert set(left.files) == set(right.files)
            for field in left.files:
                np.testing.assert_array_equal(left[field], right[field])
    exact, class_rows = {}, []
    for suite, panel in cases():
        key = (panel.name, panel.fold, panel.seed)
        truth = panel.gold if suite == "original" else panel.truth
        with np.load(sources[0] / f"{panel.name}-{panel.fold}-{panel.seed}.npz") as saved:
            for method in frames[0].xs(key).index:
                pred = saved[method + "_pred"]
                terms = []
                for label in np.unique(truth):
                    count = int(np.sum(truth == label))
                    correct = int(np.sum((truth == label) & (pred == label)))
                    terms.append(Fraction(correct, count))
                    class_rows.append(dict(zip(KEY, key), method=method, label=str(label),
                                           class_count=count, class_correct=correct))
                exact[(*key, method)] = sum(terms) / len(terms)
                np.testing.assert_allclose(float(exact[(*key, method)]),
                                           frames[0].loc[(*key, method), "score"], rtol=0, atol=1e-14)
    rows = []
    for method in states[0]["methods"]:
        left, right = (frame.xs(method, level="method") for frame in frames)
        full_left, full_right = (frame.xs("full", level="method") for frame in frames)
        result = left[["score", "width", "p"]].copy()
        result["method"] = method
        result["time_a"], result["time_b"] = left.time, right.time
        result["total_ms"] = np.maximum(left.time, right.time)
        result["select_ms"] = np.maximum(left["select"], right["select"])
        result["gap"] = left.score - full_left.score
        result["ratio"] = np.maximum(left.time/full_left.time, right.time/full_right.time)
        result["score_fraction"] = [str(exact[(*key, method)]) for key in left.index]
        differences = [exact[(*key, method)]-exact[(*key, "full")] for key in left.index]
        result["gap_fraction"] = [str(value) for value in differences]
        result["within"] = result.gap >= -.02-EPSILON
        np.testing.assert_array_equal(result.within, [value >= -MARGIN for value in differences])
        reduced = left.width < left.p
        result["joint"] = reduced & result.within & (result.ratio < 1)
        result["strict_joint"] = reduced & (result.gap >= -.02) & (result.ratio < 1)
        result["rounding_changed_joint"] = result.joint != result.strict_joint
        result["joint_10pct"] = reduced & result.within & (result.ratio <= .9)
        result["both_windows"] = result.joint & ((left.early<full_left.early) & (left.late<full_left.late) &
                                                 (right.early<full_right.early) & (right.late<full_right.late))
        rows.append(result.reset_index())
    combined = pd.concat(rows, ignore_index=True)
    vote = combined[combined.method == "reliability"].set_index(KEY)
    comparisons = []
    for method in states[0]["methods"]:
        if method == "reliability":
            continue
        baseline = combined[combined.method == method].set_index(KEY)
        pair = vote[["joint", "strict_joint", "joint_10pct"]].copy()
        pair["baseline"] = method
        pair["score_delta"] = vote.score-baseline.score
        pair["width_ratio"] = vote.width/baseline.width
        pair["time_ratio"] = np.maximum(vote.time_a/baseline.time_a, vote.time_b/baseline.time_b)
        differences = [exact[(*key, "reliability")]-exact[(*key, method)] for key in vote.index]
        pair["delta_fraction"] = [str(value) for value in differences]
        safe = pair.score_delta >= -.02-EPSILON
        np.testing.assert_array_equal(safe, [value >= -MARGIN for value in differences])
        pair["relative_joint"] = pair.joint & safe & (pair.width_ratio <= 1) & (pair.time_ratio < 1)
        pair["strict_relative_joint"] = pair.strict_joint & (pair.score_delta >= -.02) & (
            pair.width_ratio <= 1) & (pair.time_ratio < 1)
        pair["relative_joint_10pct"] = pair.joint_10pct & safe & (pair.width_ratio <= 1) & (pair.time_ratio <= .9)
        pair["relative_joint_10pct_width"] = pair.joint & safe & (pair.width_ratio <= 1.1) & (pair.time_ratio < 1)
        comparisons.append(pair.reset_index())
    comparisons = pd.concat(comparisons, ignore_index=True)
    flags = comparisons.pivot(index=KEY, columns="baseline", values="relative_joint")
    flags["half_all"] = flags.F_50 & flags.random_50
    flags["matched_all"] = flags.F_match & flags.random_match
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "cases.csv", index=False)
    combined[combined.rounding_changed_joint].to_csv(output / "boundary-corrections.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "class-counts.csv", index=False)
    comparisons.to_csv(output / "paired-baselines.csv", index=False)
    flags.to_csv(output / "joint-comparisons.csv")
    combined.groupby(["dataset", "method"]).agg(
        configs=("seed", "size"), joint=("joint", "sum"), joint_10pct=("joint_10pct", "sum"),
        both_windows=("both_windows", "sum"), mean_score=("score", "mean"),
        median_width=("width", "median"), median_time_ms=("total_ms", "median"),
        median_select_ms=("select_ms", "median"), median_ratio=("ratio", "median"),
        worst_gap=("gap", "min"), rounding_changed_joint=("rounding_changed_joint", "sum"),
    ).to_csv(output / "summary.csv")
    comparisons.groupby(["dataset", "baseline"]).agg(
        configs=("seed", "size"), joint=("relative_joint", "sum"), joint_10pct=("relative_joint_10pct", "sum"),
        joint_10pct_width=("relative_joint_10pct_width", "sum"),
        median_score_delta=("score_delta", "median"), median_time_ratio=("time_ratio", "median"),
        median_width_ratio=("width_ratio", "median"),
    ).to_csv(output / "baseline-summary.csv")
    write_json(output / "audit.json", dict(
        sources=[str(source) for source in sources], source_sha256=file_hash(Path(__file__)),
        raw_sha256={str(source / name): file_hash(source / name) for source in sources
                    for name in ("protocol.json", "cases.csv", "trials.csv", "fresh-process-audit.csv")},
        margin=str(MARGIN), epsilon=EPSILON, all_guarded_comparisons_equal_exact_fraction=True,
        all_saved_arrays_identical_between_passes=True,
        changed_own_joint=int(combined.rounding_changed_joint.sum()),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.runs, args.output)
