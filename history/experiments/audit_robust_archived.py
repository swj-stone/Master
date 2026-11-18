"""Audit stored classification predictions and floating-point threshold boundaries."""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.document_vote_archived_panels import cases
from experiments.stable_minpi_stress import file_hash, write_json


KEY = ["dataset", "fold", "seed"]
EPSILON = 1e-12
TOLERANCE = Fraction(1, 50)


def main(source, output):
    source_cases = pd.read_csv(source / "cases.csv", float_precision="round_trip").set_index(
        KEY + ["method"]).sort_index()
    raw_text = pd.read_csv(source / "cases.csv", dtype={"gap": str}).set_index(KEY + ["method"])
    trials = pd.read_csv(source / "trials.csv", float_precision="round_trip")
    measured = trials.groupby(KEY + ["method"]).agg(
        score=("score", "first"), width=("width", "first"), total_ms=("total_ms", "median"))
    pd.testing.assert_index_equal(measured.index, source_cases.index)
    for column in measured:
        np.testing.assert_allclose(measured[column], source_cases[column], rtol=0, atol=1e-12)
    exact, class_rows = {}, []
    for suite, panel in cases():
        key = (panel.name, panel.fold, panel.seed)
        truth = panel.gold if suite == "original" else panel.truth
        with np.load(source / f"{panel.name}-{panel.fold}-{panel.seed}.npz") as saved:
            for method in source_cases.xs(key).index:
                pred = saved[method + "_pred"]
                terms = []
                for label in np.unique(truth):
                    count = int(np.sum(truth == label))
                    correct = int(np.sum((truth == label) & (pred == label)))
                    terms.append(Fraction(correct, count))
                    class_rows.append(dict(zip(KEY, key), method=method, label=str(label),
                                           class_count=count, class_correct=correct))
                exact[(*key, method)] = sum(terms) / len(terms)
                np.testing.assert_allclose(float(exact[(*key, method)]), source_cases.loc[(*key, method), "score"],
                                           rtol=0, atol=1e-14)
    audited = source_cases.copy()
    audited["strict_within"] = audited.within
    audited["strict_joint"] = audited.joint
    audited["strict_gap_text"] = raw_text.gap
    audited["score_fraction"] = [str(exact[index]) for index in audited.index]
    differences = [exact[index] - exact[(*index[:3], "full")] for index in audited.index]
    audited["gap_fraction"] = [str(difference) for difference in differences]
    audited["within"] = audited.gap >= -.02 - EPSILON
    np.testing.assert_array_equal(audited.within, [difference >= -TOLERANCE for difference in differences])
    audited["boundary_corrected"] = audited.within != audited.strict_within
    audited["reduced_pass"] = audited.reduced & audited.within
    audited["joint"] = audited.reduced_pass & (audited.ratio < 1)
    audited["joint_5pct"] = audited.reduced_pass & (audited.ratio <= .95)
    audited["joint_10pct"] = audited.reduced_pass & (audited.ratio <= .90)
    audited["paired_joint"] = audited.reduced_pass & (audited.paired_ratio < 1)
    output.mkdir(parents=True, exist_ok=False)
    audited.to_csv(output / "cases.csv")
    audited[audited.boundary_corrected].to_csv(output / "boundary-corrections.csv")
    pd.DataFrame(class_rows).to_csv(output / "class-counts.csv", index=False)
    audited.reset_index().groupby(["dataset", "method"]).agg(
        configs=("seed", "size"), joint=("joint", "sum"), strict_joint=("strict_joint", "sum"),
        joint_10pct=("joint_10pct", "sum"), mean_score=("score", "mean"),
        median_width=("width", "median"), median_time_ms=("total_ms", "median"),
        median_ratio=("ratio", "median"),
    ).to_csv(output / "summary.csv")
    robust = audited.xs("robust", level="method")
    pairs = []
    for method in audited.index.get_level_values("method").unique():
        if method == "robust":
            continue
        baseline = audited.xs(method, level="method")
        pair = robust[["joint", "strict_joint", "joint_10pct"]].copy()
        pair["baseline"] = method
        pair["score_delta"] = robust.score - baseline.score
        pair["width_ratio"] = robust.width / baseline.width
        pair["time_ratio"] = robust.total_ms / baseline.total_ms
        delta = [exact[(*key, "robust")] - exact[(*key, method)] for key in robust.index]
        pair["delta_fraction"] = [str(difference) for difference in delta]
        safe_score = pair.score_delta >= -.02 - EPSILON
        np.testing.assert_array_equal(safe_score, [difference >= -TOLERANCE for difference in delta])
        safe_width = pair.width_ratio <= 1
        pair["relative_joint"] = pair.joint & safe_score & safe_width & (pair.time_ratio < 1)
        pair["strict_relative_joint"] = (
            pair.strict_joint & (pair.score_delta >= -.02) & safe_width & (pair.time_ratio < 1))
        pair["relative_joint_10pct"] = pair.joint_10pct & safe_score & safe_width & (pair.time_ratio <= .9)
        pairs.append(pair)
    pairs = pd.concat(pairs).reset_index()
    pairs.to_csv(output / "paired-baselines.csv", index=False)
    pairs.groupby(["dataset", "baseline"]).agg(
        configs=("seed", "size"), joint=("relative_joint", "sum"),
        strict_joint=("strict_relative_joint", "sum"), joint_10pct=("relative_joint_10pct", "sum"),
        median_score_delta=("score_delta", "median"), median_time_ratio=("time_ratio", "median"),
        median_width_ratio=("width_ratio", "median"),
    ).to_csv(output / "baseline-summary.csv")
    flags = pairs.pivot(index=KEY, columns="baseline", values="relative_joint")
    flags["F50_and_random50"] = flags.F_50 & flags.random_50
    flags.to_csv(output / "joint-comparisons.csv")
    write_json(output / "audit.json", {
        "raw_directory": str(source), "source_sha256": file_hash(Path(__file__)),
        "raw_sha256": {f.name: file_hash(f) for f in
                       [source / "cases.csv", source / "trials.csv", *sorted(source.glob("*.npz"))]},
        "score_margin": str(TOLERANCE), "comparison_epsilon": EPSILON,
        "checks": "Every guarded comparison equals exact classwise Fraction balanced accuracy.",
        "changed_own_within": int(audited.boundary_corrected.sum()),
        "explanation": "Binary floating-point subtraction may place an exact 1/50 score decrease "
                       "below -0.02. Raw timings, raw flags, supports and predictions remain archived.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
