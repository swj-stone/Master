"""Readable method tables for the frozen directional-minimum minPI iteration."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_refined_document import KEY, ROUNDING_TOLERANCE, read_pass


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("existing", "reliability-document-20261117a", "reliability-document-20261117b",
     "reliability-document-comparison-20261117a"),
    ("turkish_sms", "reliability-document-external-20261117a", "reliability-document-external-20261117b",
     "reliability-document-external-comparison-20261117a"),
    ("turkish_mail", "reliability-mail-20261117b", "reliability-mail-20261117c",
     "reliability-mail-audit-20261117a"),
)


def run(output):
    tables, comparisons, hashes = [], [], {}
    for cohort, first, second, audit in RUNS:
        directories = [ROOT / "results/minpi-study" / name for name in (first, second)]
        records = [read_pass(directory) for directory in directories]
        (_, a, metadata), (_, b, _) = records
        for field in ("score", "width", "p"):
            np.testing.assert_array_equal(a[field], b[field])
        full_a, full_b = (frame.xs("full", level="method") for frame in (a, b))
        for method in a.index.get_level_values("method").unique():
            left, right = (frame.xs(method, level="method") for frame in (a, b))
            rows = metadata[["suite", "stage"]].copy()
            rows["cohort"] = cohort
            if cohort == "existing":
                rows["cohort"] = np.where(rows.suite.eq("uci"), "uci", "spamassassin")
            rows["method"] = method
            for field in ("score", "width", "p"):
                rows[field] = left[field]
            rows["keep"] = left.width / left.p
            rows["select_ms"] = np.maximum(left["select"], right["select"])
            rows["total_ms"] = np.maximum(left.time, right.time)
            rows["time_full_ratio"] = np.maximum(left.time / full_a.time, right.time / full_b.time)
            rows["score_full_delta"] = left.score - full_a.score
            rows["full_joint"] = ((left.width < left.p) &
                                  (rows.score_full_delta >= -.02 - ROUNDING_TOLERANCE) &
                                  (rows.time_full_ratio < 1))
            tables.append(rows.reset_index())
        audit_path = ROOT / "results/minpi-study" / audit / "cases.csv"
        audited = pd.read_csv(audit_path)
        assert set(audited.candidate) == {"reliability"}
        audited["cohort"] = (np.where(audited.suite.eq("uci"), "uci", "spamassassin")
                             if cohort == "existing" else cohort)
        comparisons.append(audited)
        hashes[audit_path.relative_to(ROOT).as_posix()] = sha256(audit_path.read_bytes()).hexdigest()
        for directory in directories:
            for file in directory.glob("*.csv"):
                hashes[file.relative_to(ROOT).as_posix()] = sha256(file.read_bytes()).hexdigest()
            protocol = directory / "protocol.json"
            hashes[protocol.relative_to(ROOT).as_posix()] = sha256(protocol.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    cases = pd.concat(tables, ignore_index=True)
    assert len(cases) == 26 * 10
    assert not cases.duplicated(KEY + ["method"]).any()
    cases.to_csv(output / "method-cases.csv", index=False)
    aggregation = dict(
        configs=("score", "size"), mean_score=("score", "mean"),
        median_width=("width", "median"), median_keep=("keep", "median"),
        median_select_ms=("select_ms", "median"), median_total_ms=("total_ms", "median"),
        full_joint=("full_joint", "sum"), worst_full_gap=("score_full_delta", "min"),
        median_full_time_ratio=("time_full_ratio", "median"),
    )
    cases.groupby(["cohort", "method"]).agg(**aggregation).to_csv(output / "methods-by-cohort.csv")
    cases.groupby(["dataset", "method"]).agg(**aggregation).to_csv(output / "methods-by-source.csv")
    pairs = cases[cases.method.eq("reliability")].set_index(KEY)
    previous = cases[cases.method.eq("old_vote")].set_index(KEY)
    pairs = pairs[["cohort", "stage", "score", "width", "p", "full_joint"]].copy()
    pairs["old_score"] = previous.score
    pairs["old_width"] = previous.width
    pairs["score_old_delta"] = pairs.score - pairs.old_score
    pairs["width_old_ratio"] = pairs.width / pairs.old_width
    pairs["fewer_columns"] = pairs.width < pairs.old_width
    pairs["score_win"] = pairs.score_old_delta > ROUNDING_TOLERANCE
    pairs["score_tie"] = pairs.score_old_delta.abs() <= ROUNDING_TOLERANCE
    pairs["score_loss"] = pairs.score_old_delta < -ROUNDING_TOLERANCE
    comparisons = pd.concat(comparisons, ignore_index=True)
    assert not comparisons.duplicated(KEY + ["group"]).any()
    audited = comparisons[comparisons.group.eq("old_vote")].set_index(KEY)
    np.testing.assert_allclose(pairs.score_old_delta.sort_index(), audited.direct_score_delta.sort_index(),
                               atol=1e-12, rtol=0)
    for field in ("direct_time_ratio", "joint", "joint_5pct", "joint_10pct", "both_windows",
                  "direct_joint", "direct_joint_10pct_width"):
        pairs[field] = audited[field]
    pairs.to_csv(output / "new-old-cases.csv")
    pair_summary = dict(
        configs=("score", "size"), fewer_columns=("fewer_columns", "sum"),
        score_win=("score_win", "sum"), score_tie=("score_tie", "sum"), score_loss=("score_loss", "sum"),
        mean_score_delta=("score_old_delta", "mean"), median_width_ratio=("width_old_ratio", "median"),
        worst_old_gap=("score_old_delta", "min"),
        median_old_time_ratio=("direct_time_ratio", "median"), joint=("joint", "sum"),
        joint_5pct=("joint_5pct", "sum"), joint_10pct=("joint_10pct", "sum"),
        both_windows=("both_windows", "sum"), direct_joint=("direct_joint", "sum"),
    )
    pairs.groupby("cohort").agg(**pair_summary).to_csv(output / "new-old-summary.csv")
    pairs.groupby("dataset").agg(**pair_summary).to_csv(output / "new-old-by-source.csv")
    comparisons.to_csv(output / "baseline-intersections.csv", index=False)
    for field in ("cohort", "dataset"):
        comparisons.groupby([field, "group"]).agg(
            configs=("score", "size"), joint=("joint", "sum"), joint_5pct=("joint_5pct", "sum"),
            joint_10pct=("joint_10pct", "sum"), joint_10pct_width=("joint_10pct_width", "sum"),
            both_windows=("both_windows", "sum"),
        ).to_csv(output / ("baselines-by-" + field + ".csv"))
    (output / "audit.json").write_text(json.dumps(dict(
        source_sha256=hashes, driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=26, method_outputs=260, score_tolerance=.02,
        rounding_tolerance=ROUNDING_TOLERANCE,
        time="Per-configuration maximum of the two run medians; speed ratio is the maximum of paired ratios.",
        success="Actual reduction and score at least full minus .02 and faster in each independent process.",
    ), indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
