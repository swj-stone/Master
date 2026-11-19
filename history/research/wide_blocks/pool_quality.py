"""Evaluate all screened candidates together to localize prediction-score changes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.wide_blocks.method import wide_support
from research.wide_blocks.panels import KEY, cases, registry


HERE = Path(__file__).resolve().parent


def run(base):
    index = registry()
    wanted = set(index[index.p > 128].index)
    assert len(wanted) == 165
    quality = pd.read_csv(base / "pass1/quality.csv", keep_default_na=False)
    quality = quality.set_index(KEY + ["method"])
    rows = []
    for case in cases(wanted):
        cols, audit = wide_support(case.selection_X(), case.y, task=case.task, seed=case.seed, audit=True)
        candidates = np.asarray(audit["candidate_columns"], dtype=np.int64)
        ref = base / "pass1" / ("-".join(map(str, case.key)) + ".npz")
        with np.load(ref) as arrays:
            np.testing.assert_array_equal(cols, arrays["wide_cols"])
            cap_cols = arrays["cap_cols"]
        if audit["no_signal"]:
            np.testing.assert_array_equal(cols, np.arange(case.X.shape[1]))
            np.testing.assert_array_equal(cap_cols, cols)
        else:
            assert np.isin(cols, candidates).all() and np.isin(cap_cols, candidates).all()
        assert len(candidates) == int(quality.loc[case.key + ("wide",), "screen_width"])
        pool_score = case.score(*case.predict(candidates))
        row = dict(zip(KEY, case.key), p=case.X.shape[1], metric=case.metric,
                   candidate_count=len(candidates), adaptive_k=len(cols), cap_k=len(cap_cols),
                   pool_score=pool_score)
        for arm in ("wide", "cap", "full"):
            row[arm + "_score"] = quality.loc[case.key + (arm,), "score"]
        row["pool_minus_full"] = pool_score - row["full_score"]
        row["adaptive_minus_pool"] = row["wide_score"] - pool_score
        row["cap_minus_pool"] = row["cap_score"] - pool_score
        wider, audit256 = wide_support(case.selection_X(), case.y, task=case.task, seed=case.seed,
                                       candidate_limit=256, audit=True)
        larger_pool = np.asarray(audit256["candidate_columns"], dtype=np.int64)
        if audit256["no_signal"]:
            np.testing.assert_array_equal(wider, np.arange(case.X.shape[1]))
        else:
            assert len(larger_pool) == min(case.X.shape[1], 256)
            assert np.isin(wider, larger_pool).all()
        row["candidates_256"] = len(larger_pool)
        row["adaptive_k_256"] = len(wider)
        row["score_256"] = case.score(*case.predict(wider))
        row["expanded_minus_standard"] = row["score_256"] - row["wide_score"]
        rows.append(row)
    assert len(rows) == len(wanted)
    output = HERE / "evidence"
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "candidate-pool.csv", index=False)
    scopes = {"all_screened": frame, "129-256": frame[frame.p <= 256],
              "257+": frame[frame.p > 256], "dense": frame[frame.cohort.ne("text")],
              "text": frame[frame.cohort.eq("text")], "spectra": frame[frame.cohort.eq("spectra")]}
    contrasts = {"expanded_vs_adaptive": ("score_256", "wide_score"),
                 "pool_vs_full": ("pool_score", "full_score"),
                 "adaptive_vs_pool": ("wide_score", "pool_score"),
                 "filled_vs_pool": ("cap_score", "pool_score")}
    counts, metrics = [], []
    for name, group in scopes.items():
        for contrast, (left, right) in contrasts.items():
            diff = group[left] - group[right]
            counts.append(dict(scope=name, contrast=contrast, n=len(group),
                               higher=(diff > 1e-12).sum(), tied=(diff.abs() <= 1e-12).sum(),
                               lower=(diff < -1e-12).sum(), within_002=(diff >= -.02-1e-12).sum(),
                               median_k128=group.adaptive_k.median(),
                               median_k256=group.adaptive_k_256.median()))
            for metric, part in group.groupby("metric"):
                delta = part[left] - part[right]
                metrics.append(dict(scope=name, contrast=contrast, metric=metric, n=len(part),
                                    median_gap=delta.median(), mean_gap=delta.mean()))
    pd.DataFrame(counts).to_csv(output / "candidate-summary.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "candidate-by-metric.csv", index=False)
    print(f"Verified {len(frame)} candidate pools and expanded-budget evaluations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.run_dir)
