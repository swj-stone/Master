"""Audit distinct fold and seed combinations directly from frozen out-of-sample predictions."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stable_minpi_stress import DATASETS, OTHER_SEEDS, PRIMARY_SEED, TOLERANCE, metric


def audit(run):
    pairs = pd.read_csv(run/"seed-sensitivity.csv").set_index(
        ["dataset", "fold", "method", "first_seed", "second_seed"])
    output = []
    for dataset in DATASETS:
        task = "regression" if dataset in ("abalone", "crimedata") else "classification"
        for fold in (1, 2, 3):
            with np.load(run/f"{dataset}-{fold}-split.npz") as split:
                gold = split["gold"]
            with np.load(run/f"{dataset}-{fold}-outputs.npz") as saved:
                gold_score = metric(gold, saved["full_pred"], task)
                width = len(saved["full_support"])
                for method in ("stable", "environment_half"):
                    seen = {}
                    for seed in (PRIMARY_SEED, *OTHER_SEEDS):
                        prefix = method if seed == PRIMARY_SEED else f"{method}_s{seed}"
                        selected = saved[f"{prefix}_support"]
                        pred = saved[f"{prefix}_pred"]
                        score = metric(gold, pred, task)
                        gap = score - gold_score
                        seen[seed] = selected, score
                        output.append(dict(dataset=dataset, fold=fold, seed=seed, method=method, p=width,
                                           width=len(selected), score=score, full_score=gold_score,
                                           gap=gap, meets_tolerance=gap >= -TOLERANCE,
                                           reduced=len(selected) < width))
                    for a, b in combinations(seen, 2):
                        record = pairs.loc[(dataset, fold, method, a, b)]
                        col_a, score_a = seen[a]
                        col_b, score_b = seen[b]
                        overlap = len(np.intersect1d(col_a, col_b)) / len(np.union1d(col_a, col_b))
                        np.testing.assert_allclose([score_a, score_b, overlap],
                                                   [record.first_score, record.second_score, record.jaccard],
                                                   rtol=0, atol=1e-12)
    df = pd.DataFrame(output)
    assert len(df) == len(DATASETS) * 3 * 3 * 2
    assert not df.duplicated(["dataset", "fold", "seed", "method"]).any()
    df.to_csv(run/"distinct-seed-cases.csv", index=False)
    df["reduced_in_tolerance"] = df.reduced & df.meets_tolerance
    totals = df.groupby(["method", "seed"]).agg(
        cases=("fold","size"), in_tolerance=("meets_tolerance","sum"), reduced=("reduced","sum"),
        reduced_in_tolerance=("reduced_in_tolerance","sum"), worst_gap=("gap","min"))
    totals.to_csv(run/"distinct-seed-summary.csv")
    print(totals.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
