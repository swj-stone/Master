"""Check the published transfer supports with integer arithmetic and additional layer controls."""
import argparse
from fractions import Fraction
from math import ceil, comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, train
from experiments.minpi_study_new_cohorts import DIGESTS, LAYERS, SEEDS, batch_scores, load_cohort
from minppi_mofs.integer_minpi import rational_top_from_votes


def fractional_oracle(votes, h):
    """Enumerate the positive and negative draws, retaining exact per-layer denominators."""
    width = votes.shape[1]
    pos = (votes > 0).sum(1)
    neg = (votes < 0).sum(1)
    abstain = width - pos - neg
    rows = zip(pos.tolist(), neg.tolist(), abstain.tolist())
    compositions = list(rows)
    profiles = set(compositions)
    scores = []
    for k in LAYERS:
        weights = {}
        for profile in profiles:
            for sign in (-1, 1):
                p, n, z = profile
                p -= sign == 1
                n -= sign == -1
                count = 0
                for i in range(min(p, k) + 1):
                    for j in range(min(n, k-i) + 1):
                        old = int(np.sign(i-j))
                        increment = int(np.sign(i-j+sign)) - old
                        count += comb(p, i) * comb(n, j) * comb(z, k-i-j) * increment
                weights[profile, sign] = count
        num = [0] * width
        for profile, row in zip(compositions, votes.tolist()):
            for col, sign in enumerate(row):
                if sign:
                    num[col] += weights[profile, sign]
        denominator = len(votes) * comb(width-1, k)
        scores.append([Fraction(n, denominator) for n in num])
    minima = [min(column) for column in zip(*scores)]
    return sorted(sorted(range(width), key=minima.__getitem__, reverse=True)[:h])


def check(directory):
    records = []
    for name in DIGESTS:
        X, y, groups, tr, te, task = load_cohort(name)
        h = ceil(.1 * X.shape[1])
        for seed in SEEDS:
            with np.load(directory / f"{name}-{seed}.npz") as case:
                v, rates = case["train_votes"], case["train_layers"]
                columns, info = rational_top_from_votes(v, h)
                oracle = fractional_oracle(v, h)
                np.testing.assert_array_equal(oracle, columns)
                float_swaps = len(set(oracle) - set(case["columns_min"]))
                for method, vector in (("G4", rates[1]), ("layer_mean", rates.mean(axis=0))):
                    selected = np.sort(np.argsort(-vector, kind="stable")[:h])
                    pred, _ = train("ridge", X[tr][:, selected], y[tr], X[te][:, selected], task, seed)
                    quality = (np.mean(list(batch_scores(y[te], pred, groups[te]).values()))
                               if task == "classification" else r2_score(y[te], pred))
                    records.append(dict(source=name, seed=seed, method=method, quality=quality,
                                        integer_float_swaps=float_swaps, oracle_agrees=True,
                                        exact_columns=info["exact_columns"], selected=" ".join(map(str, selected))))
    pd.DataFrame(records).to_csv(directory / "secondary-checks.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/minpi-study/new-cohorts")
    with threadpool_limits(1):
        check(parser.parse_args().input)
