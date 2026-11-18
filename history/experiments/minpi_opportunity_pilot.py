"""Explore a per-layer opportunity normalization on fixed joint-ridge masking ballots."""
import argparse
from fractions import Fraction
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.minpi_loss_ballot_pilot import folds
from experiments.minpi_study import select, train
from experiments.minpi_study_new_cohorts import SEEDS, batch_scores, load_cohort
from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _layers
from minppi_mofs.masked_loss_minpi import masked_loss_votes
from minppi_mofs.word_minpi import exact_top_from_votes


def opportunity_top(votes, width, tiers=None):
    """Rank by the weakest signed layer after dividing by its mean absolute mass.

    Each signed layer numerator has common denominator w*binom(p-1,k), shared by
    the mean absolute mass. The extra factor p is constant across all layers.
    Thus we compare integer totals divided by the layer total absolute mass.
    """
    votes = _validate(votes, width)
    levels = _selection_tiers(tiers, votes.shape[1])
    positives = votes > 0
    negatives = votes < 0
    positive_counts = positives.sum(axis=1).tolist()
    negative_counts = negatives.sum(axis=1).tolist()
    p = votes.shape[1]
    numerator = np.zeros(p, dtype=object)
    divisor = np.zeros(p, dtype=object)
    for _, signed, _, _ in _layers(votes, levels, None):
        opportunity = sum(a * positive - b * negative for (a, b), positive, negative
                          in zip(signed, positive_counts, negative_counts))
        if not opportunity:
            continue
        # Python integers keep products exact, including 128-column size-16 layers.
        weight = np.asarray(signed, dtype=object)
        total = weight[:, 0] @ positives + weight[:, 1] @ negatives
        lower = (divisor == 0) | (total * divisor < numerator * opportunity)
        numerator[lower] = total[lower]
        divisor[lower] = opportunity
    if not np.any(divisor):
        return np.arange(width)
    order = sorted(range(p), key=lambda j: -Fraction(numerator[j], divisor[j]))
    return np.sort(order[:width])


def development(name):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
        outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=18617)
        for i, (fit, valid) in enumerate(outer.split(X, y), 1):
            yield i, X[fit], y[fit], X[valid], y[valid], None, "classification"
        return
    X, y, groups, fit0, test0, task = load_cohort(name)
    for split, fit, valid in folds(name, groups, fit0, test0):
        if split != "historical-test":
            yield split, X[fit], y[fit], X[valid], y[valid], groups[valid], task


def run(name):
    for split, xt, yt, xv, yv, group, task in development(name):
        h = ceil(xt.shape[1] * .1)
        for seed in SEEDS:
            votes = masked_loss_votes(xt, yt, task, seed)
            methods = dict(G0=exact_top_from_votes(votes, h, tiers=(0,))[0],
                           raw_min=exact_top_from_votes(votes, h)[0],
                           normalized_min=opportunity_top(votes, h),
                           F=select("f", xt, yt, task, seed)[0])
            for method, cols in methods.items():
                pred, _ = train("ridge", xt[:, cols], yt, xv[:, cols], task, seed)
                if name == "gas":
                    score = np.mean(list(batch_scores(yv, pred, group).values()))
                elif task == "regression":
                    score = r2_score(yv, pred)
                else:
                    score = balanced_accuracy_score(yv, pred)
                yield dict(source=name, split=split, seed=seed, method=method, score=score,
                           width=h, support=" ".join(map(str, cols)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        records = pd.DataFrame(row for name in ("gas", "supercon", "digits") for row in run(name))
    records.to_csv(out / "paired.csv", index=False)
    print(records.pivot_table(index="source", columns="method", values="score"))
