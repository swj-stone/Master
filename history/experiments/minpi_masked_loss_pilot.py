"""Development-only comparison of singleton and joint-model masked loss ballots."""
import argparse
import hashlib
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from experiments.minpi_loss_ballot_pilot import folds
from experiments.minpi_study import train
from experiments.minpi_study_new_cohorts import SEEDS, batch_scores, load_cohort
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.loss_minpi import loss_top_from_training
from minppi_mofs.masked_loss_minpi import joint_ridge_cis_scores, masked_loss_top_from_training
from minppi_mofs.word_minpi import exact_top_from_votes


def select_support(method, X, y, width, task, seed):
    if method.startswith("neighbor"):
        votes = np.sign(witness_margins(X, y, task, seed)).astype(np.int8)
        scorer = top_from_votes if method == "neighbor_float" else exact_top_from_votes
        return scorer(votes, width)[0]
    if method == "singleton":
        return loss_top_from_training(X, y, width, task=task, random_state=seed)[0]
    if method == "CIS":
        scores = joint_ridge_cis_scores(X, y, task, seed)
        return np.sort(np.argsort(-scores, kind="stable")[:width])
    layers = (0,) if method == "masked_G0" else (0, 4, 16)
    return masked_loss_top_from_training(X, y, width, task=task, random_state=seed, tiers=layers)[0]


def run(name):
    X, y, groups, outer_train, outer_test, task = load_cohort(name)
    width = int(np.ceil(X.shape[1] * .1))
    methods = ["neighbor_float", "neighbor_exact", "singleton", "masked", "masked_G0", "CIS"]
    for split, fit, valid in folds(name, groups, outer_train, outer_test):
        if split == "historical-test":
            continue
        xt, yt, xv = X[fit], y[fit], X[valid]
        for seed in SEEDS:
            samples = {method: [] for method in methods}
            outputs = {}
            for repeat in range(len(methods)):
                for method in np.roll(methods, repeat):
                    start = perf_counter()
                    codes = select_support(method, xt, yt, width, task, seed)
                    after_select = perf_counter()
                    pred, _ = train("ridge", xt[:, codes], yt, xv[:, codes], task, seed)
                    end = perf_counter()
                    samples[method].append((after_select-start, end-start))
                    if method in outputs:
                        np.testing.assert_array_equal(codes, outputs[method][0])
                        np.testing.assert_array_equal(pred, outputs[method][1])
                    outputs[method] = codes, pred
            for method in methods:
                cols, pred = outputs[method]
                if task == "classification":
                    quality = np.mean(list(batch_scores(y[valid], pred, groups[valid]).values()))
                else:
                    quality = r2_score(y[valid], pred)
                select_s, total_s = map(median, zip(*samples[method]))
                yield dict(source=name, split=split, seed=seed, method=method, score=quality,
                           width=width, selector_s=select_s, total_s=total_s, repeats=len(methods),
                           support=" ".join(map(str, cols)), prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    directory = parser.parse_args().output
    directory.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        table = pd.DataFrame(row for name in ("gas", "supercon") for row in run(name))
    table.to_csv(directory / "paired.csv", index=False)
    print(table.groupby(["source", "method"])[["score", "selector_s", "total_s"]].mean())
