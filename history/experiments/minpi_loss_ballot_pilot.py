"""Test held-out univariate squared-loss ballots in the existing weakest-layer game."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from experiments.minpi_study import select, train
from experiments.minpi_study_new_cohorts import SEEDS, batch_scores, load_cohort
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.loss_minpi import held_loss_votes, loss_top_from_training
from minppi_mofs.word_minpi import exact_top_from_votes


def folds(name, groups, fit, test):
    if name == "gas":
        for b in (4, 5, 6, 7):
            yield f"development-{b}", np.where(groups < b)[0], np.where(groups == b)[0]
    else:
        for i, (train_part, valid_part) in enumerate(
                GroupKFold(n_splits=3).split(fit, groups=groups[fit]), 1):
            yield f"development-{i}", fit[train_part], fit[valid_part]
    yield "historical-test", fit, test


def evaluate(name, repeats):
    X, y, groups, outer_train, outer_test, task = load_cohort(name)
    width = int(np.ceil(X.shape[1] * .1))
    for split, fit, test in folds(name, groups, outer_train, outer_test):
        xt, yt, xv = X[fit], y[fit], X[test]
        for seed in SEEDS:
            timers = {method: [] for method in ("triplet", "loss", "F", "full")}
            chosen = {}
            predictions = {}
            for repeat in range(repeats):
                for method in np.roll(list(timers), repeat % len(timers)):
                    start = perf_counter()
                    if method in ("F", "full"):
                        selected = select(method.lower(), xt, yt, task, seed)[0]
                    elif method == "triplet":
                        votes = np.sign(witness_margins(xt, yt, task, seed)).astype(np.int8)
                        selected = exact_top_from_votes(votes, width)[0]
                    else:
                        selected = loss_top_from_training(xt, yt, width, task=task, random_state=seed)[0]
                    after_selection = perf_counter()
                    pred, _ = train("ridge", xt[:, selected], yt, xv[:, selected], task, seed)
                    stop = perf_counter()
                    timers[method].append((after_selection - start, stop - after_selection, stop - start))
                    if method in chosen:
                        np.testing.assert_array_equal(chosen[method], selected)
                        np.testing.assert_array_equal(predictions[method], pred)
                    chosen[method] = selected
                    predictions[method] = pred
            for method, selected in chosen.items():
                pred = predictions[method]
                if name == "gas":
                    score = np.mean(list(batch_scores(y[test], pred, groups[test]).values()))
                else:
                    score = r2_score(y[test], pred)
                selection, downstream, end_to_end = map(median, zip(*timers[method]))
                yield dict(source=name, split=split, seed=seed, method=method, score=score,
                           selector_seconds=selection, ridge_seconds=downstream, total_seconds=end_to_end,
                           repeats=repeats, width=len(selected), features=" ".join(map(str, selected)))


def layer_ablation(name):
    """Keep the loss ballots fixed and isolate the effect of the peer-count minimum."""
    X, y, groups, outer_train, outer_test, task = load_cohort(name)
    width = int(np.ceil(X.shape[1] * .1))
    for split, fit, test in folds(name, groups, outer_train, outer_test):
        xt, yt = X[fit], y[fit]
        for seed in SEEDS:
            votes = held_loss_votes(xt, yt, task, seed)
            for method, layers in [("G0", (0,)), ("G16", (16,)), ("min", (0, 4, 16))]:
                cols = exact_top_from_votes(votes, width, tiers=layers)[0]
                pred, _ = train("ridge", xt[:, cols], yt, X[test][:, cols], task, seed)
                if name == "gas":
                    quality = np.mean(list(batch_scores(y[test], pred, groups[test]).values()))
                else:
                    quality = r2_score(y[test], pred)
                yield dict(source=name, split=split, seed=seed, layers=method, score=quality,
                           features=" ".join(map(str, cols)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        records = pd.DataFrame(row for name in ("gas", "supercon") for row in evaluate(name, 4))
        ablated = pd.DataFrame(row for name in ("gas", "supercon") for row in layer_ablation(name))
        ablated.to_csv(opts.output / "layers.csv", index=False)
    records.to_csv(opts.output / "scores.csv", index=False)
    view = records.assign(stage=np.where(records.split == "historical-test", "replay", "development"))
    print(view.groupby(["source", "stage", "method"])[["score", "selector_seconds", "total_seconds"]].mean())
