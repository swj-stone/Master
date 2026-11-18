"""Compare original and revised pivotal games at two budgets on wide data."""
import argparse
import json
import platform
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, file_hash, load_dataset, metrics, train
from experiments.minpi_held_permutation_control import (
    ASSESS_CAP, FIT_CAP, PERMUTATIONS, linear_permutation_losses, ridge,
)
from minppi_mofs.grouped_minpi import grouped_control_from_training
from minppi_mofs.local_refit_minpi import local_refit_top_from_training
from minppi_mofs.loss_minpi import held_loss_votes
from minppi_mofs.masked_loss_minpi import joint_ridge_cis_scores
from minppi_mofs.support_selector import PivotalSupportSelector


def held_permutation_scores(X, y, seed, *, large):
    """Rank on a stratified subset held out of the screen's ridge fit, within training."""
    assess_size = min(512 if large else ASSESS_CAP, len(y) // 4)
    fit_pool, assess = train_test_split(
        np.arange(len(y)), test_size=assess_size, random_state=seed, stratify=y)
    fit = fit_pool
    if not large and len(fit_pool) > FIT_CAP:
        fit, _ = train_test_split(fit_pool, train_size=FIT_CAP, random_state=seed,
                                  stratify=y[fit_pool])
    model = ridge("classification", 1.).fit(X[fit], y[fit])
    target = y[assess]
    _, codes, counts = np.unique(target, return_inverse=True, return_counts=True)
    weights = 1 / (len(counts) * counts[codes])
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(assess)) for _ in range(PERMUTATIONS)]
    # Subtracting the same unpermuted loss leaves the ranking unchanged.
    return linear_permutation_losses(model, X[assess], target, orders, "classification", weights)


def fit_support(method, X, y, fraction, seed):
    width = ceil(X.shape[1] * fraction)
    if method == "full":
        return np.arange(X.shape[1])
    if method == "random":
        return np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:width])
    if method == "local_refit_reference":
        return local_refit_top_from_training(X, y, width, random_state=seed)[0]
    if method == "CIS":
        scores = joint_ridge_cis_scores(X, y, "classification", seed)
        return np.sort(np.argsort(-scores, kind="stable")[:width])
    if method in ("PFI", "PFI_large"):
        scores = held_permutation_scores(X, y, seed, large=method == "PFI_large")
        return np.sort(np.argsort(-scores, kind="stable")[:width])
    if method == "grouped_no_vote":
        return grouped_control_from_training(X, y, width, random_state=seed)[0]
    if method == "loss_mean_vote":
        scores = held_loss_votes(X, y, "classification", seed).sum(axis=0)
        return np.sort(np.argsort(-scores, kind="stable")[:width])
    if method == "F":
        # Same ranking as the study's F baseline; compute all scores for both budgets.
        score = np.nan_to_num(f_classif(X, y)[0], nan=-np.inf)
        return np.sort(np.argsort(-score, kind="stable")[:width])
    rule = method if method in ("grouped", "loss", "utility", "coalition", "refit", "local_refit", "restoration") else "static"
    arithmetic = "float" if method == "original_float" else "rational"
    estimator = PivotalSupportSelector(fraction=fraction, random_state=seed,
                                       rule=rule, arithmetic=arithmetic)
    return estimator.fit(X, y).get_support(indices=True)


def run_dataset(name, output, repeats, fractions, methods):
    X, y, task, _, _, _, digest = load_dataset(name)
    rows = []
    for seed in (1129, 2027, 3469):
        tr, te = train_test_split(np.arange(len(X)), test_size=.25, stratify=y, random_state=seed)
        xt, yt, xv, yv = X[tr], y[tr], X[te], y[te]
        calls = [(fraction, method) for fraction in fractions for method in methods] + [(1., "full")]
        order = np.random.default_rng(seed).permutation(len(calls))
        for rep in range(repeats):
            for idx in np.roll(order, rep * 4):
                fraction, method = calls[idx]
                start = perf_counter()
                support = fit_support(method, xt, yt, fraction, seed)
                after_select = perf_counter()
                fit_x, test_x = (xt, xv) if method == "full" else (xt[:, support], xv[:, support])
                pred, scores = train("ridge", fit_x, yt, test_x, task, seed)
                end = perf_counter()
                quality = metrics(yv, pred, scores)
                rows.append(dict(dataset=name, source_sha256=digest, seed=seed, replicate=rep, method=method,
                                 fraction=fraction, n_features=len(support), accuracy=quality["accuracy"],
                                 balanced_accuracy=quality["balanced_accuracy"], auc=quality["auc"],
                                 selection_s=after_select-start, total_s=end-start,
                                 columns=" ".join(map(str, support))))
            print(f"{name} seed {seed}, pass {rep + 1}/{repeats}", flush=True)
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    frame = pd.DataFrame(rows)
    by_seed = frame.groupby(["fraction", "method", "seed"])[["accuracy", "auc", "total_s"]].median()
    print(by_seed.groupby(["fraction", "method"]).mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["madelon", "isolet"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.10, .25])
    choices = ["original_float", "original_exact", "grouped", "grouped_no_vote", "loss", "loss_mean_vote", "F"]
    parser.add_argument("--methods", nargs="+", choices=choices + [
        "random", "CIS", "PFI", "PFI_large", "utility", "coalition", "refit", "local_refit",
        "local_refit_reference", "restoration"],
                        default=choices)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if any(not 0 < fraction < 1 for fraction in args.fractions):
        parser.error("--fractions must be strictly between zero and one")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "experiments/minpi_study.py",
               ROOT / "experiments/minpi_held_permutation_control.py"]
    sources.extend(ROOT / "src/minppi_mofs" / name for name in (
        "support_selector.py", "grouped_minpi.py", "loss_minpi.py", "masked_loss_minpi.py", "exact_minpi.py",
        "utility_minpi.py", "coalition_minpi.py", "refit_minpi.py", "local_refit_minpi.py", "local_refit_fast.py",
        "restoration_minpi.py",
        "piv3_compact_filter.py", "integer_minpi.py", "word_minpi.py", "certified_minpi.py", "piv2.py"))
    manifest = dict(datasets=args.datasets, repeats=args.repeats, seeds=[1129, 2027, 3469],
                    fractions=args.fractions, methods=args.methods, test_fraction=.25, threads=1,
                    pfi="three shared permutations, held-out balanced error; small: up to 256 fit/128 assess; large: remaining train fit/up to 512 assess",
                    metric="accuracy, balanced accuracy and macro one-vs-rest AUC",
                    timing="fresh selection + column extraction + ridge training and prediction",
                    host=platform.platform(), numpy=np.__version__,
                    sha256={str(s.relative_to(ROOT)): file_hash(s) for s in sources})
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with threadpool_limits(limits=1):
        for dataset in args.datasets:
            run_dataset(dataset, args.output, args.repeats, args.fractions, args.methods)
