"""Cross-subject quality of fixed minPI candidates on the official UCI HAR partition."""
import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.minpi_held_permutation_control import linear_permutation_losses, ridge
from experiments.minpi_study import train
from experiments.minpi_study_subjects import DIGEST, ROOT, SEEDS, load_har, results, sha256
from minppi_mofs.local_refit_minpi import local_refit_top_from_training
from minppi_mofs.refit_minpi import refit_top_from_training
from minppi_mofs.restoration_minpi import restoration_top_from_training


BUDGETS = (10, 50)
METHODS = ("original", "grouped", "refit", "local_refit", "restoration", "matched_PFI",
           "full_fit_PFI", "F", "random")


def full_fit_pfi(X, y, people, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(people)
    held = np.isin(people, rng.permutation(unique)[:max(1, len(unique) // 4)])
    model = ridge("classification", 1.).fit(X[~held], y[~held])
    target = y[held]
    labels, codes, counts = np.unique(target, return_inverse=True, return_counts=True)
    weights = 1. / (len(labels) * counts[codes])
    orders = [rng.permutation(len(target)) for _ in range(3)]
    losses = linear_permutation_losses(model, X[held], target, orders, "classification", weights)
    return np.argsort(-losses, kind="stable"), np.flatnonzero(held)


def audit(saved_path, rows):
    """Reload supports and predictions and independently recompute all subject scores."""
    metrics = pd.DataFrame(rows).set_index(["seed", "key", "subject"])
    seen = {}
    with np.load(saved_path) as saved:
        truth, people = saved["truth"], saved["test_subjects"]
        for seed, key in metrics.index.droplevel("subject").unique():
            support = saved[f"s{seed}_{key}_support"]
            pred = saved[f"s{seed}_{key}_pred"]
            assert np.array_equal(np.unique(support), support) and support[0] >= 0 and support[-1] < 561
            assert len(support) == ceil(561 * int(key.split("_")[0]) / 100)
            assert pred.shape == truth.shape and np.isin(pred, np.arange(1, 7)).all()
            signature = tuple(support.tolist())
            if signature in seen:
                np.testing.assert_array_equal(seen[signature], pred)
            seen[signature] = pred
            for subject in np.unique(people):
                mask = people == subject
                score = balanced_accuracy_score(truth[mask], pred[mask])
                np.testing.assert_allclose(score, metrics.loc[(seed, key, subject), "balanced_accuracy"],
                                           atol=1e-12, rtol=0)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *(ROOT / "experiments" / f for f in (
        "compare_predictive_minpi.py", "minpi_study_subjects.py", "minpi_study.py",
        "minpi_held_permutation_control.py")), *(ROOT / "src/minppi_mofs" / f for f in (
        "restoration_minpi.py", "refit_minpi.py", "local_refit_minpi.py", "grouped_minpi.py",
        "masked_loss_minpi.py", "utility_minpi.py", "piv2.py", "certified_minpi.py",
        "exact_minpi.py", "word_minpi.py"))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    (out / "protocol.json").write_text(json.dumps(dict(
        archive_sha256=DIGEST, source_sha256=hashes, seeds=SEEDS, budgets=BUDGETS, methods=METHODS,
        partition="UCI HAR official 21 training, 9 held-out people. 7352 vs 2947 rows, 561 inputs.",
        metric="Held subject mean six-class recall, equal weight to subjects and selector seeds.",
        learner="Full outer training rows: standardize, ridge alpha=1 solver=lsqr tol=.001 max_iter=250.",
        screening="Restoration and matched k=0 PFI split by unordered training subject only.",
        large_pfi="Fit all rows of remaining 16 train people; assess all rows of 5 training people; 3 permutations.",
        timing="Quality only. Reuse final predictions for mathematically identical supports."),
        indent=2), encoding="utf-8")
    parts, _ = load_har(False)
    X, y, train_people = parts["train"]
    Xt, truth, people = parts["test"]
    saved = dict(truth=truth, test_subjects=people, train_subjects=train_people)
    cache, rows = {}, []
    for seed in SEEDS:
        large_rank, held = full_fit_pfi(X, y, train_people, seed)
        saved[f"s{seed}_large_pfi_assessment"] = held
        for budget, methods in [*((b, METHODS) for b in BUDGETS), (100, ("full",))]:
            count = ceil(X.shape[1] * budget / 100)
            for method in methods:
                if method in ("restoration", "matched_PFI"):
                    cols, info = restoration_top_from_training(
                        X, y, count, random_state=seed, groups=train_people,
                        tiers=(0,) if method == "matched_PFI" else None)
                    fit, assess = info["fit_rows"], info["assessment_rows"]
                    assert not np.intersect1d(train_people[fit], train_people[assess]).size
                    np.testing.assert_array_equal(np.unique(train_people[assess]), np.unique(train_people[held]))
                    saved[f"s{seed}_screen_fit"], saved[f"s{seed}_screen_assess"] = fit, assess
                elif method == "full_fit_PFI":
                    cols = np.sort(large_rank[:count])
                elif method in ("refit", "local_refit"):
                    selector = local_refit_top_from_training if method == "local_refit" else refit_top_from_training
                    cols = selector(X, y, count, random_state=seed)[0]
                else:
                    rule = {"original": "float", "F": "f"}.get(method, method)
                    cols = choose(rule, X, y, "classification", seed, count)
                key = f"{budget}_{method}"
                signature = tuple(cols.tolist())
                if signature not in cache:
                    design = (X, Xt) if method == "full" else (X[:, cols], Xt[:, cols])
                    pred, _ = train("ridge", design[0], y, design[1], "classification", seed)
                    cache[signature] = pred
                saved[f"s{seed}_{key}_support"], saved[f"s{seed}_{key}_pred"] = cols, cache[signature]
                rows.extend(results(truth, cache[signature], people, dict(
                    seed=seed, key=key, method=method, budget=budget, width=len(cols))))
        print(f"Finished UCI HAR seed {seed}", flush=True)
    file = out / "supports-and-predictions.npz"
    np.savez_compressed(file, **saved)
    audit(file, rows)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "subject_scores.csv", index=False)
    summary = metrics[metrics.subject != 0].groupby(["method", "budget"], as_index=False).agg(
        mean_subject_ba=("balanced_accuracy", "mean"), subject_seed_pairs=("balanced_accuracy", "size"))
    summary.to_csv(out / "summary.csv", index=False)
    for path in paths:
        if sha256(path) != hashes[path.relative_to(ROOT).as_posix()]:
            raise RuntimeError(f"Source changed during evaluation: {path}")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
