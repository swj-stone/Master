"""Run the fixed linear-model transfer study in protocol.json via thesis-python.ps1."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_witness import _sample, _within


ROOT = Path(__file__).resolve().parent


def fit(x, y, task):
    learner = Ridge(alpha=1.) if task == "regression" else LogisticRegression(C=1., max_iter=1000)
    return make_pipeline(StandardScaler(), learner).fit(x, y)


def predict(model, x, task):
    return model.predict(x) if task == "regression" else model.predict_proba(x)


def metrics(y, p, task):
    if task == "regression":
        return {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p)}
    auc = roc_auc_score(y, p[:, 1]) if task == "binary" else roc_auc_score(y, p, multi_class="ovr")
    return {"auc": auc, "accuracy": accuracy_score(y, p.argmax(axis=1))}


def run_split(name, seed, factorial=False):
    bundle = getattr(datasets, "load_" + name)(**({"scaled": False} if name == "diabetes" else {}))
    x, y = bundle.data, bundle.target
    task = {"breast_cancer": "binary", "wine": "multiclass", "diabetes": "regression"}[name]
    strat = y if task != "regression" else None
    train_rank, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed,
                                        stratify=strat)
    train, rank = train_test_split(train_rank, test_size=.25, random_state=seed + 1,
                                   stratify=None if strat is None else y[train_rank])
    model = fit(x[train], y[train], task)
    coefs = np.atleast_2d(model[-1].coef_)
    order = np.argsort(-np.mean(coefs**2, axis=0), kind="stable").tolist()
    fill = np.median(x[train], axis=0)
    def masked(rows, subset):
        xx = np.broadcast_to(fill, (len(rows), len(order))).copy()
        cols = sorted(subset)
        xx[:, cols] = x[np.ix_(rows, cols)]
        return predict(model, xx, task)
    def anchor_score(cols):
        result = metrics(y[rank], masked(rank, cols), task)
        if task == "regression":
            result["r2"] = max(0., result["r2"])
        return result
    reference = anchor_score(order)
    for k in range(1, len(order) + 1):
        scores = anchor_score(order[:k])
        if all((scores[m] - v if m == "mae" else v - scores[m]) <= .08 * abs(v) + 1e-12
               for m, v in reference.items()):
            break
    anchor = frozenset(order[:k])
    tails = ((), order[k:2*k], order[k:])
    idx, weights = _sample(y[rank], task, np.random.SeedSequence([seed, 659]))
    utility = {"regression": "r2", "binary": "roc_auc", "multiclass": "ovr_auc"}[task]
    scorer = make_scorer(y[rank][idx], weights, task, utility, False)
    frozen = {}
    rules = ["empty_majority", "all_point"] if factorial else ["min", "mean", "point"]
    for rule in rules:
        lookup = {}
        def score(mask):
            if mask not in lookup:
                lookup[mask] = scorer(masked(rank[idx], mask))
            return lookup[mask]
        def ballot(s, e):
            votes = tuple(_within(a, b) for a, b in zip(score(s | e), score(anchor | e), strict=True))
            return (votes[0],) * 4 if rule == "all_point" else votes
        extensions = tails[:1] if rule == "empty_majority" else tails
        selected, trace = search(order[:k], extensions, ballot, rule="min" if factorial else rule)
        frozen[rule] = (sorted(selected), len(lookup), len(trace["trace"]))
    if not factorial:
        frozen.update(anchor=(sorted(anchor), 0, 0), full=(sorted(order), 0, 0),
                      matched_prefix=(sorted(order[:len(frozen["min"][0])]), 0, 0))
    split_hash = hashlib.sha256(b"".join(np.asarray(a, dtype="<i8").tobytes()
                                          for a in [train, rank, test])).hexdigest()
    # Every arm freezes its support before any evaluation on test observations.
    rows = []
    for arm, (cols, calls, proposals) in frozen.items():
        estimator = fit(x[np.ix_(train_rank, cols)], y[train_rank], task)
        test_scores = metrics(y[test], predict(estimator, x[np.ix_(test, cols)], task), task)
        rows.append(dict(dataset=name, seed=seed, task=task, arm=arm, width=len(order),
                         anchor_k=k, retained=len(cols), witness_calls=calls, proposals=proposals,
                         selected=json.dumps(cols), partition_sha256=split_hash, **test_scores))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--factorial", action="store_true", help="Run the two missing panel-by-context ablations")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    results = []
    with threadpool_limits(limits=1):
        for dataset in protocol["datasets"]:
            for seed in protocol["seeds"]:
                results.extend(run_split(dataset, seed, args.factorial))
    arms = 2 if args.factorial else len(protocol["methods"])
    if len(results) != len(protocol["datasets"])*len(protocol["seeds"])*arms:
        raise RuntimeError("Incomplete transfer grid")
    fields = list(dict.fromkeys(key for row in results for key in row))
    output = args.output or ROOT / ("factorial.csv" if args.factorial else "per-seed.csv")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} evaluations to {output}")


if __name__ == "__main__":
    main()
