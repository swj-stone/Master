"""Descriptive paired test-row bootstrap, conditional on the four fitted seed-42 pipelines."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.test.run_piv3_lite_six import DATASETS


ARMS = ("pi", "pi_pfi")
COMPARATORS = ("full", "pfi")
REPEATS = 399
SEED = 42


def _rank_metric(y, score, label, include_ap):
    # Collapse equal scores before building the ROC or computing precision at a threshold.
    values, groups = np.unique(score, return_inverse=True)
    positive = y == label
    n_groups = len(values)

    def weighted(weights):
        pos = np.bincount(groups[positive], weights=weights[positive], minlength=n_groups)
        neg = np.bincount(groups[~positive], weights=weights[~positive], minlength=n_groups)
        n_pos, n_neg = pos.sum(), neg.sum()
        auc = (pos @ (np.cumsum(neg) - .5*neg)) / (n_pos*n_neg)
        if not include_ap:
            return auc, None
        total_above = np.cumsum((pos+neg)[::-1])
        precision = np.divide(np.cumsum(pos[::-1]), total_above, out=np.zeros_like(total_above),
                              where=total_above > 0)
        return auc, float(pos[::-1] @ precision / n_pos)

    return weighted


def _metric_function(y, pred, dataset):
    if dataset == "abalone":
        error = y-pred
        sse, sae = error**2, abs(error)
        def weighted(weights):
            n = weights.sum()
            avg_y = weights @ y / n
            total = weights @ ((y-avg_y)**2)
            residual = weights @ sse
            r2 = max(0., 1-residual/total) if total else float(not residual)
            return dict(r2=r2, mae=weights @ sae / n)
        return weighted

    include_ap = dataset == "credit_card_fraud"
    correct = pred.argmax(axis=1) == y
    labels = range(pred.shape[1]) if pred.shape[1] > 2 else [1]
    scorers = [_rank_metric(y, pred[:, label], label, include_ap) for label in labels]
    def weighted(weights):
        ranks = [scorer(weights) for scorer in scorers]
        result = dict(auc=np.mean([auc for auc, _ in ranks]), accuracy=weights @ correct / weights.sum())
        if include_ap:
            result["ap"] = ranks[0][1]
        return result
    return weighted


def _relative_loss(value, comparator, metric):
    loss = value-comparator if metric == "mae" else comparator-value
    return loss/abs(comparator) if comparator else (0. if loss <= 0 else float("inf"))


def _quantile(v):
    # Empirical upper order statistic also supports a zero reference with positive loss (+inf).
    return float(np.quantile(v, .95, method="inverted_cdf"))


def _paired_scores(dataset, y, predictions, n_boot, rng):
    # Identical predictions must receive identical weights and yield exactly equal scores.
    unique = {}
    aliases = {}
    for name, pred in predictions.items():
        old = next((key for key, arr in unique.items() if np.array_equal(pred, arr)), None)
        if old is None:
            unique[name] = pred
        aliases[name] = old or name
    scorers = {name:_metric_function(y, p, dataset) for name, p in unique.items()}
    ones = np.ones(len(y))
    points = {name:scorer(ones) for name, scorer in scorers.items()}
    bootstrap = {name:{metric:np.empty(n_boot) for metric in scores} for name,scores in points.items()}
    strata = ([np.arange(len(y))] if dataset == "abalone" else
              [np.flatnonzero(y == label) for label in np.unique(y)])
    if len(scorers) == 1:
        # Every contrast is identically zero, independently of row resampling.
        for name, scores in points.items():
            for metric, val in scores.items():
                bootstrap[name][metric].fill(val)
    for rep in range(n_boot if len(scorers) > 1 else 0):
        weights = np.zeros(len(y), dtype=float)
        for rows in strata:
            sampled = rng.integers(len(rows), size=len(rows))
            weights[rows] = np.bincount(sampled, minlength=len(rows))
        for name, scorer in scorers.items():
            for metric, value in scorer(weights).items():
                bootstrap[name][metric][rep] = value
    return {name:points[src] for name,src in aliases.items()}, {name:bootstrap[src] for name,src in aliases.items()}


def _csv(path, rows):
    temp = path.with_suffix(".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def report(run, *, n_boot=REPEATS, seed=SEED, reference_run=None):
    if n_boot < 1:
        raise ValueError("At least one bootstrap replicate required")
    run = Path(run)
    reference_run = run if reference_run is None else Path(reference_run)
    per_metric, joint = [], []
    scope = "paired test-row bootstrap, conditional on fixed fitted models; descriptive"
    for i, ds in enumerate(DATASETS):
        labels, raw, declared = None, {}, {}
        for arm in (*ARMS, *COMPARATORS):
            folder = (reference_run if arm in COMPARATORS else run)/ds/arm
            timer = json.loads((folder/"watchdog.json").read_bytes())
            if timer["status"] != "completed":
                continue
            declared[arm] = json.loads((folder/"result.json").read_bytes())["test_scores"]
            with np.load(folder/"predictions.npz") as archive:
                y, pred = archive["y_test"], archive["test_pred"]
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise ValueError(f"{ds}/{arm}: test row labels differ")
            if not np.isfinite(pred).all():
                raise ValueError(f"{ds}/{arm}: nonfinite test prediction")
            raw[arm] = pred
        with threadpool_limits(limits=1):
            points, samples = ({}, {}) if labels is None else _paired_scores(
                ds, labels, raw, n_boot, np.random.default_rng([seed, i]))
        for arm, scores in points.items():
            for key,val in scores.items():
                if not np.isclose(val, declared[arm][key], atol=1e-10, rtol=0):
                    raise ValueError(f"{ds}/{arm}: weighted calculation disagrees with archived {key}")
        for challenger in ARMS:
            meta = dict(dataset=ds, challenger=challenger, replicates=n_boot, seed=seed, scope=scope)
            all_point, all_rep = [], []
            for baseline in COMPARATORS:
                pair = dict(**meta, comparator=baseline)
                if challenger not in raw or baseline not in raw:
                    per_metric.append(dict(**pair, status="incomplete"))
                    continue
                for metric, value in points[challenger].items():
                    point = _relative_loss(value, points[baseline][metric], metric)
                    reps = np.fromiter((_relative_loss(a,b,metric) for a,b in zip(
                        samples[challenger][metric], samples[baseline][metric], strict=True)), dtype=float, count=n_boot)
                    upper = _quantile(reps)
                    per_metric.append(dict(**pair, status="completed", metric=metric, candidate_score=value,
                                      comparator_score=points[baseline][metric], relative_loss=point,
                                      upper95_relative_loss=upper))
                    all_point.append(point)
                    all_rep.append(reps)
            row = dict(**meta, comparator="full+pfi", status="incomplete")
            if all(arm in raw for arm in (challenger, *COMPARATORS)):
                upper = _quantile(np.max(all_rep, axis=0))
                row.update(status="completed", worst_point_relative_loss=max(all_point),
                           upper95_worst_relative_loss=upper, upper_within_5pct=upper <= .05 + 1e-12)
            joint.append(row)
    _csv(run/"bootstrap.csv", per_metric)
    _csv(run/"joint.csv", joint)
    return joint


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--reference-run", type=Path, help="Run holding matched full and direct PFI predictions")
    p.add_argument("--replicates", type=int, default=REPEATS)
    args = p.parse_args()
    for row in report(args.run, n_boot=args.replicates, reference_run=args.reference_run):
        print(row["dataset"], row["challenger"], row.get("upper95_worst_relative_loss", row["status"]))
