"""Frozen transfer experiment on later gas batches and held-out material compositions."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, file_hash, select, train
from experiments.minpi_study_matched import reference_weights
from minppi_mofs.exact_minpi import pivotal_rates, witness_margins

DATA = ROOT / "data/revision-cohorts"
PROTOCOL = ROOT / "docs/minpi-study/new-cohorts-protocol.md"
DIGESTS = {
    "gas": "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4",
    "supercon": "87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e",
}
SEEDS = (1129, 2027, 3469)
LAYERS = (0, 4, 16)


def load_cohort(name):
    archive = DATA / f"{name}.zip"
    if file_hash(archive) != DIGESTS[name]:
        raise ValueError(f"Unexpected source archive for {name}")
    with ZipFile(archive) as z:
        if name == "gas":
            matrices, labels, blocks = [], [], []
            for batch in range(1, 11):
                with z.open(f"Dataset/batch{batch}.dat") as f:
                    matrix, y = load_svmlight_file(f, n_features=128, zero_based=False)
                matrices.append(matrix.toarray())
                labels.append(y.astype(int))
                blocks.append(np.full(len(y), batch))
            groups = np.concatenate(blocks)
            return (np.vstack(matrices), np.concatenate(labels), groups,
                    np.flatnonzero(groups <= 7), np.flatnonzero(groups > 7), "classification")
        with z.open("train.csv") as f:
            descriptors = pd.read_csv(f)
        with z.open("unique_m.csv") as f:
            materials = pd.read_csv(f)
    y = descriptors.pop("critical_temp").to_numpy()
    np.testing.assert_array_equal(y, materials.pop("critical_temp"))
    elements = materials.drop(columns="material").to_numpy()
    proportions = np.round(elements / elements.sum(axis=1, keepdims=True), 8)
    X = descriptors.to_numpy()
    _, first = np.unique(X, axis=0, return_inverse=True)
    _, second = np.unique(proportions, axis=0, return_inverse=True)
    second = second + first.max() + 1
    graph = coo_matrix((np.ones(len(first)), (first, second)),
                       shape=(second.max() + 1, second.max() + 1)).tocsr()
    _, labels = connected_components(graph, directed=False)
    groups = labels[first]
    train_rows, test_rows = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=61429)
                                 .split(X, y, groups))
    assert not np.intersect1d(groups[train_rows], groups[test_rows]).size
    return X, y, groups, train_rows, test_rows, "regression"


def batch_scores(y, pred, groups):
    scores = {}
    for group in np.unique(groups):
        rows = groups == group
        gold, estimated = y[rows], pred[rows]
        recall = [np.mean(estimated[gold == cls] == cls) for cls in np.unique(gold)]
        scores[int(group)] = float(np.mean(recall))
    return scores


def bootstrap_difference(y, pred_a, pred_b, groups):
    _, codes = np.unique(groups, return_inverse=True)
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=y)
    squares = np.bincount(codes, weights=y**2)
    error_difference = np.bincount(codes, weights=(y - pred_b)**2 - (y - pred_a)**2)
    rng = np.random.default_rng(18617)
    draws = []
    for _ in range(2000):
        g = rng.integers(len(counts), size=len(counts))
        sst = squares[g].sum() - sums[g].sum()**2 / counts[g].sum()
        draws.append(error_difference[g].sum() / sst)
    return np.quantile(draws, [.025, .975]).tolist()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    records, comparisons = [], []
    manifest = {"archive_sha256": DIGESTS, "protocol_sha256": file_hash(PROTOCOL),
                "driver_sha256": file_hash(Path(__file__)), "cohorts": {}}
    for name in DIGESTS:
        X, y, groups, tr, te, task = load_cohort(name)
        h = ceil(X.shape[1] * .1)
        manifest["cohorts"][name] = dict(p=X.shape[1], h=h, train=len(tr), test=len(te),
            train_groups=int(np.unique(groups[tr]).size), test_groups=int(np.unique(groups[te]).size))
        np.savez_compressed(output / f"{name}-split.npz", train=tr, test=te, groups=groups, y=y)
        xt, yt = X[tr], y[tr]
        for seed in SEEDS:
            start = perf_counter()
            margins = witness_margins(xt, yt, task, seed)
            v = np.sign(margins).astype(np.int8)
            witness_s = perf_counter() - start
            start = perf_counter()
            rates = pivotal_rates(v, LAYERS)
            score_s = perf_counter() - start
            vectors = {"distance": margins.mean(0), "sign": v.mean(0), "largest": rates[-1],
                       "min": rates.min(0), "geometry": reference_weights(xt, yt, task, seed)}
            supports = {m: np.sort(np.argsort(-s, kind="stable")[:h]) for m, s in vectors.items()}
            for method in ("f", "random", "full"):
                supports[method] = select(method, xt, yt, task, seed)[0]
            predictions = {}
            times = {}
            for method, cols in supports.items():
                start = perf_counter()
                pred, _ = train("ridge", xt[:, cols], yt, X[te][:, cols], task, seed)
                times[method] = perf_counter() - start
                predictions[method] = pred
            held = np.sign(witness_margins(X[te], y[te], task, seed + 10000)).astype(np.int8)
            held_min = pivotal_rates(held, LAYERS).min(0)
            scores = {}
            for method, pred in predictions.items():
                record = dict(source=name, seed=seed, method=method, width=len(supports[method]),
                              mean_held_min=float(held_min[supports[method]].mean()),
                              downstream_s=times[method], shared_witness_s=witness_s, shared_layers_s=score_s)
                if task == "classification":
                    batch = batch_scores(y[te], pred, groups[te])
                    score = np.mean(list(batch.values()))
                    record.update({f"batch_{g}": val for g, val in batch.items()})
                else:
                    score = r2_score(y[te], pred)
                    record["mae"] = mean_absolute_error(y[te], pred)
                scores[method] = score
                records.append(dict(**record, quality=score))
            for a, b in (("largest", "sign"), ("min", "largest"), ("min", "f")):
                row = dict(source=name, seed=seed, contrast=f"{a}-{b}",
                           delta=scores[a]-scores[b], swapped=len(set(supports[a])-set(supports[b])),
                           delta_held_min=held_min[supports[a]].mean()-held_min[supports[b]].mean())
                if task == "regression" and seed == SEEDS[0]:
                    row["lower"], row["upper"] = bootstrap_difference(y[te], predictions[a], predictions[b], groups[te])
                comparisons.append(row)
            payload = {f"columns_{m}": cols for m, cols in supports.items()}
            payload.update({f"prediction_{m}": pred for m, pred in predictions.items()})
            np.savez_compressed(output / f"{name}-{seed}.npz", **payload,
                                train_votes=v, train_layers=rates, held_votes=held, held_min=held_min)
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/minpi-study/new-cohorts")
    with threadpool_limits(1):
        run(parser.parse_args().output)
