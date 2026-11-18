"""Evaluate frozen selectors on the published Arcene and Dexter validation partitions."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import cdist
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from minppi_mofs.exact_minpi import ExactMinPI


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1129, 2039, 3049)
METHODS = ("exact", "f", "relieff", "l1", "random", "full")
FILES = {
    "arcene": (10000, "c7a40e39edf23a9b744eb570592d6e917fd9a1976aaf20ad581e09afd47d5220"),
    "dexter": (20000, "70400df292178d8d3627411a5631a4cb13fb6b19da4a200b9a206f25255cddcc"),
}


def load_challenge(name):
    width, digest = FILES[name]
    path = ROOT / "data" / "raw" / "nips2003" / (name + ".zip")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected source archive {path}")
    splits = []
    with ZipFile(path) as archive:
        for partition in ("train", "valid"):
            prefix = f"{name.upper()}/{name}_{partition}"
            target = prefix + ".labels" if partition == "train" else f"{name}_valid.labels"
            with archive.open(target) as stream:
                y = np.loadtxt(stream, dtype=int)
            with archive.open(prefix + ".data") as stream:
                if name == "arcene":
                    X = np.loadtxt(stream, dtype=np.float32)
                else:
                    # Dexter stores one-based column:value pairs and implicit zeroes.
                    X = np.zeros((len(y), width), dtype=np.float32)
                    rows = 0
                    for i, line in enumerate(stream):
                        pairs = np.fromstring(line.replace(b":", b" "), sep=" ", dtype=int)
                        indices = pairs[::2] - 1
                        if np.any(indices < 0) or np.any(indices >= width):
                            raise ValueError("Dexter index outside declared dimensions")
                        X[i, indices] = pairs[1::2]
                        rows += 1
                    if rows != len(y):
                        raise ValueError("Dexter row count differs from labels")
            if X.shape != (len(y), width) or set(y) != {-1, 1}:
                raise ValueError("Unexpected archive dimensions or labels")
            splits.append((X, y))
    return splits


def relief_binary(X, y):
    ranges = np.ptp(X, axis=0)
    scaled = X / np.where(ranges > 0, ranges, 1)
    distances = cdist(scaled, scaled, metric="cityblock")
    np.fill_diagonal(distances, np.inf)
    scores = np.zeros(X.shape[1], dtype=float)
    for i in range(len(y)):
        for mask, sign in ((y != y[i], 1), (y == y[i], -1)):
            neighbors = np.flatnonzero(mask)
            order = np.argsort(distances[i, neighbors], kind="stable")[:10]
            scores += sign * np.abs(scaled[neighbors[order]] - scaled[i]).mean(axis=0)
    return scores / len(y)


def select_features(method, X, y, seed):
    width = X.shape[1]
    k = math.ceil(0.1 * width)
    if method == "full":
        return np.arange(width)
    if method == "random":
        return np.sort(np.random.default_rng(seed).permutation(width)[:k])
    if method == "exact":
        ranker = ExactMinPI(task="classification", random_state=seed, fraction=0.1).fit(X, y)
        return np.sort(ranker.get_support(indices=True))
    if method == "relieff":
        scores = relief_binary(X, y)
    elif method == "l1":
        scaled = StandardScaler(with_mean=False).fit_transform(X)
        estimator = LogisticRegression(
            C=1, l1_ratio=1, solver="liblinear", max_iter=5000, random_state=seed
        ).fit(scaled, y)
        scores = np.abs(estimator.coef_[0])
    elif method == "f":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.filterwarnings("ignore", message="Features.*constant", category=UserWarning)
            scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    else:
        raise ValueError(method)
    return np.sort(np.argsort(-scores, kind="stable")[:k])


def evaluation(name, out):
    (train, labels), (held, truth) = load_challenge(name)
    permutation = np.random.default_rng(391).permutation(train.shape[1])
    train = np.ascontiguousarray(train[:, permutation])
    held = np.ascontiguousarray(held[:, permutation])
    rows = []
    traces = {"labels": truth, "train_labels": labels, "permutation": permutation}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(METHODS)
        for method in order:
            start = time.perf_counter()
            columns = select_features(method, train, labels, seed)
            selected = time.perf_counter()
            expected = train.shape[1] if method == "full" else math.ceil(0.1 * train.shape[1])
            if (len(columns) != expected or len(set(columns)) != expected
                    or min(columns) < 0 or max(columns) >= train.shape[1]):
                raise ValueError(f"Invalid selection from {method}")
            scaler = StandardScaler(with_mean=False)
            scaled = scaler.fit_transform(train[:, columns])
            estimator = LogisticRegression(
                C=1, l1_ratio=0, dual=True, solver="liblinear", max_iter=5000, random_state=seed
            ).fit(scaled, labels)
            prediction = estimator.decision_function(scaler.transform(held[:, columns]))
            end = time.perf_counter()
            suffix = f"{method}_{seed}"
            traces[f"decision_{suffix}"] = prediction
            traces[f"columns_{suffix}"] = np.sort(permutation[columns])
            rows.append(dict(dataset=name, seed=seed, method=method, n_train=len(train),
                             n_valid=len(held), p=train.shape[1], k=len(columns),
                             auc=roc_auc_score(truth, prediction),
                             balanced_accuracy=balanced_accuracy_score(truth, np.where(prediction > 0, 1, -1)),
                             selection_s=selected-start, end_to_end_s=end-start))
            print(name, seed, method, f"{end-start:.3f}s", flush=True)
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / f"{name}-trace.npz", **traces)


def paired_interval(trace):
    # The Mann--Whitney definition allows all bootstrap replicates to share pair comparisons.
    y = trace["labels"]
    diffs = []
    for seed in SEEDS:
        areas = []
        for method in ("exact", "f"):
            scores = trace[f"decision_{method}_{seed}"]
            delta = scores[y == 1, None] - scores[None, y == -1]
            areas.append((delta > 0).astype(float) + 0.5 * (delta == 0))
        diffs.append(areas[0] - areas[1])
    matrix = np.mean(diffs, axis=0)
    rng = np.random.default_rng(2718)
    npos, nneg = matrix.shape
    pos = rng.multinomial(npos, np.full(npos, 1/npos), 2000)
    neg = rng.multinomial(nneg, np.full(nneg, 1/nneg), 2000)
    samples = ((pos @ matrix) * neg).sum(axis=1) / (npos*nneg)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(contrast="exact_minus_f_auc", difference=matrix.mean(), lower=lo, upper=hi,
                draws=len(samples), seed=2718, conditioning="fixed fits and official split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    contrasts = {}
    with threadpool_limits(limits=1):
        for name in FILES:
            evaluation(name, out)
            with np.load(out / f"{name}-trace.npz") as trace:
                contrasts[name] = paired_interval(trace)
    info = dict(datasets=FILES, seeds=SEEDS, cpu=os.environ.get("PROCESSOR_IDENTIFIER"),
                platform=platform.platform(), sklearn=sklearn.__version__, scipy=scipy.__version__,
                numpy=np.__version__, threads=1, contrasts=contrasts)
    (out / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
