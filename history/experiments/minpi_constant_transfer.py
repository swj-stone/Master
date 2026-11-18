"""Independent transfer with native sparse text, dense URL indicators and controlled constants."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.exchange_vote_benchmark import constant_columns
from experiments.fast_minpi_five_source import summarize_timings
from experiments.minpi_study import P, ROOT, file_hash, train
from experiments.stable_minpi_stress import digest, metric
from minppi_mofs.exchange_vote_minpi import exchange_vote_top_from_training
from minppi_mofs.stable_minpi import _impute_validation


FILES = {"farm-ads": "be279ec71b033b40e49e6b87873f1a94acdb023badca478cc186a113570080d8",
         "internet-advertisements": "ec24f1b9285079146dad6af5edb1f9c1bde3811576cc53368e60708d2165e47c"}
SEEDS = (50131, 50147, 50153)
METHODS = ("exchange", "random", "F", "constant", "full")


def observed_data(name):
    archive = ROOT/"data/raw"/f"{name}.zip"
    if file_hash(archive).lower() != FILES[name]:
        raise ValueError(f"Incorrect {name} archive checksum")
    with ZipFile(archive) as z:
        if name == "farm-ads":
            with z.open("farm-ads-vect") as stream:
                X, y = load_svmlight_file(stream, n_features=54877, zero_based=False, dtype=np.float32)
            X.sum_duplicates()
            X.eliminate_zeros()
            X.sort_indices()
            def key(i):
                a, b = X.indptr[i:i+2]
                return X.indices[a:b].tobytes() + X.data[a:b].tobytes()
        else:
            with z.open("ad.data") as stream:
                frame = pd.read_csv(stream, header=None, skipinitialspace=True, na_values=["?"], low_memory=False)
            y = frame.pop(frame.columns[-1]).str.strip().map({"ad.": 1, "nonad.": -1}).to_numpy()
            if not np.isin(y, [-1, 1]).all():
                raise ValueError("Unrecognized Internet Advertisements label")
            X = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
            def key(i):
                return X[i].tobytes()
    # Use the first record for each exact feature vector, independent of its label.
    seen, unique = set(), []
    for i in range(len(y)):
        fingerprint = hashlib.sha256(key(i)).digest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(i)
    return X[unique], y[unique], np.asarray(unique)


def panels():
    for name in FILES:
        X, y, original = observed_data(name)
        splits = StratifiedKFold(3, shuffle=True, random_state=429710).split(np.empty(len(y)), y)
        for fold, ((train_pool, test_pool), seed) in enumerate(zip(splits, SEEDS, strict=True), 1):
            large, _ = train_test_split(train_pool, train_size=768, stratify=y[train_pool], random_state=seed)
            small, _ = train_test_split(large, train_size=256, stratify=y[large], random_state=seed)
            test, _ = train_test_split(test_pool, train_size=256, stratify=y[test_pool], random_state=seed)
            for fit in (small, large):
                yield (f"{name}-{len(fit)}", fold, seed, X[fit], y[fit], X[test], y[test],
                       "classification", {"train_rows": original[fit], "test_rows": original[test]})
    for task in ("regression", "classification"):
        for k in (0, 32, 64, 96):
            for fold, seed in enumerate(SEEDS[:2], 1):
                rng = np.random.default_rng([767393, seed, int(task == "classification")])
                X = rng.normal(size=(2400, 128)).astype(np.float32)
                y = X[:, :8].sum(axis=1)/np.sqrt(8) + .35*rng.normal(size=len(X))
                order = rng.permutation(128)
                # The variable columns carry the same signal in all four regimes.
                # Deliberately let some formerly constant measurements vary at test time.
                if k:
                    X[:1600, -k:] = rng.integers(-3, 4, size=k)
                X = np.ascontiguousarray(X[:, order])
                if task == "classification":
                    y = (y > 0).astype(int)
                yield (f"generated-{task}-{k}", fold, seed, X[:1600], y[:1600], X[1600:], y[1600:],
                       task, {"constant_train_features": np.array(k)})


def select(method, X, y, task, seed, width, impute):
    if method == "exchange":
        return exchange_vote_top_from_training(X, y, task=task, random_state=seed)
    if method == "constant":
        return constant_columns(X), {}
    if method == "zero_scan":
        # Specialized one-pass baseline for the canonical, explicit-zero-free Farm Ads CSR.
        used = np.flatnonzero(np.bincount(X.indices, minlength=X.shape[1]))
        return used if len(used) else np.array([0]), {}
    if method == "F" and impute:
        X, _ = _impute_validation(X, X[:0])
    return choose(method.lower(), X, y, task, seed, width), {}


def fit_predict(X, y, Xt, cols, task, seed, impute):
    A, B = (X, Xt) if len(cols) == X.shape[1] else (X[:, cols], Xt[:, cols])
    if sparse.issparse(X):
        model = make_pipeline(StandardScaler(with_mean=False),
                              RidgeClassifier(alpha=P["linear_alpha"], solver="lsqr",
                                              tol=P["ridge_tol"], max_iter=P["ridge_max_iter"]))
        return model.fit(A, y).predict(B)
    if impute:
        A, B = _impute_validation(A, B)
    return train("ridge", A, y, B, task, seed)[0]


def run(out, verify):
    files = (Path(__file__), ROOT/"src/minppi_mofs/exchange_vote_minpi.py",
             ROOT/"experiments/exchange_vote_benchmark.py", ROOT/"experiments/minpi_study.py",
             ROOT/"src/minppi_mofs/tournament_minpi.py", ROOT/"configs/minpi-study.json",
             ROOT/"src/minppi_mofs/stable_minpi.py", ROOT/"src/minppi_mofs/sketch_minpi.py",
             ROOT/"src/minppi_mofs/utility_minpi.py", ROOT/"experiments/compare_predictive_minpi.py",
             ROOT/"experiments/fast_minpi_five_source.py", ROOT/"experiments/stable_minpi_stress.py")
    spec = dict(sources={f.relative_to(ROOT).as_posix(): file_hash(f) for f in files}, data=FILES,
                seeds=list(SEEDS), repeats=5, methods=list(METHODS), csr_extra="zero_scan", threads=1,
                python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                scipy=scipy.__version__, platform=platform.platform(),
                eligibility="p>=16 and at least 25% training columns finite constant (CSR all-zero)",
                interpretation="Two UCI real sources at nested labeled training sizes, plus controlled "
                               "classification and regression. Count each configuration once.")
    manifest = out/"protocol.json"
    if verify:
        assert spec == json.loads(manifest.read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(spec, indent=2), encoding="utf8")
    records, audit, metadata = [], [], []
    for name, fold, seed, X, y, Xt, truth, task, split in panels():
        path = out/f"{name}-{fold}.npz"
        p = X.shape[1]
        methods = METHODS + ("zero_scan",) if sparse.issparse(X) else METHODS
        impute = name.startswith("internet-advertisements")
        traces, scores, width = {**split, "truth": truth}, {}, p
        for method in methods:
            cols, info = select(method, X, y, task, seed, width, impute)
            if method == "exchange":
                width = len(cols)
                branch = info["branch"]
                certified = len(info.get("constant_columns", []))
            if method in ("exchange", "F", "random"):
                assert len(cols) == width
            pred = fit_predict(X, y, Xt, cols, task, seed, impute)
            traces[method+"_cols"], traces[method+"_pred"] = cols, pred
            scores[method] = metric(truth, pred, task)
        metadata.append(dict(dataset=name, fold=fold, seed=seed, p=p, n_train=len(y), task=task,
                             layout="CSR" if sparse.issparse(X) else "dense", certified=certified,
                             branch=branch, varying=len(traces["constant_cols"]),
                             same_constant_support=np.array_equal(traces["exchange_cols"], traces["constant_cols"]),
                             exchange_full_equal=np.array_equal(traces["exchange_pred"], traces["full_pred"]),
                             max_prediction_change=np.abs(traces["exchange_pred"] - traces["full_pred"]).max(),
                             score_difference=scores["exchange"]-scores["full"]))
        if verify:
            with np.load(path) as saved:
                for key in traces:
                    np.testing.assert_array_equal(traces[key], saved[key])
            audit.extend(dict(dataset=name, fold=fold, seed=seed, method=m,
                              support_sha256=digest(traces[m+"_cols"]), prediction_sha256=digest(traces[m+"_pred"]))
                         for m in methods)
        else:
            order = np.random.default_rng([seed, fold, len(y)]).permutation(methods)
            for repeat in range(5):
                for position, m in enumerate(np.roll(order, repeat)):
                    start = perf_counter_ns()
                    cols, _ = select(m, X, y, task, seed, width, impute)
                    middle = perf_counter_ns()
                    pred = fit_predict(X, y, Xt, cols, task, seed, impute)
                    stop = perf_counter_ns()
                    np.testing.assert_array_equal(cols, traces[m+"_cols"])
                    np.testing.assert_array_equal(pred, traces[m+"_pred"])
                    records.append(dict(dataset=name, fold=fold, seed=seed, repeat=repeat, position=position,
                                        method=m, width=len(cols), p=p, score=scores[m], select_ms=(middle-start)/1e6,
                                        refit_ms=(stop-middle)/1e6, total_ms=(stop-start)/1e6))
            np.savez_compressed(path, **traces)
        print(name, fold, "verified" if verify else "measured", flush=True)
    if verify:
        pd.DataFrame(audit).to_csv(out/"fresh-process-audit.csv", index=False)
    else:
        cases = summarize_timings(out, records)
        panels_frame = pd.DataFrame(metadata)
        panels_frame.to_csv(out/"panels.csv", index=False)
        cases = cases.merge(panels_frame, on=["dataset", "fold", "seed", "p"], validate="many_to_one")
        cases["family"] = np.where(cases.dataset.str.startswith("generated-"), "simulation", "UCI")
        cases.groupby(["family","layout","branch","method"]).agg(
            configs=("fold","size"), reduced=("reduced","sum"), within=("within","sum"),
            joint=("joint","sum"), joint_5pct=("joint_5pct","sum"), joint_10pct=("joint_10pct","sum"),
            total_ratio=("ratio","median"), select_ms=("select_ms","median")).to_csv(out/"by-branch.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
