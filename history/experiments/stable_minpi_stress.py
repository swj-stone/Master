"""Frozen five-source stress test for adaptive and fixed-budget minPI."""
import argparse
import hashlib
import json
import platform
import warnings
from itertools import combinations
from math import ceil, floor
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.minpi_held_permutation_control import linear_permutation_losses, ridge
from experiments.minpi_study import CONFIG, ROOT, file_hash, load_dataset, train
from minppi_mofs.environment_minpi import InsufficientEnvironmentsError, environment_top_from_training
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training
from minppi_mofs.stable_minpi import StableMinPISelector, _calibrate_supports, _partition


DATASETS = ("spambase", "digits", "madelon", "abalone", "crimedata")
SPLIT_SEED = 60217
PRIMARY_SEED = 1129
OTHER_SEEDS = (2027, 3469)
FRACTIONS = (.10, .50, .75)
TOLERANCE = .02
METHODS = ("stable", "environment_half", "local_half", "full", "F", "PFI", "random", "F_policy")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inputs(name):
    groups = None
    if name == "abalone":
        config = yaml.safe_load((ROOT / "configs/abalone.yaml").read_bytes())["dataset"]
        path = ROOT / config["raw_path"]
        source_hash = file_hash(path)
        if source_hash != config["raw_sha256"]:
            raise ValueError("Abalone checksum differs from the study configuration")
        df = pd.read_csv(path)
        y = df.pop("Rings").to_numpy(dtype=float)
        sex = df.pop("Sex").to_numpy()
        if not np.isin(sex, (0, 1, 2)).all():
            raise ValueError("Unexpected abalone category encoding")
        X = np.column_stack((sex[:, None] == [0, 1, 2], df.to_numpy())).astype(np.float32)
        task = "regression"
    else:
        X, y, task, names, _, _, source_hash = load_dataset(name)
        if name == "crimedata":
            frame = pd.read_csv(ROOT / "data/raw/crimedata.csv")
            if set(names) != set(frame.loc[:, "population":"LemasPctOfficDrugUn"].columns):
                raise AssertionError("Unexpected community predictors")
            groups = frame["state_encoded"].to_numpy()
            if not np.array_equal(frame.nonViolPerPop.to_numpy(), y) or not np.isfinite(groups).all():
                raise AssertionError("State grouping rows do not match regression targets")
    _, duplicate_ids = np.unique(X, axis=0, return_inverse=True)
    if groups is None and duplicate_ids.max() + 1 < len(X):
        groups = duplicate_ids
    return X, y, task, groups, source_hash, duplicate_ids


def folds(X, y, task, groups):
    if groups is None:
        cls = StratifiedKFold if task == "classification" else KFold
    else:
        cls = StratifiedGroupKFold if task == "classification" else GroupKFold
    return cls(3, shuffle=True, random_state=SPLIT_SEED).split(X, y, groups=groups)


def run_stable(X, y, task, seed, groups):
    selector = StableMinPISelector(task=task, random_state=seed, tolerance=TOLERANCE, fractions=FRACTIONS)
    selector.fit(X, y, groups=groups)
    return selector.get_support(indices=True), selector.diagnostics_


def f_policy(X, y, task, seed, groups):
    """The identical training/calibration protocol with an F ranking in discovery."""
    info = dict(discovery_rows=[], validation_rows=[], rule="full", reason="insufficient_environments")
    try:
        discovery, valid = _partition(y, groups, False, task, seed)
        info.update(discovery_rows=discovery.tolist(), validation_rows=[v.tolist() for v in valid])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            measure = f_classif if task == "classification" else f_regression
            scores = measure(X[discovery], y[discovery])[0]
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
        widths = {ceil(X.shape[1] * f) for f in FRACTIONS}
        supports = {width: np.sort(order[:width]) for width in widths}
        columns, decisions = _calibrate_supports(X, y, discovery, valid, supports, task, seed, TOLERANCE, "F")
        return columns, {**info, **decisions}
    except InsufficientEnvironmentsError as exc:
        return np.arange(X.shape[1]), {**info, "detail": str(exc)}


def permutation_order(X, y, task, seed, groups):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if groups is None:
        fit, check = train_test_split(indices, test_size=.25, random_state=seed,
                                     stratify=y if task == "classification" else None)
    else:
        fit, check = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(X, y, groups))
        assert not np.intersect1d(groups[fit], groups[check]).size
    model = ridge(task, 1.).fit(X[fit], y[fit])
    orders = [rng.permutation(len(check)) for _ in range(3)]
    weights = None
    if task == "classification":
        labels, codes, counts = np.unique(y[check], return_inverse=True, return_counts=True)
        weights = 1 / (len(labels) * counts[codes])
    losses = linear_permutation_losses(model, X[check], y[check], orders, task, weights)
    return np.argsort(-losses, kind="stable")


def choose_columns(method, X, y, task, seed, groups, matched_width):
    p = X.shape[1]
    if method == "stable":
        return run_stable(X, y, task, seed, groups)
    if method == "F_policy":
        return f_policy(X, y, task, seed, groups)
    if method == "environment_half":
        return environment_top_from_training(X, y, ceil(p / 2), task=task, random_state=seed, groups=groups)
    if method == "local_half":
        return local_refit_fast_top_from_training(X, y, ceil(p / 2), task=task, random_state=seed)
    if method == "full" or matched_width == p:
        return np.arange(p), {}
    if method == "PFI":
        return np.sort(permutation_order(X, y, task, seed, groups)[:matched_width]), {}
    return choose(method.lower(), X, y, task, seed, matched_width), {}


def prediction(X, y, Xt, columns, task, seed):
    if len(columns) == X.shape[1]:
        return train("ridge", X, y, Xt, task, seed)[0]
    return train("ridge", X[:, columns], y, Xt[:, columns], task, seed)[0]


def metric(y, pred, task):
    return (balanced_accuracy_score if task == "classification" else r2_score)(y, pred)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path.write_text(json.dumps(data, default=jsonable, indent=2), encoding="utf-8")


def manifest():
    modules = ("minppi_mofs/stable_minpi.py", "minppi_mofs/environment_minpi.py",
               "minppi_mofs/local_refit_fast.py", "minppi_mofs/local_refit_minpi.py",
               "minppi_mofs/masked_loss_minpi.py", "minppi_mofs/piv2.py", "minppi_mofs/utility_minpi.py",
               "minppi_mofs/exact_minpi.py", "minppi_mofs/restoration_minpi.py")
    files = [ROOT/"src"/module for module in modules]
    files += [Path(__file__), CONFIG, ROOT/"experiments/minpi_study.py",
              ROOT/"experiments/minpi_held_permutation_control.py", ROOT/"experiments/compare_predictive_minpi.py"]
    return dict(sources={str(f.relative_to(ROOT)): file_hash(f) for f in files},
                datasets=DATASETS, split_seed=SPLIT_SEED, primary_seed=PRIMARY_SEED, sensitivity_seeds=OTHER_SEEDS,
                fractions=FRACTIONS, tolerance=TOLERANCE, sklearn=sklearn.__version__, numpy=np.__version__,
                python=platform.python_version(), platform=platform.platform(), threads=1,
                protocol="Three fixed outer folds per dataset. Preserve identical predictor vectors within fold; communities grouped by state with all identifiers and crime outcomes excluded from predictors.",
                quality="Classification balanced accuracy, regression R2. Pass iff the individual outer-fold score is at least full score minus .02. Width-matched controls receive the width decided on training data only.",
                timing="Fresh selection including calibration plus extraction, final ridge fit and prediction. Stable timed first in repeat zero to obtain matched width. Methods rotate in later repeats. Three repeats per primary configuration; no timing-based choice.",
                reuse="Repeated ridge training and prediction on equivalent workloads with a frozen support. Strict break-even N=floor(selection_ms/(full_refit_ms-reduced_refit_ms))+1. Full fallback and nonpositive saving have no break-even.",
                reproducibility="Same input and primary seed must reproduce columns, predictions and policy decisions in each repeat and a separate Python process; two additional selector seeds measure sensitivity.")


def audit_inner(info, groups):
    discovery = np.asarray(info["discovery_rows"], dtype=int)
    verification = np.asarray([i for fold in info["validation_rows"] for i in fold], dtype=int)
    assert not np.intersect1d(discovery, verification).size
    if groups is not None:
        assert not np.intersect1d(groups[discovery], groups[verification]).size
    for held in info.get("environment_path", {}).get("folds", []):
        train_rows = discovery[held["fit_rows"]]
        held_rows = discovery[held["held_rows"]]
        assert not np.intersect1d(train_rows, held_rows).size
        if groups is not None:
            assert not np.intersect1d(groups[train_rows], groups[held_rows]).size


def benchmark_fold(output, name, fold, X, y, Xt, gold, task, groups, repeats):
    records, sensitivity, saved, decisions = [], [], {}, {}
    widths = {}
    order = np.random.default_rng([SPLIT_SEED, fold]).permutation(METHODS[1:]).tolist()
    for rep in range(repeats):
        schedule = np.roll(["stable", *order], rep * 3)
        for position, method in enumerate(schedule):
            start = perf_counter_ns()
            cols, info = choose_columns(method, X, y, task, PRIMARY_SEED, groups, widths.get("stable"))
            selected = perf_counter_ns()
            pred = prediction(X, y, Xt, cols, task, PRIMARY_SEED)
            end = perf_counter_ns()
            assert np.array_equal(np.unique(cols), cols) and len(cols) > 0
            if method in ("stable", "F_policy"):
                audit_inner(info, groups)
                if rep == 0:
                    decisions[method] = info
                elif json.dumps(info, default=jsonable, sort_keys=True) != json.dumps(decisions[method], default=jsonable, sort_keys=True):
                    raise AssertionError("Policy decisions changed on identical inputs")
            widths[method] = len(cols)
            for suffix, array in (("support", cols), ("pred", pred)):
                key = f"{method}_{suffix}"
                if rep:
                    np.testing.assert_array_equal(saved[key], array)
                else:
                    saved[key] = array
            records.append(dict(dataset=name, fold=fold, method=method, seed=PRIMARY_SEED,
                                repeat=rep, position=position, width=len(cols), p=X.shape[1],
                                rule=info.get("rule", method), score=metric(gold, pred, task),
                                select_ms=(selected-start)/1e6, refit_ms=(end-selected)/1e6,
                                total_ms=(end-start)/1e6))
    for method in ("stable", "environment_half"):
        supports = {PRIMARY_SEED: saved[f"{method}_support"]}
        scores = {PRIMARY_SEED: metric(gold, saved[f"{method}_pred"], task)}
        for seed in OTHER_SEEDS:
            cols, info = choose_columns(method, X, y, task, seed, groups, None)
            pred = prediction(X, y, Xt, cols, task, seed)
            saved[f"{method}_s{seed}_support"], saved[f"{method}_s{seed}_pred"] = cols, pred
            if method == "stable":
                audit_inner(info, groups)
                decisions[f"{method}_s{seed}"] = info
            supports[seed], scores[seed] = cols, metric(gold, pred, task)
        for a, b in combinations(supports, 2):
            ca, cb = supports[a], supports[b]
            sensitivity.append(dict(dataset=name, fold=fold, method=method, first_seed=a, second_seed=b,
                                    first_score=scores[a], second_score=scores[b], first_width=len(ca),
                                    second_width=len(cb), jaccard=len(np.intersect1d(ca,cb))/len(np.union1d(ca,cb))))
    np.savez_compressed(output/f"{name}-{fold}-outputs.npz", **saved)
    write_json(output/f"{name}-{fold}-decisions.json", decisions)
    return records, sensitivity


def summarize(output, records):
    df = pd.DataFrame(records)
    df.to_csv(output/"trials.csv", index=False)
    keys = ["dataset", "fold", "method", "rule", "width", "p"]
    cases = df.groupby(keys, as_index=False)[["score","select_ms","refit_ms","total_ms"]].median()
    full = cases[cases.method=="full"][["dataset","fold","score","refit_ms","total_ms"]].rename(
        columns={"score":"full_score","refit_ms":"full_refit_ms","total_ms":"full_total_ms"})
    cases = cases.merge(full, on=["dataset","fold"], validate="many_to_one")
    cases["gap"] = cases.score - cases.full_score
    cases["within"] = cases.gap >= -TOLERANCE
    cases["reduced"] = cases.width < cases.p
    cases["joint"] = cases.within & cases.reduced & (cases.total_ms < cases.full_total_ms)
    cases["break_even"] = [floor(r.select_ms / (r.full_refit_ms-r.refit_ms))+1 if
                           r.reduced and r.full_refit_ms > r.refit_ms else np.nan
                           for r in cases.itertuples()]
    cases.to_csv(output/"cases.csv", index=False)
    summary = cases.groupby(["dataset", "method"], as_index=False).agg(
        folds=("fold","size"), mean_score=("score","mean"), mean_gap=("gap","mean"), worst_gap=("gap","min"),
        within_folds=("within","sum"), reduced_folds=("reduced","sum"), joint_folds=("joint","sum"),
        mean_width=("width","mean"), p=("p","first"), select_ms=("select_ms","median"),
        total_ms=("total_ms","median"), reused_refit_ms=("refit_ms","median"),
        folds_with_break_even=("break_even","count"), median_finite_break_even=("break_even","median"))
    summary.to_csv(output/"summary.csv", index=False)


def run(output, verify):
    current = manifest()
    if verify:
        protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
        if protocol["sources"] != current["sources"]:
            raise ValueError("Source mismatch during replay")
    else:
        output.mkdir(parents=True, exist_ok=False)
        protocol = current
        protocol["input_hashes"] = {}
        write_json(output/"protocol.json", protocol)
    rows, sensitivity, audited = [], [], []
    for name in DATASETS:
        X, y, task, groups, source, duplicates = inputs(name)
        hashes = dict(source=source, X=digest(X), y=digest(y),
                      groups=None if groups is None else digest(groups))
        if verify:
            assert hashes == protocol["input_hashes"][name]
        else:
            protocol["input_hashes"][name] = hashes
        coverage = np.zeros(len(y),dtype=int)
        for fold, (tr, te) in enumerate(folds(X, y, task, groups), 1):
            assert not np.intersect1d(tr, te).size
            if groups is not None:
                assert not np.intersect1d(groups[tr], groups[te]).size
            if np.intersect1d(duplicates[tr], duplicates[te]).size:
                raise ValueError(f"Identical inputs crossed {name} outer folds")
            coverage[te] += 1
            training_groups = None if groups is None else groups[tr]
            split_path = output/f"{name}-{fold}-split.npz"
            if verify:
                with np.load(split_path) as split:
                    np.testing.assert_array_equal(split["train"], tr)
                    np.testing.assert_array_equal(split["test"], te)
                    np.testing.assert_array_equal(split["gold"], y[te])
                with np.load(output/f"{name}-{fold}-outputs.npz") as saved:
                    ref = json.loads((output/f"{name}-{fold}-decisions.json").read_text(encoding="utf-8"))
                    for method in ("stable", "environment_half"):
                        c, info = choose_columns(method, X[tr], y[tr], task, PRIMARY_SEED, training_groups, None)
                        pred = prediction(X[tr], y[tr], X[te], c, task, PRIMARY_SEED)
                        np.testing.assert_array_equal(saved[f"{method}_support"], c)
                        np.testing.assert_array_equal(saved[f"{method}_pred"], pred)
                        if method == "stable":
                            assert ref[method] == json.loads(json.dumps(info, default=jsonable))
                        audited.append(dict(dataset=name,fold=fold,method=method,width=len(c),
                                            score=metric(y[te],pred,task),support_sha256=digest(c),
                                            prediction_sha256=digest(pred)))
            else:
                np.savez_compressed(split_path, train=tr,test=te,gold=y[te])
                r,s = benchmark_fold(output,name,fold,X[tr],y[tr],X[te],y[te],task,training_groups,3)
                rows.extend(r)
                sensitivity.extend(s)
            print(f"{'Verified' if verify else 'Finished'} {name} fold {fold}", flush=True)
        assert np.all(coverage == 1)
    if verify:
        pd.DataFrame(audited).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        summarize(output, rows)
        pd.DataFrame(sensitivity).to_csv(output/"seed-sensitivity.csv", index=False)
        if current["sources"] != manifest()["sources"]:
            raise RuntimeError("Source changed during scoring")
        write_json(output/"protocol.json", protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
