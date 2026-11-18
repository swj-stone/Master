"""Re-evaluate the two-class and continuous outcomes on the original thesis splits."""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_selection import f_classif, f_regression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, ROOT, load_dataset, metrics, train
from research.conditional_dense.vote_width.method import vote_support
from research.conditional_dense.original_tasks.regression import regression_support
from research.feature_game_eval.audit_controls import effect_size

SOURCES = ("breast_cancer", "madelon", "spambase", "diabetes", "crimedata", "gisette")
ARMS = ("minpi", "full", "f", "random")
HERE = Path(__file__).resolve().parent


def f_support(X, y, task, width):
    """Rank the official univariate scores after centering in double precision."""
    scores = np.empty(X.shape[1])
    scorer = f_classif if task == "classification" else f_regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        for start in range(0, X.shape[1], 256):
            part = slice(start, start + 256)
            block = X[:, part].astype(np.float64)
            block -= block.mean(axis=0)
            scores[part] = scorer(block, y)[0]
    return np.sort(np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")[:width])


def run_pipeline(arm, X, y, Xt, task, seed, width):
    start = perf_counter()
    if arm == "minpi":
        selector = vote_support if task == "classification" else regression_support
        columns, info = selector(X, y, seed=seed)
        ranking = info["rank"]
    else:
        ranking = None
        if arm == "full":
            columns = np.arange(X.shape[1])
        elif arm == "f":
            columns = f_support(X, y, task, width)
        elif arm == "random":
            columns = np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:width])
        else:
            raise ValueError(arm)
    selected = perf_counter()
    A, B = (X, Xt) if len(columns) == X.shape[1] else (X[:, columns], Xt[:, columns])
    prediction, decision = train("ridge", A, y, B, task, seed)
    finished = perf_counter()
    return (columns, prediction, decision, ranking,
            (selected - start) * 1000, (finished - start) * 1000)


def metrics_row(name, seed, arm, width, truth, pred, decision):
    values = metrics(truth, pred, decision)
    return dict(source=name, seed=seed, arm=arm, width=width, **values)


def check_f(X, y, task, columns):
    effects = np.concatenate([effect_size(X[:, j:j+256], y, task)
                              for j in range(0, X.shape[1], 256)])
    outside = np.setdiff1d(np.arange(X.shape[1]), columns)
    if outside.size and effects[outside].max() - effects[columns].min() > 1e-9:
        raise ValueError("F boundary disagrees with independent variance decomposition")


def run_source(name, folder, trial_writer, score_writer, pass_id, previous):
    features, labels, task, _, order_map, _, _ = load_dataset(name)
    main_metric = "auc" if task == "classification" else "r2"
    for seed in P["formal_seeds"]:
        original = OUTPUT / "formal" / name / f"s{seed}"
        record = json.loads((original / "record.json").read_text())
        np.testing.assert_array_equal(order_map, record["original_column_indices"])
        with np.load(original / "predictions.npz") as reference:
            tr, te = reference["train_indices"], reference["test_indices"]
            expected = train_test_split(np.arange(len(labels)), test_size=P["test_fraction"],
                                        random_state=seed, stratify=labels if task=="classification" else None)
            np.testing.assert_array_equal(tr, expected[0])
            np.testing.assert_array_equal(te, expected[1])
            np.testing.assert_array_equal(labels[te], reference["y_test"])
        X, Xt, y, truth = features[tr], features[te], labels[tr], labels[te]
        p = X.shape[1]
        arrays = dict(train=tr, test=te, truth=truth)
        width = None
        for arm in ARMS:  # warm-up and deterministic reference for timed calls
            cols, pred, decision, scores, _, _ = run_pipeline(arm, X, y, Xt, task, seed, width)
            measure = metrics_row(name, seed, arm, len(cols), truth, pred, decision)
            score_writer.writerow(measure)
            arrays[arm+"_cols"], arrays[arm+"_pred"] = cols, pred
            if decision is not None:
                arrays[arm+"_decision"] = decision
            if arm == "full":
                archived = record["arms"]["full"]["backends"]["ridge"][main_metric]
                np.testing.assert_allclose(measure[main_metric], archived, atol=5e-5, rtol=0)
            if arm == "f":
                check_f(X, y, task, cols)
            if arm == "minpi":
                width = len(cols)
                budget = record["arms"]["exact"]["k"]
                fixed = np.sort(np.argsort(-scores, kind="stable")[:budget])
                prediction, response = train("ridge", X[:, fixed], y, Xt[:, fixed], task, seed)
                score_writer.writerow(metrics_row(name, seed, "minpi_10pct", budget, truth, prediction, response))
                arrays["minpi_10pct_cols"], arrays["minpi_10pct_pred"] = fixed, prediction
                if response is not None:
                    arrays["minpi_10pct_decision"] = response
                for label in ("exact", "random", "f"):
                    historical = record["arms"][label]
                    row = {key: val for key, val in historical["backends"]["ridge"].items()
                           if key in ("auc","accuracy","balanced_accuracy","r2","mae")}
                    score_writer.writerow(dict(source=name, seed=seed, arm="archived_"+label,
                                               width=historical["k"], **row))
        filename = f"{name}-{seed}.npz"
        if previous is not None:
            with np.load(previous / filename) as replay:
                if set(arrays) != set(replay.files):
                    raise ValueError("Replay fields differ")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, replay[key])
        np.savez_compressed(folder / filename, **arrays)
        order = np.random.default_rng([20261118, seed, pass_id]).permutation(ARMS)
        for repeat in range(len(ARMS)):
            for position, arm in enumerate(np.roll(order, repeat)):
                cols, pred, decision, _, selection_ms, total_ms = run_pipeline(arm, X, y, Xt, task, seed, width)
                np.testing.assert_array_equal(cols, arrays[arm+"_cols"])
                np.testing.assert_array_equal(pred, arrays[arm+"_pred"])
                if decision is not None:
                    np.testing.assert_array_equal(decision, arrays[arm+"_decision"])
                trial_writer.writerow(dict(source=name, seed=seed, p=p, arm=arm, width=len(cols),
                                           repeat=repeat, position=position, select_ms=selection_ms, total_ms=total_ms))
    print(f"Completed {name}", flush=True)


def run(folder, pass_id, previous):
    folder.mkdir(parents=True, exist_ok=False)
    files = (Path(__file__), HERE/"paper-protocol.md", HERE/"regression.py",
             HERE.parent/"shared_method.py", HERE.parent/"vote_width/method.py",
             ROOT/"experiments/minpi_study.py", ROOT/"configs/minpi-study.json")
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    if previous and json.loads((previous/"protocol.json").read_text())["sha256"] != hashes:
        raise ValueError("Expected identical code in both passes")
    with (folder/"times.csv").open("x", newline="") as t, (folder/"scores.csv").open("x", newline="") as s:
        trials = csv.DictWriter(t, fieldnames="source seed p arm width repeat position select_ms total_ms".split())
        scores = csv.DictWriter(s, fieldnames="source seed arm width auc accuracy balanced_accuracy r2 mae".split())
        trials.writeheader()
        scores.writeheader()
        for name in SOURCES:
            run_source(name, folder, trials, scores, pass_id, previous)
            t.flush()
            s.flush()
    (folder/"protocol.json").write_text(json.dumps(dict(pass_id=pass_id, sha256=hashes), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pass-id", required=True, type=int, choices=(1, 2))
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if (args.pass_id == 2) != (args.previous is not None):
        parser.error("The second pass needs the first-pass directory")
    with threadpool_limits(1):
        run(args.folder, args.pass_id, args.previous)
