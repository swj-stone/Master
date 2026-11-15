"""Run a frozen PMLB fold in isolation; the outer outcomes enter after all supports freeze."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_study.data import DATA, OUT, config, digest
from experiments.context_study.selection import select_l1, select_vote
from minppi_mofs.piv3_compact_tree import select as gain_anchor
from minppi_mofs.utility import compute_utility


def store(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf8")
    os.replace(tmp, path)


def fit(x, y, seed):
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1, tree_method="hist",
                         device="cpu", n_jobs=1, random_state=seed).fit(x, y)


def run(name, fold):
    folder = OUT / name / f"fold-{fold}"
    folder.mkdir(exist_ok=True, parents=True)
    spec = config()
    table = DATA / name / f"{name}.tsv.gz"
    source = json.loads((OUT / "sources.json").read_bytes())["sources"][name][table.name]
    if digest(table) != source["sha256"]:
        raise ValueError(f"Source hash mismatch for {name}")
    frame = pd.read_csv(table, sep="\t")
    y = frame.pop("target").to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(x[~np.isnan(x)]).all():
        raise ValueError("Infinite features")
    splits = StratifiedKFold(3, shuffle=True, random_state=20261115)
    development, test = list(splits.split(x, y))[fold]
    seed = 20421 + fold
    train, rank = train_test_split(development, test_size=.25, stratify=y[development], random_state=seed)
    classes, counts = np.unique(y[train], return_counts=True)
    if len(classes) == 2 and counts[0] < counts[1]:
        classes = classes[::-1]
    code = {label: i for i,label in enumerate(classes)}
    y = np.array([code[label] for label in y])
    if set(y[rank]) != set(range(len(classes))):
        raise ValueError("Ranking fold missing a training class")
    task = "binary" if len(classes) == 2 else "multiclass"
    start = perf_counter()
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xv = imp.transform(x[train]), imp.transform(x[rank])
    prep_s = perf_counter()-start
    start = perf_counter()
    model = fit(xt, y[train], seed)
    model_s = perf_counter()-start
    start = perf_counter()
    cols = [f"c{i}" for i in range(x.shape[1])]
    _, info = gain_anchor(model, pd.DataFrame(xt, columns=cols), y[train],
                          pd.DataFrame(xv, columns=cols), y[rank], task=task,
                          utility="ovr_auc", seed=seed, aggregate="gain")
    order = [int(key[1:]) for key in info["order"]]
    k = info["k"]
    anchor_s = perf_counter()-start
    arms = spec["arms"]
    offset = (fold+spec["datasets"].index(name))%len(arms)
    arms = arms[offset:]+arms[:offset]
    selected, traces, selection_times = {}, {}, {}
    for arm in arms:
        start = perf_counter()
        if arm == "matched_gain":
            continue
        if arm == "full":
            support, trace = list(range(x.shape[1])), {}
        elif arm == "gain":
            support, trace = sorted(order[:k]), {}
        elif arm == "l1":
            support, trace = select_l1(xt, y[train], xv, y[rank], seed)
        else:
            support, trace = select_vote(arm, model, xt, xv, y[rank], order, k, fold)
        selection_times[arm] = perf_counter()-start
        selected[arm], traces[arm] = support, trace
    start = perf_counter()
    selected["matched_gain"] = sorted(order[:len(selected["min"])])
    selection_times["matched_gain"] = perf_counter()-start
    traces["matched_gain"] = {}
    store(folder/"selection.json", dict(train=train.tolist(), rank=rank.tolist(), test=test.tolist(),
          class_order=classes.tolist(), feature_names=frame.columns.tolist(), order=order,
          anchor_k=k, selected=selected, traces=traces))
    receipt = digest(folder/"selection.json")
    start = perf_counter()
    refit_imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[development])
    xd, xe = refit_imputer.transform(x[development]), refit_imputer.transform(x[test])
    final_prep = perf_counter()-start
    predictions = {"row_id": test, "y": y[test]}
    records = []
    for arm in arms:
        support = selected[arm]
        start = perf_counter()
        refit = fit(xd[:,support], y[development], seed)
        pred = refit.predict_proba(xe[:,support])
        final_s = perf_counter()-start + final_prep
        predictions[arm] = pred
        auc = compute_utility(y[test], pred, metric="ovr_auc")
        acc = float(accuracy_score(y[test], pred.argmax(axis=1)))
        initial_s = 0. if arm=="full" else prep_s if arm=="l1" else prep_s+model_s+anchor_s
        select_s = initial_s+selection_times[arm]
        records.append(dict(dataset=name, fold=fold, arm=arm, n=len(x), width=x.shape[1],
            n_test=len(test), classes=len(classes), k=len(support), anchor_k=k,
            mask_calls=traces[arm].get("mask_calls",0), auc=auc, accuracy=acc,
            selection_s=select_s, deletion_s=selection_times[arm], final_s=final_s,
            total_s=select_s+final_s, selected=json.dumps(support)))
    np.savez_compressed(folder/"predictions.npz", **predictions)
    store(folder/"completed.json", dict(selection_sha256=receipt, predictions_sha256=digest(folder/"predictions.npz"),
          protocol_sha256=digest(OUT/"protocol-lock.json"), data_sha256=source["sha256"], records=records))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=config()["datasets"])
    p.add_argument("fold", type=int, choices=range(3))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.dataset, args.fold)

