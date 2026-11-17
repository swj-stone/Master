"""Select supports without outer scoring, seal predictions, then score the frozen run."""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import Timer
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import sources
from experiments.minpi_compression.sources import ROOT, DATA, load
from minppi_mofs import sampled_linear_minpi as engine


PROTOCOL = Path(__file__).with_name("protocol.json")
DESIGN = json.loads(PROTOCOL.read_text(encoding="utf8"))
RUN = ROOT / "results/minpi_compression"
EVIDENCE = ROOT / "thesis-paper/evidence/minpi_compression"
METHODS = DESIGN["methods"]


def digest(path):
    with Path(path).open("rb") as handle:
        import hashlib
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stamp():
    return {p.name: sha256(p.read_text(encoding="utf8").encode("utf8")).hexdigest()
            for p in [PROTOCOL, Path(__file__), Path(sources.__file__), Path(engine.__file__)]}


def split(name, seed):
    if name in DESIGN["source_sha256"] and digest(DATA / f"{name}.zip") != DESIGN["source_sha256"][name]:
        raise ValueError(f"Raw data changed: {name}")
    x, y, task, groups = load(name)
    unique, reps = np.unique(groups, return_index=True)
    def partition(pool, proportion, salt):
        labels = y[reps[pool]] if task == "classification" else None
        return train_test_split(pool, test_size=proportion, random_state=seed+salt, stratify=labels)
    development, test = partition(unique, .2, 0)
    base, width = partition(development, .2, 101)
    fit, vote = partition(base, .25, 201)
    indices = {k: np.flatnonzero(np.isin(groups, a)) for k, a in
               dict(fit=fit, vote=vote, width=width, test=test).items()}
    assert np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(len(x)))
    if task == "classification":
        assert all(len(np.unique(y[i])) == len(np.unique(y)) for i in indices.values())
    return x, y, task, indices


def transform(x, train):
    prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    return prep.fit(x[train])


def prepare(name, seed):
    x, y, task, idx = split(name, seed)
    prep = transform(x, idx["fit"])
    return dict(X_fit=prep.transform(x[idx["fit"]]), y_fit=y[idx["fit"]],
                X_vote=prep.transform(x[idx["vote"]]), y_vote=y[idx["vote"]], task=task, seed=seed)


def cost_jobs(seed):
    for name in DESIGN["confirmation"]:
        data = prepare(name, seed)
        yield name, data["task"], tuple(data[k] for k in ("X_fit","y_fit","X_vote","y_vote"))


def metrics(y, prediction, task, scale):
    if task == "classification":
        # One-versus-rest AUC on each raw ridge score; no probabilities needed.
        auc = np.mean([roc_auc_score(y == c, prediction[:, c]) for c in range(prediction.shape[1])])
        return dict(auc=float(auc), accuracy=accuracy_score(y, prediction.argmax(axis=1)))
    mae = mean_absolute_error(y, prediction)
    return dict(r2=r2_score(y, prediction), mae=mae, scaled_mae=mae/scale)


def eligible(candidate, full, task):
    if task == "classification":
        return all(candidate[k] >= full[k] - .01 for k in ("auc", "accuracy"))
    return candidate["r2"] >= full["r2"] - .02 and candidate["scaled_mae"] <= full["scaled_mae"] + .02


def targets(y, task, classes):
    return np.eye(classes)[y] if task == "classification" else y


def ranked(method, x, y, task, idx, seed):
    fit, vote = idx["fit"], idx["vote"]
    if method in {"minpi", "coefficient", "mean"}:
        prep = transform(x, fit)
        result = engine.rank_features(prep.transform(x[fit]), y[fit], prep.transform(x[vote]), y[vote],
                                      task=task, method=method, seed=seed, **DESIGN["engine"])
        meta = dict(work=result.work)
        if result.minimum is not None:
            meta["minimum"] = result.minimum.tolist()
        return result.order, meta
    pool = np.sort(np.r_[fit, vote])
    z = transform(x, pool).transform(x[pool])
    if method == "univariate":
        scores = (f_classif if task == "classification" else f_regression)(z, y[pool])[0]
        return np.lexsort((np.arange(x.shape[1]), -np.nan_to_num(scores))), {}
    if method == "rfe":
        selector = RFE(Ridge(alpha=1.), n_features_to_select=1, step=1)
        selector.fit(z, targets(y[pool], task, len(np.unique(y[pool]))))
        return np.argsort(selector.ranking_), {}
    raise ValueError(method)


def predict(x, y, task, train, evaluation, order, widths):
    prep = transform(x, train)
    fit, query = prep.transform(x[train]), prep.transform(x[evaluation])
    yt = targets(y[train], task, len(np.unique(y[train])) if task == "classification" else 0)
    result = {}
    for k in widths:
        cols = np.sort(order[:k])
        result[k] = Ridge(alpha=1.).fit(fit[:, cols], yt).predict(query[:, cols])
    return result


def select_one(method, x, y, task, idx, seed):
    p = x.shape[1]
    start = perf_counter()
    if method == "full":
        order, k, trace, details = np.arange(p), p, {}, {}
        rank_s = 0.
    else:
        order, details = ranked(method, x, y, task, idx, seed)
        rank_s = perf_counter() - start
        assert np.array_equal(np.sort(order), np.arange(p))
        widths = sorted({1, *(max(1, math.ceil(p*f)) for f in DESIGN["width_fractions"])})
        pool = np.sort(np.r_[idx["fit"], idx["vote"]])
        pred = predict(x, y, task, pool, idx["width"], order, widths)
        scale = max(np.std(y[pool]), 1e-12)
        trace = {k: metrics(y[idx["width"]], v, task, scale) for k, v in pred.items()}
        k = next(k for k in widths if eligible(trace[k], trace[p], task))
    selection_s = perf_counter() - start
    pool = np.sort(np.r_[idx["fit"], idx["vote"], idx["width"]])
    outputs = predict(x, y, task, pool, idx["test"], order, [k])[k]
    deploy_s = perf_counter() - start
    return dict(method=method, p=p, k=k, order=order.tolist(), inner=trace, rank_s=rank_s,
                selection_s=selection_s, deployment_s=deploy_s, **details), outputs


def run(phase):
    out = RUN / phase
    out.mkdir(parents=True, exist_ok=True)
    signatures = stamp()
    tasks = [(name, seed) for name in DESIGN[phase] for seed in DESIGN["seeds"]]
    start = perf_counter()
    for index, (name, seed) in enumerate(tasks):
        root = out / f"{name}-{seed}"
        root.mkdir(exist_ok=True)
        receipt = root / "selection.json"
        if receipt.exists():
            saved = json.loads(receipt.read_bytes())
            if saved["sources"] != signatures or saved["predictions"] != digest(root/"predictions.npz"):
                raise ValueError(f"Stale run {root}; select a new version")
            continue
        timer = Timer(3600, lambda: os._exit(124))
        timer.daemon = True
        timer.start()
        begin = perf_counter()
        try:
            x, y, task, idx = split(name, seed)
            rng = np.random.default_rng(seed+index)
            ordering = rng.permutation(METHODS)
            predictions, records = {}, []
            for repeat in range(DESIGN["repetitions"]):
                for method in np.roll(ordering, repeat*2):
                    rec, prediction = select_one(method, x, y, task, idx, seed)
                    if repeat:
                        np.testing.assert_array_equal(predictions[method], prediction)
                    predictions[method] = prediction
                    records.append(dict(**rec, repeat=repeat))
            np.savez_compressed(root/"predictions.npz", **predictions)
            info = dict(dataset=name, seed=seed, task=task, splits={k: v.tolist() for k,v in idx.items()},
                        sources=signatures, predictions=digest(root/"predictions.npz"), rows=records)
            staged = root/"selection.tmp"
            staged.write_text(json.dumps(info), encoding="utf8")
            staged.replace(receipt)
        finally:
            timer.cancel()
        elapsed = perf_counter()-start
        print(f"[{index+1}/{len(tasks)}] {name}/{seed} {perf_counter()-begin:.2f}s, "
              f"elapsed {elapsed:.1f}s, ETA {elapsed*(len(tasks)-index-1)/(index+1):.1f}s", flush=True)
    lock = {f"{name}-{seed}": digest(out/f"{name}-{seed}"/"selection.json") for name,seed in tasks}
    (out/"lock.json").write_text(json.dumps(lock, indent=2), encoding="utf8")


def score(phase):
    out = RUN/phase
    lock = json.loads((out/"lock.json").read_bytes())
    tasks = [(name, seed) for name in DESIGN[phase] for seed in DESIGN["seeds"]]
    assert len(lock) == len(tasks)
    rows = []
    for name, seed in tasks:
        root = out/f"{name}-{seed}"
        assert lock[root.name] == digest(root/"selection.json")
        state = json.loads((root/"selection.json").read_bytes())
        assert digest(root/"predictions.npz") == state["predictions"] and state["sources"] == stamp()
        x, y, task, idx = split(name, seed)
        assert all(np.array_equal(idx[k], v) for k,v in state["splits"].items())
        dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
        with np.load(root/"predictions.npz") as pred:
            for method in METHODS:
                runs = [r for r in state["rows"] if r["method"] == method]
                rec = runs[0]
                assert all(r["k"] == rec["k"] and r["order"] == rec["order"] for r in runs)
                clocks = {k: np.median([r[k] for r in runs]) for k in ("rank_s","selection_s","deployment_s")}
                outcome = metrics(y[idx["test"]], pred[method], task, max(np.std(y[dev]), 1e-12))
                rows.append(dict(dataset=name, task=task, seed=seed, method=method, p=x.shape[1], k=rec["k"],
                                 retained=rec["k"]/x.shape[1], selected=" ".join(map(str,sorted(rec["order"][:rec["k"]]))),
                                 negative=rec.get("minimum", []).count(-1),**clocks,**outcome))
    dst = EVIDENCE/phase
    dst.mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(dst/"scores.csv",index=False)
    summary = frame.groupby(["task","dataset","method"]).mean(numeric_only=True).drop(columns="seed")
    summary.to_csv(dst/"sources.csv")
    cols = ["retained","auc","accuracy","r2","scaled_mae","selection_s","deployment_s"]
    macro = summary.groupby(["task","method"]).mean()[cols]
    macro.to_csv(dst/"summary.csv")
    print(macro.round(4).to_string())
    (dst/"lock.json").write_text(json.dumps(lock, indent=2),encoding="utf8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["select","score"])
    cli.add_argument("phase",choices=["pilot","confirmation"])
    args = cli.parse_args()
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.c_size_t(1 << 6)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(limits=1):
        (run if args.action == "select" else score)(args.phase)
