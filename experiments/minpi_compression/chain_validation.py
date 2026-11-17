"""Freeze four-source predictions separately from scoring and verify their individual endpoints."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from threading import Timer
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, chain_validation_sources as data
from minppi_mofs import chain_minpi

METHODS = ["full", "coefficient", "univariate", "rfe", "minpi", "chain"]
OUT = b.RUN / "chain-validation"
EVIDENCE = b.EVIDENCE / "chain-validation"
SELECTOR = b.ranked


def rank(method, x, y, task, idx, seed):
    if method != "chain":
        return SELECTOR(method, x, y, task, idx, seed)
    fit, vote = idx["fit"], idx["vote"]
    prep = b.transform(x, fit)
    result = chain_minpi.rank_chain(prep.transform(x[fit]), y[fit], prep.transform(x[vote]), y[vote],
                                    task=task, seed=seed, **b.DESIGN["engine"])
    return result.order, dict(work=result.work, minimum=result.minimum.tolist())


def signatures():
    return {**b.stamp(), **{p.name: b.digest(p) for p in
                           (Path(__file__), Path(data.__file__), Path(chain_minpi.__file__),
                            Path(__file__).with_name("chain_validation_protocol.md"))}}


def bind_cpu():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        available = [i for i in range(process.value.bit_length()) if (1 << i) & process.value]
        cpu = available[min(8, len(available)-1)]
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.c_size_t(1 << cpu)):
            raise ctypes.WinError(ctypes.get_last_error())
        return cpu
    if hasattr(os, "sched_getaffinity"):
        cpu = sorted(os.sched_getaffinity(0))[0]
        os.sched_setaffinity(0, {cpu})
        return cpu
    return None


def select():
    cpu = bind_cpu()
    signature = signatures()
    jobs = [(name, seed) for name in data.SOURCES for seed in b.DESIGN["seeds"]]
    receipts = {}
    begin = perf_counter()
    for job, (name, seed) in enumerate(jobs):
        started = perf_counter()
        dest = OUT / f"{name}-{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        receipt = dest / "selection.json"
        if receipt.exists():
            state = json.loads(receipt.read_bytes())
            if state["hashes"] != signature or state["predictions"] != b.digest(dest / "predictions.npz"):
                raise ValueError(f"Stale receipt at {dest}")
        else:
            timeout = Timer(3600, lambda: os._exit(124))
            timeout.daemon = True
            timeout.start()
            try:
                x, truth, task, idx = data.split(name, seed)
                y = truth.copy()
                y[idx["test"]] = -1
                records, predictions = [], {}
                order = np.random.default_rng(seed + job).permutation(METHODS)
                with patch.object(b, "ranked", rank):
                    for rep in range(3):
                        for method in np.roll(order, rep*2):
                            rec, pred = b.select_one(method, x, y, task, idx, seed)
                            if rep:
                                np.testing.assert_array_equal(predictions[method], pred)
                            predictions[method] = pred
                            records.append(dict(**rec, repeat=rep))
                np.savez_compressed(dest / "predictions.npz", **predictions)
                state = dict(dataset=name, seed=seed, task=task, cpu=cpu, rows=records,
                             hashes=signature, predictions=b.digest(dest / "predictions.npz"),
                             splits={k: v.tolist() for k, v in idx.items()})
                temporary = dest / "selection.tmp"
                temporary.write_text(json.dumps(state), encoding="utf8")
                temporary.replace(receipt)
            finally:
                timeout.cancel()
        receipts[dest.name] = b.digest(receipt)
        spent = perf_counter()-begin
        print(f"[{job+1}/{len(jobs)}] {dest.name} {perf_counter()-started:.2f}s, elapsed {spent:.2f}s, "
              f"ETA {spent*(len(jobs)-job-1)/(job+1):.2f}s", flush=True)
    (OUT / "lock.json").write_text(json.dumps(receipts, indent=2), encoding="utf8")


def verify_predictions(x, y, task, idx, rec, pred):
    dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
    # Complete inputs on these four sources. Verify a different solver and explicit train scaling.
    scale = x[dev].std(axis=0)
    z = (x - x[dev].mean(axis=0)) / np.where(scale == 0, 1., scale)
    selected = np.sort(rec["order"][:rec["k"]])
    target = b.targets(y[dev], task, len(np.unique(y[dev])))
    verified = Ridge(alpha=1., solver="svd").fit(z[dev][:, selected], target).predict(z[idx["test"]][:, selected])
    np.testing.assert_allclose(verified, pred, rtol=1e-8, atol=1e-8)


def score():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    lock = OUT / "lock.json"
    receipts = json.loads(lock.read_bytes())
    assert set(receipts) == {f"{n}-{s}" for n in data.SOURCES for s in b.DESIGN["seeds"]}
    rows = []
    signature = signatures()
    with ZipFile(EVIDENCE / "forecasts.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(lock, "lock.json")
        for key, digest in receipts.items():
            dest = OUT / key
            assert digest == b.digest(dest / "selection.json")
            state = json.loads((dest / "selection.json").read_bytes())
            assert state["hashes"] == signature and state["predictions"] == b.digest(dest / "predictions.npz")
            for filename in ("selection.json","predictions.npz"):
                archive.write(dest / filename, f"{key}/{filename}")
            name, seed = state["dataset"], state["seed"]
            x, y, task, idx = data.split(name, seed)
            for part in idx:
                np.testing.assert_array_equal(idx[part], state["splits"][part])
            dev = np.r_[idx["fit"],idx["vote"],idx["width"]]
            with np.load(dest / "predictions.npz") as predictions:
                for method in METHODS:
                    runs = [r for r in state["rows"] if r["method"] == method]
                    rec = runs[0]
                    assert len(runs) == 3 and all(r["k"] == rec["k"] and r["order"] == rec["order"] for r in runs)
                    verify_predictions(x, y, task, idx, rec, predictions[method])
                    clocks = {col: np.median([r[col] for r in runs]) for col in ("rank_s","selection_s","deployment_s")}
                    metrics = b.metrics(y[idx["test"]], predictions[method], task, y[dev].std())
                    rows.append(dict(dataset=name,seed=seed,task=task,method=method,p=rec["p"],k=rec["k"],
                                retained=rec["k"]/rec["p"],selected=" ".join(map(str, sorted(rec["order"][:rec["k"]]))),
                                negative=rec.get("minimum",[]).count(-1),**metrics,**clocks))
    frame = pd.DataFrame(rows)
    frame.to_csv(EVIDENCE / "scores.csv", index=False)
    frame.groupby(["task","dataset","method"]).mean(numeric_only=True).drop(columns="seed").to_csv(EVIDENCE / "sources.csv")
    effects = []
    for domain, group in frame.groupby("task"):
        a = group[group.method == "chain"].set_index(["dataset","seed"])
        for method in set(METHODS)-{"chain"}:
            c = group[group.method == method].set_index(["dataset","seed"])
            deltas = a[["auc","accuracy","r2","scaled_mae"]] - c[["auc","accuracy","r2","scaled_mae"]]
            ratios = ["retained","deployment_s"] if method == "full" else ["retained","rank_s","selection_s","deployment_s"]
            for field in ratios:
                deltas[field] = np.log(a[field]/c[field])
            estimate = deltas.groupby(level="dataset").mean().mean().dropna()
            estimate[ratios] = np.exp(estimate[ratios])
            quality = (estimate.auc >= -.01 and estimate.accuracy >= -.01) if domain == "classification" else (
                     estimate.r2 >= -.02 and estimate.scaled_mae <= .02)
            small = quality and estimate.retained <= (.85 if method == "full" else .90)
            fast = None if method == "full" else quality and estimate.retained <= 1 and estimate.selection_s <= .9
            effects.append(dict(task=domain,baseline=method,**estimate.to_dict(),same_support=int((a.selected==c.selected).sum()),
                                source_seeds=len(a),quality_pass=quality,smaller_pass=small,faster_pass=fast))
    effects = pd.DataFrame(effects).sort_values(["task","baseline"])
    effects.to_csv(EVIDENCE / "effects.csv", index=False)
    (EVIDENCE / "lock.json").write_bytes(lock.read_bytes())
    print(effects.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select","score"])
    with threadpool_limits(limits=1):
        (select if parser.parse_args().action == "select" else score)()
