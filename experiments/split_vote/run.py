"""Time independent selector arms, recording one immutable receipt per arm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import legacy_gain as legacy
from experiments.split_vote.path import select_width
from minppi_mofs.modeling import make_model
from minppi_mofs.split_vote import pivotal_order, split_voters
from minppi_mofs.utility import compute_utility


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/split_vote"
ORIGINAL = ("abalone","madelon","spambase","gisette","credit_card_fraud","isolet")
METHODS = ("min","pooled","gain_grid","exhaustive")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(x, y, task, idx, seed, method):
    started = perf_counter()
    tr = idx["fit"]
    va = np.sort(np.r_[idx["vote"],idx["width"]])
    full = tuple(range(x.shape[1]))
    model_task = task if task == "regression" else ("binary" if len(np.unique(y[tr])) == 2 else "multiclass")
    metric = {"binary":"roc_auc", "multiclass":"ovr_auc", "regression":"r2"}[model_task]
    def fitted(rows, cols):
        estimator = make_model(task=model_task, seed=seed, backend="cuda", model_params=legacy.CONFIG["model_params"])
        estimator.fit(x[np.ix_(rows,cols)], y[rows])
        return estimator
    def predict(model, rows, cols):
        matrix = x[np.ix_(rows,cols)]
        return model.predict(matrix) if task == "regression" else model.predict_proba(matrix)
    model = fitted(tr,full)
    booster = model.get_booster()
    if not json.loads(booster.save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
        raise RuntimeError("Lost the registered CUDA backend")
    full_score = compute_utility(y[va], predict(model,va,full), metric=metric)
    gain_map = booster.get_score(importance_type="gain")
    gain = np.array([gain_map.get(f"f{j}",0.) for j in full])
    if method == "gain_grid":
        order = np.lexsort((-np.arange(len(full)), -gain))
    elif method in {"min","pooled"}:
        gain_map = booster.get_score(importance_type="total_gain")
        total = np.array([gain_map.get(f"f{j}",0.) for j in full])
        voters, weights = split_voters(x[tr], y[tr], total, task=="classification", seed)
        order = pivotal_order(voters, weights, gain, rule=method)
    else:
        raise ValueError(method)
    ranking_seconds = perf_counter()-started
    def evaluate(cols):
        if len(cols) == len(full):
            return full_score
        return compute_utility(y[va], predict(fitted(tr,cols),va,cols), metric=metric)
    k,scores = select_width(order,evaluate)
    cols = sorted(order[:k].tolist())
    selection = perf_counter()-started
    dev = np.sort(np.r_[tr,va])
    pred = predict(fitted(dev,cols),idx["test"],cols)
    return dict(k=k,selected=cols,order=order.tolist(),scores=scores,fits=len(scores)+1,
                selection_s=selection,ranking_s=ranking_seconds,seconds=perf_counter()-started),pred


def load(name,seed):
    if name in ORIGINAL:
        folder,row = legacy.jobs()[name,seed]
        x,y,t,idx = legacy.read_split(folder,row,seed)
    else:
        from experiments.minpi_replication.data import split
        x,y,t,idx = split(name,seed)
    return x.astype(np.float32),y,t,idx


def execute(args):
    from minppi_mofs import split_vote
    from experiments.split_vote import path
    version = {f.name:digest(f) for f in [Path(__file__), Path(split_vote.__file__),Path(path.__file__)]}
    jobs = [(name,seed) for seed in args.seeds for name in args.names]
    beginning = perf_counter()
    for i,(name,seed) in enumerate(jobs):
        x,y,task,idx = load(name,seed)
        arms = np.asarray(args.methods)
        cyclic = (args.seeds.index(seed)+args.names.index(name)) % len(arms)
        for method in np.roll(arms,cyclic):
            folder = RUN/args.tag/f"{name}-{seed}"/method
            folder.mkdir(parents=True,exist_ok=True)
            receipt = folder/"receipt.json"
            forecast = folder/"forecast.npz"
            if receipt.exists():
                rec = json.loads(receipt.read_bytes())
                if version != rec["version"] or rec["forecast_sha256"] != digest(forecast):
                    raise RuntimeError(f"Stale receipt: {receipt}")
                continue
            print(f"{name}/{seed} {method} start",flush=True)
            timer = Timer(3600, lambda: os._exit(124))
            timer.daemon = True
            timer.start()
            try:
                if method == "exhaustive":
                    from experiments.minpi_replication.direct_gain import run_direct_gain
                    row,pred = run_direct_gain(x,y,task,idx,seed)
                else:
                    row,pred = run_arm(x,y,task,idx,seed,method)
                np.savez_compressed(forecast,**(pred if isinstance(pred,dict) else {"primary":pred}))
                rec = dict(dataset=name,seed=seed,method=str(method),task=task,p=x.shape[1],
                           forecast_sha256=digest(forecast),version=version,record=row)
                temp = receipt.with_suffix(".tmp")
                temp.write_text(json.dumps(rec, allow_nan=False),encoding="utf8")
                temp.replace(receipt)
                print(f"{name}/{seed} {method} finished {row.get('seconds',row.get('wall_s'))}s",flush=True)
            finally:
                timer.cancel()
        elapsed = perf_counter()-beginning
        print(f"{i+1}/{len(jobs)} jobs | {elapsed:.1f}s | ETA {elapsed*(len(jobs)-i-1)/(i+1):.1f}s",flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names",nargs="+",default=list(ORIGINAL))
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--methods",choices=METHODS,nargs="+",default=list(METHODS))
    p.add_argument("--tag",required=True)
    with threadpool_limits(1):
        execute(p.parse_args())
