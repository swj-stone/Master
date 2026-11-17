"""Refit the archived nonlinear receiver on frozen supports and through a common width guard."""
import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as bench
from experiments.minpi_compression import legacy_gain as legacy
from minppi_mofs.modeling import make_model


OUT = legacy.RUN / "xgb-receiver"
DEST = legacy.DEST / "xgb-receiver"
PROTOCOLS = [Path(__file__).with_name(k+".md") for k in ("legacy_gain_receiver", "legacy_gain_strong_guard")]


def predict(x, y, train, query, columns, model_task, seed):
    model = make_model(task=model_task, seed=seed, backend=legacy.CONFIG["backend"],
                       model_params=legacy.CONFIG["model_params"])
    cols = np.sort(columns)
    model.fit(x[np.ix_(train,cols)], y[train])
    eval_x = x[np.ix_(query, cols)]
    return model.predict(eval_x) if model_task == "regression" else model.predict_proba(eval_x)


def strong_guard(method, x, y, task, idx, seed, model_task, expected_order):
    start = perf_counter()
    if method == "full":
        order = np.arange(x.shape[1])
    else:
        rank = legacy.gain_rank if method == "gain" else bench.ranked
        order, _ = rank(method, x, y, task, idx, seed)
    rank_s = perf_counter()-start
    np.testing.assert_array_equal(expected_order,order)
    k, trace = len(order), {}
    if method != "full":
        pool = np.sort(np.r_[idx["fit"],idx["vote"]])
        widths = sorted({1, *(math.ceil(k*f) for f in bench.DESIGN["width_fractions"])})
        for width in widths:
            pred = predict(x,y,pool,idx["width"],order[:width],model_task,seed)
            trace[width] = bench.metrics(y[idx["width"]],pred,task,max(np.std(y[pool]),1e-12))
        k = next(width for width in widths if bench.eligible(trace[width],trace[k],task))
    selection_s = perf_counter()-start
    dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
    forecast = predict(x,y,dev,idx["test"],order[:k],model_task,seed)
    return dict(k=k,selected=np.sort(order[:k]).tolist(),trace=trace,rank_s=rank_s,
                selection_s=selection_s,deployment_s=perf_counter()-start),forecast


def select():
    start = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads((legacy.RUN / "lock.json").read_bytes())
    assert len(parent) == 30
    signature = {p.name:bench.digest(p) for p in [Path(__file__),*PROTOCOLS]}
    lock = {}
    for index, ((name,seed),(directory,row)) in enumerate(sorted(legacy.jobs().items())):
        key = f"{name}-{seed}"
        dest = OUT / key
        dest.mkdir(exist_ok=True)
        receipt = dest / "selection.json"
        if receipt.exists():
            stored = json.loads(receipt.read_bytes())
            accepted = [signature]
            filename = Path(__file__).name
            compatibility = json.loads(Path(__file__).with_name("legacy_gain_compat.json").read_bytes())
            if signature[filename] == compatibility["revised"]:
                accepted.append({**signature, filename:compatibility["original"]})
            assert (stored["signature"] in accepted and stored["parent"] == parent[key]
                    and stored["forecast"] == bench.digest(dest / "predictions.npz"))
            lock[key] = bench.digest(receipt)
            continue
        timer = Timer(3600,lambda:os._exit(124))
        timer.daemon = True
        timer.start()
        try:
            begin = perf_counter()
            state_path = legacy.RUN / key / "selection.json"
            assert bench.digest(state_path) == parent[key]
            state = json.loads(state_path.read_bytes())
            rows = {r["method"]:r for r in state["rows"] if r["repeat"]==0}
            x,y,task,idx = legacy.read_split(directory,row,seed)
            dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
            info, forecasts = {}, {}
            for method in np.random.default_rng(seed+index).permutation(legacy.METHODS):
                previous = rows[method]
                fresh, forecast = strong_guard(method,x,y,task,idx,seed,row["task"],previous["order"])
                info["guard_"+method], forecasts["guard_"+method] = fresh,forecast
                fixed = np.sort(previous["order"][:previous["k"]])
                t = perf_counter()
                pred = predict(x,y,dev,idx["test"],fixed,row["task"],seed)
                info["fixed_"+method] = dict(k=len(fixed),selected=fixed.tolist(),refit_s=perf_counter()-t)
                forecasts["fixed_"+method] = pred
                if fresh["selected"] == fixed.tolist():
                    np.testing.assert_array_equal(forecast,pred)
                print(f"{key} {method} complete pipeline {fresh['deployment_s']:.1f}s; "
                      f"split elapsed {perf_counter()-begin:.1f}s", flush=True)
            np.savez_compressed(dest/"predictions.npz",**forecasts)
            tmp = receipt.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(signature=signature,parent=parent[key],task=task,
                forecast=bench.digest(dest/"predictions.npz"),records=info)),encoding="utf8")
            tmp.replace(receipt)
            lock[key] = bench.digest(receipt)
            elapsed = perf_counter()-start
            print(f"sealed [{index+1}/30] {key} {perf_counter()-begin:.1f}s, elapsed {elapsed:.1f}s "
                  f"ETA {elapsed*(29-index)/(index+1):.1f}s",flush=True)
        finally:
            timer.cancel()
    (OUT/"lock.json").write_text(json.dumps(lock,indent=2),encoding="utf8")


def as_score_matrix(pred,task):
    if task == "classification" and pred.ndim == 1:
        return np.column_stack([1-pred,pred])
    return pred


def score():
    tasks = legacy.jobs()
    lock = json.loads((OUT/"lock.json").read_bytes())
    assert set(lock) == {f"{n}-{s}" for n,s in tasks}
    DEST.mkdir(exist_ok=True,parents=True)
    scores = []
    masks = np.load(legacy.DEST/"novel-input-rows.npz")
    with ZipFile(bench.ROOT/"packages/thesis-piv1-evidence.zip") as old, ZipFile(
        DEST/"forecasts.zip","w",ZIP_DEFLATED) as pack:
        pack.write(OUT/"lock.json","lock.json")
        for (n,s),(directory,row) in tasks.items():
            key = f"{n}-{s}"
            root = OUT/key
            rec = json.loads((root/"selection.json").read_bytes())
            assert bench.digest(root/"selection.json") == lock[key]
            assert bench.digest(root/"predictions.npz") == rec["forecast"]
            for filename in ("selection.json","predictions.npz"):
                pack.write(root/filename,key+"/"+filename)
            base = json.loads((legacy.RUN/key/"selection.json").read_bytes())
            path = directory/row["test_relpath"]
            assert bench.digest(path) == base["inputs"]["test"]
            y = pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()
            task = rec["task"]
            with np.load(root/"predictions.npz") as current, np.load(legacy.RUN/key/"predictions.npz") as ridge:
                preds = {**{k:(a,rec["records"][k]) for k,a in current.items()},
                         **{"ridge_"+r["method"]:(ridge[r["method"]], dict(k=r["k"]))
                            for r in base["rows"] if r["repeat"]==0}}
                for arm in ("direct_gain","PIv1"):
                    suffix = f"/seed-{s}/{n}/{arm}/predictions.npz"
                    member, = [f for f in old.namelist() if "PIv1-all-seeds-" in f and f.endswith(suffix)]
                    with np.load(BytesIO(old.read(member))) as a:
                        np.testing.assert_array_equal(y,a["y"])
                        frozen = json.loads(old.read(member.replace("predictions.npz","frozen.json")))
                        choice = frozen["choices"]["vote"] if arm == "PIv1" else frozen
                        preds["old_"+arm] = as_score_matrix(a["selected"],task),dict(k=choice["retained"])
                for method,(pred,entry) in preds.items():
                    common = dict(dataset=n,seed=s,method=method,task=task,k=entry["k"],p=len(base["rows"][0]["order"]))
                    for part,mask in (("primary",slice(None)),("novel",masks[key])):
                        vals = legacy.independent_metrics(y[mask],pred[mask],task,base["scale"])
                        clocks = {k:v for k,v in entry.items() if k.endswith("_s")}
                        scores.append(dict(**common,part=part,rows=len(y[mask]),**vals,**clocks))
    masks.close()
    frame = pd.DataFrame(scores)
    frame.to_csv(DEST/"scores.csv",index=False)
    sources = frame.groupby(["part","task","dataset","method"]).mean(numeric_only=True).drop(columns="seed")
    sources.to_csv(DEST/"sources.csv")
    rows = []
    for (part,task),group in frame.groupby(["part","task"]):
        g = group.set_index(["dataset","seed","method"])
        for arm, controls in (("guard_minpi",("guard_gain","guard_full","guard_coefficient","old_direct_gain","old_PIv1")),
                              ("fixed_minpi",("fixed_gain","fixed_full","fixed_coefficient","old_direct_gain","old_PIv1")),
                              ("ridge_minpi",("old_direct_gain","old_PIv1"))):
            ours = g.xs(arm,level="method")
            for control in controls:
                other = g.xs(control,level="method").reindex(ours.index)
                metrics = ["auc","accuracy"] if task == "classification" else ["r2","mae","scaled_mae"]
                differences = pd.DataFrame({k:ours[k]-other[k] for k in metrics})
                ratios = ["k"]
                if arm=="guard_minpi" and control in ("guard_gain","guard_coefficient"):
                    ratios += ["rank_s","selection_s","deployment_s"]
                for k in ratios:
                    differences[k] = np.log(ours[k]/other[k])
                source = differences.groupby(level="dataset").mean()
                effect = source.mean()
                effect[ratios] = np.exp(effect[ratios])
                rows.append(dict(part=part,task=task,arm=arm,control=control,**effect.to_dict()))
    summary = pd.DataFrame(rows)
    summary.to_csv(DEST/"contrasts.csv",index=False)
    print(summary[summary.part=="primary"].round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("select","score"))
    action = p.parse_args().action
    if os.name=="nt":
        import ctypes
        if not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<8)):
            raise ctypes.WinError()
    with threadpool_limits(limits=1):
        (select if action=="select" else score)()
