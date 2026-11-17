"""Apply one matched, guarded deletion to a ridge-coefficient prefix on 12 exposed sources."""
import argparse
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

from experiments.minpi_compression import benchmark as b, chain_validation as panel
from minppi_mofs import dictator_minpi


NAMES = [*b.DESIGN["confirmation"], *panel.data.SOURCES]
METHODS = ["full","coefficient","witness","tail","uniform"]
OUT = b.RUN / "witness-guard-r1"
EVIDENCE = b.EVIDENCE / "witness-guard"


def split(name, seed):
    return (panel.data.split if name in panel.data.SOURCES else b.split)(name, seed)


def signature():
    paths = [Path(__file__), Path(__file__).with_suffix(".md"), Path(dictator_minpi.__file__), Path(panel.data.__file__)]
    return {**b.stamp(), **{p.name:b.digest(p) for p in paths}}


def run_one(method, x, y, task, idx, seed):
    if method in ("full","coefficient"):
        rec, pred = b.select_one(method, x, y, task, idx, seed)
        rec.update(columns=sorted(rec["order"][:rec["k"]]), optional_refits=0, accepted=False,
                   width_fits=len(rec["inner"]), deployment_fits=1)
        return rec, pred
    start = perf_counter()
    p = x.shape[1]
    train, vote = idx["fit"], idx["vote"]
    prep = b.transform(x,train)
    rank = dictator_minpi.rank_bounds(prep.transform(x[train]), y[train], prep.transform(x[vote]), y[vote],
                                     task=task, seed=seed, closure=True, **b.DESIGN["engine"])
    order = np.lexsort((np.arange(p), -rank.tie_scores))
    rank_s = perf_counter()-start
    widths = sorted({1, *(math.ceil(p*f) for f in b.DESIGN["width_fractions"])})
    pool = np.sort(np.r_[train,vote])
    scale = max(y[pool].std(),1e-12)
    predictions = b.predict(x,y,task,pool,idx["width"],order,widths)
    scores = {k:b.metrics(y[idx["width"]], pred, task, scale) for k,pred in predictions.items()}
    k = next(k for k in widths if b.eligible(scores[k], scores[p], task))
    common = order[:k]
    negative = common[rank.minimum[common] == -1]
    h = len(negative)
    removed = np.empty(0,dtype=int)
    optional = None
    accepted = False
    cols = common
    if 0 < h < k:
        if method == "witness":
            removed = negative
        elif method == "tail":
            removed = common[-h:]
        else:
            removed = np.random.default_rng(seed+307).choice(common, h, replace=False)
        candidate = common[~np.isin(common,removed)]
        prediction = b.predict(x,y,task,pool,idx["width"],candidate,[len(candidate)])[len(candidate)]
        optional = b.metrics(y[idx["width"]],prediction,task,scale)
        accepted = bool(b.eligible(optional,scores[p],task))
        if accepted:
            cols = candidate
    selection_s = perf_counter()-start
    dev = np.sort(np.r_[pool,idx["width"]])
    pred = b.predict(x,y,task,dev,idx["test"],cols,[len(cols)])[len(cols)]
    deployment_s = perf_counter()-start
    kept = set(int(i) for i in cols)
    rec = dict(method=method,p=p,k=len(cols),order=cols.tolist()+[int(i) for i in order if int(i) not in kept],
               columns=sorted(kept),prefix=common.tolist(),negative=negative.tolist(),removed=removed.tolist(),
               work=rank.work,h=h,accepted=accepted,optional=optional,inner=scores,optional_refits=int(optional is not None),
               width_fits=len(widths),deployment_fits=1,rank_s=rank_s,selection_s=selection_s,deployment_s=deployment_s)
    return rec, pred


def select():
    sig = signature()
    cpu = panel.bind_cpu()
    run_start = perf_counter()
    receipts = {}
    for name in NAMES:
        for seed in b.DESIGN["seeds"]:
            key = f"{name}-{seed}"
            dest = OUT / key
            dest.mkdir(parents=True,exist_ok=True)
            receipt = dest / "selection.json"
            if receipt.exists():
                record = json.loads(receipt.read_bytes())
                assert record["hashes"] == sig and record["predictions"] == b.digest(dest / "predictions.npz")
                receipts[key] = b.digest(receipt)
                continue
            start = perf_counter()
            timer = Timer(3600, lambda: os._exit(124))
            timer.daemon = True
            timer.start()
            try:
                x,truth,task,idx = split(name,seed)
                y = truth.copy()
                y[idx["test"]] = -1
                ordering = np.random.default_rng(seed+len(receipts)).permutation(METHODS)
                predictions, records = {}, []
                for rep in range(3):
                    for method in np.roll(ordering,rep*2):
                        record, pred = run_one(method,x,y,task,idx,seed)
                        if rep:
                            np.testing.assert_array_equal(pred,predictions[method])
                        predictions[method] = pred
                        records.append(dict(**record,repeat=rep))
                archive_phase = "chain-validation" if name in panel.data.SOURCES else "confirmation"
                with ZipFile(b.EVIDENCE / archive_phase / "forecasts.zip") as previous:
                    import io
                    with np.load(io.BytesIO(previous.read(f"{key}/predictions.npz"))) as old:
                        for method in ["coefficient","full"]:
                            np.testing.assert_allclose(predictions[method],old[method],rtol=1e-12,atol=1e-12)
                    ref = next(r for r in json.loads(previous.read(f"{key}/selection.json"))["rows"]
                               if r["method"] == "coefficient")
                    for rec in records:
                        if rec["method"] in ("witness","tail","uniform"):
                            assert rec["prefix"] == ref["order"][:ref["k"]]
                np.savez_compressed(dest/"predictions.npz", **predictions)
                payload = dict(dataset=name,seed=seed,cpu=cpu,rows=records,hashes=sig,
                               splits={k:v.tolist() for k,v in idx.items()}, predictions=b.digest(dest/"predictions.npz"))
                temporary = dest/"selection.tmp"
                temporary.write_text(json.dumps(payload),encoding="utf8")
                temporary.replace(receipt)
                receipts[key] = b.digest(receipt)
            finally:
                timer.cancel()
            duration = perf_counter()-start
            elapsed = perf_counter()-run_start
            remaining = elapsed*(60-len(receipts))/len(receipts)
            print(f"[{len(receipts)}/60] {key} {duration:.2f}s, elapsed {elapsed:.1f}s, ETA {remaining:.1f}s", flush=True)
    (OUT/"lock.json").write_text(json.dumps(receipts,indent=2),encoding="utf8")


def score():
    lock = OUT/"lock.json"
    receipts = json.loads(lock.read_bytes())
    assert set(receipts) == {f"{n}-{s}" for n in NAMES for s in b.DESIGN["seeds"]}
    EVIDENCE.mkdir(exist_ok=True)
    rows = []
    with ZipFile(EVIDENCE/"forecasts.zip","w",ZIP_DEFLATED) as archived:
        archived.write(lock,"lock.json")
        for key,digest in receipts.items():
            dest=OUT/key
            assert b.digest(dest/"selection.json") == digest
            state = json.loads((dest/"selection.json").read_bytes())
            assert state["predictions"] == b.digest(dest/"predictions.npz") and state["hashes"]==signature()
            for f in ("selection.json","predictions.npz"):
                archived.write(dest/f,f"{key}/{f}")
            name,seed = state["dataset"],state["seed"]
            x,y,task,idx = split(name,seed)
            for part in idx:
                np.testing.assert_array_equal(state["splits"][part],idx[part])
            dev = np.r_[idx["fit"],idx["vote"],idx["width"]]
            with np.load(dest/"predictions.npz") as pred:
                for method in METHODS:
                    runs = [r for r in state["rows"] if r["method"]==method]
                    rec = runs[0]
                    assert all(r["columns"]==rec["columns"] for r in runs)
                    panel.verify_predictions(x,y,task,idx,rec,pred[method])
                    clocks = {k:np.median([r[k] for r in runs]) for k in ("selection_s","deployment_s","rank_s")}
                    scores = b.metrics(y[idx["test"]],pred[method],task,y[dev].std())
                    rows.append(dict(dataset=name,seed=seed,task=task,method=method,p=x.shape[1],k=rec["k"],
                              selected=" ".join(map(str,rec["columns"])),retained=rec["k"]/x.shape[1],
                              removed=" ".join(map(str,sorted(rec.get("removed",[])))),accepted=rec["accepted"],
                              h=rec.get("h",0),width_fits=rec["width_fits"],optional_refits=rec["optional_refits"],
                              deployment_fits=rec["deployment_fits"],factorizations=rec.get("work",{}).get("factorizations",0),
                              **scores,**clocks))
    frame=pd.DataFrame(rows)
    frame.to_csv(EVIDENCE/"scores.csv",index=False)
    frame.groupby(["task","dataset","method"]).mean(numeric_only=True).drop(columns="seed").to_csv(EVIDENCE/"sources.csv")
    effects=[]
    for domain, group in frame.groupby("task"):
        voting=group[group.method=="witness"].set_index(["dataset","seed"])
        for method in METHODS:
            other=group[group.method==method].set_index(["dataset","seed"])
            d=voting[["auc","accuracy","r2","scaled_mae"]]-other[["auc","accuracy","r2","scaled_mae"]]
            for k in ("retained","deployment_s","selection_s"):
                d[k]=np.log(voting[k]/other[k])
            mean=d.groupby(level="dataset").mean().mean().dropna()
            for k in ("retained","deployment_s","selection_s"):
                mean[k]=np.exp(mean[k])
            if method=="full":
                mean=mean.drop("selection_s")
            opp=voting.optional_refits.astype(bool)
            effects.append(dict(task=domain,baseline=method,witness_opportunities=int(opp.sum()),
                                accepted=int(other.accepted.sum()),
                                different_proposals=int((opp&(voting.removed!=other.removed)).sum())
                                   if method in ("tail","uniform","witness") else None,
                                different_sets=int((voting.selected!=other.selected).sum()),**mean.to_dict()))
    effects=pd.DataFrame(effects)
    effects.to_csv(EVIDENCE/"effects.csv",index=False)
    print(effects.round(4).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["select","score"])
    with threadpool_limits(limits=1):
        (select if parser.parse_args().action=="select" else score)()
