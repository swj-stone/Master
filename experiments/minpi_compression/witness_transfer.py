"""Measure fixed witness deletion on a six-source panel and seal the design bounds before scoring."""
import argparse
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import witness_guard as prior, witness_transfer_sources as data


b=prior.b
OUT=b.RUN/"witness-transfer"
EVIDENCE=b.EVIDENCE/"witness-transfer"
JOBS=[(n,s) for n in data.SOURCES for s in b.DESIGN["seeds"]]


def hashes():
    paths=[Path(__file__),Path(data.__file__),Path(__file__).with_suffix(".md")]
    return {**prior.signature(), **{p.name:b.digest(p) for p in paths}}


def select():
    cpu=prior.panel.bind_cpu()
    signature=hashes()
    EVIDENCE.mkdir(exist_ok=True,parents=True)
    receipts,opportunities={},[]
    start=perf_counter()
    for name,seed in JOBS:
        key=f"{name}-{seed}"
        dest=OUT/key
        dest.mkdir(exist_ok=True,parents=True)
        receipt=dest/"selection.json"
        before=perf_counter()
        if not receipt.exists():
            timer=Timer(3600,lambda:os._exit(124))
            timer.daemon=True
            timer.start()
            try:
                x,truth,task,idx=data.split(name,seed)
                y=truth.copy()
                y[idx["test"]]=-1
                order=np.random.default_rng(seed+len(receipts)).permutation(prior.METHODS)
                records,predictions=[],{}
                for repeat in range(3):
                    for method in np.roll(order,2*repeat):
                        rec,pred=prior.run_one(method,x,y,task,idx,seed)
                        if repeat:
                            np.testing.assert_array_equal(pred,predictions[method])
                        predictions[method]=pred
                        records.append(dict(**rec,repeat=repeat))
                used={r["method"]:r for r in records if r["repeat"]==0}
                base=used["coefficient"]
                for method in ("witness","uniform","tail"):
                    rec=used[method]
                    assert rec["prefix"]==base["order"][:base["k"]] and rec["h"]==used["witness"]["h"]
                    assert rec["work"]==used["witness"]["work"]
                np.savez_compressed(dest/"predictions.npz",**predictions)
                state=dict(dataset=name,seed=seed,task=task,cpu=cpu,rows=records,
                           hashes=signature,splits={k:v.tolist() for k,v in idx.items()},predictions=b.digest(dest/"predictions.npz"))
                temp=dest/"selection.tmp"
                temp.write_text(json.dumps(state),encoding="utf8")
                temp.replace(receipt)
            finally:
                timer.cancel()
        state=json.loads(receipt.read_bytes())
        assert state["hashes"]==signature and state["predictions"]==b.digest(dest/"predictions.npz")
        receipts[key]=b.digest(receipt)
        rec=next(r for r in state["rows"] if r["method"]=="witness")
        h,k=rec["h"],len(rec["prefix"])
        opportunities.append(dict(dataset=name,seed=seed,task=state["task"],prefix_width=k,inside_negatives=h,
                                  eligible=0<h<k,best_ratio=1-h/k if 0<h<k else 1.))
        elapsed=perf_counter()-start
        left=len(JOBS)-len(receipts)
        print(f"[{len(receipts)}/{len(JOBS)}] {key} {perf_counter()-before:.2f}s, elapsed {elapsed:.1f}s,"
              f" ETA {elapsed*left/len(receipts):.1f}s",flush=True)
    lock=OUT/"lock.json"
    lock.write_text(json.dumps(receipts,indent=2),encoding="utf8")
    frame=pd.DataFrame(opportunities)
    frame.to_csv(EVIDENCE/"opportunity.csv",index=False)
    rows=[]
    for by,group in [*(frame.groupby(["task","dataset"])),*(( (t,"ALL"),g) for t,g in frame.groupby("task"))]:
        means=group.groupby("dataset").best_ratio.apply(lambda x:np.log(x).mean())
        best=np.exp(means.mean())
        rows.append(dict(task=by[0],dataset=by[1],splits=len(group),eligible_splits=int(group.eligible.sum()),
                         inside_negatives=int(group.inside_negatives.sum()),best_width_ratio=best,max_saving=1-best))
    bound=pd.DataFrame(rows)
    bound.to_csv(EVIDENCE/"opportunity-summary.csv",index=False)
    (EVIDENCE/"lock.json").write_bytes(lock.read_bytes())
    print(bound.to_string(index=False))


def estimates(frame):
    rows=[]
    for task,g in frame.groupby("task"):
        witness=g[g.method=="witness"].set_index(["dataset","seed"])
        for method in ("coefficient","tail","uniform","full"):
            control=g[g.method==method].set_index(["dataset","seed"])
            fields=["auc","accuracy"] if task=="classification" else ["r2","scaled_mae"]
            ratios=["retained","deployment_s"] if method=="full" else ["retained","deployment_s","selection_s"]
            d=witness[fields]-control[fields]
            for f in ratios:
                d[f]=np.log(witness[f]/control[f])
            d["distinct_support"]=(witness["selected"]!=control["selected"]).astype(float)
            means=d.groupby(level="dataset").mean()
            draws=np.array(list(product(range(len(means)),repeat=len(means))))
            for col in means:
                s=means[col].to_numpy()
                vals=np.r_[s.mean(),np.quantile(s[draws].mean(axis=1),[.025,.975])]
                est,lo,hi=np.exp(vals) if col in ratios else vals
                rows.append(dict(task=task,baseline=method,field=col,estimate=est,lo=lo,hi=hi))
    return pd.DataFrame(rows)


def score():
    lock=EVIDENCE/"lock.json"
    assert lock.read_bytes()==(OUT/"lock.json").read_bytes() and (EVIDENCE/"opportunity-summary.csv").is_file()
    receipts=json.loads(lock.read_bytes())
    assert set(receipts)=={f"{n}-{s}" for n,s in JOBS}
    rows=[]
    signature=hashes()
    with ZipFile(EVIDENCE/"forecasts.zip","w",ZIP_DEFLATED) as archive:
        archive.write(lock,"lock.json")
        for key,digest in receipts.items():
            dest=OUT/key
            assert b.digest(dest/"selection.json")==digest
            state=json.loads((dest/"selection.json").read_bytes())
            assert state["hashes"]==signature and state["predictions"]==b.digest(dest/"predictions.npz")
            for f in ("selection.json","predictions.npz"):
                archive.write(dest/f,f"{key}/{f}")
            name,seed=state["dataset"],state["seed"]
            x,y,task,idx=data.split(name,seed)
            for part in idx:
                np.testing.assert_array_equal(idx[part],state["splits"][part])
            test=idx["test"]
            dev=np.r_[idx["fit"],idx["vote"],idx["width"]]
            with np.load(dest/"predictions.npz") as forecasts:
                for method in prior.METHODS:
                    runs=[r for r in state["rows"] if r["method"]==method]
                    rec=runs[0]
                    assert all(r["columns"]==rec["columns"] for r in runs)
                    pred=forecasts[method]
                    prior.panel.verify_predictions(x,y,task,idx,rec,pred)
                    metric=b.metrics(y[test],pred,task,y[dev].std())
                    if name=="musk1":
                        molecules=data.load(name)[3][test]
                        bags=pd.DataFrame(dict(y=y[test],score=pred[:,1]-pred[:,0],id=molecules)).groupby("id")
                        assert bags.y.nunique().max()==1
                        bs=bags.score.max()
                        metric.update(molecule_auc=roc_auc_score(bags.y.first(),bs),
                                      molecule_accuracy=accuracy_score(bags.y.first(),bs>0))
                    if name=="fires":
                        area=np.expm1(y[test])
                        clipped=np.maximum(0.,np.expm1(pred))
                        metric.update(hectares_mae=mean_absolute_error(area,clipped),hectares_r2=r2_score(area,clipped))
                    times={k:np.median([r[k] for r in runs]) for k in ("rank_s","selection_s","deployment_s")}
                    rows.append(dict(dataset=name,seed=seed,task=task,method=method,p=x.shape[1],k=rec["k"],
                                     selected=" ".join(map(str,rec["columns"])),retained=rec["k"]/x.shape[1],
                                     h=rec.get("h",0),optional_refits=rec["optional_refits"],accepted=rec["accepted"],
                                     width_fits=rec["width_fits"],deployment_fits=rec["deployment_fits"],
                                     factorizations=rec.get("work",{}).get("factorizations",0),**times,**metric))
    frame=pd.DataFrame(rows)
    frame.to_csv(EVIDENCE/"scores.csv",index=False)
    frame.groupby(["task","dataset","method"]).mean(numeric_only=True).drop(columns="seed").to_csv(EVIDENCE/"sources.csv")
    effects=estimates(frame)
    effects.to_csv(EVIDENCE/"paired-effects.csv",index=False)
    print(effects.pivot(index=["task","baseline"],columns="field",values="estimate").round(4))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["select","score"])
    with threadpool_limits(limits=1):
        (select if parser.parse_args().action=="select" else score)()
