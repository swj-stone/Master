"""Seal forecasts from restricted boosted trees before opening historical outcomes."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import DMatrix, XGBClassifier, XGBRegressor

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy, legacy_gain_receiver as receiver
from experiments.minpi_rework import quota_compress_tree as trees
from minppi_mofs.modeling import make_model


OUT=b.ROOT/"results/minpi_rework/quota-tree"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/quota-tree"
ARMS=("gain_prefix","gain_sweep","votes","exact_greedy","full")
CONTROLS=("one_vote","direct_sweep")
RHO=.20


def check():
    rng=np.random.default_rng(303)
    x=rng.normal(size=(120,5)).astype("float32")
    x[::11,0]=np.nan
    y=np.nan_to_num(x[:,0])+x[:,1]*x[:,2]
    allcols=np.arange(5)
    for task,model,target in [
        ("regression",XGBRegressor,y),
        ("classification",XGBClassifier,(y>0).astype(int)),
        ("classification",XGBClassifier,np.digitize(y,[-.4,.4]))]:
        model=model(n_estimators=7,max_depth=3,n_jobs=1).fit(x,target)
        f=trees.extract(model,x[:1])
        raw=f.predict(x,allcols)
        native=model.get_booster().predict(DMatrix(x),output_margin=True,strict_shape=True)
        np.testing.assert_allclose(raw,native,atol=1e-6)
        g=f.cache(x)
        for mask in range(1<<5):
            cols=allcols[(mask>>allcols)&1==1]
            cached=cached_margin(f,g,x,cols)
            np.testing.assert_allclose(cached,f.predict(x[:,cols],cols),atol=1e-7)
        score=trees.as_scores(raw,task)
        ref=model.predict(x) if task=="regression" else model.predict_proba(x)
        np.testing.assert_allclose(score,ref,atol=1e-6)
        for arm in ARMS[:-1]:
            cols,_=f.gain_prefix(x,allcols,RHO) if arm=="gain_prefix" else g.search(arm,allcols)
            budget=RHO*np.linalg.norm(raw-f.empty)
            assert np.linalg.norm(raw-f.predict(x[:,cols],cols))<=budget+1e-7
            if arm!="gain_prefix":
                for j in cols:
                    rest=cols[cols!=j]
                    assert np.linalg.norm(raw-f.predict(x[:,rest],rest))>budget-1e-7


def cached_margin(f,g,x,cols):
    omit=np.setdiff1d(np.arange(f.p),cols)
    active=np.asarray(g.incidence[:,omit].sum(axis=1)).ravel()==0
    return f.empty+np.asarray(g.a[:,active].sum(axis=1)).reshape(len(x),f.c)


def select(name,seed,job,controls=False):
    directory,row=job
    x,y,task,idx=legacy.read_split(directory,row,seed)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    xval=x[idx["width"]]
    p=x.shape[1]
    fullcols=np.arange(p)
    t=perf_counter()
    model=make_model(task=row["task"],seed=seed,backend=legacy.CONFIG["backend"],
                     model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    fit_s=perf_counter()-t
    t=perf_counter()
    f=trees.extract(model,xval[:1])
    parse_s=perf_counter()-t
    t=perf_counter()
    gains=model.get_booster().get_score(importance_type="gain")
    order=np.array(sorted(fullcols,key=lambda j:(gains.get(f"f{j}",0.),j),reverse=True))
    rank_s=perf_counter()-t
    raw=f.predict(xval,fullcols)
    native=model.get_booster().predict(DMatrix(xval),output_margin=True,strict_shape=True)
    np.testing.assert_allclose(raw,native,rtol=2e-5,atol=3e-5)
    scores=trees.as_scores(raw,task)
    original=model.predict(xval) if task=="regression" else model.predict_proba(xval)
    np.testing.assert_allclose(scores,original,atol=2e-5)
    time=perf_counter()
    g=f.cache(xval)
    cache_s=perf_counter()-time
    np.testing.assert_allclose(raw-f.empty,g.full.reshape(raw.shape),atol=4e-8,rtol=1e-6)
    info,forecasts={},{}
    key=f"{name}-{seed}"
    for method in np.random.default_rng(seed).permutation(CONTROLS if controls else ARMS):
        t=perf_counter()
        if method=="gain_prefix":
            cols,diag=f.gain_prefix(xval,order,RHO)
        elif method=="direct_sweep":
            cols,diag=f.direct_sweep(xval,order,RHO)
            original=json.loads((OUT/key/"selection.json").read_bytes())
            np.testing.assert_array_equal(original["records"]["gain_sweep"]["support"],cols)
        elif method=="full":
            cols,diag=fullcols,{}
        else:
            cols,diag=g.search(method,order,RHO)
        search_s=perf_counter()-t
        direct=f.predict(xval[:,cols],cols)
        np.testing.assert_allclose(direct,cached_margin(f,g,xval,cols),atol=3e-7,rtol=1e-6)
        error=np.linalg.norm(raw-direct)/max(g.norm,1e-100)
        assert error<=RHO+1e-7
        t=perf_counter()
        forecasts[method]=trees.as_scores(f.predict(x[np.ix_(idx["test"],cols)],cols),task)
        pred_s=perf_counter()-t
        pipeline=fit_s+parse_s+rank_s+search_s+pred_s
        cache_used=method not in ("full","gain_prefix","direct_sweep")
        if cache_used:
            pipeline+=cache_s
        info[method]=dict(p=p,k=len(cols),support=cols.tolist(),relative_error=float(error),fit_s=fit_s,
            parse_s=parse_s,rank_s=rank_s,cache_s=cache_s if cache_used else 0.,
            search_s=search_s,pred_s=pred_s,pipeline_s=pipeline,**diag)
        print(name,seed,method,len(cols),f"error={error:.4f}, {pipeline:.3f}s",flush=True)
    sub=(OUT/"controls" if controls else OUT)/key
    sub.mkdir(parents=True,exist_ok=True)
    scale=max(float(np.std(y[pool])),1e-12)
    archived=json.loads((receiver.OUT/key/"selection.json").read_bytes())["records"]["guard_gain"]
    reference=archived["trace"][str(p)]
    observed=b.metrics(y[idx["width"]],scores,task,scale)
    for m in reference:
        np.testing.assert_allclose(observed[m],reference[m],atol=8e-5)
    if not controls:
        with np.load(receiver.OUT/key/"predictions.npz") as saved:
            forecasts["refit_gain"]=saved["guard_gain"]
        info["refit_gain"]=dict(p=p,k=archived["k"],support=archived["selected"],pipeline_s=archived["deployment_s"])
    np.savez_compressed(sub/"predictions.npz",**forecasts)
    receipt=dict(task=task,seed=seed,p=p,scale=scale,groups=g.a.shape[1],calibration=len(xval),
        native_margin_max_abs=float(np.max(np.abs(raw-native))),records=info,
        forecast_hash=b.digest(sub/"predictions.npz"))
    (sub/"selection.json").write_text(json.dumps(receipt),encoding="utf8")


def score(jobs,seeds,controls=False):
    output=OUT/"controls" if controls else OUT
    keys=[(n,s) for n in legacy.CONFIG["datasets"] for s in seeds]
    lock={}
    for n,s in keys:
        key=f"{n}-{s}"
        sub=output/key
        receipt=sub/"selection.json"
        rec=json.loads(receipt.read_bytes())
        assert rec["forecast_hash"]==b.digest(sub/"predictions.npz")
        lock[key]=b.digest(receipt)
    DEST.mkdir(parents=True,exist_ok=True)
    suffix=("controls-" if controls else "")+"-".join(map(str,seeds))
    (DEST/f"lock-{suffix}.json").write_text(json.dumps(lock,indent=2),encoding="utf8")
    rows=[]
    with ZipFile(DEST/f"receipts-{suffix}.zip","w",ZIP_DEFLATED) as archive:
        for n,s in keys:
            key=f"{n}-{s}"
            sub=output/key
            receipt=sub/"selection.json"
            assert b.digest(receipt)==lock[key]
            archive.write(receipt,key+"/selection.json")
            rec=json.loads(receipt.read_bytes())
            d,r=jobs[n,s]
            path=d/r["test_relpath"]
            assert b.digest(path)==r["test_file_sha256"]
            y=pd.read_csv(path,usecols=[r["target"]])[r["target"]].to_numpy()
            with np.load(legacy.DEST/"novel-input-rows.npz") as inp:
                masks=dict(primary=slice(None),novel=inp[key])
            with np.load(sub/"predictions.npz") as pred:
                for arm,entry in rec["records"].items():
                    for part,mask in masks.items():
                        metrics=b.metrics(y[mask],pred[arm][mask],rec["task"],rec["scale"])
                        rows.append(dict(dataset=n,seed=s,method=arm,part=part,
                            **{k:v for k,v in entry.items() if k!="support"},**metrics))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/f"historical-{suffix}.csv",index=False)
    print(frame[frame.part=="primary"].pivot_table(index="dataset",columns="method",values="k"),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["check","select","score"])
    parser.add_argument("--seeds",type=int,nargs="+",default=[42])
    parser.add_argument("--datasets",nargs="+",default=legacy.CONFIG["datasets"])
    parser.add_argument("--controls",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.action in ("check","select"):
            check()
            print("Restricted traversal, all toy supports and fidelity endpoints verified.",flush=True)
        if args.action=="select":
            jobs=legacy.jobs()
            for seed in args.seeds:
                for name in args.datasets:
                    select(name,seed,jobs[name,seed],args.controls)
        if args.action=="score":
            score(legacy.jobs(),args.seeds,args.controls)


if __name__=="__main__":
    main()
