"""Lock classification-only scaling and exchange forecasts on five new sources."""
import argparse
import ctypes
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import validation_exchange as panel,quota_compress_tree as tree
from experiments.minpi_rework import quota_compress_scale as scaler,quota_compress_rescue as rescue
from experiments.minpi_rework import quota_compress_scale_run as old,quota_compress_rescue_check as checks
from experiments.minpi_rework import quota_compress_tree_native_exact as native
from minppi_mofs import modeling


ROOT=b.ROOT/"results/minpi_rework/quota-scale-confirmation"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/quota-scale-confirmation"
ALL_METHODS=("exact","ablation","gain","treeshap")


def signature():
    modules=(old,scaler,rescue,checks,tree,native,panel,b,legacy,modeling)
    files=[Path(m.__file__) for m in modules]+[Path(__file__),Path(__file__).with_suffix(".md"),panel.MANIFEST,legacy.ORIGINAL]
    return {p.name:b.digest(p) for p in files}


def select(name,seed,d,sign):
    x,y,task,idx=panel.split(name,seed)
    outer=y[idx["test"]]
    assert (outer==-1).all() if task=="classification" else np.isnan(outer).all()
    old.select("fresh",name,seed,panel,None,d)
    original=json.loads((d/"receipt.json").read_bytes())
    records=original["records"]
    with np.load(d/"forecasts.npz") as pred:
        forecasts={k:pred[k] for k in pred}
    val=x[idx["width"]]
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
    t=perf_counter()
    model=modeling.make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    fit_s=perf_counter()-t
    t=perf_counter()
    native_forecast=model.predict(x[idx["test"]]) if task=="regression" else model.predict_proba(x[idx["test"]])
    predict_s=perf_counter()-t
    atol=3e-5*max(1.,np.abs(native_forecast).max())
    np.testing.assert_allclose(native_forecast,forecasts["full"],rtol=3e-5,atol=atol)
    records["native_full"]=dict(p=x.shape[1],k=x.shape[1],lam=1.,relative_error=0.,
        cols=list(range(x.shape[1])),predict_s=predict_s,pipeline_s=fit_s+predict_s)
    forecasts["native_full"]=native_forecast
    f=tree.extract(model,val[:1])
    g=f.cache(val)
    gain=model.get_booster().get_score(importance_type="gain")
    order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
    t=perf_counter()
    cols,path,risk,lam,counts=scaler.search(g,order,False,True,True,True)
    search_s=perf_counter()-t
    baseline=records["exact_raw"]
    np.testing.assert_array_equal(cols,baseline["cols"])
    plain=scaler.search(g,order,False,True,True,False)
    np.testing.assert_array_equal(path,plain[1])
    records["exact_raw_votes"]={**baseline,"search_s":search_s,
        "pipeline_s":baseline["pipeline_s"]-baseline["search_s"]+search_s,
        "checks":int(counts[0]),"vetoes":int(counts[1])}
    forecasts["exact_raw_votes"]=forecasts["exact_raw"]
    for method in ALL_METHODS:
        chosen=method+("_scaled" if task=="classification" else "_raw")
        key=method+"_policy"
        records[key]={**records[chosen],"from":chosen}
        forecasts[key]=forecasts[chosen]
    chosen="exact_scaled_votes" if task=="classification" else "exact_raw_votes"
    records["exact_policy_votes"]={**records[chosen],"from":chosen}
    forecasts["exact_policy_votes"]=forecasts[chosen]
    for method in ("exact","ablation"):
        original=records[method+"_raw"]
        canonical=None
        for ballot in (False,True):
            t=perf_counter()
            keep,moves,cs,stop=rescue.search(g,original["cols"],ballot=ballot)
            rescue_s=perf_counter()-t
            if canonical is not None:
                np.testing.assert_array_equal(canonical[0],keep)
                np.testing.assert_array_equal(canonical[1],moves)
            canonical=keep,moves
            risk=checks.physical(g,keep)
            assert risk<=(.2*g.norm+1e-12*max(1.,g.norm))**2
            t=perf_counter()
            pred=tree.as_scores(f.predict(x[np.ix_(idx["test"],keep)],keep),task)
            pred_s=perf_counter()-t
            key=method+("_rescue_votes" if ballot else "_rescue")
            records[key]=dict(p=g.p,k=len(keep),cols=keep.tolist(),lam=1.,moves=moves.tolist(),
                relative_error=np.sqrt(risk)/max(g.norm,1e-100),vetoes=int(cs[0]),checks=int(cs[1]),stop=stop,
                search_s=original["search_s"]+rescue_s,rescue_s=rescue_s,
                pipeline_s=original["pipeline_s"]-original["predict_s"]+rescue_s+pred_s)
            forecasts[key]=pred
    np.savez_compressed(d/"all-forecasts.npz",**forecasts)
    receipt=dict(task=task,source=sign,split={k:v.tolist() for k,v in idx.items()},records=records,
        forecast=b.digest(d/"all-forecasts.npz"))
    tmp=d/"complete.tmp"
    tmp.write_text(json.dumps(receipt),encoding="utf8")
    tmp.replace(d/"complete.json")


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("collect","score"))
    args=cli.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    jobs=[(n,s) for n in panel.SOURCES for s in panel.SEEDS]
    assert len(jobs)==25
    if args.action=="collect":
        scaler.check()
        checks.check()
        sign=signature()
        lock={}
        start=perf_counter()
        for i,(n,s) in enumerate(jobs):
            d=ROOT/f"{n}-{s}"
            if not (d/"complete.json").exists():
                select(n,s,d,sign)
            rec=json.loads((d/"complete.json").read_bytes())
            assert rec["source"]==sign and rec["forecast"]==b.digest(d/"all-forecasts.npz")
            lock[d.name]=b.digest(d/"complete.json")
            elapsed=perf_counter()-start
            print(f"[{i+1}/25] {n}/{s}: {elapsed:.1f}s elapsed; {elapsed*(24-i)/(i+1):.1f}s ETA",flush=True)
        (DEST/"lock.json").write_text(json.dumps(dict(source=sign,receipts=lock),indent=2),encoding="utf8")
        archive=b.ROOT/"packages/minpi-scale-five-forecasts.zip"
        with ZipFile(archive,"w",ZIP_DEFLATED) as out:
            out.write(DEST/"lock.json","lock.json")
            for name in lock:
                for file in ("all-forecasts.npz","complete.json"):
                    out.write(ROOT/name/file,name+"/"+file)
        (DEST/"forecast-sha256.txt").write_text(b.digest(archive)+"  "+archive.name+"\n",encoding="utf8")
    else:
        lock=json.loads((DEST/"lock.json").read_bytes())
        assert lock["source"]==signature() and set(lock["receipts"])=={f"{n}-{s}" for n,s in jobs}
        rows=[]
        for n,s in jobs:
            d=ROOT/f"{n}-{s}"
            rec=json.loads((d/"complete.json").read_bytes())
            assert b.digest(d/"complete.json")==lock["receipts"][d.name]
            metrics=panel.score(n,s,d/"all-forecasts.npz",rec["forecast"])
            for m,item in rec["records"].items():
                rows.append(dict(dataset=n,seed=s,task=rec["task"],method=m,**metrics[m],
                    **{k:v for k,v in item.items() if k not in ("cols","moves")}))
        pd.DataFrame(rows).to_csv(DEST/"scores.csv",index=False)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        main()
