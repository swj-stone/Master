"""Blind forecast collection for registered tree compression; score is a separate invocation."""
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
from xgboost import DMatrix
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as tree,quota_compress_tree_exact as exact
from experiments.minpi_rework import quota_compress_tree_guard as guard,quota_compress_tree_exact_run as timing
from experiments.minpi_rework import quota_compress_tree_run as historical,validation_panel,validation_wide
from minppi_mofs import modeling


OUT=b.ROOT/"results/minpi_rework/quota-tree-confirmation"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/quota-tree-confirmation"
ARMS=("full","gain_prefix","gain_sweep","dense","cache_only","directions")
ALL=(*ARMS,"guard_directions","guard_gain_sweep","guard_gain_prefix")


def jobs():
    for panel,source in (("core",validation_panel),("wide",validation_wide)):
        for name in source.SOURCES:
            for seed in range(42,47):
                yield panel,source,name,seed


def signatures():
    modules=[tree,exact,guard,timing,historical,validation_panel,validation_wide,b,legacy,modeling]
    paths=[Path(m.__file__) for m in modules]+[Path(__file__),Path(__file__).with_suffix(".md"),
        validation_panel.MANIFEST,Path(validation_wide.__file__).with_name("validation_wide_sources.json"),legacy.ORIGINAL]
    return {p.name:b.digest(p) for p in paths}


def select_one(source,name,seed,sign):
    x,y,task,idx=source.split(name,seed)
    outer=y[idx["test"]]
    assert (outer==-1).all() if task=="classification" else np.isnan(outer).all()
    val=x[idx["width"]]
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
    scale=max(float(np.std(y[pool])),1e-12)
    t=perf_counter()
    model=modeling.make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    fit_s=perf_counter()-t
    t=perf_counter()
    f=tree.extract(model,val[:1])
    parse_s=perf_counter()-t
    t=perf_counter()
    gain=model.get_booster().get_score(importance_type="gain")
    order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
    rank_s=perf_counter()-t
    t=perf_counter()
    game=f.cache(val)
    cache_s=perf_counter()-t
    full=np.arange(x.shape[1])
    raw=f.predict(val,full)
    native=model.get_booster().predict(DMatrix(val),output_margin=True,strict_shape=True)
    tol=3e-5*max(1.,float(np.abs(raw).max()))
    np.testing.assert_allclose(raw,native,rtol=2e-5,atol=tol)
    pred=tree.as_scores(raw,task)
    native_pred=model.predict(val) if task=="regression" else model.predict_proba(val)
    np.testing.assert_allclose(pred,native_pred,rtol=2e-5,atol=tol if task=="regression" else 2e-5)
    np.testing.assert_allclose(raw-f.empty,game.full.reshape(raw.shape),atol=1e-9*max(1.,np.abs(raw).max()))
    runtimes={m:[] for m in ALL}
    support={}
    path={}
    stats={}
    rng=np.random.default_rng(seed)
    for _ in range(3):
        for m in rng.permutation(ARMS):
            t=perf_counter()
            if m=="full":
                cols,diag=full,{}
            elif m=="gain_prefix":
                cols,diag=f.gain_prefix(val,order,.2)
            elif m in ("dense","gain_sweep"):
                cols,diag=game.search("exact_greedy" if m=="dense" else m,order)
            else:
                cols,diag=exact.search(game,order,m=="directions")
            runtimes[m].append(perf_counter()-t)
            if m in support:
                np.testing.assert_array_equal(support[m],cols)
            support[m]=cols
            path[m]=diag.pop("deletion_order",[])
            stats[m]=diag
        assert path["directions"]==path["dense"]==path["cache_only"]
        for m in rng.permutation(("directions","gain_sweep","gain_prefix")):
            t=perf_counter()
            if m=="gain_prefix":
                ranking=order
                k=0
            else:
                ranking=np.r_[support[m],path[m][::-1]].astype(int)
                k=len(support[m])
            cols,trace=guard.guarded(f,val,ranking,k,y[idx["width"]],task,scale)
            arm="guard_"+m
            runtimes[arm].append(perf_counter()-t+runtimes[m][-1])
            if arm in support:
                np.testing.assert_array_equal(support[arm],cols)
            support[arm]=cols
            stats[arm]=dict(predictive_checks=len(trace))
    records,forecasts={},{}
    for m in ALL:
        cols=support[m]
        cal=f.predict(val[:,cols],cols)
        np.testing.assert_allclose(cal,historical.cached_margin(f,game,val,cols),atol=2e-9*max(1.,np.abs(raw).max()))
        loss=np.linalg.norm(raw-cal)/max(1e-100,game.norm)
        assert loss<=.2+2e-8
        t=perf_counter()
        forecasts[m]=tree.as_scores(f.predict(x[np.ix_(idx["test"],cols)],cols),task)
        pred_s=perf_counter()-t
        ranked=m!="full"
        cached=m not in ("full","gain_prefix","guard_gain_prefix")
        search_s=float(np.median(runtimes[m]))
        postfit=parse_s+rank_s*ranked+cache_s*cached+search_s+pred_s
        records[m]=dict(p=len(full),k=len(cols),support=cols.tolist(),search_s=search_s,postfit_s=postfit,
            fit_s=fit_s,parse_s=parse_s,rank_s=rank_s*ranked,cache_s=cache_s*cached,
            pipeline_s=postfit+fit_s,pred_s=pred_s,search_repeats=runtimes[m],
            relative_error=loss,**stats[m])
    for m in ("dense","cache_only"):
        np.testing.assert_array_equal(forecasts[m],forecasts["directions"])
    sub=OUT/f"{name}-{seed}"
    sub.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(sub/"forecasts.npz",**forecasts)
    receipt=dict(records=records,split={k:v.tolist() for k,v in idx.items()},task=task,source=sign,
        path=path["directions"],forecast=b.digest(sub/"forecasts.npz"),
        max_native_margin_error=float(np.max(np.abs(raw-native))))
    temp=sub/"selection.tmp"
    temp.write_text(json.dumps(receipt),encoding="utf8")
    temp.replace(sub/"selection.json")


def collect():
    DEST.mkdir(exist_ok=True,parents=True)
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<7)):
            raise ctypes.WinError(ctypes.get_last_error())
    historical.check()
    timing.warm()
    sign=signatures()
    lock={}
    start=perf_counter()
    for index,(_,source,name,seed) in enumerate(jobs()):
        t=perf_counter()
        sub=OUT/f"{name}-{seed}"
        receipt=sub/"selection.json"
        if not receipt.exists():
            select_one(source,name,seed,sign)
        rec=json.loads(receipt.read_bytes())
        assert rec["source"]==sign and rec["forecast"]==b.digest(sub/"forecasts.npz")
        lock[sub.name]=b.digest(receipt)
        elapsed=perf_counter()-start
        print(f"[{index+1}/65] {name}/{seed} {perf_counter()-t:.2f}s | {elapsed:.1f}s | ETA {elapsed*(64-index)/(index+1):.1f}s",flush=True)
    (DEST/"lock.json").write_text(json.dumps(dict(source=sign,receipts=lock),indent=2),encoding="utf8")
    archive=b.ROOT/"packages/minpi-quota-tree-forecasts.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        z.write(DEST/"lock.json","lock.json")
        for key in lock:
            for file in ("selection.json","forecasts.npz"):
                z.write(OUT/key/file,key+"/"+file)
    (DEST/"forecast-sha256.txt").write_text(b.digest(archive)+"  "+archive.name+"\n",encoding="utf8")


def score():
    lock=json.loads((DEST/"lock.json").read_bytes())
    assert signatures()==lock["source"]
    assert set(lock["receipts"])=={f"{n}-{s}" for _,_,n,s in jobs()}
    rows=[]
    for panel,source,name,seed in jobs():
        sub=OUT/f"{name}-{seed}"
        receipt=sub/"selection.json"
        assert b.digest(receipt)==lock["receipts"][sub.name]
        rec=json.loads(receipt.read_bytes())
        metrics=source.score(name,seed,sub/"forecasts.npz",rec["forecast"])
        for m in ALL:
            rows.append(dict(panel=panel,dataset=name,seed=seed,task=rec["task"],method=m,**rec["records"][m],**metrics[m]))
    pd.DataFrame(rows).to_csv(DEST/"scores.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["collect","score"])
    with threadpool_limits(1):
        (collect if parser.parse_args().action=="collect" else score)()
