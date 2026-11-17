"""Apply the archived predictive tolerance to calibration predictions on frozen paths."""
import argparse
import json
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b, legacy_gain as legacy, legacy_gain_receiver as receiver
from experiments.minpi_rework import quota_compress_tree as tree, quota_compress_tree_exact as exact
from experiments.minpi_rework import quota_compress_tree_run as run
from minppi_mofs.modeling import make_model


def guarded(f,val,order,min_k,target,task,scale):
    ranks=np.argsort(order)+1
    increments=tree._prefix(np.asarray(val,dtype=np.float32),*f.buffers(),ranks,f.c)
    raw=np.cumsum(increments,axis=1)
    raw+=np.broadcast_to(f.empty,(len(val),f.c)).reshape(-1,1)
    full=raw[:,-1].reshape(len(val),f.c)
    full_score=b.metrics(target,tree.as_scores(full,task),task,scale)
    threshold=.2*np.linalg.norm(full-f.empty)
    chosen=order
    trace=[]
    # Increasing width implies the first eligible path support is the sparsest.
    for k in range(min_k,len(order)+1):
        pred=raw[:,k].reshape(full.shape)
        if np.linalg.norm(pred-full)>threshold+1e-8:
            continue
        scores=b.metrics(target,tree.as_scores(pred,task),task,scale)
        good=b.eligible(scores,full_score,task)
        trace.append(dict(k=k,eligible=bool(good),**scores))
        if good and k<len(chosen):
            chosen=order[:k]
    return np.sort(chosen),trace


def select(name,seed,job):
    directory,row=job
    x,y,task,idx=legacy.read_split(directory,row,seed)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    dev=np.sort(np.r_[pool,idx["width"]])
    val=x[idx["width"]]
    scale=max(np.std(y[pool]),1e-12)
    t=perf_counter()
    model=make_model(task=row["task"],seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    f=tree.extract(model,val[:1])
    gain=model.get_booster().get_score(importance_type="gain")
    order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
    common=perf_counter()-t
    t=perf_counter()
    game=f.cache(val)
    cache_s=perf_counter()-t
    info,pred={},{}
    key=f"{name}-{seed}"
    seal=json.loads((run.OUT/key/"selection.json").read_bytes())["records"]
    for method in np.random.default_rng(seed).permutation(("exact_greedy","gain_sweep","gain_prefix")):
        t=perf_counter()
        if method=="gain_prefix":
            path_order=order
            min_k=0
        else:
            if method=="exact_greedy":
                cols,details=exact.search(game,order)
            else:
                cols,details=game.search(method,order)
            np.testing.assert_array_equal(cols,seal[method]["support"])
            path_order=np.r_[cols,details["deletion_order"][::-1]].astype(int)
            min_k=len(cols)
        cols,trace=guarded(f,val,path_order,min_k,y[idx["width"]],task,scale)
        search=perf_counter()-t+(cache_s if method!="gain_prefix" else 0.)
        t=perf_counter()
        scores=tree.as_scores(f.predict(x[np.ix_(idx["test"],cols)],cols),task)
        sparse_s=perf_counter()-t
        base=dict(k=len(cols),p=x.shape[1],support=cols.tolist(),selection_s=common+search,
            predictive_checks=len(trace),trace=trace)
        pred[method]=scores
        info[method]=dict(**base,pipeline_s=common+search+sparse_s)
        t=perf_counter()
        ref=receiver.predict(x,y,dev,idx["test"],cols,row["task"],seed)
        refit_s=perf_counter()-t
        pred[method+"_refit"]=ref
        info[method+"_refit"]=dict(**base,pipeline_s=common+search+refit_s)
        print(key,method,len(cols),f"selection {common+search:.2f}s",flush=True)
    sub=run.OUT/"guard"/key
    sub.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(sub/"predictions.npz",**pred)
    (sub/"selection.json").write_text(json.dumps(dict(task=task,seed=seed,p=x.shape[1],scale=scale,
        records=info,forecast_hash=b.digest(sub/"predictions.npz"))),encoding="utf8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["select","score"])
    parser.add_argument("--seeds",nargs="+",type=int,default=legacy.CONFIG["seeds"])
    args=parser.parse_args()
    jobs=legacy.jobs()
    if args.action=="score":
        run.score(jobs,args.seeds,series="guard")
    else:
        with threadpool_limits(1):
            run.check()
            # Warm the compiled solver before measured calls.
            from experiments.minpi_rework.quota_compress_tree_exact_run import warm
            warm()
            for seed in args.seeds:
                for name in legacy.CONFIG["datasets"]:
                    select(name,seed,jobs[name,seed])


if __name__=="__main__":
    main()
