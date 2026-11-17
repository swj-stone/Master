"""Historical six-source comparison of literal signed min votes and mean-panel safety."""
import argparse
import json
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import quota_coalition_mean as cm,quota_compress_tree as trees
from experiments.minpi_rework import quota_compress_tree_cost as cost,quota_compress_rescue_run as starts
from minppi_mofs.modeling import make_model


OUT=b.ROOT/"results/minpi_rework/coalition-mean"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/coalition-mean"


def noise_forecast(f,x,cols,assign,mask):
    result=np.empty((len(x),f.c))
    for k,feature_mask in enumerate(mask):
        which=np.flatnonzero(assign==k)
        use=np.intersect1d(cols,feature_mask)
        if len(which):
            result[which]=f.predict(x[np.ix_(which,use)],use)
    return result


def select(name,seed,job,variant=False):
    directory,row=job
    x,y,task,idx=legacy.read_split(directory,row,seed)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    val=x[idx["width"]]
    xt=x[idx["test"]]
    t=perf_counter()
    model=make_model(task=row["task"],seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    f=trees.extract(model,val[:1])
    g=f.cache(val)
    shared=perf_counter()-t
    t=perf_counter()
    score=cm.votes(g)
    vote_s=perf_counter()-t
    t=perf_counter()
    d=(g.a@g.incidence).toarray()
    ablation=np.einsum("ij,ij->j",d,d)
    ab_s=perf_counter()-t
    np.testing.assert_allclose(score[-1],ablation,rtol=1e-7,atol=1e-8)
    gain=model.get_booster().get_score(importance_type="gain")
    native=np.array([gain.get(f"f{j}",0.) for j in range(g.p)])
    ranks={m:np.lexsort((-np.arange(g.p),-s)) for m,s in
        dict(min=score.min(axis=0),pool=score.mean(axis=0),last=ablation,gain=native,greedy=native).items()}
    if variant:
        t=perf_counter()
        ranks=cm.refined(g,score)
        vote_s+=perf_counter()-t
    t=perf_counter()
    q,w,norm,a=cm.panels(g)
    panel_s=perf_counter()-t
    diff=np.asarray(g.incidence.sum(axis=1)).ravel()
    assert np.all(diff>0)
    arms=[(m,"full") for m in ranks]+[(m,"max") for m in ranks]
    if not variant:
        arms.append(("greedy","pool"))
    records={}
    forecasts={}
    cached={}
    rng=np.random.default_rng(51118+seed)
    assignments=[rng.integers(0,16,len(xt)) for _ in q[:2]]
    masks=[[np.sort(rng.choice(g.p,k,replace=False)) for _ in range(16)] for k in q[:2]]
    for m,mode in arms+[("teacher","full")]:
        t=perf_counter()
        if m=="teacher":
            keep=np.arange(g.p)
        elif mode=="full":
            if m=="greedy":
                keep=starts.starting(g,ranks[m],"exact")
            else:
                keep=cost.compiled(g,ranks[m])[0]
        else:
            keep,_,_=cm.solve(g,a,ranks[m],m=="greedy",mode)
        search_s=perf_counter()-t
        setup=vote_s if m in ("min","pool") or m in ranks and variant else ab_s if m=="last" else 0.
        key=m+"_"+mode
        lost=np.setdiff1d(np.arange(g.p),keep)
        dead=np.asarray(g.incidence[:,lost].sum(axis=1)).ravel()>0
        err=np.array([np.linalg.norm(g.a[:,dead]@s[dead])/n for s,n in zip(w,norm)])
        if mode=="max":
            assert err.max()<=.2+1e-8
        if mode=="full":
            assert err[-1]<=.2+1e-8
        if mode=="pool":
            assert np.sqrt(np.mean(err*err))<=.2+1e-8
        if tuple(keep) not in cached:
            t=perf_counter()
            normal=f.predict(xt[:,keep],keep)
            pred_s=perf_counter()-t
            mean=cm.predict(f,xt,keep,w)
            np.testing.assert_allclose(mean[:,:,-1],normal,rtol=1e-6,atol=1e-7)
            result={"normal":trees.as_scores(normal,task)}
            for i in (0,1):
                result["mean"+str(i)]=trees.as_scores(mean[:,:,i],task)
                in_mask=noise_forecast(f,xt,keep,assignments[i],masks[i])
                result["outage"+str(i)]=trees.as_scores(in_mask,task)
            cached[tuple(keep)]=pred_s,result
        pred_s,result=cached[tuple(keep)]
        forecasts.update({key+"_"+s:val for s,val in result.items()})
        records[key]=dict(p=g.p,k=len(keep),cols=keep.tolist(),err25=err[0],err50=err[1],err100=err[2],
            search_s=search_s,rank_s=setup,panels_s=panel_s if mode!="full" else 0.,shared_s=shared,
            pipeline_s=shared+search_s+setup+(panel_s if mode!="full" else 0.)+pred_s)
    sub=(OUT/"variant" if variant else OUT)/f"{name}-{seed}"
    sub.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(sub/"forecasts.npz",**forecasts)
    rec=dict(task=task,records=records,signed_votes=score.tolist(),negative_counts=(score<(-1e-12*g.norm**2)).sum(axis=1).tolist(),
        min_negative=int((score.min(axis=0)<(-1e-12*g.norm**2)).sum()),
        scale=max(float(np.std(y[pool])),1e-12),forecast=b.digest(sub/"forecasts.npz"))
    (sub/"receipt.json").write_text(json.dumps(rec),encoding="utf8")
    print(name,seed,[(k,records[k+"_full"]["k"],records[k+"_max"]["k"]) for k in ranks],
        "negatives",rec["min_negative"],flush=True)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=["select","score"])
    cli.add_argument("--all",action="store_true")
    cli.add_argument("--variant",action="store_true")
    args=cli.parse_args()
    jobs=legacy.jobs()
    tag=("variant-" if args.variant else "")+("all" if args.all else "pilot")
    out=OUT/"variant" if args.variant else OUT
    work=[(n,s) for s in (legacy.CONFIG["seeds"] if args.all else [42]) for n in legacy.CONFIG["datasets"]]
    DEST.mkdir(parents=True,exist_ok=True)
    if args.action=="select":
        cm.check()
        locked={}
        for n,s in work:
            sub=out/f"{n}-{s}"
            if not (sub/"receipt.json").exists():
                select(n,s,jobs[n,s],args.variant)
            locked[sub.name]=b.digest(sub/"receipt.json")
        (DEST/f"lock-{tag}.json").write_text(json.dumps(locked,indent=2),encoding="utf8")
    else:
        locked=json.loads((DEST/f"lock-{tag}.json").read_bytes())
        assert set(locked)=={f"{n}-{s}" for n,s in work}
        rows=[]
        for n,s in work:
            sub=out/f"{n}-{s}"
            assert b.digest(sub/"receipt.json")==locked[sub.name]
            rec=json.loads((sub/"receipt.json").read_bytes())
            assert b.digest(sub/"forecasts.npz")==rec["forecast"]
            directory,row=jobs[n,s]
            path=directory/row["test_relpath"]
            assert b.digest(path)==row["test_file_sha256"]
            y=pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()
            with np.load(sub/"forecasts.npz") as preds:
                for method,arm in rec["records"].items():
                    for mode in ("normal","mean0","mean1","outage0","outage1"):
                        metrics=b.metrics(y,preds[method+"_"+mode],rec["task"],rec["scale"])
                        rows.append(dict(dataset=n,seed=s,method=method,mode=mode,min_negative=rec["min_negative"],
                            **{k:v for k,v in arm.items() if k!="cols"},**metrics))
        pd.DataFrame(rows).to_csv(DEST/f"scores-{tag}.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
