"""Supervised proper-loss votes on mean coalition predictions, using calibration labels only."""
import argparse
import json
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import quota_coalition_mean as cm,quota_compress_tree as tree,quota_compress_tree_cost as cost
from minppi_mofs.modeling import make_model


OUT=b.ROOT/"results/minpi_rework/coalition-proper"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/coalition-mean"


def loss(raw,y,task):
    if task=="regression":
        return np.mean((raw[:,0,:]-y[:,None])**2,axis=0)
    if raw.shape[1]==1:
        z=raw[:,0,:]
        return np.mean(np.logaddexp(0.,z)-y[:,None]*z,axis=0)
    return np.mean(logsumexp(raw,axis=1)-raw[np.arange(len(y)),y.astype(int),:],axis=0)


def proper(g,base,y,task):
    deg=np.diff(g.incidence.indptr)
    p=g.p
    votes=[]
    for sizes in np.array_split(np.arange(p),3):
        w=np.mean([cm.contain(p-1,s,deg) for s in sizes],axis=0)
        add=np.mean([cm.contain(p-1,s,deg-1) for s in sizes],axis=0)
        drop=(g.a@g.incidence.multiply(w[:,None])).toarray()
        delta=(g.a@g.incidence.multiply(add[:,None])).toarray()
        pre=(g.a@w)[:,None]-drop
        m0=pre.reshape(len(y),len(base),p)+base[None,:,None]
        m1=m0+delta.reshape(m0.shape)
        votes.append(loss(m0,y,task)-loss(m1,y,task))
    delta=(g.a@g.incidence).toarray()
    full=g.full.reshape(len(y),len(base),1)+base[None,:,None]
    last=loss(full-delta.reshape(len(y),len(base),p),y,task)-loss(full,y,task)
    return np.asarray(votes),last,np.sum(delta*delta,axis=0)


def select(n,s,job):
    d,row=job
    x,y,task,idx=legacy.read_split(d,row,s)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    val=x[idx["width"]]
    t=perf_counter()
    model=make_model(task=row["task"],seed=s,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    f=tree.extract(model,val[:1])
    g=f.cache(val)
    shared=perf_counter()-t
    t=perf_counter()
    scores,last,energy=proper(g,f.empty,y[idx["width"]],task)
    rank_s=perf_counter()-t
    j=np.arange(g.p)
    sorted_vote=np.sort(scores,axis=0)
    m=scores.min(axis=0)
    ordered={nm:np.lexsort((-j,-v)) for nm,v in dict(min=m,pool=scores.mean(axis=0),last=last,energy=energy).items()}
    ordered["harm"]=np.lexsort((-j,-energy,m < -1e-12))
    ordered["lex"]=np.lexsort((-j,-sorted_vote[2],-sorted_vote[1],-sorted_vote[0]))
    rec={}
    forecasts={}
    for name,order in ordered.items():
        t=perf_counter()
        keep=cost.compiled(g,order)[0]
        elapsed=perf_counter()-t
        for budget,cols in (("guard",keep),("half",np.sort(order[:int(np.ceil(g.p*.5))])),("quarter",np.sort(order[:int(np.ceil(g.p*.25))]))):
            cols=np.asarray(cols)
            t=perf_counter()
            raw=f.predict(x[np.ix_(idx["test"],cols)],cols)
            key=name+"_"+budget
            forecasts[key]=tree.as_scores(raw,task)
            rec[key]=dict(k=len(cols),p=g.p,cols=cols.tolist(),rank_assembly_s=rank_s,shared_s=shared,
                sweep_s=elapsed if budget=="guard" else 0.,prediction_s=perf_counter()-t)
    sub=OUT/f"{n}-{s}"
    sub.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(sub/"forecasts.npz",**forecasts)
    info=dict(task=task,records=rec,negative=int(np.sum(m < -1e-12)),votes=scores.tolist(),
        scale=max(float(np.std(y[pool])),1e-12),forecast=b.digest(sub/"forecasts.npz"))
    (sub/"receipt.json").write_text(json.dumps(info),encoding="utf8")
    print(n,s,[(name,rec[name+"_guard"]["k"]) for name in ordered],info["negative"],flush=True)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=["select","score"])
    cli.add_argument("--all",action="store_true")
    args=cli.parse_args()
    alljobs=legacy.jobs()
    tag="all" if args.all else "pilot"
    jobs=[(n,s) for s in (legacy.CONFIG["seeds"] if args.all else [42]) for n in legacy.CONFIG["datasets"]]
    if args.action=="select":
        lock={}
        for n,s in jobs:
            sub=OUT/f"{n}-{s}"
            if not (sub/"receipt.json").exists():
                select(n,s,alljobs[n,s])
            lock[sub.name]=b.digest(sub/"receipt.json")
        (DEST/f"proper-lock-{tag}.json").write_text(json.dumps(lock,indent=2),encoding="utf8")
    else:
        lock=json.loads((DEST/f"proper-lock-{tag}.json").read_bytes())
        assert set(lock)=={f"{n}-{s}" for n,s in jobs}
        results=[]
        for n,s in jobs:
            sub=OUT/f"{n}-{s}"
            assert b.digest(sub/"receipt.json")==lock[sub.name]
            rec=json.loads((sub/"receipt.json").read_bytes())
            assert b.digest(sub/"forecasts.npz")==rec["forecast"]
            d,row=alljobs[n,s]
            path=d/row["test_relpath"]
            assert b.digest(path)==row["test_file_sha256"]
            y=pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()
            with np.load(sub/"forecasts.npz") as pred:
                for arm,item in rec["records"].items():
                    results.append(dict(dataset=n,seed=s,arm=arm,negative=rec["negative"],
                        **{k:v for k,v in item.items() if k!="cols"},**b.metrics(y,pred[arm],rec["task"],rec["scale"])))
        pd.DataFrame(results).to_csv(DEST/f"proper-{tag}.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
