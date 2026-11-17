"""A predictive MinPI game for a forest with sporadically available measurements."""
import argparse
import ctypes
import os
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,csc_matrix
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework.availability_trial import SOURCES,DEST
from minppi_mofs.predictive_quota import PredictiveGame


def centered(m,y,task):
    yy=np.eye(len(m.empty))[y.astype(int)] if task=="classification" else y[:,None]
    return (yy-m.empty).reshape(-1)


def curve(g,cols):
    p=g.structure.incidence.shape[1]
    keep=np.zeros(p,bool)
    keep[cols]=True
    return g.conditional_risk(keep)[1:]*(p/np.arange(1,p+1))/max(g.null,1e-100)


def one(source,name,seed,epsilon,slow):
    x,y,task,idx=source.split(name,seed)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    t=perf_counter()
    model=rf.fit(x[pool],y[pool],task,seed)
    game=model.game(x[idx["width"]])
    setup_s=perf_counter()-t
    t=perf_counter()
    sel=PredictiveGame.from_paths(game.a,game.incidence,centered(model,y[idx["width"]],task))
    moment_s=perf_counter()-t
    p=game.p
    full_order=np.argsort(sel.initial.sum(axis=1),kind="stable") # signed intact predictive cost
    arms={"min":("min",None),"pooled":("pooled",None),"intact":("intact",None),
          "static_min":("min",full_order),"static_intact":("intact",full_order)}
    mdi=np.argsort(model.model.feature_importances_,kind="stable")
    arms.update({"mdi_min":("min",mdi),"mdi_intact":("intact",mdi)})
    chosen={}
    for arm,(rule,order) in arms.items():
        start=perf_counter()
        cols,path=sel.search(epsilon,rule,order)
        chosen[arm]=(cols,path,perf_counter()-start)
    if slow:
        start=perf_counter()
        cols,path=sel.search(epsilon,recount=True)
        assert path==chosen["min"][1]
        chosen["recount"]=(cols,path,perf_counter()-start)
    width=len(chosen["min"][0])
    for arm in ("pooled","intact","static_intact","mdi_intact"):
        cols,path=sel.search(epsilon,*arms[arm],k=width)
        chosen[arm+"_width"]=(cols,path,np.nan)
    chosen["full"]=(np.arange(p),[],np.nan)
    test=x[idx["test"]]
    truth=source._partition(name,seed)[1][idx["test"]]
    out=model.game(test)
    validated=PredictiveGame.from_paths(out.a,out.incidence,centered(model,truth,task))
    scale=max(float(np.std(y[np.r_[pool,idx["width"]]])),1e-12)
    rows=[]
    for arm,(cols,path,seconds) in chosen.items():
        test_curve=curve(validated,cols)
        cal=curve(sel,cols)
        q=lambda a,f:a[max(0,round(f*p)-1)]
        preds=model.predict(test[:,cols],cols)
        scores=b.metrics(truth,preds,task,scale)
        rows.append(dict(dataset=name,seed=seed,method=arm,p=p,k=len(cols),task=task,
             setup_s=setup_s,moment_s=moment_s,search_s=seconds,epsilon=epsilon,nnz=sel.structure.gram.nnz,
             outer_worst=test_curve.max(),outer_mean=test_curve.mean(),outer_intact=test_curve[-1],
             quarter=q(test_curve,.25),half=q(test_curve,.5),cal_worst=cal.max(),
             cal_intact=cal[-1],worst_size=int(test_curve.argmax()+1),
             cal_worst_size=int(cal.argmax()+1),support=cols.tolist(),**scores))
    return rows


def check():
    from itertools import combinations
    r=np.random.default_rng(624)
    checks=paths=0
    for p in range(1,8):
        groups=[h for d in range(1,min(3,p)+1) for h in combinations(range(p),d)]
        inc=csr_matrix([[j in h for j in range(p)] for h in groups],dtype=float)
        bits=np.array([sum(1<<j for j in h) for h in groups])
        ids=np.arange(1<<p)
        mask=np.array([[(m>>j)&1 for j in range(p)] for m in ids],bool)
        for _ in range(4):
            a=r.integers(-3,4,(3,len(groups))).astype(float)
            target=r.integers(-5,6,3)
            y=np.array([a[:,(m&bits)==bits].sum(axis=1) for m in ids])
            g=PredictiveGame.from_paths(csc_matrix(a),inc,target)
            for m,S in zip(ids,mask):
                risks=np.sum((target-y[ids&m])**2,axis=1)-np.sum((target-y)**2,axis=1)
                exact=np.array([risks[mask.sum(axis=1)==s].mean() for s in range(p+1)])
                np.testing.assert_allclose(g.conditional_risk(S),exact,atol=1e-10)
                checks+=p+1
            for mode in ("min","pooled","intact"):
                counted,_=g.ranked_path(mode,screen=False)
                screened,_=g.ranked_path(mode)
                np.testing.assert_array_equal(counted,screened)
                profiles=g.path_curves(counted)
                for depth in range(p+1):
                    remaining=np.ones(p,bool)
                    remaining[counted[:depth]]=False
                    np.testing.assert_allclose(profiles[depth],g.conditional_risk(remaining),atol=1e-9)
                for rank in (None,r.permutation(p)):
                    for k in (None,p//2):
                        direct=g.search(.02,mode,rank,k,recount=True)
                        actual=g.search(.02,mode,rank,k)
                        assert actual[1]==direct[1]
                        paths+=1
    print(f"{checks} predictive risks and {paths} incremental vs recount paths verified")


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--check",action="store_true")
    cli.add_argument("--seeds",nargs="+",type=int,default=[42])
    cli.add_argument("--names",nargs="+")
    cli.add_argument("--tag",default="predictive-pilot")
    cli.add_argument("--epsilon",type=float,default=.02)
    cli.add_argument("--recount",action="store_true")
    args=cli.parse_args()
    if args.check:
        check()
        return
    path=DEST/f"{args.tag}.csv"
    DEST.mkdir(parents=True,exist_ok=True)
    rows=[]
    jobs=[(s,n,seed) for s in SOURCES for n in s.SOURCES if args.names is None or n in args.names for seed in args.seeds]
    start=perf_counter()
    for i,(source,name,seed) in enumerate(jobs):
        t=perf_counter()
        result=one(source,name,seed,args.epsilon,args.recount)
        rows+=result
        pd.DataFrame(rows).to_csv(path,index=False)
        elapsed=perf_counter()-start
        print(f"[{i+1}/{len(jobs)}] {name}/{seed} {perf_counter()-t:.1f}s, elapsed {elapsed:.0f}s ETA {elapsed*(len(jobs)-1-i)/(i+1):.0f}s",flush=True)


if __name__=="__main__":
    if os.name=="nt":
        ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6))
    with threadpool_limits(1):
        main()
