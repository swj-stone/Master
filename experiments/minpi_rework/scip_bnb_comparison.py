"""Run and independently verify the specialized cardinality bound."""
import argparse
import ctypes
from itertools import combinations
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_rework import scip_comparison as c
from minppi_mofs import exchange_minpi as ex
from minppi_mofs.exchange_bnb import bnb_best


def check():
    count=0
    for p in range(2,10):
        rng=np.random.default_rng(323+p)
        for kind in ("psd","signed","integer","tiny","tied"):
            x=rng.normal(size=(p,p))
            gram=x.T@x if kind=="psd" else (x+x.T)/2
            d=rng.normal(size=p)
            if kind=="integer":
                gram,d=np.rint(gram),np.rint(d)
            if kind=="tiny":
                gram,d=gram*1e-15,d*1e-15
            if kind=="tied":
                gram,d=gram*0,d*0
            permutation=rng.permutation(p)
            dag=np.triu(rng.uniform(size=(p,p))<.2,1)[np.ix_(permutation,permutation)]
            for edges in (None,dag):
                for k in range(p+1):
                    support,_=bnb_best(gram,d,k,edges)
                    truth=-np.inf
                    for subset in combinations(range(p),k):
                        z=np.zeros(p,dtype=bool)
                        z[list(subset)]=True
                        if edges is not None and edges[np.ix_(~z,z)].any():
                            continue
                        truth=max(truth,ex.value(gram,d,np.flatnonzero(z)))
                    observed=ex.value(gram,d,support)
                    np.testing.assert_allclose(observed,truth,atol=max(np.abs(d).sum()+np.abs(gram).sum(),1e-100)*1e-12,rtol=1e-12)
                    count+=1
    print(f"Independently enumerated {count} signed, PSD, tied and scaled games/widths/orders.",flush=True)
    for p in (63,64,65,96,100,128):
        g=np.eye(p)
        d=np.r_[-100.-np.arange(p-9),np.arange(9.)]
        for k in (1,2,5,p-5,p-2,p-1):
            initial=np.arange(p)<k
            answer,_=bnb_best(g,d,k,initial=initial)
            np.testing.assert_array_equal(answer,np.sort(np.argsort(d)[-k:]))
        # The top item needs column zero, which is too costly for this pair.
        edges=np.zeros((p,p),dtype=bool)
        edges[0,p-1]=True
        for k in (2,p-2):
            answers=[]
            for positions in combinations(range(p),2):
                selected=np.zeros(p,dtype=bool)
                selected[list(positions)]=True
                if k!=2:
                    selected=~selected
                if not selected[-1] or selected[0]:
                    answers.append(d[selected].sum())
            chosen,_=bnb_best(g,d,k,edges)
            np.testing.assert_allclose(d[chosen].sum(),max(answers),atol=1e-12)
        tied,_=bnb_best(g,np.zeros(p),2,edges)
        assert len(tied)==2 and (p-1 not in tied or 0 in tied)
    print("Wide keep/omit checks passed through 128 columns, including boundary bits, ties and precedence.",flush=True)


def one(data,mode):
    begin=perf_counter()
    g,d,k,p,width,fixed,band,base=c.game(data)
    ready=perf_counter()
    initial=c.warm_start(g,d,k)
    t=perf_counter()
    s=(ex.Screen(np.empty(0,int),np.arange(len(d)),None,0)
       if mode=="direct" else ex.screen(g,d,k,mode))
    screen_s=perf_counter()-t
    f,kept=s.free,s.keep
    remain=k-len(kept)
    t=perf_counter()
    idx,stat=bnb_best(g[np.ix_(f,f)],d[f]-2*g[np.ix_(f,kept)].sum(axis=1),remain,s.order,initial[f])
    opt=perf_counter()-t
    local=np.sort(np.r_[kept,f[idx]])
    objective=ex.value(g,d,local)+base
    chosen=np.sort(np.r_[fixed,band[local]])
    done=perf_counter()
    return dict(mode=mode,p=p,width=width,band=len(d),free=len(f),fixed_in=len(kept),
        edges=int(s.order.sum()) if s.order is not None else 0,teacher_s=ready-begin,
        screen_s=screen_s,opt_s=opt,search_s=done-ready,total_s=done-begin,objective=objective,
        support=" ".join(map(str,chosen)),**stat)


def run():
    c.RUN.mkdir(parents=True,exist_ok=True)
    c.warm()
    stamp=c.signature()
    bnb_best(np.eye(3),np.ones(3),1)
    g,d=np.eye(4),np.arange(4,dtype=float)
    e=np.zeros((4,4),dtype=bool)
    e[3,0]=True
    bnb_best(g,d,2,e)
    jobs=c.legacy_gain.jobs()
    all_rows=[]
    target=c.RUN/"branch-and-bound.csv"
    for n in c.NAMES:
        data=c.prepare(n,jobs)
        for rep in range(3):
            rng=np.random.default_rng(9110+73*rep+c.NAMES.index(n))
            for mode in rng.permutation(c.MODES).tolist():
                row=one(data,mode)
                all_rows.append(dict(dataset=n,seed=42,repeat=rep,**row))
                print(n,rep,mode,round(row["search_s"],5),row["nodes"],flush=True)
        pd.DataFrame(all_rows).to_csv(target,index=False)
    (c.RUN/"branch-and-bound-version.json").write_text(json.dumps(stamp,indent=2),encoding="utf8")


def export():
    data=pd.read_csv(c.RUN/"branch-and-bound.csv")
    assert len(data)==72
    assert json.loads((c.RUN/"branch-and-bound-version.json").read_bytes())==json.loads(
        (c.EVIDENCE/"versions.json").read_bytes())
    truth=pd.read_csv(c.EVIDENCE/"runs.csv").query("mode=='direct' and backend=='enum'")[["dataset","repeat","objective","scale"]]
    joined=data.merge(truth,on=["dataset","repeat"],suffixes=("","_oracle"),validate="many_to_one")
    joined["oracle_gap"]=joined.objective_oracle-joined.objective
    tol=2e-10*np.maximum(abs(joined.objective_oracle),joined.scale)
    assert (abs(joined.oracle_gap)<=tol).all()
    joined.to_csv(c.EVIDENCE/"bnb-runs.csv",index=False)
    agg=joined.groupby(["dataset","mode"])[["total_s","search_s","opt_s","nodes","pruned"]].median()
    agg.to_csv(c.EVIDENCE/"bnb-by-source.csv")
    print(agg.search_s.unstack().round(5).to_string())
    common=["dataset","backend","mode","total_s","search_s"]
    timing=pd.concat([agg.reset_index().assign(backend="branch_bound")[common],
        pd.read_csv(c.EVIDENCE/"by-source.csv")[common]],ignore_index=True)
    summary=[]
    for backend,group in timing.groupby("backend"):
        reference=group[group["mode"]=="direct"].set_index("dataset")
        for mode,arm in group.groupby("mode"):
            aligned=arm.set_index("dataset").reindex(reference.index)
            row=dict(backend=backend,mode=mode)
            for metric in ("search_s","total_s"):
                ratio=aligned[metric]/reference[metric]
                row[metric+"_sum"]=aligned[metric].sum()
                row[metric+"_geometric_ratio"]=np.exp(np.log(ratio).mean())
                row[metric+"_faster_sources"]=int((ratio<1).sum())
            summary.append(row)
    pd.DataFrame(summary).to_csv(c.EVIDENCE/"policy-summary.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["check","run","export"])
    args=p.parse_args()
    logical_core=10 if args.action=="check" else 8
    if os.name=="nt" and not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<logical_core)):
        raise ctypes.WinError()
    with threadpool_limits(1):
        {"check":check,"run":run,"export":export}[args.action]()
