"""Matched native exact greedy kernels with and without minimum directional screening."""
import ctypes
import json
import os
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as tree,quota_compress_tree_confirmation as conf
from minppi_mofs.modeling import make_model


@njit(cache=True)
def initialize(n,ap,ai,av,ip,ii,p):
    d=np.zeros((p,n))
    for group in range(len(ip)-1):
        for pos in range(ip[group],ip[group+1]):
            j=ii[pos]
            for z in range(ap[group],ap[group+1]):
                d[j,ai[z]]+=av[z]
    return d


@njit(cache=True)
def direct(n,ap,ai,av,ip,ii,gp,gi,order,limit,block):
    p=len(order)
    d=initialize(n,ap,ai,av,ip,ii,p)
    residual=np.zeros(n)
    active=np.ones(len(ap)-1,np.bool_)
    keep=np.ones(p,np.bool_)
    path=np.empty(p,np.int32)
    checked=0
    count=0
    for _ in range(p):
        best=limit*limit
        win=-1
        # Visiting lower Gain priority first gives the original tie order.
        for pos in range(p-1,-1,-1):
            j=order[pos]
            if keep[j]:
                checked+=1
                cost=0.
                if block:
                    for start in range(0,n,32):
                        for row in range(start,min(n,start+32)):
                            v=residual[row]+d[j,row]
                            cost+=v*v
                        if cost>best:
                            break
                else:
                    for row in range(n):
                        v=residual[row]+d[j,row]
                        cost+=v*v
                if cost<best or (cost==best and win<0):
                    best=cost
                    win=j
        if win<0:
            break
        residual+=d[win]
        keep[win]=False
        path[count]=win
        count+=1
        for ptr in range(gp[win],gp[win+1]):
            group=gi[ptr]
            if active[group]:
                active[group]=False
                for pos in range(ip[group],ip[group+1]):
                    j=ii[pos]
                    if keep[j]:
                        for z in range(ap[group],ap[group+1]):
                            d[j,ai[z]]-=av[z]
    return path[:count],checked


@njit(cache=True)
def sliced(n,ap,ai,av,ip,ii,gp,gi,order,limit,bands):
    p=len(order)
    d=initialize(n,ap,ai,av,ip,ii,p)
    squared=np.zeros(p)
    den=np.zeros((p,bands))
    binmap=np.zeros((p,n),np.int8)
    for j in range(p):
        for v in d[j]:
            squared[j]+=v*v
        if bands:
            rms=np.sqrt(squared[j]/n)
            for i in range(n):
                v=d[j,i]
                h=0 if bands==1 else int(v>0)*2+int(abs(v)>rms)
                den[j,h]+=v*v
                binmap[j,i]=h
    mass=np.zeros((len(ii),bands))
    if bands:
        for g in range(len(ip)-1):
            for pos in range(ip[g],ip[g+1]):
                j=ii[pos]
                for z in range(ap[g],ap[g+1]):
                    row=ai[z]
                    mass[pos,binmap[j,row]]+=av[z]*d[j,row]
    acc=den.copy()
    residual=np.zeros(n)
    delta=np.zeros(n)
    active=np.ones(len(ap)-1,np.bool_)
    keep=np.ones(p,np.bool_)
    known=keep.copy()
    path=np.empty(p,np.int32)
    ranks=np.argsort(order)
    calls=0
    count=0
    tol=2e-12*max(1.,(limit/.2)**2)
    for _ in range(p):
        rr=np.sum(residual**2)
        fixed=np.full(p,rr)
        if count:
            for g in range(len(ip)-1):
                if active[g]:
                    dot=0.
                    for z in range(ap[g],ap[g+1]):
                        dot+=residual[ai[z]]*av[z]
                    for pos in range(ip[g],ip[g+1]):
                        fixed[ii[pos]]+=2*dot
        lower=np.full(p,np.inf)
        for j in range(p):
            if keep[j]:
                bound=0.
                if known[j]:
                    bound=squared[j]
                else:
                    for h in range(bands):
                        if den[j,h]>0.:
                            bound+=acc[j,h]*acc[j,h]/den[j,h]
                lower[j]=fixed[j]+bound
        rank=np.argsort(lower)
        win=-1
        best=limit*limit
        for j in rank:
            if lower[j]>best+tol:
                break
            if not known[j]:
                calls+=1
                delta[:]=0.
                for ptr in range(gp[j],gp[j+1]):
                    g=gi[ptr]
                    if active[g]:
                        for z in range(ap[g],ap[g+1]):
                            delta[ai[z]]+=av[z]
                squared[j]=np.sum(delta**2)
                known[j]=True
            cost=fixed[j]+squared[j]
            if cost<best or (cost==best and (win<0 or ranks[j]>ranks[win])):
                best=cost
                win=j
        if win<0:
            break
        keep[win]=False
        path[count]=win
        count+=1
        for ptr in range(gp[win],gp[win+1]):
            g=gi[ptr]
            if active[g]:
                active[g]=False
                for z in range(ap[g],ap[g+1]):
                    residual[ai[z]]+=av[z]
                for pos in range(ip[g],ip[g+1]):
                    j=ii[pos]
                    known[j]=False
                    for h in range(bands):
                        acc[j,h]-=mass[pos,h]
    return path[:count],calls


def compare(game,order,oracle,seed):
    rng=np.random.default_rng(seed)
    rows=[]
    for rep in range(-1,3):
        for m in rng.permutation(("dense","partial","zero","one","four")):
            t=perf_counter()
            a=game.a
            inc=game.incidence
            by=inc.tocsc()
            limit=.2*game.norm+1e-12*max(1.,game.norm)
            args=(len(game.full),a.indptr,a.indices,a.data,inc.indptr,inc.indices,by.indptr,by.indices,order,limit)
            path,calls=direct(*args,m=="partial") if m in ("dense","partial") else sliced(*args,{"zero":0,"one":1,"four":4}[m])
            elapsed=perf_counter()-t
            np.testing.assert_array_equal(path,oracle)
            if rep>=0:
                rows.append(dict(method=m,repeat=rep,seconds=elapsed,checked=calls))
    return rows


def run():
    rows=[]
    for panel,source,name,seed in conf.jobs():
        rec=json.loads((conf.OUT/f"{name}-{seed}"/"selection.json").read_bytes())
        x,y,task,idx=source.split(name,seed)
        val=x[idx["width"]]
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model=make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        f=tree.extract(model,val[:1])
        g=f.cache(val)
        gain=model.get_booster().get_score(importance_type="gain")
        order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
        for item in compare(g,order,rec["path"],seed):
            item.update(dataset=name,seed=seed,panel=panel)
            rows.append(item)
        print(name,seed,"native exact paths verified",flush=True)
    pd.DataFrame(rows).to_csv(conf.DEST/"compiled-exact-partial.csv",index=False)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<7)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        run()
