"""Global coalition-size projection votes fused into direct partial-distance search."""
import ctypes
import json
import os
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as trees,quota_compress_tree_confirmation as conf
from experiments.minpi_rework.quota_compress_tree_native_exact import initialize,direct
from minppi_mofs.modeling import make_model


@njit(cache=True)
def angle(n,ap,ai,av,ip,ii,gp,gi,order,limit,sizes):
    p=len(order)
    d=initialize(n,ap,ai,av,ip,ii,p)
    count_s=len(sizes)
    group_count=len(ap)-1
    u=np.zeros((count_s,n))
    for s,k in enumerate(sizes):
        for g in range(group_count):
            deg=ip[g+1]-ip[g]
            prob=1.
            for i in range(deg):
                prob*=max(0,k-i)/(p-i)
            for z in range(ap[g],ap[g+1]):
                u[s,ai[z]]+=prob*av[z]
        scale=np.sqrt(np.sum(u[s]*u[s]))
        if scale>0.:
            u[s]/=scale
    weight=np.zeros((group_count,count_s))
    projected=np.zeros((p,count_s))
    cur=np.zeros(count_s)
    for g in range(group_count):
        for s in range(count_s):
            val=0.
            for z in range(ap[g],ap[g+1]):
                val+=u[s,ai[z]]*av[z]
            weight[g,s]=val
            for pos in range(ip[g],ip[g+1]):
                projected[ii[pos],s]+=val
    sr=np.zeros(p)
    for j in range(p):
        for val in d[j]:
            sr[j]+=val*val
    residual=np.zeros(n)
    active=np.ones(group_count,np.bool_)
    keep=np.ones(p,np.bool_)
    path=np.empty(p,np.int32)
    count=0
    rr=0.
    broad=0
    extra=0
    checks=0
    tol=2e-12*max(1.,(limit/.2)**2)
    for _ in range(p):
        best=limit*limit
        win=-1
        for pos in range(p-1,-1,-1):
            j=order[pos]
            if not keep[j]:
                continue
            # Cauchy's lower bound in the whole residual space is the zero-axis vote.
            cutoff=rr+sr[j]-best-tol
            if cutoff>0 and cutoff*cutoff>4*rr*max(0.,sr[j]):
                broad+=1
                continue
            veto=False
            for s in range(count_s):
                v=projected[j,s]
                a=cur[s]
                lhs=cutoff+2*a*v
                if lhs>0. and lhs*lhs>4*max(0.,rr-a*a)*max(0.,sr[j]-v*v):
                    veto=True
                    break
            if veto:
                extra+=1
                continue
            checks+=1
            cost=0.
            for start in range(0,n,32):
                for row in range(start,min(n,start+32)):
                    val=residual[row]+d[j,row]
                    cost+=val*val
                if cost>best:
                    break
            if cost<best or (cost==best and win<0):
                best=cost
                win=j
        if win<0:
            break
        rr=best
        path[count]=win
        count+=1
        keep[win]=False
        residual+=d[win]
        for ptr in range(gp[win],gp[win+1]):
            g=gi[ptr]
            if active[g]:
                active[g]=False
                for s in range(count_s):
                    cur[s]+=weight[g,s]
                for pos in range(ip[g],ip[g+1]):
                    j=ii[pos]
                    if not keep[j]:
                        continue
                    for s in range(count_s):
                        projected[j,s]-=weight[g,s]
                    update=0.
                    for z in range(ap[g],ap[g+1]):
                        row=ai[z]
                        val=av[z]
                        update+=val*(val-2*d[j,row])
                        d[j,row]-=val
                    sr[j]+=update
    return path[:count],broad,extra,checks


def run():
    rows=[]
    for panel,source,name,seed in conf.jobs():
        rec=json.loads((conf.OUT/f"{name}-{seed}"/"selection.json").read_bytes())
        x,y,task,idx=source.split(name,seed)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        val=x[idx["width"]]
        problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model=make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        g=trees.extract(model,val[:1]).cache(val)
        gain=model.get_booster().get_score(importance_type="gain")
        order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
        rng=np.random.default_rng(seed)
        for rep in range(-1,3):
            for m in rng.permutation(("partial","null","one","three")):
                t=perf_counter()
                inc=g.incidence
                by=inc.tocsc()
                a=g.a
                args=(len(g.full),a.indptr,a.indices,a.data,inc.indptr,inc.indices,by.indptr,by.indices,
                    order,.2*g.norm+1e-12*max(1.,g.norm))
                if m=="partial":
                    path,checks=direct(*args,True)
                    zero=votes=0
                else:
                    p=g.p
                    sizes=dict(null=[],one=[p],three=[max(1,p//4),max(1,p//2),p])[m]
                    path,zero,votes,checks=angle(*args,np.array(sizes,np.int64))
                elapsed=perf_counter()-t
                np.testing.assert_array_equal(path,rec["path"])
                if rep>=0:
                    rows.append(dict(panel=panel,dataset=name,seed=seed,method=m,repeat=rep,seconds=elapsed,
                        zero_vetoes=zero,directional_vetoes=votes,remaining_checks=checks))
        print(name,seed,"angle-certified paths equal",flush=True)
    pd.DataFrame(rows).to_csv(conf.DEST/"angle.csv",index=False)


def summarize():
    f=pd.read_csv(conf.DEST/"angle.csv",keep_default_na=False)
    results=f.groupby(["panel","dataset","seed","method"]).median(numeric_only=True)
    rows=[]
    for metric in ("seconds","remaining_checks"):
        wide=results[metric].unstack()
        for method,reference in [("three","one"),("three","partial"),("one","partial"),
                                 ("null","partial"),("three","null")]:
            for panel in ("all","core","wide"):
                ratio=wide[method]/wide[reference]
                if panel!="all":
                    ratio=ratio.xs(panel,level="panel")
                v=np.log(ratio).groupby("dataset").mean().to_numpy()
                rng=np.random.default_rng(5002)
                draws=rng.choice(v,(20000,len(v)),replace=True).mean(axis=1)
                point,lo,hi=np.exp([v.mean(),*np.quantile(draws,[.025,.975])])
                rows.append(dict(metric=metric,panel=panel,method=method,reference=reference,
                    value=point,low=lo,high=hi))
    frame=pd.DataFrame(rows)
    frame.to_csv(conf.DEST/"angle-contrasts.csv",index=False)
    print(frame[frame.panel=="all"].to_string(index=False))


if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--summarize",action="store_true")
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<7)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        (summarize if cli.parse_args().summarize else run)()
