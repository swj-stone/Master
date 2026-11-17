"""Develop all-size input compression on declared, already opened panels."""
import argparse
import ctypes
import os
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as tree
from experiments.minpi_rework import validation_panel,validation_wide,validation_exchange
from minppi_mofs.availability_minpi import AvailabilityGame
from minppi_mofs.modeling import make_model


SOURCES=(validation_panel,validation_wide,validation_exchange)
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/availability"
ARMS={"min":("min",None),"pooled":("pooled",None),"intact":("intact",None),
      "gain_min":("min","gain"),"ablation_min":("min","ablation"),
      "gain_intact":("intact","gain"),"ablation_intact":("intact","ablation")}


def ratios(game,cols):
    mask=np.zeros(game.incidence.shape[1],bool)
    mask[cols]=True
    absolute=game.conditional(game.coefficients(mask))
    base=game.conditional(game.totals)
    valid=base>max(0,base.max())*1e-12
    return np.sqrt(np.maximum(0,np.divide(absolute,base,out=np.zeros_like(base),where=valid)))


def run(name,seed,api,rho,slow):
    x,y,task,idx=api.split(name,seed)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    val=x[idx["width"]]
    t=perf_counter()
    prob=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
    model=make_model(task=prob,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    f=tree.extract(model,val[:1])
    game=f.cache(val)
    common_s=perf_counter()-t
    t=perf_counter()
    availability=AvailabilityGame.from_paths(game.a,game.incidence)
    setup=perf_counter()-t
    t=perf_counter()
    gain=model.get_booster().get_score(importance_type="gain")
    gain_rank=np.lexsort((np.arange(game.p),[gain.get(f"f{j}",0.) for j in range(game.p)]))
    gain_s=perf_counter()-t
    t=perf_counter()
    marginal=(game.a@game.incidence).toarray()
    ablation=np.lexsort((np.arange(game.p),np.einsum("ij,ij->j",marginal,marginal)))
    ablation_s=perf_counter()-t
    models={}
    clocks={}
    paths={}
    for arm,(rule,rank) in ARMS.items():
        order=None if rank is None else gain_rank if rank=="gain" else ablation
        t=perf_counter()
        cols,diag=availability.search(rho,rule=rule,order=order)
        clocks[arm]=perf_counter()-t
        models[arm]=cols
        paths[arm]=diag["path"]
    if slow:
        t=perf_counter()
        cols,diag=availability.search(rho,algorithm="recount")
        clocks["recount"]=perf_counter()-t
        assert diag["path"]==paths["min"]
        models["recount"]=cols
    # Hold widths fixed while comparing policies that use different risk quotas.
    k=len(models["min"])
    for arm in ("pooled","intact"):
        _,diag=availability.search(10.,rule=arm)
        order=diag["path"]
        models[arm+"_width"]=np.array(sorted(set(range(game.p))-set(order[:game.p-k])),dtype=int)
    models["full"]=np.arange(game.p)
    test=x[idx["test"]]
    true=api._partition(name,seed)[1][idx["test"]]
    scale=max(float(np.std(y[np.r_[pool,idx["width"]]])),1e-12)
    test_game=f.cache(test)
    test_vote=AvailabilityGame.from_paths(test_game.a,test_game.incidence)
    rows=[]
    for arm,cols in models.items():
        ct=ratios(availability,cols)
        ot=ratios(test_vote,cols)
        raw=f.predict(test[:,cols],cols)
        native=b.metrics(true,tree.as_scores(raw,task),task,scale)
        rows.append(dict(dataset=name,seed=seed,method=arm,p=game.p,k=len(cols),task=task,groups=game.a.shape[1],
            nnz=availability.gram.nnz,build_s=setup,shared_s=common_s,search_s=clocks.get(arm,np.nan),
            rank_s=gain_s if arm.startswith("gain") else ablation_s if arm.startswith("ablation") else 0.,
            worst_cal=ct.max(),worst_outer=ot.max(),full_outer=ot[-1],
            worst_size=int(np.argmax(ot)),**native))
    return rows


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--seeds",nargs="+",type=int,default=[42])
    cli.add_argument("--names",nargs="+")
    cli.add_argument("--tag",default="pilot")
    cli.add_argument("--rho",type=float,default=.2)
    cli.add_argument("--recount",action="store_true")
    args=cli.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    matrix=np.eye(2)
    from scipy.sparse import csc_matrix,csr_matrix
    g=AvailabilityGame.from_paths(csc_matrix(matrix),csr_matrix(matrix))
    g.search()
    rows=[]
    jobs=[(a,n,s) for a in SOURCES for n in a.SOURCES if args.names is None or n in args.names for s in args.seeds]
    start=perf_counter()
    for i,(source,name,seed) in enumerate(jobs):
        t=perf_counter()
        new=run(name,seed,source,args.rho,args.recount)
        rows+=new
        pd.DataFrame(rows).to_csv(DEST/f"{args.tag}.csv",index=False)
        elapsed=perf_counter()-start
        print(f"[{i+1}/{len(jobs)}] {name}/{seed} {perf_counter()-t:.1f}s, elapsed {elapsed:.1f}s "
              f"ETA {elapsed*(len(jobs)-i-1)/(i+1):.1f}s, k "+str([r["k"] for r in new[:5]]),flush=True)


if __name__=="__main__":
    if os.name=="nt":
        ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6))
    with threadpool_limits(1):
        main()
