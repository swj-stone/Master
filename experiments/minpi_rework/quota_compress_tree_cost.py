"""Compiled Gain sweep and native teacher baselines on the signed prospective supports."""
import json
import ctypes
import os
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain as legacy,benchmark as b
from experiments.minpi_rework import quota_compress_tree_confirmation as conf,quota_compress_tree as tree
from minppi_mofs.modeling import make_model


@njit(cache=True)
def sweep(ap,ai,av,gp,gi,n,order,limit):
    p=len(order)
    keep=np.ones(p,np.bool_)
    active=np.ones(len(ap)-1,np.bool_)
    path=np.empty(p,np.int32)
    residual=np.zeros(n)
    trial=np.zeros(n)
    step=0
    for _ in range(p):
        before=step
        for pos in range(p-1,-1,-1):
            j=order[pos]
            if not keep[j]:
                continue
            trial[:]=residual
            for a in range(gp[j],gp[j+1]):
                g=gi[a]
                if active[g]:
                    for k in range(ap[g],ap[g+1]):
                        trial[ai[k]]+=av[k]
            cost=0.
            for value in trial:
                cost+=value*value
            if cost<=limit*limit:
                for a in range(gp[j],gp[j+1]):
                    active[gi[a]]=False
                path[step]=j
                step+=1
                keep[j]=False
                residual[:]=trial
        if step==before:
            break
    return np.where(keep)[0],path[:step]


def compiled(game,order):
    t=perf_counter()
    inc=game.incidence.tocsc()
    a=game.a
    cols,path=sweep(a.indptr,a.indices,a.data,inc.indptr,inc.indices,len(game.full),order,
        .2*game.norm+1e-12*max(1.,game.norm))
    return cols,path,perf_counter()-t


def main():
    sign=json.loads((conf.DEST/"lock.json").read_bytes())
    assert sign["source"]==conf.signatures()
    rows=[]
    for panel,source,name,seed in conf.jobs():
        key=f"{name}-{seed}"
        sub=conf.OUT/key
        assert b.digest(sub/"selection.json")==sign["receipts"][key]
        rec=json.loads((sub/"selection.json").read_bytes())
        x,y,task,idx=source.split(name,seed)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        val=x[idx["width"]]
        test=x[idx["test"]]
        prob=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model=make_model(task=prob,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        f=tree.extract(model,val[:1])
        g=f.cache(val)
        gain=model.get_booster().get_score(importance_type="gain")
        order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
        reference,info=g.search("gain_sweep",order)
        np.testing.assert_array_equal(reference,rec["records"]["gain_sweep"]["support"])
        native=model.predict if task=="regression" else model.predict_proba
        with np.load(sub/"forecasts.npz") as fcast:
            saved=fcast["full"]
        tol=3e-5*max(1.,np.abs(saved).max())
        np.testing.assert_allclose(saved,native(test),rtol=3e-5,atol=tol)
        for repeat in range(-1,3):
            cols,sequence,search_s=compiled(g,order)
            np.testing.assert_array_equal(cols,reference)
            np.testing.assert_array_equal(sequence,info["deletion_order"])
            t=perf_counter()
            native(test)
            native_s=perf_counter()-t
            if repeat>=0:
                old=rec["records"]["gain_sweep"]
                matched=old["pipeline_s"]-old["search_s"]+search_s
                rows.append(dict(panel=panel,dataset=name,seed=seed,repeat=repeat,k=len(cols),
                    compiled_s=search_s,matched_gain_pipeline_s=matched,
                    native_prediction_s=native_s,native_full_pipeline_s=old["fit_s"]+native_s))
        print(key,"compiled full order matches",flush=True)
    pd.DataFrame(rows).to_csv(conf.DEST/"native-costs.csv",index=False)


def summarize():
    results=pd.read_csv(conf.DEST/"scores.csv").set_index(["panel","dataset","seed","method"])
    exact=results.xs("directions",level="method")
    sweep=results.xs("gain_sweep",level="method")
    costs=pd.read_csv(conf.DEST/"native-costs.csv").groupby(["panel","dataset","seed"]).median(numeric_only=True)
    ratios=dict(search_vs_compiled=exact.search_s/costs.compiled_s,
        pipeline_vs_compiled=exact.pipeline_s/costs.matched_gain_pipeline_s,
        postfit_vs_compiled=exact.postfit_s/(costs.matched_gain_pipeline_s-sweep.fit_s),
        pipeline_vs_native=exact.pipeline_s/costs.native_full_pipeline_s)
    rows=[]
    for metric,vals in ratios.items():
        for panel in ("all","wide","core"):
            val=vals if panel=="all" else vals.xs(panel,level="panel")
            log=np.log(val).groupby("dataset").mean().to_numpy()
            rng=np.random.default_rng(5002)
            samples=rng.choice(log,(20000,len(log)),replace=True).mean(axis=1)
            centre,low,high=np.exp([log.mean(),*np.quantile(samples,[.025,.975])])
            rows.append(dict(metric=metric,panel=panel,value=centre,low=low,high=high))
    frame=pd.DataFrame(rows)
    frame.to_csv(conf.DEST/"native-cost-contrasts.csv",index=False)
    print(frame.to_string(index=False))


if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--summarize",action="store_true")
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<7)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        (summarize if cli.parse_args().summarize else main)()
