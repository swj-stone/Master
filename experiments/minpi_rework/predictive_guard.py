"""Choose the width on a second labeled sample, separately from ranking inputs."""
import argparse
import ctypes
import os
from time import perf_counter
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework.availability_trial import SOURCES,DEST
from experiments.minpi_rework.predictive_mask_study import centered
from minppi_mofs.predictive_quota import PredictiveGame


def margin_path(model,game,order):
    inc=game.incidence
    p=inc.shape[1]
    ranks=np.empty(p,int)
    ranks[order]=np.arange(1,p+1)
    first=np.minimum.reduceat(ranks[inc.indices],inc.indptr[:-1]) if inc.shape[0] else np.empty(0,int)
    allocation=sparse.csr_matrix((np.ones(len(first)),(np.arange(len(first)),first)),shape=(len(first),p+1))
    removed=np.cumsum((game.a@allocation).toarray(),axis=1)
    all_values=game.full.reshape(-1,len(model.empty))+model.empty
    return (all_values.ravel()[:,None]-removed).T.reshape(p+1,-1,len(model.empty))


def passing_native(y,raw,task,scale):
    scores=[b.metrics(y,r if task=="classification" else r[:,0],task,scale) for r in raw]
    return np.array([b.eligible(s,scores[0],task) for s in scores])


def select(x,y,task,idx,seed,epsilon=.02):
    itt=perf_counter()
    model=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    fit_s=perf_counter()-itt
    games={}
    teacher_scale=max(float(np.std(y[idx["fit"]])),1e-12)
    for key in ("vote","width"):
        t=perf_counter()
        rows=idx[key]
        raw=model.game(x[rows])
        extract=perf_counter()-t
        t=perf_counter()
        target=centered(model,y[rows],task)
        pg=PredictiveGame.from_paths(raw.a,raw.incidence,target)
        games[key]=(raw,pg,extract,perf_counter()-t)
    vote,vg,ve,vm=games["vote"]
    cal,cg,ce,cm=games["width"]
    p=vote.p
    ranks={}
    for rule in ("min","pooled","intact"):
        t=perf_counter()
        order,checks=vg.ranked_path(rule)
        ranks[rule]=(order,perf_counter()-t,checks,ve+vm)
    t=perf_counter()
    initial=(vote.a@vote.incidence).toarray()
    residual=centered(model,y[idx["vote"]],task)-vote.full
    marginal=2*(residual@initial)+np.einsum("ij,ij->j",initial,initial)
    ranks["static"]=(np.argsort(marginal,kind="stable"),perf_counter()-t,p,ve)
    t=perf_counter()
    ranks["mdi"]=(np.argsort(model.model.feature_importances_,kind="stable"),perf_counter()-t,p,0.)
    records={}
    forecasts={}
    for rank,(order,rank_s,checks,requirements) in ranks.items():
        t=perf_counter()
        preds=margin_path(model,cal,order)
        good=passing_native(y[idx["width"]],preds,task,teacher_scale)
        guard_s=perf_counter()-t
        t=perf_counter()
        excess=cg.path_curves(order)[:,1:]*(p/np.arange(1,p+1))
        risk_s=perf_counter()-t
        for guard,curve in (("min",excess.max(axis=1)),("pooled",excess.mean(axis=1)),
                            ("intact",excess[:,-1]),("clean",np.zeros(p+1))):
            acceptable=good&(curve<=epsilon*cg.null+1e-12*max(1,cg.null))
            d=np.flatnonzero(acceptable)[-1]
            arm=rank+"_"+guard
            cols=np.sort(order[d:])
            t=perf_counter()
            forecasts[arm]=model.predict(x[np.ix_(idx["test"],cols)],cols)
            pred_s=perf_counter()-t
            records[arm]=dict(support=cols.tolist(),k=len(cols),p=p,fit_s=fit_s,rank_s=rank_s,
                rank_setup_s=requirements,checks=checks,pred_s=pred_s,search_s=rank_s+guard_s+(0 if guard=="clean" else risk_s),
                pipeline_s=fit_s+requirements+ce+rank_s+guard_s+pred_s+(0 if guard=="clean" else cm+risk_s),
                cal_worst=excess[d].max()/max(cg.null,1e-100),cal_intact=excess[d,-1]/max(cg.null,1e-100),
                guard_value=curve[d]/max(cg.null,1e-100))
    full=np.arange(p)
    t=perf_counter()
    forecasts["full"]=model.native(x[idx["test"]])
    records["full"]=dict(support=full.tolist(),k=p,p=p,fit_s=fit_s,pipeline_s=fit_s+perf_counter()-t)
    for partner in ("min","pooled","intact","static"):
        for guard in ("min","pooled","intact","clean"):
            alternatives=[rank+"_"+guard for rank in ("mdi",partner)]
            winner=min(alternatives,key=lambda arm:(records[arm]["k"],records[arm]["guard_value"],arm))
            arm="mdi_plus_"+partner+"_"+guard
            record=dict(records[winner],winner=winner)
            # Fit, the two forest path dictionaries and the calibration Gram are reusable across paths.
            # The partner's ranking work and both paths' validation passes are charged.
            mdi=records[alternatives[0]]
            other=records[alternatives[1]]
            record["pipeline_s"]=other["pipeline_s"]-other["pred_s"]+mdi["search_s"]+record["pred_s"]
            records[arm],forecasts[arm]=record,forecasts[winner]
    return model,records,forecasts


def score(model,source,name,seed,idx,task,records,forecasts):
    x,truth,_,idx_actual=source._partition(name,seed)
    np.testing.assert_array_equal(idx_actual["test"],idx["test"])
    test=x[idx["test"]]
    target=truth[idx["test"]]
    scale=max(float(np.std(truth[np.r_[idx["fit"],idx["vote"],idx["width"]]])),1e-12)
    out=model.game(test)
    audit=PredictiveGame.from_paths(out.a,out.incidence,centered(model,target,task))
    p=out.p
    rows=[]
    for arm,record in records.items():
        selected=np.zeros(p,bool)
        selected[record["support"]]=True
        risk=audit.conditional_risk(selected)[1:]*p/np.arange(1,p+1)/max(audit.null,1e-100)
        native=b.metrics(target,forecasts[arm],task,scale)
        rows.append(dict(dataset=name,seed=seed,method=arm,task=task,
            outer_worst=risk.max(),outer_positive=max(0,risk.max()),outer_intact=risk[-1],
            outer_mean=risk.mean(),outer_worst_size=int(risk.argmax()+1),**record,**native))
    return rows


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--seeds",nargs="+",type=int,default=[42])
    cli.add_argument("--tag",default="two-sample")
    args=cli.parse_args()
    DEST.mkdir(exist_ok=True,parents=True)
    rows=[]
    jobs=[(s,n,seed) for s in SOURCES for n in s.SOURCES for seed in args.seeds]
    start=perf_counter()
    for i,(source,name,seed) in enumerate(jobs):
        t=perf_counter()
        x,y,task,idx=source.split(name,seed)
        model,records,forecast=select(x,y,task,idx,seed)
        rows+=score(model,source,name,seed,idx,task,records,forecast)
        pd.DataFrame(rows).to_csv(DEST/f"{args.tag}.csv",index=False)
        elapsed=perf_counter()-start
        print(f"[{i+1}/{len(jobs)}] {name}/{seed} {perf_counter()-t:.1f}s elapsed {elapsed:.0f}s ETA {elapsed*(len(jobs)-i-1)/(i+1):.0f}s",flush=True)


if __name__=="__main__":
    if os.name=="nt":
        ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6))
    with threadpool_limits(1):
        main()
