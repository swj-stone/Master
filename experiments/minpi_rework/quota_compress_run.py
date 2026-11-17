"""Select on disjoint calibration rows, seal predictions, and evaluate historical panels."""
import argparse
import json
from time import perf_counter
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import quota_compress as q
from experiments.minpi_rework.forward import development_split
from minppi_mofs.additive_minpi import fit_additive


OUT=b.ROOT/"results/minpi_rework/quota-compress"
EVIDENCE=b.ROOT/"thesis-paper/evidence/minpi_rework/quota-compress"
OLD=tuple(legacy.CONFIG["datasets"])
NAMES=(*OLD,*b.DESIGN["confirmation"])
SEED=42
ARMS=("energy","gain","greedy","ballots","ballots_fix","cone","full")


def load(name,jobs):
    if name in OLD:
        return legacy.read_split(*jobs[name,SEED],SEED)
    x,y,task,idx=development_split(name,SEED)
    y=y.copy()
    y[idx["test"]]=-1
    return x,y,task,idx


def select(name,jobs):
    x,y,task,idx=load(name,jobs)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    started=perf_counter()
    teacher=fit_additive(x[pool],y[pool],task,"linear")
    fit_s=perf_counter()-started
    a=teacher.parts(x[idx["width"]])-teacher.center
    matrix=a.transpose(0,2,1).reshape(-1,x.shape[1])
    gram=matrix.T@matrix
    norm=matrix.sum(axis=1)@matrix.sum(axis=1)
    rho=q.RHO if norm>0 else 0.
    gram=gram/norm if norm>0 else gram/max(1.,float(np.abs(gram).max()))
    base_s=perf_counter()-started
    t=perf_counter()
    greedy,path=q.greedy(gram,rho)
    greedy_s=perf_counter()-t
    t=perf_counter()
    w,rhs=q.initial_ballots(gram,path,rho)
    states=q.fix(w,rhs,int(greedy.sum()))
    bound=q.lower_bound(w,rhs)
    diag_s=perf_counter()-t
    info,forecasts={},{}
    arms=np.random.default_rng(511+NAMES.index(name)).permutation(ARMS).tolist()
    for method in arms:
        details={}
        t=perf_counter()
        if method=="full":
            keep=np.ones(x.shape[1],bool)
        elif method=="energy":
            order=np.lexsort((np.arange(len(gram)),-gram.diagonal()))
            keep=q.prefix(gram,order,rho)
        elif method=="gain":
            order,_=legacy.gain_rank("gain",x,y,task,dict(fit=pool),SEED)
            keep=q.prefix(gram,order,rho)
        elif method=="greedy":
            keep=greedy
        elif method=="cone":
            keep,details=q.cone(gram,greedy,rho)
        else:
            keep,details=q.master(gram,greedy,path,method=="ballots_fix",rho)
        branch_s=perf_counter()-t
        if method in ("greedy","cone","ballots","ballots_fix"):
            branch_s+=greedy_s
        selection=(fit_s if method=="full" else base_s)+branch_s
        error=q.loss(gram,keep)
        assert q.feasible(gram,keep,rho)
        t=perf_counter()
        cols=np.flatnonzero(keep)
        forecast=teacher.restrict(cols).predict(x[np.ix_(idx["test"],cols)])
        forecasts[method]=forecast if task=="classification" else forecast.ravel()
        info[method]=dict(p=len(gram),k=len(cols),cols=cols.tolist(),relative_error=np.sqrt(max(0.,error)),
            train_s=fit_s,train_gram_s=base_s,selection_component_s=selection,
            pipeline_component_s=selection+perf_counter()-t,**details)
        print(name,method,"width",len(cols),"lower",details.get("lb"),f"{selection:.3f}s",flush=True)
    sub=OUT/name
    sub.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(sub/"forecasts.npz",**forecasts)
    info=dict(dataset=name,task=task,seed=SEED,tol=q.RHO,scale=max(np.std(y[pool]),1e-12),
        initial_lb=bound,initial_state=states.tolist(),initial_ballots=len(w),diagnostic_s=diag_s,
        forecast_sha256=b.digest(sub/"forecasts.npz"),arms=info)
    (sub/"selection.json").write_text(json.dumps(info),encoding="utf8")


def seal():
    locked={}
    for name in NAMES:
        directory=OUT/name
        info=json.loads((directory/"selection.json").read_bytes())
        assert set(info["arms"])==set(ARMS) and info["forecast_sha256"]==b.digest(directory/"forecasts.npz")
        locked[name]=b.digest(directory/"selection.json")
    (OUT/"lock.json").write_text(json.dumps(locked,indent=2),encoding="utf8")


def score(jobs):
    lock=json.loads((OUT/"lock.json").read_bytes())
    assert set(lock)==set(NAMES)
    rows=[]
    for name in NAMES:
        sub=OUT/name
        assert lock[name]==b.digest(sub/"selection.json")
        rec=json.loads((sub/"selection.json").read_bytes())
        assert rec["forecast_sha256"]==b.digest(sub/"forecasts.npz")
        if name in OLD:
            dire,row=jobs[name,SEED]
            path=dire/row["test_relpath"]
            assert b.digest(path)==row["test_file_sha256"]
            y=pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()
            with np.load(legacy.DEST/"novel-input-rows.npz") as a:
                masks=dict(primary=slice(None),novel=a[f"{name}-{SEED}"])
        else:
            _,target,_,idx=development_split(name,SEED)
            y=target[idx["test"]]
            masks=dict(primary=slice(None))
        with np.load(sub/"forecasts.npz") as pred:
            for method,arm in rec["arms"].items():
                arm=arm.copy()
                if method.startswith("ballots") and "last_master_status" not in arm:
                    # Initial receipts retained the last MILP subproblem's status.
                    arm["last_master_status"]=arm["status"]
                    arm["status"]="optimal" if arm["certified"] else "timelimit"
                for part,mask in masks.items():
                    metrics=b.metrics(y[mask],pred[method][mask],rec["task"],rec["scale"])
                    state=np.asarray(rec["initial_state"])
                    rows.append(dict(dataset=name,task=rec["task"],part=part,method=method,seed=SEED,
                        **{key:val for key,val in arm.items() if key!="cols"},support=" ".join(map(str,arm["cols"])),
                        initial_lb=rec["initial_lb"],initial_forced_in=int((state==1).sum()),
                        initial_forced_out=int((state==-1).sum()),**metrics))
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(EVIDENCE/"historical.csv",index=False)
    (EVIDENCE/"lock.json").write_bytes((OUT/"lock.json").read_bytes())
    with ZipFile(EVIDENCE/"selection-receipts.zip","w",ZIP_DEFLATED) as archive:
        for name in NAMES:
            archive.write(OUT/name/"selection.json",name+"/selection.json")
    print(result[result.part=="primary"].pivot(index="dataset",columns="method",values="k"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["select","seal","score"])
    parser.add_argument("--names",nargs="+",choices=NAMES,default=NAMES)
    args=parser.parse_args()
    jobs=legacy.jobs()
    with threadpool_limits(1):
        if args.action=="select":
            for n in args.names:
                select(n,jobs)
        elif args.action=="seal":
            seal()
        else:
            score(jobs)
