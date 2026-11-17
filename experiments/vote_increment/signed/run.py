"""Independent timed arms for masked votes versus native Gain, with sealed forecasts."""
import argparse
import ctypes
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import accuracy_score,mean_absolute_error,r2_score,roc_auc_score

from experiments.minpi_replication.direct_gain import fit_native,predict_native,validation_utility
from experiments.split_vote.run import ROOT,ORIGINAL,load,legacy,digest
from . import games

ARM=("min","mean","gain")
HERE=Path(__file__).resolve().parent
DEST=ROOT/"thesis-paper/evidence/vote_increment/signed"


def arm(x,y,task,idx,seed,method,family):
    begin=perf_counter()
    fit,valid=idx["fit"],idx["width"]
    full=fit_native(x[fit],y[fit],task,seed,"cpu")
    p=x.shape[1]
    raw=full.get_booster().get_score(importance_type="gain")
    gain=np.array([raw.get(f"f{i}",0.) for i in range(p)])
    start=perf_counter()
    aux=0
    if method=="gain":
        values=gain
    else:
        bank,aux=getattr(games,family)(x[fit],y[fit],x[idx["vote"]],y[idx["vote"]],task,seed)
        values=bank.min(axis=0) if method=="min" else bank.mean(axis=0)
    order=np.lexsort((-np.arange(p),-gain,-values))
    mass=np.cumsum(np.maximum(values[order],0))
    if mass[-1]<=0:
        order=np.lexsort((-np.arange(p),-gain))
        mass=np.cumsum(gain[order])
        if mass[-1]<=0:
            mass=np.arange(1,p+1,dtype=float)
    widths=sorted({p,*(1+int(np.searchsorted(mass,mass[-1]*q)) for q in (.5,.75,.9))})
    rank_seconds=perf_counter()-start
    reference=validation_utility(y[valid],predict_native(full,x[valid],task),task)
    fits=1
    trace={}
    for k in widths:
        cols=np.sort(order[:k])
        utility=reference
        if k!=p:
            trial=fit_native(x[np.ix_(fit,cols)],y[fit],task,seed,"cpu")
            fits+=1
            utility=validation_utility(y[valid],predict_native(trial,x[np.ix_(valid,cols)],task),task)
        trace[k]=utility
        if utility>=reference-.005-1e-14:
            break
    dev=np.sort(np.r_[fit,idx["vote"],valid])
    final=fit_native(x[np.ix_(dev,cols)],y[dev],task,seed,"cpu")
    pred=predict_native(final,x[np.ix_(idx["test"],cols)],task)
    return dict(k=len(cols),columns=cols.tolist(),fits=fits+1,aux=aux,rank_seconds=rank_seconds,
                time=perf_counter()-begin,trace=trace,reference=reference,widths=widths,
                order=order.tolist(),support_values=values.tolist()),pred


def outcome(name,seed):
    folder,info=legacy.jobs()[name,seed]
    target=info["target"]
    return pd.read_csv(folder/info["test_relpath"],usecols=[target])[target].to_numpy()


def metrics(y,pred):
    if pred.ndim==1:
        return r2_score(y,pred),mean_absolute_error(y,pred)
    scores=pred[:,1] if pred.shape[1]==2 else pred
    return roc_auc_score(y,scores,multi_class="ovr"),accuracy_score(y,pred.argmax(axis=1))


def run(args):
    out=DEST/args.tag
    out.mkdir(parents=True,exist_ok=True)
    t=perf_counter()
    for i,name in enumerate(ORIGINAL):
        x,y,task,idx=load(name,args.seed)
        for method in np.roll(ARM,i%len(ARM)):
            d=out/name/method
            d.mkdir(parents=True,exist_ok=True)
            if (d/"record.json").exists():
                continue
            print(name,args.seed,method,"started",flush=True)
            info,pred=arm(x,y,task,idx,args.seed,str(method),args.family)
            np.savez_compressed(d/"predictions.npz",values=pred,ids=idx["test"])
            info.update(name=name,seed=args.seed,task=task,method=str(method),family=args.family,
                        forecast_sha256=digest(d/"predictions.npz"))
            tmp=d/"record.tmp"
            tmp.write_text(json.dumps(info,allow_nan=False),encoding="utf8")
            tmp.replace(d/"record.json")
            print(name,method,"k",info["k"],"fits",info["fits"],"sec",round(info["time"],3),flush=True)
    rows=[]
    for name in ORIGINAL:
        labels=outcome(name,args.seed)
        for method in ARM:
            d=out/name/method
            r=json.loads((d/"record.json").read_bytes())
            if digest(d/"predictions.npz")!=r["forecast_sha256"] or r["family"]!=args.family:
                raise ValueError(f"Invalid receipt {d}")
            with np.load(d/"predictions.npz") as arr:
                primary,second=metrics(labels,arr["values"])
            rows.append(dict(dataset=name,seed=args.seed,method=method,k=r["k"],time=r["time"],
                             primary=primary,secondary=second,fits=r["fits"],aux=r["aux"],rank_s=r["rank_seconds"]))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"scores.csv",index=False)
    print(frame.round(4).to_string(index=False),flush=True)
    print("Elapsed",round(perf_counter()-t,1),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--family",default="random_subspace")
    p.add_argument("--tag",required=True)
    args=p.parse_args()
    if not ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<12)):
        raise RuntimeError("Requested core unavailable")
    with threadpool_limits(1):
        run(args)
