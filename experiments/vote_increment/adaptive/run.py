"""Development runner for quota proposals and true, budget-matched refits."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.anchored_vote.audit import score
from experiments.minpi_replication import direct_gain as native
from experiments.minpi_design.study import load
from experiments.split_vote.run import ORIGINAL, ROOT
from experiments.vote_increment.adaptive import game


DEST=ROOT/"thesis-paper/evidence/vote_increment/adaptive"
RUN=ROOT/"results/vote_increment/adaptive"
ARMS=("min","mean","gain","mean_scale")


def select(x,y,task,idx,seed,arm,design):
    start=perf_counter()
    tr=np.sort(idx["fit"])
    v=np.sort(idx["vote"])
    va=np.sort(idx["width"])
    dev=np.sort(np.r_[tr,v,va])
    p=x.shape[1]
    full=tuple(range(p))
    def fit(rows,cols):
        return native.fit_native(x[np.ix_(rows,cols)],y[rows],task,seed,"cpu")
    def predict(model,rows,cols):
        return native.predict_native(model,x[np.ix_(rows,cols)],task)
    model=fit(tr,full)
    gain_map=model.get_booster().get_score(importance_type="gain")
    gain=np.array([gain_map.get(f"f{j}",0.) for j in full])
    scores={full:native.validation_utility(y[va],predict(model,va,full),task)}
    floor=scores[full]-.005
    t=perf_counter()
    if arm=="gain":
        ranks=game.normalized(gain)
    else:
        if design=="margin":
            votes=game.margin_votes(model,x[v],y[v],task,seed)
        else:
            raise ValueError(design)
        ranks=votes.min(axis=0) if arm=="min" else votes.mean(axis=0)
        if arm=="mean_scale" and ranks.sum()>0:
            ranks*=votes.min(axis=0).sum()/ranks.sum()
    vote_s=perf_counter()-t
    order=np.lexsort((-np.arange(p),-gain,-ranks))
    def check(q):
        k=max(1,int(np.count_nonzero(ranks>=q)))
        c=tuple(np.sort(order[:k]).tolist())
        if c not in scores:
            scores[c]=native.validation_utility(y[va],predict(fit(tr,c),va,c),task)
        return scores[c]>=floor-1e-14
    if check(1.):
        check(2.)
    else:
        for q in (.5,.25,.125):
            if check(q):
                break
    cols=min((c for c in scores if scores[c]>=floor-1e-14),key=len)
    pred=predict(fit(dev,cols),idx["test"],cols)
    elapsed=perf_counter()-start
    return dict(cols=list(cols),k=len(cols),fits=len(scores)+1,seconds=elapsed,vote_s=vote_s,
                floor=floor,validation=scores[cols],order=order.tolist(),
                trace=[(list(c),u) for c,u in scores.items()]),pred


def analyze(frame):
    keys=["dataset","seed"]
    minrows=frame[frame.arm=="min"]
    pairs=[]
    for ctrl in ("gain","mean","mean_scale"):
        base=frame[frame.arm==ctrl]
        joined=minrows.merge(base,on=keys,suffixes=("_min","_base"),validate="one_to_one")
        width=joined.k_min-joined.k_base
        fits=joined.fits_min-joined.fits_base
        ratio=joined.seconds_min/joined.seconds_base
        delta=joined.primary_min-joined.primary_base
        joined["outcome"]=((delta>=-.01)&((width<0)|((width<=np.maximum(1,.05*joined.k_base))&(fits<0)&(ratio<1))))
        joined["strict"]=((delta>=-.01)&((width<0)|((width<=0)&(fits<0)&(ratio<=.95))))
        joined["control"]=ctrl
        joined["primary_delta"]=delta
        joined["time_ratio"]=ratio
        pairs.append(joined)
    return pd.concat(pairs,ignore_index=True)


def run(args):
    sub=RUN/args.tag
    sub.mkdir(parents=True,exist_ok=True)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()+Path(game.__file__).read_bytes()).hexdigest()
    for i,name in enumerate(args.names):
        for j,seed in enumerate(args.seeds):
            x,y,task,idx,read_labels=load(name,seed)
            x=np.asarray(x,dtype=np.float32)
            for arm in np.roll(ARMS,i+j):
                folder=sub/f"{name}-{seed}"/arm
                folder.mkdir(parents=True,exist_ok=True)
                receipt=folder/"record.json"
                if receipt.exists():
                    if json.loads(receipt.read_bytes())["source"]!=source_sha:
                        raise ValueError(f"Stale selector at {receipt}")
                    continue
                rec,pred=select(x,y,task,idx,seed,arm,args.design)
                np.savez_compressed(folder/"pred.npz",pred=pred,indices=idx["test"])
                rec.update(dataset=name,seed=seed,arm=str(arm),design=args.design,source=source_sha,
                           task=task,forecast=hashlib.sha256((folder/"pred.npz").read_bytes()).hexdigest())
                tmp=receipt.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec,allow_nan=False),encoding="utf8")
                tmp.replace(receipt)
                print(f"{name}/{seed}/{arm} k={rec['k']} fits={rec['fits']} {rec['seconds']:.2f}s",flush=True)
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            _,_,task,idx,read_labels=load(name,seed)
            y=read_labels()
            for arm in ARMS:
                folder=sub/f"{name}-{seed}"/arm
                rec=json.loads((folder/"record.json").read_bytes())
                assert rec["forecast"]==hashlib.sha256((folder/"pred.npz").read_bytes()).hexdigest()
                with np.load(folder/"pred.npz") as pred:
                    np.testing.assert_array_equal(idx["test"],pred["indices"])
                    data=score(y,pred["pred"],task)
                rows.append({k:v for k,v in rec.items() if k not in ("trace","order")}|data)
    dest=DEST/args.tag
    dest.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(dest/"seeds.csv",index=False)
    pairs=analyze(frame)
    pairs.to_csv(dest/"pairs.csv",index=False)
    required=pairs[pairs.control.isin(["gain","mean"])]
    print("joint",required.groupby(["dataset","seed"])[["outcome","strict"]].all(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag",required=True)
    p.add_argument("--design",default="margin")
    p.add_argument("--names",nargs="+",default=list(ORIGINAL))
    p.add_argument("--seeds",nargs="+",type=int,default=[42])
    args=p.parse_args()
    if os.name=="nt" and not ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.c_void_p(-1),1<<10):
        raise RuntimeError("Could not pin to requested processor")
    with threadpool_limits(1):
        run(args)
