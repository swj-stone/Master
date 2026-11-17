"""Sequential, counterbalanced, resumable paired measurements with live ETA."""
import argparse
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.anchored_vote import select as policy
from experiments.minpi_replication import direct_gain
from experiments.split_vote import path
from experiments.split_vote.run import ROOT, digest, load
from minppi_mofs import split_vote


RUN=ROOT/"results/anchored_vote"
ARMS=("min","mean","gain","gain_grid","exhaustive")


def run(args):
    if args.names is None:
        from experiments.minpi_replication.second_panel.data import SOURCES
        names=list(SOURCES)
    else:
        names=args.names
    files=[Path(__file__),Path(__file__).with_name("protocol.md"),*(Path(f.__file__) for f in (policy,direct_gain,path,split_vote))]
    if args.names is None:
        from experiments.minpi_replication.second_panel import data
        files.append(Path(data.__file__))
    signature={f.name:digest(f) for f in files}
    jobs=[(n,s) for s in args.seeds for n in names]
    t=perf_counter()
    rng=np.random.default_rng(20261117)
    warm_x=rng.normal(size=(96,4)).astype("float32")
    warm_y=np.arange(len(warm_x))%2
    warm=direct_gain.fit_native(warm_x,warm_y,"classification",42,args.device)
    direct_gain.predict_native(warm,warm_x,"classification")
    for i,(name,seed) in enumerate(jobs):
        if args.names is None:
            from experiments.minpi_replication.second_panel.data import split
            x,y,task,idx=split(name,seed)
        else:
            x,y,task,idx=load(name,seed)
        x=np.asarray(x,dtype=np.float32)
        methods=np.roll(args.arms,(args.offset+args.seeds.index(seed)+names.index(name))%len(args.arms))
        for method in methods:
            folder=RUN/args.tag/f"{name}-{seed}"/method
            folder.mkdir(parents=True,exist_ok=True)
            receipt,forecast=folder/"receipt.json",folder/"forecast.npz"
            if receipt.exists():
                old=json.loads(receipt.read_bytes())
                if old["signature"]!=signature or old["forecast"]!=digest(forecast) or old["device"]!=args.device:
                    raise RuntimeError(f"Stale {receipt}")
                continue
            print(f"{name}/{seed} {method} started",flush=True)
            timer=Timer(3600,lambda:os._exit(124))
            timer.daemon=True
            timer.start()
            try:
                r,pred=policy.select(x,y,task,idx,seed,method,args.device)
                out=dict(primary=pred,indices=idx["test"])
                if method=="exhaustive":
                    near=r["near_support"]
                    if near==r["selected"]:
                        out["near"]=pred
                        r["near_seconds"]=r["seconds"]
                    else:
                        dev=np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
                        start=perf_counter()
                        model=direct_gain.fit_native(x[np.ix_(dev,near)],y[dev],task,seed,args.device)
                        out["near"]=direct_gain.predict_native(model,x[np.ix_(idx["test"],near)],task)
                        r["near_seconds"]=r["selection_s"]+perf_counter()-start
                np.savez_compressed(forecast,**out)
                row=dict(dataset=name,seed=int(seed),method=str(method),task=task,p=x.shape[1],
                         record=r,forecast=digest(forecast),signature=signature,device=args.device)
                tmp=receipt.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,allow_nan=False),encoding="utf8")
                tmp.replace(receipt)
                print(f"{name}/{seed} {method} {r['seconds']:.2f}s, {r['k']} columns",flush=True)
            finally:
                timer.cancel()
        elapsed=perf_counter()-t
        print(f"{i+1}/{len(jobs)} | {elapsed:.1f}s | ETA {elapsed*(len(jobs)-i-1)/(i+1):.1f}s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names",nargs="+")
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44,45,46])
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--tag",required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--offset",type=int,default=0,help="Rotate the arm start for a repeated timing")
    p.add_argument("--cpu-mask",type=int,help="Optional Windows process affinity mask")
    args=p.parse_args()
    if args.cpu_mask is not None:
        import ctypes
        if not ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(args.cpu_mask)):
            raise RuntimeError("Cannot pin process")
    with threadpool_limits(1):
        run(args)
