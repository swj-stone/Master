"""Independent finite-budget comparison of directional, native-conic and cheap selectors."""
import argparse
import ctypes
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b,legacy_gain
from experiments.minpi_rework import quota_compress as q,quota_compress_control as local
from experiments.minpi_rework import validation_panel,validation_wide,directional_pilot
from minppi_mofs import additive_minpi,directional_quota as dq


ROOT=b.ROOT/"results/minpi_rework/directional-confirmation"
OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/directional-confirmation"
PROTOCOL=Path(__file__).with_suffix(".md")
METHODS=("shared","independent","cone","local","energy","gain","full","direct")
RHO=.2


def initial(game):
    g=game.gram
    order=np.argsort(-g.diagonal(),kind="stable")
    energy=local.from_feasible(g,q.prefix(g,order))
    backward=local.from_feasible(g,np.ones(len(g),bool))
    forward=np.zeros(len(g),bool)
    while not q.feasible(g,forward,RHO):
        candidates=np.flatnonzero(~forward)
        returns=2*g[:,candidates].sum(axis=1)-g.diagonal()
        forward[candidates[np.argmax(returns[candidates])]]=True
        forward=local.improve(g,forward)
    options=[energy,backward,local.from_feasible(g,forward),np.ones(len(g),bool)]
    return min((s for s in options if game.feasible(s,RHO)),
               key=lambda s:(int(s.sum()),dq.gram_loss(g,s),tuple(np.flatnonzero(s))))


def enumerate_small(game,best):
    p=len(best)
    bits=np.arange(p,dtype=np.uint64)
    for lo in range(0,2**p,4096):
        ids=np.arange(lo,min(2**p,lo+4096),dtype=np.uint64)
        masks=((ids[:,None]>>bits)&1).astype(bool)
        masks=masks[masks.sum(axis=1)<best.sum()]
        omit=(~masks).astype(float)
        risks=np.einsum("ij,ij->i",omit@game.gram,omit)
        valid=masks[risks<=RHO**2+(1+RHO**2)*game.roundoff]
        for s in valid[np.argsort(valid.sum(axis=1),kind="stable")]:
            if s.sum()>=best.sum():
                break
            if game.feasible(s,RHO):
                best=s.copy()
                break
    return best,dict(certified=True,lb=int(best.sum()),status="enumerated")


def choose(x,y,task,idx,seed,method):
    start=perf_counter()
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    if method=="full":
        model=additive_minpi.fit_additive(x[pool],y[pool],task,"linear")
        keep=np.ones(x.shape[1],bool)
        details={}
    elif method in ("energy","gain"):
        model=additive_minpi.fit_additive(x[pool],y[pool],task,"linear")
        parts=model.parts(x[idx["width"]])-model.center
        a=parts.transpose(0,2,1).reshape(-1,x.shape[1])
        if method=="energy":
            order=np.argsort(-np.einsum("ij,ij->j",a,a),kind="stable")
        else:
            order,_=legacy_gain.gain_rank(method,x,y,task,{"fit":pool},seed)
        residual=a.sum(axis=1)
        limit=RHO*np.linalg.norm(residual)
        keep=np.zeros(len(order),bool)
        for column in order:
            if np.linalg.norm(residual)<=limit and np.linalg.norm(a[:,~keep].sum(axis=1))<=limit:
                break
            keep[column]=True
            residual-=a[:,column]
        details={}
    else:
        model,game=directional_pilot.game(x,y,task,idx)
        keep=initial(game)
        if method=="local":
            details={}
        elif method=="direct" and len(keep)<=16:
            keep,details=enumerate_small(game,keep)
        elif method in ("cone","direct"):
            candidate,details=q.cone(game.gram,keep,rho=RHO,seconds=3.)
            if game.feasible(candidate,RHO):
                keep=candidate
            else:
                details=dict(certified=False,lb=0,status="numerically rejected")
        else:
            use_shared=method=="shared"
            keep,details=dq.directional_search(game,keep,RHO,max_seconds=3.,shared=use_shared,cache=use_shared)
        if not game.feasible(keep,RHO):
            raise ArithmeticError("An infeasible support reached deployment")
    elapsed=perf_counter()-start
    cols=np.flatnonzero(keep)
    pred=model.restrict(cols).predict(x[np.ix_(idx["test"],cols)])
    return dict(method=method,p=x.shape[1],k=len(cols),support=cols.tolist(),selection_s=elapsed,
                deployment_s=perf_counter()-start,**details),pred if task=="classification" else pred.ravel()


def jobs():
    for panel,source in [("core",validation_panel),("wide",validation_wide)]:
        for name in source.SOURCES:
            for seed in range(42,47):
                yield panel,source,name,seed


def signatures():
    modules=[q,local,directional_pilot,additive_minpi,dq,validation_panel,validation_wide,b,legacy_gain]
    paths=[Path(m.__file__) for m in modules]+[Path(__file__),PROTOCOL,validation_panel.MANIFEST,
          Path(validation_wide.__file__).with_name("validation_wide_sources.json"),legacy_gain.ORIGINAL]
    return {p.name:b.digest(p) for p in paths}


def select():
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<5)):
            raise ctypes.WinError(ctypes.get_last_error())
    dq.directional_search(dq.FidelityGame.from_factor(np.eye(5)),np.ones(5,bool),shared=True,cache=True)
    dq._bank_reject(np.zeros((1,3)),np.zeros(1),np.arange(3)[None,:],1,np.zeros(3,np.int8),2)
    ROOT.mkdir(exist_ok=True,parents=True)
    OUT.mkdir(exist_ok=True,parents=True)
    sources=signatures()
    lock={}
    start=perf_counter()
    for j,(panel,source,name,seed) in enumerate(jobs()):
        sub=ROOT/f"{name}-{seed}"
        sub.mkdir(exist_ok=True)
        receipt=sub/"selection.json"
        if not receipt.exists():
            begin=perf_counter()
            x,y,task,idx=source.split(name,seed)
            predictions,records={},{}
            for method in np.random.default_rng(seed+j).permutation(METHODS):
                if method=="direct" and x.shape[1]>16:
                    continue
                records[method],predictions[method]=choose(x,y,task,idx,seed,method)
            if "direct" not in records:
                records["direct"]={**records["cone"],"method":"direct"}
                predictions["direct"]=predictions["cone"]
            np.savez_compressed(sub/"forecasts.npz",**predictions)
            staged=receipt.with_suffix(".tmp")
            staged.write_text(json.dumps(dict(task=task,source=sources,records=records,
                        split={k:v.tolist() for k,v in idx.items()},forecast=b.digest(sub/"forecasts.npz"))),encoding="utf8")
            staged.replace(receipt)
            elapsed=perf_counter()-start
            print(f"[{j+1}/65] {name}/{seed} {perf_counter()-begin:.2f}s, total {elapsed:.1f}s ETA {elapsed*(64-j)/(j+1):.1f}s",flush=True)
        info=json.loads(receipt.read_bytes())
        if info["source"]!=sources or info["forecast"]!=b.digest(sub/"forecasts.npz"):
            raise ValueError(f"Stale {sub}")
        lock[sub.name]=b.digest(receipt)
    (OUT/"lock.json").write_text(json.dumps(dict(sources=sources,receipts=lock),indent=2),encoding="utf8")
    archive=b.ROOT/"packages/minpi-directional-forecasts.zip"
    with ZipFile(archive,"w",compression=ZIP_DEFLATED) as z:
        z.write(OUT/"lock.json","lock.json")
        for name in lock:
            for file in ("forecasts.npz","selection.json"):
                z.write(ROOT/name/file,f"{name}/{file}")
    (OUT/"forecast-sha256.txt").write_text(b.digest(archive)+"  "+archive.name+"\n",encoding="utf8")


def score():
    lock=json.loads((OUT/"lock.json").read_bytes())
    if lock["sources"]!=signatures() or set(lock["receipts"])!={f"{n}-{s}" for _,_,n,s in jobs()}:
        raise ValueError("Incomplete or obsolete forecast set")
    rows=[]
    for panel,source,name,seed in jobs():
        sub=ROOT/f"{name}-{seed}"
        path=sub/"selection.json"
        assert lock["receipts"][sub.name]==b.digest(path)
        info=json.loads(path.read_bytes())
        out=source.score(name,seed,sub/"forecasts.npz",info["forecast"])
        assert set(out)==set(METHODS)
        for m in METHODS:
            rows.append(dict(panel=panel,dataset=name,task=info["task"],seed=seed,**info["records"][m],**out[m]))
    pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["select","score"])
    with threadpool_limits(1):
        (select if cli.parse_args().action=="select" else score)()
