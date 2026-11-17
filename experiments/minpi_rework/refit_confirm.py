"""Prepare or seal two frozen workflows using the public panels' masked interfaces only."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework import validation_panel as panel,validation_wide as wide
from experiments.minpi_rework.budget import warm_budget
from experiments.minpi_rework.pair_outage import exchange_half,stats
from experiments.minpi_rework.refit_delivery import refit_select
from minppi_mofs.additive_minpi import fit_additive


HERE=Path(__file__).resolve().parent
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/refit-outage-confirmation"
WORK=b.ROOT/"results/minpi_rework/refit-outage-confirmation"
CONFIG=DEST/"preflight.json"
REFIT=("width","greedy","energy","gain","full")
OUTAGE=("exchange_min","exchange_mean","exchange_point","energy","full")
NAMES=tuple(panel.SOURCES)+tuple(wide.SOURCES)
JOBS=[(n,s) for n in NAMES for s in panel.SEEDS]
FILES=("refit.npz","outage.npz","terms.npz")
SOURCES=[
    HERE/f for f in ("refit_confirm.py","refit_confirm.md","refit_delivery.py","pair_outage.py",
        "budget.py","exchange_benchmark.py","validation_panel.py","validation_wide.py",
        "validation_protocol.md","validation_wide.md","validation_sources.json","validation_wide_sources.json")]
SOURCES += [b.ROOT/p for p in (
    "src/minppi_mofs/additive_minpi.py","src/minppi_mofs/exchange_minpi.py","src/minppi_mofs/modeling.py",
    "experiments/minpi_compression/benchmark.py","experiments/minpi_compression/legacy_gain.py",
    "experiments/minpi_compression/protocol.json","docs/thesis/archives/piv1/protocol.json")]


def hash_file(path):
    return b.digest(path)


def text_hashes():
    return {p.relative_to(b.ROOT).as_posix():hashlib.sha256(p.read_text(encoding="utf8").encode()).hexdigest()
            for p in SOURCES}


def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode())
    h.update(a.data)
    return h.hexdigest()


def write_json(p,data):
    tmp=p.with_suffix(".partial")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False),encoding="utf8")
    tmp.replace(p)


def masked(n,s):
    provider=panel if n in panel.SOURCES else wide
    x,y,task,idx=provider.split(n,s)
    outer=y[idx["test"]]
    assert (outer==-1).all() if task=="classification" else np.isnan(outer).all()
    meta=dict(archive=provider.SOURCES[n][-1],x=array_hash(x),blinded_y=array_hash(y),p=x.shape[1],task=task,
              indices={k:array_hash(np.asarray(v,dtype="<i8")) for k,v in idx.items()},
              sizes={k:len(v) for k,v in idx.items()})
    return x,y,task,idx,meta


def prepare():
    if CONFIG.exists(): raise FileExistsError(CONFIG)
    assert len(NAMES)==13 and len(JOBS)==65
    splits={f"{n}-{s}":masked(n,s)[-1] for n,s in JOBS}
    DEST.mkdir(parents=True,exist_ok=True)
    write_json(CONFIG,dict(refit=REFIT,outage=OUTAGE,seeds=panel.SEEDS,names=NAMES,source_hashes=text_hashes(),
        splits=splits,cpu=4,refit_repeats=3,blas_threads=1,python=platform.python_version(),
        libraries={p:importlib.metadata.version(p) for p in ("xgboost","numpy","pandas","scikit-learn","scipy","numba","threadpoolctl")}))
    print(f"Preflight locked {len(splits)} blinded input/split fingerprints")


def pin():
    if os.name=="nt":
        lib=ctypes.WinDLL("kernel32",use_last_error=True)
        if not lib.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<4)):
            raise ctypes.WinError(ctypes.get_last_error())
    else: os.sched_setaffinity(0,{4})


def outage_forecast(x,y,task,idx):
    fit,vote,test=(idx[k] for k in ("fit","vote","test"))
    model=fit_additive(x[fit],y[fit],task,"spline")
    center=model.parts(x[fit]).mean(axis=0)
    offset=model.intercept+center.sum(axis=0)
    v=model.parts(x[vote])-center
    target=np.eye(len(offset))[y[vote]] if task=="classification" else y[vote,None]
    gram,d,_=stats(v,target,offset)
    parts=model.parts(x[test])-center
    normal,selected={},{}
    for arm in OUTAGE:
        if arm=="full": cols=np.arange(x.shape[1])
        else:
            order=(np.argsort(-gram.diagonal(),kind="stable") if arm=="energy" else
                   exchange_half(gram,d,arm.removeprefix("exchange_")))
            cols=order[:(x.shape[1]+1)//2]
        pred=offset+parts[:,cols].sum(axis=1)
        normal[arm]=pred if task=="classification" else pred.ravel()
        selected[arm]=cols.tolist()
    return normal,selected,dict(centered_shapes=parts,offset=offset)


def verify_job(folder,fingerprint):
    rec=json.loads((folder/"receipt.json").read_bytes())
    assert rec["preflight"]==fingerprint
    for f in FILES: assert hash_file(folder/f)==rec["forecasts"][f]
    assert set(rec["refit"])==set(REFIT) and set(rec["outage"])==set(OUTAGE)
    with np.load(folder/"refit.npz") as z:
        assert set(z.files)=={m+"_"+p for m in REFIT for p in ("adaptive","fixed")}
        for key in z:
            assert z[key].shape[0]==len(rec["test"]) and np.isfinite(z[key]).all()
    with np.load(folder/"outage.npz") as z,np.load(folder/"terms.npz") as parts:
        assert set(z.files)==set(OUTAGE) and set(parts.files)=={"centered_shapes","offset"}
        shapes,offset=parts["centered_shapes"],parts["offset"]
        assert shapes.shape[:2]==(len(rec["test"]),rec["p"]) and np.isfinite(shapes).all()
        for arm,cols in rec["outage"].items():
            assert len(set(cols))==len(cols) and all(0<=j<rec["p"] for j in cols)
            pred=offset+shapes[:,cols].sum(axis=1)
            np.testing.assert_allclose(z[arm],pred if rec["task"]=="classification" else pred.ravel(),rtol=0,atol=1e-12)
    return rec


def run():
    config=json.loads(CONFIG.read_bytes())
    assert config["source_hashes"]==text_hashes()
    fingerprint=hash_file(CONFIG)
    pin()
    warm_budget()
    rng=np.random.default_rng(683)
    legacy.gain_rank("gain",rng.normal(size=(64,4)),np.arange(64)%2,"classification",{"fit":np.arange(64)},42)
    WORK.mkdir(parents=True,exist_ok=True)
    begin=perf_counter()
    for j,(n,s) in enumerate(JOBS):
        key=f"{n}-{s}"
        folder=WORK/key; folder.mkdir(exist_ok=True)
        if (folder/"receipt.json").exists():
            verify_job(folder,fingerprint)
            continue
        clock=perf_counter()
        timer=Timer(3600,lambda:os._exit(124)); timer.daemon=True; timer.start()
        try:
            x,y,task,idx,meta=masked(n,s)
            assert config["splits"][key]==meta
            refit,records,clocks={},{},{}
            order=np.random.default_rng(3107+s+j).permutation(REFIT)
            for rep in range(3):
                for arm in np.roll(order,rep):
                    rec,pred=refit_select(x,y,task,idx,s,arm)
                    times={k:rec[k] for k in ("selection_s","deployment_s","fixed_extra_s")}
                    clocks.setdefault(arm,[]).append(dict(repeat=rep,**times))
                    if rep:
                        assert rec["support"]==records[arm]["support"] and rec["fixed"]==records[arm]["fixed"]
                    else: records[arm]=rec
                    for policy,array in pred.items():
                        k=arm+"_"+policy
                        if rep: np.testing.assert_array_equal(refit[k],array)
                        else: refit[k]=array
            normal,selected,parts=outage_forecast(x,y,task,idx)
            for file,arrays in zip(FILES,(refit,normal,parts),strict=True):
                tmp=folder/(file+".partial")
                with tmp.open("wb") as f: np.savez_compressed(f,**arrays)
                tmp.replace(folder/file)
            write_json(folder/"receipt.json",dict(name=n,seed=s,preflight=fingerprint,p=x.shape[1],task=task,
                refit=records,clocks=clocks,outage=selected,test=idx["test"].tolist(),
                forecasts={f:hash_file(folder/f) for f in FILES},repeat_order=[np.roll(order,i).tolist() for i in range(3)]))
            verify_job(folder,fingerprint)
            took=perf_counter()-begin
            print(f"Sealed {j+1}/{len(JOBS)} {key} ({perf_counter()-clock:.1f}s); elapsed {took:.0f}s, ETA {took*(len(JOBS)-j-1)/(j+1):.0f}s",flush=True)
        finally: timer.cancel()


def seal():
    fingerprint=hash_file(CONFIG)
    assert json.loads(CONFIG.read_bytes())["source_hashes"]==text_hashes()
    expected={f"{n}-{s}" for n,s in JOBS}
    assert {p.name for p in WORK.iterdir() if p.is_dir()}==expected
    locks={}
    bundle=DEST/"forecasts.zip"
    with ZipFile(bundle.with_suffix(".partial"),"w",ZIP_DEFLATED,compresslevel=6) as z:
        for key in sorted(expected):
            folder=WORK/key
            r=verify_job(folder,fingerprint)
            locks[key]={"receipt.json":hash_file(folder/"receipt.json"),**r["forecasts"]}
            for file in locks[key]: z.write(folder/file,key+"/"+file)
    bundle.with_suffix(".partial").replace(bundle)
    with ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==4*len(expected)
        for key,row in locks.items():
            for file,digest in row.items(): assert hashlib.sha256(z.read(key+"/"+file)).hexdigest()==digest
    write_json(DEST/"sealed.json",dict(preflight=fingerprint,bundle=hash_file(bundle),files=locks))
    print(f"Sealed {len(locks)} splits; bundle {bundle.stat().st_size:,} bytes")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["prepare","run","seal"])
    op=cli.parse_args().action
    with threadpool_limits(1): {"prepare":prepare,"run":run,"seal":seal}[op]()
