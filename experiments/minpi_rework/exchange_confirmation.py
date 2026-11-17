"""Select on blinded sources, freeze all predictions, then score the committed lock."""
import argparse
import ctypes
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b,legacy_gain
from experiments.minpi_rework import (radius_study,validation_panel,validation_wide,
                                      exchange_benchmark,budget)
from minppi_mofs import additive_minpi,exchange_minpi,exchange_bnb,radius_minpi,sector_minpi


METHODS=("width","unrestricted","bnb","sector_both","sector_plane","energy","greedy","gain","full")
EXACT=METHODS[:5]
RUN=b.ROOT/"results/minpi_rework/exchange-confirmation"
OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/exchange-confirmation"
PROTOCOL=Path(__file__).with_suffix(".md")


def signature():
    modules=[b,legacy_gain,radius_study,validation_panel,validation_wide,additive_minpi,exchange_minpi,
             exchange_bnb,radius_minpi,sector_minpi,exchange_benchmark,budget]
    paths=[Path(__file__),PROTOCOL,validation_panel.MANIFEST, *[Path(m.__file__) for m in modules],
           Path(validation_wide.__file__).with_name("validation_wide_sources.json"),legacy_gain.ORIGINAL]
    return {p.name:b.digest(p) for p in paths}


def jobs():
    for label,source in (("core",validation_panel),("wide",validation_wide)):
        for name in source.SOURCES:
            for seed in range(42,47):
                yield label,source,name,seed


def affinity():
    if os.name=="nt":
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6)):
            raise ctypes.WinError(ctypes.get_last_error())


def select():
    affinity()
    radius_study.warm()
    start=perf_counter()
    sources=signature()
    all_jobs=list(jobs())
    RUN.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    lock={}
    for n,(panel,source,name,seed) in enumerate(all_jobs):
        path=RUN/f"{name}-{seed}"
        path.mkdir(exist_ok=True)
        receipt=path/"selection.json"
        if receipt.exists():
            info=json.loads(receipt.read_bytes())
            if info["source"]!=sources or info["forecast"]!=b.digest(path/"predictions.npz"):
                raise ValueError(f"Stale receipt at {path}")
        else:
            timer=Timer(3600,lambda:os._exit(124))
            timer.daemon=True
            timer.start()
            try:
                x,y,task,idx=source.split(name,seed)
                records,forecasts={},{}
                order=np.random.default_rng(707+seed+n).permutation(METHODS)
                begin=perf_counter()
                for repeat in range(3):
                    for method in np.roll(order,repeat*3):
                        rec,pred=radius_study.run_one(x,y,task,idx,seed,method)
                        for budget_type,item in rec.items():
                            key=method+"_"+budget_type
                            if repeat:
                                assert records[key][0]["support"]==item["support"]
                                np.testing.assert_array_equal(forecasts[key],pred[budget_type])
                            records.setdefault(key,[]).append(item)
                            forecasts[key]=pred[budget_type]
                for mode in ("sparse","dense"):
                    vals={records[m+"_"+mode][0]["utility_integer"] for m in EXACT}
                    if len(vals)!=1:
                        raise ArithmeticError(f"Distinct exact objective values: {name}, {seed}, {mode}: {vals}")
                np.savez_compressed(path/"predictions.npz",**forecasts)
                info=dict(panel=panel,dataset=name,seed=seed,task=task,source=sources,
                          splits={k:v.tolist() for k,v in idx.items()},forecast=b.digest(path/"predictions.npz"),
                          records=records)
                tmp=receipt.with_suffix(".tmp")
                tmp.write_text(json.dumps(info),encoding="utf8")
                tmp.replace(receipt)
                total=perf_counter()-start
                print(f"[{n+1}/{len(all_jobs)}] {name}/{seed} {perf_counter()-begin:.1f}s, elapsed {total:.1f}s ETA {total*(len(all_jobs)-n-1)/(n+1):.1f}s",flush=True)
            finally:
                timer.cancel()
        lock[path.name]=b.digest(receipt)
    target=OUT/"lock.json"
    target.write_text(json.dumps(dict(source=sources,receipts=lock),indent=2),encoding="utf8")
    package=b.ROOT/"packages/minpi-exchange-forecasts.zip"
    with ZipFile(package,"w",compression=ZIP_DEFLATED,compresslevel=6) as z:
        z.write(target,"lock.json")
        for key in lock:
            for file in ("selection.json","predictions.npz"):
                z.write(RUN/key/file,f"{key}/{file}")
    (OUT/"forecast-sha256.txt").write_text(b.digest(package)+"  "+package.name+"\n",encoding="utf8")


def score():
    declared=list(jobs())
    seal=json.loads((OUT/"lock.json").read_bytes())
    if seal["source"]!=signature() or set(seal["receipts"])!={f"{n}-{s}" for _,_,n,s in declared}:
        raise ValueError("Incomplete or stale panel")
    data=[]
    for panel,source,name,seed in declared:
        sub=RUN/f"{name}-{seed}"
        path=sub/"selection.json"
        if b.digest(path)!=seal["receipts"][sub.name]:
            raise ValueError(path)
        info=json.loads(path.read_bytes())
        metrics=source.score(name,seed,sub/"predictions.npz",info["forecast"])
        if set(metrics)!={m+"_"+k for m in METHODS for k in ("sparse","dense")}:
            raise ValueError("Arm missing from frozen archive")
        for key,scores in metrics.items():
            method,mode=key.rsplit("_",1)
            reps=info["records"][key]
            rec=reps[0].copy()
            for col in ("search_s","selection_s","deployment_s","setup_s"):
                rec[col]=np.median([r[col] for r in reps])
            data.append(dict(panel=panel,dataset=name,task=info["task"],seed=seed,mode=mode,**rec,**scores))
    pd.DataFrame(data).to_csv(OUT/"scores.csv",index=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["select","score"])
    args=cli.parse_args()
    with threadpool_limits(1):
        (select if args.action=="select" else score)()
