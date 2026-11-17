"""Evaluate the supervised RF selector on the original six hashed thesis datasets."""
import ctypes
import json
import os
from threading import Timer
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain as legacy,benchmark as b
from experiments.minpi_rework import predictive_guard as guard


RUN=b.ROOT/"results/minpi_rework/predictive-prepared"
OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-prepared"


def main():
    jobs=legacy.jobs()
    RUN.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    receipt={}
    start=perf_counter()
    for i,((name,seed),(directory,row)) in enumerate(sorted(jobs.items())):
        folder=RUN/f"{name}-{seed}"
        folder.mkdir(exist_ok=True)
        t=perf_counter()
        path=folder/"selection.json"
        if not path.exists():
            timer=Timer(3600,lambda:os._exit(124))
            timer.daemon=True
            timer.start()
            try:
                x,y,_,idx=legacy.read_split(directory,row,seed)
                task="regression" if row["task"]=="regression" else "classification"
                model,records,pred=guard.select(x,y,task,idx,seed)
                np.testing.assert_allclose(model.predict(x[idx["test"]]),pred["full"],rtol=2e-10,atol=2e-10)
                np.savez_compressed(folder/"forecasts.npz",**pred)
                current=dict(data=row["test_file_sha256"],forecast=b.digest(folder/"forecasts.npz"),
                       task=task,scale=float(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])),records=records)
                tmp=path.with_suffix(".tmp")
                tmp.write_text(json.dumps(current),encoding="utf8")
                tmp.replace(path)
            finally:
                timer.cancel()
        receipt[folder.name]=b.digest(path)
        elapsed=perf_counter()-start
        print(f"[{i+1}/30] {name}/{seed} {perf_counter()-t:.1f}s elapsed {elapsed:.1f}s ETA {elapsed*(29-i)/(i+1):.1f}s",flush=True)
    (OUT/"receipts.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")
    rows=[]
    for (name,seed),(directory,row) in jobs.items():
        folder=RUN/f"{name}-{seed}"
        rec=json.loads((folder/"selection.json").read_bytes())
        assert receipt[folder.name]==b.digest(folder/"selection.json")
        assert rec["forecast"]==b.digest(folder/"forecasts.npz") and rec["data"]==b.digest(directory/row["test_relpath"])
        truth=pd.read_csv(directory/row["test_relpath"],usecols=[row["target"]])[row["target"]].to_numpy()
        with np.load(folder/"forecasts.npz") as ar:
            for arm,record in rec["records"].items():
                score=b.metrics(truth,ar[arm],rec["task"],rec["scale"])
                rows.append(dict(dataset=name,seed=seed,method=arm,task=rec["task"],**record,**score))
    pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)


if __name__=="__main__":
    if os.name=="nt":
        ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<8))
    with threadpool_limits(1):
        main()
