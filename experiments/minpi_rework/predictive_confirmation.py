"""Freeze supported models and outer game increments before evaluating the seven-source panel."""
import argparse
import ctypes
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import predictive_guard as guard, predictive_mask_study as study
from experiments.minpi_rework import quota_coalition_rf as rf,quota_compress_tree as tree,validation_predictive as source
from minppi_mofs import predictive_quota,availability_minpi


OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-confirmation"
RUN=b.ROOT/"results/minpi_rework/predictive-confirmation"
PACKAGE=b.ROOT/"packages/minpi-predictive-seven-forecasts.zip"
FILES=("forecasts.npz","increments.npz","incidence.npz","selection.json")


def jobs():
    return [(name,seed) for name in source.SOURCES for seed in range(42,47)]


def hashes():
    paths=[Path(m.__file__) for m in (b,guard,study,rf,tree,source,predictive_quota,availability_minpi)]
    paths.extend([Path(__file__),Path(__file__).with_suffix(".md")])
    paths.extend(p for p in Path(source.__file__).parent.glob("validation_predictive*") if p.is_file())
    return {p.name:b.digest(p) for p in paths}


def collect():
    # Warm compiled code before the recorded per-job timers.
    tiny=predictive_quota.PredictiveGame.from_paths(sparse.eye(3,format="csc"),sparse.eye(3,format="csr"),[1.,1.,1.])
    tiny.ranked_path()
    tiny.path_curves(np.arange(3))
    dummy=np.arange(60,dtype=float).reshape(20,3)
    for task,y in (("classification",np.arange(20)%2),("regression",np.arange(20))):
        learner=rf.fit(dummy,y,task,42,n_estimators=2,max_depth=2)
        learner.game(dummy[:4])
        learner.predict(dummy[:4,:1],[0])
    OUT.mkdir(exist_ok=True,parents=True)
    RUN.mkdir(exist_ok=True,parents=True)
    lock={}
    src=hashes()
    start=perf_counter()
    for i,(name,seed) in enumerate(jobs()):
        folder=RUN/f"{name}-{seed}"
        folder.mkdir(exist_ok=True)
        receipt=folder/"selection.json"
        if not receipt.exists():
            tick=perf_counter()
            timer=Timer(3600,lambda:os._exit(124))
            timer.daemon=True
            timer.start()
            try:
                x,y,task,idx=source.split(name,seed)
                held=y[idx["test"]]
                if not (np.isnan(held).all() if task=="regression" else (held==-1).all()):
                    raise ValueError("Test targets must be masked")
                model,records,preds=guard.select(x,y,task,idx,seed)
                np.testing.assert_allclose(model.predict(x[idx["test"]]),preds["full"],atol=2e-10,rtol=2e-10)
                test_game=model.game(x[idx["test"]])
                sparse.save_npz(folder/"increments.npz",test_game.a)
                sparse.save_npz(folder/"incidence.npz",test_game.incidence)
                np.savez_compressed(folder/"forecasts.npz",**preds)
                data=dict(source=src,dataset=name,seed=seed,task=task,idx={k:v.tolist() for k,v in idx.items()},
                   empty=model.empty.tolist(),records=records,files={n:b.digest(folder/n) for n in FILES[:-1]})
                tmp=receipt.with_suffix(".tmp")
                tmp.write_text(json.dumps(data),encoding="utf8")
                tmp.replace(receipt)
                elapsed=perf_counter()-start
                print(f"[{i+1}/{len(jobs())}] {name}/{seed} {perf_counter()-tick:.1f}s, total {elapsed:.1f}s ETA {elapsed*(len(jobs())-i-1)/(i+1):.1f}s",flush=True)
            finally:
                timer.cancel()
        rec=json.loads(receipt.read_bytes())
        if src!=rec["source"] or any(b.digest(folder/n)!=h for n,h in rec["files"].items()):
            raise ValueError("Stale predictive selection")
        lock[folder.name]=b.digest(receipt)
    (OUT/"lock.json").write_text(json.dumps(dict(source=src,receipts=lock),indent=2),encoding="utf8")
    PACKAGE.parent.mkdir(exist_ok=True)
    with ZipFile(PACKAGE,"w",ZIP_DEFLATED) as bundle:
        bundle.write(OUT/"lock.json","lock.json")
        for key in lock:
            for file in FILES:
                bundle.write(RUN/key/file,f"{key}/{file}")
    (OUT/"forecast-sha256.txt").write_text(b.digest(PACKAGE)+"  "+PACKAGE.name+"\n",encoding="utf8")


def score():
    lock=json.loads((OUT/"lock.json").read_bytes())
    if set(lock["receipts"])!={f"{name}-{seed}" for name,seed in jobs()} or lock["source"]!=hashes():
        raise ValueError("Incomplete or obsolete lock")
    rows=[]
    for name,seed in jobs():
        folder=RUN/f"{name}-{seed}"
        if b.digest(folder/"selection.json")!=lock["receipts"][folder.name]:
            raise ValueError("Receipt mismatch")
        rec=json.loads((folder/"selection.json").read_bytes())
        assert all(b.digest(folder/n)==h for n,h in rec["files"].items())
        metrics=source.score(name,seed,folder/"forecasts.npz",rec["files"]["forecasts.npz"])
        x,y,task,idx=source._partition(name,seed)
        assert {k:v.tolist() for k,v in idx.items()}==rec["idx"]
        test=y[idx["test"]]
        empty=np.array(rec["empty"])
        target=np.eye(len(empty))[test.astype(int)] if task=="classification" else test[:,None]
        st=predictive_quota.PredictiveGame.from_paths(sparse.load_npz(folder/"increments.npz"),
               sparse.load_npz(folder/"incidence.npz"),(target-empty).reshape(-1))
        p=st.structure.incidence.shape[1]
        for arm,record in rec["records"].items():
            mask=np.zeros(p,bool)
            mask[record["support"]]=True
            risk=st.conditional_risk(mask)[1:]*(p/np.arange(1,p+1))/max(st.null,1e-100)
            rows.append(dict(dataset=name,seed=seed,task=task,method=arm,outer_worst=risk.max(),
                outer_positive=max(0,risk.max()),outer_intact=risk[-1],outer_mean=risk.mean(),
                outer_worst_size=int(risk.argmax()+1),**record,**metrics[arm]))
    pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=["collect","score"])
    if os.name=="nt":
        ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6))
    with threadpool_limits(1):
        (collect if cli.parse_args().action=="collect" else score)()
