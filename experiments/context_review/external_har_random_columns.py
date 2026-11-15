"""Two outcome-blind random-column nulls at the separating selector's width on UCI HAR."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.context_review.external_har import OUT as BASE, ROOT, SEED, fit, load_member, original_archive, digest, store

SPEC = ROOT/"configs/context_review/external_har_random_columns.json"
OUT = ROOT/"results/context_review/external-har-random-columns"


def freeze():
    spec = json.loads(SPEC.read_bytes())
    separation, anchor = [BASE/a/"support.json" for a in ("antichain","gain")]
    k = len(json.loads(separation.read_bytes())["selected"])
    pool = [list(range(561)), json.loads(anchor.read_bytes())["selected"]]
    groups = {}
    for j,name in enumerate(spec["families"]):
        for d in range(spec["draws_each"]):
            rng = np.random.default_rng(np.random.SeedSequence([918273,j,d]))
            cols = sorted(map(int,rng.choice(sorted(pool[j]),k,replace=False)))
            groups[f"{name}_{d:02}"] = cols
    lock = dict(spec=digest(SPEC), runner=digest(Path(__file__)),primary_lock=digest(BASE/"lock.json"),
                donor=digest(separation),anchor=digest(anchor),supports=groups)
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/"supports.json"
    if path.exists() and json.loads(path.read_bytes())!=lock:
        raise ValueError("Changed frozen inputs for random-draw sensitivity")
    store(path,lock)
    print(f"Saved {len(groups)} outcome-blind supports at k={k}",flush=True)


def run():
    lock=json.loads((OUT/"supports.json").read_bytes())
    if lock["spec"]!=digest(SPEC) or lock["runner"]!=digest(Path(__file__)):
        raise ValueError("Changed random-column specification or runner")
    with original_archive() as z:
        x=load_member(z,"train","X")
        y=load_member(z,"train","y")-1
        xt=load_member(z,"test","X")
        subject=load_member(z,"test","subject")
    preds, times = {}, {}
    for name,cols in lock["supports"].items():
        start=perf_counter()
        preds[name]=fit(x[:,cols],y,SEED).predict_proba(xt[:,cols])
        times[name]=perf_counter()-start
    np.savez_compressed(OUT/"predictions.npz",**preds)
    store(OUT/"frozen.json",dict(support_sha256=digest(OUT/"supports.json"),
          probabilities_sha256=digest(OUT/"predictions.npz"),refit_times=times))
    with original_archive() as z:
        yt=load_member(z,"test","y")-1
    rows,subject_rows=[],[]
    for name,p in preds.items():
        correct=p.argmax(axis=1)==yt
        accuracies = []
        for s in np.unique(subject):
            value=correct[subject==s].mean()
            accuracies.append(value)
            subject_rows.append(dict(draw=name,subject=s,accuracy=value))
        rows.append(dict(draw=name,pool=name.rsplit("_",1)[0],k=len(lock["supports"][name]),
                         mean_subject_accuracy=np.mean(accuracies),pooled_accuracy=correct.mean(),
                         pooled_auc=roc_auc_score(yt,p,multi_class="ovr"),refit_s=times[name]))
    pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)
    pd.DataFrame(subject_rows).to_csv(OUT/"by-subject.csv",index=False)
    print(pd.DataFrame(rows).groupby("pool").mean_subject_accuracy.agg(["mean","min","max"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["freeze","run","worker"],default="run")
    args=parser.parse_args()
    if args.stage=="freeze":
        freeze()
    elif args.stage=="worker":
        with threadpool_limits(limits=1):
            run()
    else:
        freeze()
        subprocess.run([sys.executable,"-m","experiments.context_review.external_har_random_columns","--stage","worker"],
                       check=True, timeout=3600)
