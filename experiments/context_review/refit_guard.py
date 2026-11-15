"""Adjudicate a frozen vote proposal with two fresh refits on disjoint guard rows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.context_review.refit_target import metrics
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import ROOT, DATA, OUT as PRIMARY, config, digest
from experiments.context_study.selection import select_vote
from minppi_mofs.piv3_compact_tree import select as compact_gain
from minppi_mofs.piv3_compact_witness import _within


DESIGN = ROOT/"configs/context_review/refit_guard.json"
OUT = ROOT/"results/context_review/refit-guard"


def predict(cols, x_train, x_eval, y_train, seed):
    return fit(x_train[:,cols],y_train,seed).predict_proba(x_eval[:,cols])


def run(name,fold):
    source = PRIMARY/name/f"fold-{fold}"
    folder = OUT/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    old = json.loads((source/"selection.json").read_bytes())
    receipt = json.loads((source/"completed.json").read_bytes())
    table = DATA/name/f"{name}.tsv.gz"
    assert digest(table)==receipt["data_sha256"] and digest(source/"selection.json")==receipt["selection_sha256"]
    frame = pd.read_csv(table,sep="\t")
    y = frame.pop("target").map({c:i for i,c in enumerate(old["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train, rank, test = [np.asarray(old[k]) for k in ("train","rank","test")]
    proposal, guard = train_test_split(rank,test_size=.5,random_state=39823+fold,stratify=y[rank])
    if set(y[proposal])!=set(y[train]) or set(y[guard])!=set(y[train]):
        raise ValueError(f"{name} fold {fold}: one of the prespecified partitions lacks a class")
    seed = 20421+fold
    task = "binary" if len(old["class_order"])==2 else "multiclass"
    rare = bool(task=="binary" and min(np.bincount(y[proposal]))<=.05*len(proposal))
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xr = imp.transform(x[train]),imp.transform(x[proposal])
    base = fit(xt,y[train],seed)
    names = [f"c{i}" for i in range(x.shape[1])]
    _,info = compact_gain(base,pd.DataFrame(xt,columns=names),y[train],pd.DataFrame(xr,columns=names),y[proposal],
                          task=task,utility="ovr_auc",seed=seed,aggregate="gain")
    order = [int(c[1:]) for c in info["order"]]
    k = info["k"]
    voted,vote_trace = select_vote("antichain",base,xt,xr,y[proposal],order,k,fold)
    candidates = dict(anchor=tuple(sorted(order[:k])), vote=tuple(voted),
                      same_width=tuple(sorted(order[:len(voted)])),half=tuple(sorted(order[:(k+1)//2])))
    store(folder/"proposals.json",dict(train=train.tolist(),proposal=proposal.tolist(),guard=guard.tolist(),
          donor=receipt["selection_sha256"],test=test.tolist(),order=order,candidates=candidates,rare=rare,
          checks=vote_trace["mask_calls"]))
    inner = np.sort(np.r_[train,proposal])
    transform = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[inner])
    xi,xg = transform.transform(x[inner]),transform.transform(x[guard])
    guard_probs = {s:predict(list(s),xi,xg,y[inner],seed) for s in set(candidates.values())}
    judgment = {arm:metrics(y[guard],guard_probs[s],rare) for arm,s in candidates.items()}
    selected = candidates.copy()
    passes = {}
    for arm,s in candidates.items():
        if arm=="anchor":
            continue
        passes[arm] = _within(judgment[arm],judgment["anchor"])
        selected[arm+"_guard"] = s if passes[arm] else candidates["anchor"]
    store(folder/"selection.json",dict(selected=selected,guard_metrics=judgment,guard_pass=passes,
          unique_guard_fits=len(guard_probs),proposals_sha256=digest(folder/"proposals.json")))
    dev = np.sort(np.r_[train,rank])
    final = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe = final.transform(x[dev]),final.transform(x[test])
    prediction = {s:predict(list(s),xd,xe,y[dev],seed) for s in set(selected.values())}
    results = {arm:metrics(y[test],prediction[s],rare) for arm,s in selected.items()}
    rows = [dict(dataset=name,fold=fold,arm=arm,k=len(s),width=x.shape[1],anchor_k=k,
                 fallback=(arm.endswith("_guard") and not passes[arm.removesuffix("_guard")]),
                 exceeds=not _within(results[arm],results["anchor"]),**results[arm]) for arm,s in selected.items()]
    np.savez_compressed(folder/"predictions.npz",row_id=test,y=y[test],guard_row=guard,guard_y=y[guard],
                        **{arm:prediction[s] for arm,s in selected.items()},
                        **{"inner_"+arm:guard_probs[s] for arm,s in candidates.items()})
    store(folder/"completed.json",dict(rows=rows,protocol=digest(OUT/"protocol.json"),
          selection=digest(folder/"selection.json"),predictions=digest(folder/"predictions.npz")))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",nargs=2)
    args=p.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            return run(args.worker[0],int(args.worker[1]))
    OUT.mkdir(parents=True,exist_ok=True)
    files=[DESIGN,Path(__file__),PRIMARY/"protocol-lock.json",ROOT/"experiments/context_review/refit_target.py"]
    terms={f.name:digest(f) for f in files}
    base=OUT/"protocol.json"
    if base.exists() and json.loads(base.read_bytes())!=terms:
        raise ValueError("A frozen study input changed")
    store(base,terms)
    rows=[]
    for name in config()["datasets"]:
        for fold in range(3):
            folder=OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                subprocess.run([sys.executable,"-m","experiments.context_review.refit_guard","--worker",
                                name,str(fold)],check=True,timeout=3600)
            rec=json.loads((folder/"completed.json").read_bytes())
            assert rec["protocol"]==digest(base) and rec["selection"]==digest(folder/"selection.json")
            assert rec["predictions"]==digest(folder/"predictions.npz")
            rows.extend(rec["rows"])
        print(f"{name}: {len(rows)} arm-folds",flush=True)
    df=pd.DataFrame(rows)
    df["retained"]=df.k/df.width
    df.to_csv(OUT/"per-fold.csv",index=False)
    data=df.groupby(["dataset","arm"])[["auc","accuracy","retained","fallback","exceeds"]].mean()
    data.to_csv(OUT/"by-dataset.csv")
    data.groupby("arm").mean().to_csv(OUT/"summary.csv")


if __name__=="__main__":
    main()
