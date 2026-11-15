"""Refit selected methods on exact-feature-vector disjoint folds and audit repeated rows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import ROOT, DATA, OUT, config, digest
from experiments.context_study.selection import select_vote, select_l1
from minppi_mofs.piv3_compact_tree import select as gain_anchor
from minppi_mofs.utility import compute_utility

DESIGN = ROOT/"configs/context_study/grouped.json"
STUDY = OUT.parent/"prospective-pmlb-duplicate-groups-r1"


def data(name):
    table = DATA/name/(name+".tsv.gz")
    sources = json.loads((OUT/"sources.json").read_bytes())
    if digest(table) != sources["sources"][name][table.name]["sha256"]:
        raise ValueError("Input table differs")
    frame = pd.read_csv(table, sep="\t")
    y = frame.pop("target").to_numpy()
    groups = pd.util.hash_pandas_object(frame,index=False).to_numpy()
    if len(np.unique(groups)) != len(frame.drop_duplicates()):
        raise ValueError("A feature-vector hash collides")
    return frame, y, groups


def audit():
    summary, restricted = [], []
    for name in config()["datasets"]:
        x, y, groups = data(name)
        conflicts = pd.DataFrame({"g":groups,"y":y}).groupby("g").y.nunique()
        count = 0
        for fold,(dev,test) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261115).split(x,y)):
            held = ~np.isin(groups[test],groups[dev])
            count += int((~held).sum())
            with np.load(OUT/name/f"fold-{fold}"/"predictions.npz") as pred:
                np.testing.assert_array_equal(test,pred["row_id"])
                labels = pred["y"][held]
                for arm in config()["arms"]:
                    p = pred[arm][held]
                    complete = len(np.unique(labels)) == p.shape[1]
                    restricted.append(dict(dataset=name,fold=fold,arm=arm,n_kept=len(labels),
                        n_test=len(test),auc=compute_utility(labels,p,metric="ovr_auc") if complete else np.nan,
                        accuracy=float((p.argmax(axis=1)==labels).mean()) if len(labels) else np.nan))
        summary.append(dict(dataset=name,n=len(y),distinct_inputs=len(np.unique(groups)),conflicting_groups=int((conflicts>1).sum()),
                            seen_test_rows=count,seen_fraction=count/len(y)))
    pd.DataFrame(summary).to_csv(STUDY/"duplicates.csv",index=False)
    pd.DataFrame(restricted).to_csv(STUDY/"unique-test-only.csv",index=False)
    return [r["dataset"] for r in summary if r["seen_fraction"]>.10]


def split(ids, y, groups, n, seed):
    allclasses = set(np.unique(y[ids]))
    for offset in range(128):
        cv = StratifiedGroupKFold(n,shuffle=True,random_state=seed+offset)
        folds = [(ids[a],ids[b]) for a,b in cv.split(ids,y[ids],groups[ids])]
        valid = [set(y[a])==allclasses and set(y[b])==allclasses for a,b in folds]
        if n==3 and all(valid):
            return folds,seed+offset
        if n==4 and any(valid):
            return folds[valid.index(True)],seed+offset
    raise ValueError("Unable to cover all classes with disjoint groups")


def run(name,fold):
    frame,y,groups = data(name)
    x = frame.to_numpy(dtype=np.float32)
    outer,outer_seed = split(np.arange(len(y)),y,groups,3,20261115)
    dev,test = outer[fold]
    seed = 20421+fold
    (train,rank),inner_seed = split(dev,y,groups,4,seed)
    for a,b in [(train,rank),(dev,test)]:
        if np.intersect1d(groups[a],groups[b]).size:
            raise ValueError("Feature vectors occur in both partitions")
    classes,counts = np.unique(y[train],return_counts=True)
    if len(classes)==2 and counts[0]<counts[1]:
        classes = classes[::-1]
    inverse = {v:i for i,v in enumerate(classes)}
    y = np.array([inverse[v] for v in y])
    task = "binary" if len(classes)==2 else "multiclass"
    path = STUDY/name/f"fold-{fold}"
    path.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    imp=SimpleImputer(strategy="median",keep_empty_features=True)
    xt,xv=imp.fit_transform(x[train]),imp.transform(x[rank])
    prep=perf_counter()-start
    model=fit(xt,y[train],seed)
    cols=[f"c{i}" for i in range(x.shape[1])]
    _,info=gain_anchor(model,pd.DataFrame(xt,columns=cols),y[train],pd.DataFrame(xv,columns=cols),
                       y[rank],task=task,utility="ovr_auc",seed=seed,aggregate="gain")
    order=[int(c[1:]) for c in info["order"]]
    k=info["k"]
    anchor_t=perf_counter()-start
    design=json.loads(DESIGN.read_text(encoding="utf-8-sig"))
    supports,traces,times={},{},{}
    for arm in design["arms"]:
        start=perf_counter()
        if arm=="full":
            s,t,shared=list(range(x.shape[1])),{},0.
        elif arm=="gain":
            s,t,shared=sorted(order[:k]),{},anchor_t
        elif arm=="l1":
            s,t=select_l1(xt,y[train],xv,y[rank],seed)
            shared=prep
        else:
            s,t=select_vote(arm,model,xt,xv,y[rank],order,k,fold)
            shared=anchor_t
        times[arm]=perf_counter()-start+shared
        supports[arm],traces[arm]=s,t
    store(path/"selection.json",dict(selected=supports,train=train.tolist(),rank=rank.tolist(),test=test.tolist(),
          outer_seed=outer_seed,inner_seed=inner_seed,class_order=classes.tolist(),order=order,anchor_k=k,traces=traces))
    start=perf_counter()
    imp=SimpleImputer(strategy="median",keep_empty_features=True)
    xd,xe=imp.fit_transform(x[dev]),imp.transform(x[test])
    prep=perf_counter()-start
    predictions={"y":y[test],"row_id":test}
    rows=[]
    for arm,s in supports.items():
        start=perf_counter()
        pred=fit(xd[:,s],y[dev],seed).predict_proba(xe[:,s])
        final=perf_counter()-start+prep
        predictions[arm]=pred
        rows.append(dict(dataset=name,fold=fold,arm=arm,k=len(s),width=x.shape[1],n_test=len(test),
                   auc=compute_utility(y[test],pred,metric="ovr_auc"),accuracy=float((pred.argmax(axis=1)==y[test]).mean()),
                   selection_s=times[arm],final_s=final,total_s=times[arm]+final,selected=json.dumps(s)))
    np.savez_compressed(path/"predictions.npz",**predictions)
    store(path/"completed.json",dict(records=rows,selection_sha256=digest(path/"selection.json"),
          protocol_sha256=digest(STUDY/"protocol-lock.json"),predictions_sha256=digest(path/"predictions.npz")))


def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    tasks=audit()
    lock=dict(primary_sha256=digest(OUT/"protocol-lock.json"),design_sha256=digest(DESIGN),
              source_sha256=digest(Path(__file__)),tasks=tasks)
    p=STUDY/"protocol-lock.json"
    if p.exists() and json.loads(p.read_bytes())!=lock:
        raise RuntimeError("Grouped protocol changed")
    store(p,lock)
    records,failures=[],[]
    for name in tasks:
        for fold in range(3):
            folder=STUDY/name/f"fold-{fold}"
            done=folder/"completed.json"
            if not done.exists():
                print(f"Running grouped {name} fold {fold+1}/3",flush=True)
                try:
                    subprocess.run([sys.executable,"-m","experiments.context_study.grouped","--worker",name,str(fold)],
                                   cwd=ROOT,timeout=3600,check=True)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                    failures.append((name,fold,str(exc)))
                    continue
            r=json.loads(done.read_bytes())
            if (r["protocol_sha256"]!=digest(p) or r["selection_sha256"]!=digest(folder/"selection.json")
                    or r["predictions_sha256"]!=digest(folder/"predictions.npz")):
                raise ValueError(f"Bad receipt: {folder}")
            records.extend(r["records"])
    pd.DataFrame(records).to_csv(STUDY/"per-fold.csv",index=False)
    store(STUDY/"failures.json",failures)
    if failures:
        raise RuntimeError(f"{len(failures)} failed folds")
    print(f"Completed {len(records)} grouped evaluations")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            run(args.worker[0],int(args.worker[1]))
    else:
        main()

