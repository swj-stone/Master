"""Compare a short-circuit schedule and a reference ddmin with the same vote predicate."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from experiments.context_study.working_set import oracle, check_local
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_witness import _sample
from minppi_mofs.utility import compute_utility


OUT=ROOT/"results/context_study/schedules"


def ddmin(columns, accepts):
    """Zeller's complement-only algorithm, with a feasible initial input."""
    live=list(columns)
    granularity=2
    while len(live)>1:
        span=len(live)//granularity
        for start in range(0,len(live),span):
            candidate=live[:start]+live[start+span:]
            if accepts(frozenset(candidate)):
                live=candidate
                granularity=min(len(live),max(2,granularity-1))
                break
        else:
            if granularity==len(live):
                break
            granularity=min(len(live),2*granularity)
    return frozenset(live)


def solve(arm, order, family, ballot):
    sequence=list(family)
    trajectory=[]
    def accepts(support):
        for i,extension in enumerate(sequence):
            if sum(ballot(support,extension))<3:
                if arm=="recent" and i>1:
                    sequence.insert(1,sequence.pop(i))
                trajectory.append((tuple(sorted(support)),False))
                return False
        trajectory.append((tuple(sorted(support)),True))
        return True
    if arm=="ddmin":
        support=ddmin(reversed(order),accepts)
    else:
        # The stock search needs only a Boolean decision. All real ballots live in accepts().
        support,_=search(order,[()],lambda s,_: (accepts(s),)*4)
    return support,trajectory


def trial(name,fold):
    folder=OUT/name/f"fold-{fold}"
    source=PRIMARY/name/f"fold-{fold}"
    rec=json.loads((source/"selection.json").read_bytes())
    old=json.loads((source/"completed.json").read_bytes())
    path=DATA/name/f"{name}.tsv.gz"
    assert digest(path)==old["data_sha256"] and digest(source/"selection.json")==old["selection_sha256"]
    frame=pd.read_csv(path,sep="\t")
    y=frame.pop("target").map({v:i for i,v in enumerate(rec["class_order"])}).to_numpy()
    x=frame.to_numpy(dtype=np.float32)
    train,rank,test=(np.asarray(rec[t]) for t in ("train","rank","test"))
    seed=20421+fold
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xr=imp.transform(x[train]),imp.transform(x[rank])
    estimator=fit(xt,y[train],seed)
    task="binary" if estimator.n_classes_==2 else "multiclass"
    rows,weights=_sample(y[rank],task,np.random.SeedSequence([seed,659]))
    ids=np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    bg=np.median(xt[ids],axis=0)
    rare=task=="binary" and min(np.bincount(y[rank]))<=.05*len(rank)
    ordered=rec["order"]
    k=rec["anchor_k"]
    anchor=frozenset(ordered[:k])
    family=separating_contexts(ordered[k:])
    def fresh():
        return oracle(estimator,xr[rows],bg,y[rank][rows],weights,anchor,rare)
    arms=["direct","recent","ddmin"]
    offset=(fold+config()["datasets"].index(name))%len(arms)
    results,selected,traces={}, {}, {}
    for arm in arms[offset:]+arms[:offset]:
        start=perf_counter()
        ballot,cache=fresh()
        live,trace=solve(arm,ordered[:k],family,ballot)
        results[arm]=dict(dataset=name,fold=fold,arm=arm,k=len(live),queries=len(cache),
                          seconds=perf_counter()-start,proposals=len(trace))
        selected[arm]=sorted(live)
        traces[arm]=trace
        referee,_=fresh()
        check_local(live,family,referee)
    if traces["recent"]!=traces["direct"] or selected["direct"]!=rec["selected"]["antichain"]:
        raise AssertionError("The direct and move-to-front decisions or reference differ")
    if results["direct"]["queries"]!=rec["traces"]["antichain"]["mask_calls"]:
        raise AssertionError("Canonical baseline query count differs")
    folder.mkdir(parents=True,exist_ok=True)
    store(folder/"selection.json",dict(selected=selected,traces=traces,source=old["selection_sha256"]))
    dev=np.sort(np.r_[train,rank])
    transform=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe=transform.transform(x[dev]),transform.transform(x[test])
    preds={}
    for arm in arms:
        cols=selected[arm]
        start=perf_counter()
        p=fit(xd[:,cols],y[dev],seed).predict_proba(xe[:,cols])
        results[arm].update(refit_s=perf_counter()-start,auc=compute_utility(y[test],p,metric="ovr_auc"),
                            accuracy=accuracy_score(y[test],p.argmax(axis=1)))
        preds[arm]=p
    with np.load(source/"predictions.npz") as frozen:
        for arm in ("direct","recent"):
            np.testing.assert_array_equal(preds[arm],frozen["antichain"])
    np.savez_compressed(folder/"predictions.npz",row_id=test,y=y[test],**preds)
    store(folder/"completed.json",dict(rows=list(results.values()),predictions=digest(folder/"predictions.npz"),
                                       selection=digest(folder/"selection.json")))


def main():
    if len(sys.argv)==3:
        return trial(sys.argv[1],int(sys.argv[2]))
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/"configs/context_study/schedule_controls.json",PRIMARY/"protocol-lock.json",
           ROOT/"experiments/context_study/working_set.py",ROOT/"src/minppi_mofs/context_vote.py",
           ROOT/"src/minppi_mofs/context_design.py"]
    terms={p.name:digest(p) for p in paths}
    spec=OUT/"protocol.json"
    if spec.exists() and json.loads(spec.read_bytes())!=terms:
        raise ValueError("A frozen schedule or input changed")
    store(spec,terms)
    records=[]
    for name in config()["datasets"]:
        for fold in range(3):
            folder=OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").is_file():
                subprocess.run([sys.executable,"-m","experiments.context_study.schedule_controls",name,str(fold)],
                               check=True,timeout=3600)
            rec=json.loads((folder/"completed.json").read_bytes())
            if digest(folder/"selection.json")!=rec["selection"] or digest(folder/"predictions.npz")!=rec["predictions"]:
                raise ValueError("A schedule receipt changed")
            records.extend(rec["rows"])
        print(f"Replayed {name}",flush=True)
    with (OUT/"scores.csv").open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
