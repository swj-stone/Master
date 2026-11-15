"""Test a one-deletion maximin challenger against dropping the weakest Gain feature."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review.refit_guard import OUT as GUARD, predict
from experiments.context_review.refit_target import metrics
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import ROOT, DATA, OUT as PRIMARY, config, digest
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.maximin_challenger import nominate
from minppi_mofs.piv3_compact_witness import _sample, _within

DESIGN = ROOT/"configs/context_review/challenger.json"
OUT = ROOT/"results/context_review/challenger"


def run(name,fold):
    source = PRIMARY/name/f"fold-{fold}"
    previous = GUARD/name/f"fold-{fold}"
    folder = OUT/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    old = json.loads((source/"selection.json").read_bytes())
    parent = json.loads((previous/"proposals.json").read_bytes())
    adjudicated = json.loads((previous/"selection.json").read_bytes())
    table = DATA/name/f"{name}.tsv.gz"
    receipt = json.loads((source/"completed.json").read_bytes())
    checks = json.loads((previous/"completed.json").read_bytes())
    assert digest(table)==receipt["data_sha256"] and digest(source/"selection.json")==parent["donor"]
    assert digest(previous/"proposals.json")==adjudicated["proposals_sha256"]
    assert digest(previous/"selection.json")==checks["selection"]
    frame = pd.read_csv(table,sep="\t")
    y = frame.pop("target").map({v:i for i,v in enumerate(old["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train, prop, guard, test = [np.asarray(parent[k]) for k in ("train","proposal","guard","test")]
    seed = 20421+fold
    order, rare = parent["order"], parent["rare"]
    anchor = frozenset(parent["candidates"]["anchor"])
    live = frozenset(parent["candidates"]["vote"])
    live_order = [i for i in order if i in live]
    family = separating_contexts(order[len(anchor):])
    transform = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xr = transform.transform(x[train]),transform.transform(x[prop])
    model = fit(xt,y[train],seed)
    task = "binary" if len(old["class_order"])==2 else "multiclass"
    ids, weights = _sample(y[prop],task,np.random.SeedSequence([seed,659]))
    bg = np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    args = (model,xr[ids],np.median(xt[bg],axis=0),y[prop][ids],weights,anchor,rare)
    ballot, seen = oracle(*args)
    best, trace = nominate(live_order,family,ballot)
    eager, all_masks = oracle(*args)
    exact, full_trace = nominate(live_order,family,eager,prune=False)
    assert best==exact and trace["min_yes"]==full_trace["min_yes"]
    weak = live-{live_order[-1]} if len(live)>1 else live
    candidates = {"maximin":sorted(best),"weak_gain":sorted(weak)}
    store(folder/"proposals.json",dict(candidates=candidates,incumbent=sorted(live),anchor=sorted(anchor),
          counts=trace,eager=full_trace,masks=len(seen),eager_masks=len(all_masks),
          source=digest(previous/"proposals.json"),source_selection=checks["selection"]))
    inner = np.sort(np.r_[train,prop])
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[inner])
    xi,xg = imp.transform(x[inner]),imp.transform(x[guard])
    guard_probs = {s:predict(list(s),xi,xg,y[inner],seed) for s in set(map(tuple,candidates.values()))}
    reference = adjudicated["guard_metrics"]["anchor"]
    fallback = adjudicated["selected"]["vote_guard"]
    selected = candidates.copy()
    for arm,s in candidates.items():
        ok = _within(metrics(y[guard],guard_probs[tuple(s)],rare),reference)
        selected[arm+"_guard"] = s if ok else fallback
    store(folder/"selection.json",dict(selected=selected,unique_extra_inner_refits=len(guard_probs),
          proposals=digest(folder/"proposals.json")))
    dev = np.sort(np.r_[inner,guard])
    final = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe = final.transform(x[dev]),final.transform(x[test])
    pred = {s:predict(list(s),xd,xe,y[dev],seed) for s in set(map(tuple,selected.values()))}
    assert digest(previous/"predictions.npz")==checks["predictions"]
    with np.load(previous/"predictions.npz") as previous_p:
        reference = metrics(y[test],previous_p["anchor"],rare)
        for arm,s in selected.items():
            if s==fallback:
                np.testing.assert_array_equal(pred[tuple(s)],previous_p["vote_guard"])
    rows=[]
    for arm,s in selected.items():
        score=metrics(y[test],pred[tuple(s)],rare)
        rows.append(dict(dataset=name,fold=fold,arm=arm,k=len(s),width=x.shape[1],current_k=len(live),
            reduced=len(s)<len(live),different=best!=weak,exceeds=not _within(score,reference),**score))
    np.savez_compressed(folder/"predictions.npz",row_id=test,y=y[test],guard_row=guard,guard_y=y[guard],
          **{arm:pred[tuple(s)] for arm,s in selected.items()},
          **{"inner_"+arm:guard_probs[tuple(s)] for arm,s in candidates.items()})
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
    paths=[DESIGN,Path(__file__),GUARD/"protocol.json",ROOT/"src/minppi_mofs/maximin_challenger.py"]
    lock={p.name:digest(p) for p in paths}
    base=OUT/"protocol.json"
    if base.exists() and json.loads(base.read_bytes())!=lock:
        raise ValueError("A frozen protocol input changed")
    store(base,lock)
    rows, costs=[],[]
    for name in config()["datasets"]:
        for fold in range(3):
            folder=OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                subprocess.run([sys.executable,"-m","experiments.context_review.challenger","--worker",name,str(fold)],
                               timeout=3600,check=True)
            rec=json.loads((folder/"completed.json").read_bytes())
            assert rec["protocol"]==digest(base) and rec["selection"]==digest(folder/"selection.json")
            assert rec["predictions"]==digest(folder/"predictions.npz")
            proposal=json.loads((folder/"proposals.json").read_bytes())
            costs.append(dict(dataset=name,fold=fold,masks=proposal["masks"],eager_masks=proposal["eager_masks"],
                              visits=proposal["counts"]["visits"],eager_visits=proposal["eager"]["visits"]))
            rows.extend(rec["rows"])
        print(f"{name}: {len(rows)} arm-folds",flush=True)
    df=pd.DataFrame(rows)
    df["retained"]=df.k/df.width
    df.to_csv(OUT/"per-fold.csv",index=False)
    pd.DataFrame(costs).to_csv(OUT/"costs.csv",index=False)
    df.groupby("arm")[["auc","accuracy","retained","reduced","exceeds","different"]].mean().to_csv(OUT/"summary.csv")


if __name__=="__main__":
    main()
