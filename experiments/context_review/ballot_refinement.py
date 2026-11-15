"""Resolve ties in the worst ballot by one continuous mask measurement, then refit the nominee."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as primary
from experiments.context_review.timing_process import bind_logical_cpu
from experiments.context_study.data import digest

ROOT = bg.ROOT
DESIGN = ROOT / "configs/context_review/ballot_refinement.json"
EVIDENCE = ROOT / "thesis-paper/evidence/refinement"


def vote_first(ballot, candidates, family):
    """The running minimum can only fall; later candidates lose every equal-count tie."""
    order = sorted(candidates, key=lambda cols: -primary.rating(ballot,cols,[frozenset()],"point"))
    maximum, winner = -1, order[0]
    for candidate in order:
        count = primary.rating(ballot,candidate,family,"vote",best=maximum)
        if count > maximum:
            maximum,winner = count,candidate
        if maximum == 4:
            break
    return winner


def fold(src,dest):
    layout=bg.read(src/"layout.json")
    base=bg.read(src/"selection.json")
    assert base["layout"]==digest(src/"layout.json") and layout["inputs_sha256"]==digest(src/"inputs.npz")
    with np.load(src/"inputs.npz") as data:
        x,y,outer,train,prop,guard=[data[k] for k in ("x","y","x_outer","train","proposal","guard")]
    seed=20421+layout["fold"]
    start=perf_counter()
    transform=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xp=transform.transform(x[train]),transform.transform(x[prop])
    model=bg.fit(xt,y[train],seed)
    task="binary" if model.n_classes_==2 else "multiclass"
    rare=bool(task=="binary" and min(np.bincount(y[prop]))<=.05*len(prop))
    names=[f"c{i}" for i in range(x.shape[1])]
    _, info=bg.gain_anchor(model,pd.DataFrame(xt,columns=names),y[train],pd.DataFrame(xp,columns=names),
                           y[prop],task=task,utility="ovr_auc",seed=seed,aggregate="gain")
    order=[int(s[1:]) for s in info["order"]]
    k=info["k"]
    anchor=tuple(sorted(order[:k]))
    nominee=selected=anchor
    w=(k+1)//2
    masks=0
    passed=False
    nomination_start=perf_counter()
    if w<k:
        candidates=bg.candidate_pool(order[:k],w,1000+layout["id"],layout["fold"])[1]
        ballot=primary.build_ballot(model,xt,xp,y[prop],seed,anchor,rare)
        nominee=vote_first(ballot,candidates,primary.separating_contexts(order[k:]))
        masks=len(ballot.scores)
    nomination_seconds=perf_counter()-nomination_start
    if w<k:
        inner=np.sort(np.r_[train,prop])
        probabilities,_,_=bg.predict_sets(x,y,inner,x[guard],[anchor,nominee],seed)
        g={c:bg.metrics(y[guard],p,rare) for c,p in probabilities.items()}
        passed=bool(bg._within(g[nominee],g[anchor]))
        if passed:
            selected=nominee
    final,_,_=bg.predict_sets(x,y,np.arange(len(y)),outer,[selected],seed)
    wall=perf_counter()-start
    assert base["choices"]["gain"]["order"] == order and base["choices"]["gain"]["anchor"]==list(anchor)
    if base["pool"]:
        assert tuple(max(base["pool"],key=lambda c:(c["vote"],c["point"]))["columns"])==nominee
    with np.load(src/"predictions.npz") as earlier:
        for arm,choice in base["choices"].items():
            if choice["selected"]==list(selected):
                np.testing.assert_array_equal(final[selected],earlier[arm])
    dest.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(dest/"prediction.npz",p=final[selected])
    bg.store(dest/"sealed.json",dict(anchor=anchor,nominee=nominee,selected=selected,passed=passed,rare=rare,
             masks=masks,wall_s=wall,nomination_s=nomination_seconds,prediction=digest(dest/"prediction.npz"),
             base=digest(src/"selection.json")))


def settings(panel):
    key={"first":"transport-uci","wide":"wide-uci"}[panel]
    src=ROOT/"results/context_review"/key
    out=ROOT/"results/context_review"/("ballot-refinement-"+panel)
    return src,out


def run(panel):
    src,out=settings(panel)
    out.mkdir(parents=True,exist_ok=True)
    bind_logical_cpu()
    if panel=="first":
        keys = list(bg.read(primary.EVIDENCE/"prediction-lock.json")["selections"])
    else:
        # The wide cohort records ineligible folds before any selection is scored.
        keys = [p.parent.relative_to(src).as_posix() for p in sorted(src.glob("*/fold-*/selection.json"))]
        if not (src/"protocol.json").is_file() or not keys:
            raise ValueError("Missing wide prediction records")
    lock={k:digest(src/k/"selection.json") for k in keys}
    lock.update(design=digest(DESIGN),implementation=digest(Path(__file__)),
                inputs=digest(src/"protocol.json"))
    if (out/"protocol.json").exists() and bg.read(out/"protocol.json")!=lock:
        raise ValueError("Changed cohort or inputs; seal requires the complete primary cohort")
    bg.store(out/"protocol.json",lock)
    receipts={}
    for i,k in enumerate(keys,1):
        if not (out/k/"sealed.json").exists():
            fold(src/k,out/k)
        receipts[k]=digest(out/k/"sealed.json")
        print(f"[{i}/{len(keys)}] sealed {k}",flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    bg.store(EVIDENCE/(panel+"-lock.json"),dict(protocol=digest(out/"protocol.json"),selections=receipts))


def score(panel):
    src,out=settings(panel)
    lock=bg.read(EVIDENCE/(panel+"-lock.json"))
    assert lock["protocol"]==digest(out/"protocol.json")
    for key,sha in lock["selections"].items():
        rec=bg.read(out/key/"sealed.json")
        assert sha==digest(out/key/"sealed.json") and rec["prediction"]==digest(out/key/"prediction.npz")
        assert rec["base"]==digest(src/key/"selection.json")
    rows=[]
    for key in lock["selections"]:
        name,fold_key=key.split("/")
        new=bg.read(out/key/"sealed.json")
        base=bg.read(src/key/"selection.json")
        with np.load(src/key/"targets.npz") as truth,np.load(src/key/"predictions.npz") as priors,np.load(out/key/"prediction.npz") as new_pred:
            np.testing.assert_array_equal(priors["test_row"],truth["test_row"])
            y=truth["y"]
            anchor_metrics=bg.metrics(y,priors["anchor"],new["rare"])
            for arm in ("gain","vote","point","slack","refined"):
                ref=arm=="refined"
                c=new if ref else base["choices"][arm]
                costs=new if ref else base["costs"][arm]
                metrics=bg.metrics(y,new_pred["p"] if ref else priors[arm],new["rare"])
                fail=not bg._within(metrics,anchor_metrics)
                rows.append(dict(dataset=name,fold=int(fold_key[5:]),arm=arm,
                     nominee=" ".join(map(str,c["nominee"])),selected=" ".join(map(str,c["selected"])),
                     u=(1-len(c["selected"])/len(c["anchor"]))*(not fail),breach=int(fail),
                     masks=costs["masks"],wall_s=costs["wall_s"],nomination_s=costs["nomination_s"],**metrics))
    frame=pd.DataFrame(rows)
    frame.to_csv(EVIDENCE/(panel+"-folds.csv"),index=False)
    fields=["u","breach","auc","accuracy","masks","wall_s","nomination_s"]
    tasks=frame.groupby(["arm","dataset"])[fields].mean()
    tasks.groupby("arm").mean().to_csv(EVIDENCE/(panel+"-means.csv"))
    rows=[]
    for arm in ("gain","vote","point","slack"):
        diff=tasks.loc["refined"]-tasks.loc[arm]
        draws=np.random.default_rng(33119).integers(len(diff),size=(30000,len(diff)))
        for field in fields:
            values=diff[field].to_numpy()
            lo,hi=np.percentile(values[draws].mean(axis=1),[2.5,97.5])
            rows.append(dict(reference=arm,metric=field,delta=values.mean(),lo95=lo,hi95=hi))
    pd.DataFrame(rows).to_csv(EVIDENCE/(panel+"-contrasts.csv"),index=False)
    print(tasks.groupby("arm").mean().round(4))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("first","wide"))
    parser.add_argument("--score",action="store_true")
    opts=parser.parse_args()
    with threadpool_limits(limits=1):
        (score if opts.score else run)(opts.panel)
