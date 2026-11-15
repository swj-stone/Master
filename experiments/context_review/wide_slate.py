"""Seal every wide-panel half-width guard and final refit, including an eight-round nominee."""
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.low_fidelity_control import small_scores, minimum_slack

ROOT=bg.ROOT
RUN=ROOT/"results/context_review/wide-uci"
OUT=ROOT/"results/context_review/wide-slate"
EVIDENCE=ROOT/"thesis-paper/evidence/wide_slate"
DESIGN=ROOT/"configs/context_review/chance_nomination.json"


def run_fold(key):
    src,dest=RUN/key,OUT/key
    layout=bg.read(src/"layout.json")
    selection=bg.read(src/"selection.json")
    assert selection["layout"]==bg.digest(src/"layout.json") and layout["inputs_sha256"]==bg.digest(src/"inputs.npz")
    with np.load(src/"inputs.npz") as data:
        x,y,outer,train,prop,guard=[data[k] for k in ("x","y","x_outer","train","proposal","guard")]
    direct=selection["choices"]["gain"]
    supports=[tuple(direct["anchor"]),*(tuple(row["columns"]) for row in selection["pool"])]
    seed=20421+layout["fold"]
    rare=direct["rare"]
    scores,sec=[],0.
    nominee=0
    if len(supports)>1:
        scores,sec=small_scores(x,y,train,prop,supports,seed,rare)
        nominee=max(range(1,len(supports)),key=lambda i:minimum_slack(scores[i],scores[0]))
    g,_,_=bg.predict_sets(x,y,np.sort(np.r_[train,prop]),x[guard],supports,seed)
    o,_,_=bg.predict_sets(x,y,np.arange(len(y)),outer,supports,seed)
    guard_metrics=[bg.metrics(y[guard],g[s],rare) for s in supports]
    with np.load(src/"predictions.npz") as previous:
        np.testing.assert_array_equal(previous["anchor"],o[supports[0]])
        for arm,choice in selection["choices"].items():
            i=supports.index(tuple(choice["nominee"]))
            passed=bool(bg._within(guard_metrics[i],guard_metrics[0]))
            selected=supports[i] if passed else supports[0]
            assert list(selected)==choice["selected"]
            np.testing.assert_array_equal(previous[arm],o[selected])
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/"predictions.npz",**{f"{phase}{i}":yhat[c] for i,c in enumerate(supports)
                        for phase,yhat in (("g",g),("o",o))})
    bg.store(dest/"sealed.json",dict(dataset=layout["dataset"],fold=layout["fold"],design=bg.digest(OUT/"protocol.json"),
             layout=bg.digest(src/"layout.json"),predictions=bg.digest(dest/"predictions.npz"),
             supports=supports,guard_metrics=guard_metrics,proposal_scores_8=scores,seconds_8=sec,
             fits_8=len(scores),rare=rare,low_fidelity=nominee,gain=int(len(supports)>1)))


def seal():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=ROOT/"thesis-paper/evidence/wide"
    files=[DESIGN,Path(__file__),Path(sys.modules[small_scores.__module__].__file__),evidence/"prediction-lock.json",
           evidence/"fold-status.json",ROOT/"configs/context_review/transport_refit_control.json"]
    lock={p.relative_to(ROOT).as_posix():bg.digest(p) for p in files}
    manifest=OUT/"protocol.json"
    if manifest.exists() and bg.read(manifest)!=lock:
        raise ValueError("Changed wide-slate protocol")
    bg.store(manifest,lock)
    seals={}
    jobs=[k for k,s in bg.read(evidence/"fold-status.json").items() if s["feasible"]]
    start=perf_counter()
    for n,k in enumerate(jobs,1):
        receipt=OUT/k/"sealed.json"
        if not receipt.exists():
            subprocess.run([sys.executable,"-m","experiments.context_review.wide_slate",k],check=True,timeout=3600)
        rec=bg.read(receipt)
        assert rec["design"]==bg.digest(manifest) and rec["predictions"]==bg.digest(receipt.parent/"predictions.npz")
        seals[k]=bg.digest(receipt)
        elapsed=perf_counter()-start
        print(f"[{n}/{len(jobs)}] {k} elapsed {elapsed:.0f}s, remaining ~{elapsed*(len(jobs)-n)/n:.0f}s",flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    bg.store(EVIDENCE/"prediction-lock.json",seals)


if __name__=="__main__":
    if len(sys.argv)==1:
        seal()
    else:
        with threadpool_limits(limits=1):
            run_fold(sys.argv[1])
