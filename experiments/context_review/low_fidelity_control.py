"""Seal an eight-round nomination and full-slate counterfactual predictions on the external panel."""
import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_data as transport
from experiments.context_review.refit_opportunity import outcome

SPEC = bg.ROOT / "configs/context_review/transport_refit_control.json"
OUT = bg.ROOT / "results/context_review/low-fidelity"
EVIDENCE = bg.ROOT / "thesis-paper/evidence/low_fidelity"


def small_scores(x, y, fitting, proposal, supports, seed, rare):
    start = perf_counter()
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[fitting])
    train, validation = imp.transform(x[fitting]), imp.transform(x[proposal])
    scores = []
    for cols in supports:
        model = XGBClassifier(n_estimators=8, max_depth=4, learning_rate=.1, tree_method="hist",
                              device="cpu", n_jobs=1, random_state=seed)
        model.fit(train[:,cols], y[fitting])
        scores.append(bg.metrics(y[proposal], model.predict_proba(validation[:,cols]), rare))
    return scores, perf_counter() - start


def minimum_slack(scores, anchor):
    return min((scores[m] - ref + (.05 if m == "ap" else .03)*abs(ref) + 1e-12) /
               max((.05 if m == "ap" else .03)*abs(ref), 1e-12) for m,ref in anchor.items())


def fit_one(dataset, fold, index):
    src = transport.RUN/dataset/f"fold-{fold}"
    dest = OUT/dataset/f"fold-{fold}"
    dest.mkdir(exist_ok=True, parents=True)
    layout = bg.read(src/"layout.json")
    assert bg.digest(src/"inputs.npz") == layout["inputs_sha256"]
    with np.load(src/"inputs.npz") as data:
        x,y,future,train,prop,guard = [data[k] for k in ("x","y","x_outer","train","proposal","guard")]
    dev = np.arange(len(y))
    seed = 20421 + fold
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt, xp = imp.transform(x[train]), imp.transform(x[prop])
    model = bg.fit(xt,y[train],seed)
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    rare = bool(task == "binary" and min(np.bincount(y[prop])) <= .05*len(prop))
    names = [f"c{i}" for i in range(x.shape[1])]
    _, info = bg.gain_anchor(model,pd.DataFrame(xt,columns=names),y[train],pd.DataFrame(xp,columns=names),
                             y[prop],task=task,utility="ovr_auc",seed=seed,aggregate="gain")
    order = [int(s[1:]) for s in info["order"][:info["k"]]]
    width = (len(order)+1)//2
    anchor = tuple(sorted(order))
    candidates = bg.candidate_pool(order,width,index,fold)[1] if width < len(order) else []
    supports = [anchor,*candidates]
    scores, seconds = [], 0.
    nominee = 0
    if candidates:
        scores, seconds = small_scores(x,y,train,prop,supports,seed,rare)
        nominee = max(range(1,len(supports)),key=lambda i:minimum_slack(scores[i],scores[0]))
    g,_,_ = bg.predict_sets(x,y,np.sort(np.r_[train,prop]),x[guard],supports,seed)
    o,_,_ = bg.predict_sets(x,y,dev,future,supports,seed)
    guards = [bg.metrics(y[guard],g[s],rare) for s in supports]
    arrays = {f"{phase}{i}":probs[s] for i,s in enumerate(supports) for phase,probs in (("g",g),("o",o))}
    np.savez_compressed(dest/"predictions.npz",**arrays)
    bg.store(dest/"sealed.json",dict(dataset=dataset,fold=fold,design=bg.digest(OUT/"protocol.json"),
             layout=bg.digest(src/"layout.json"),predictions=bg.digest(dest/"predictions.npz"),
             supports=supports,guard_metrics=guards,proposal_scores_8=scores,seconds_8=seconds,
             fits_8=len(scores),rare=rare,low_fidelity=nominee,gain=int(bool(candidates))))


def jobs():
    return [(name,fold,1000+number) for name,number,*_ in bg.read(transport.SPEC)["sources"] for fold in range(3)]


def seal():
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies = [SPEC, transport.SPEC, Path(__file__), Path(bg.__file__), transport.EVIDENCE/"sources.json"]
    lock = {p.relative_to(bg.ROOT).as_posix(): bg.digest(p) for p in dependencies}
    path = OUT/"protocol.json"
    if path.exists() and lock != bg.read(path):
        raise ValueError("Changed experiment inputs")
    bg.store(path,lock)
    start=perf_counter()
    seals={}
    for n,(name,fold,index) in enumerate(jobs(),1):
        receipt=OUT/name/f"fold-{fold}"/"sealed.json"
        if not receipt.exists():
            subprocess.run([sys.executable,"-m","experiments.context_review.low_fidelity_control",
                            "--worker",name,str(fold),str(index)],check=True,timeout=3600)
        rec=bg.read(receipt)
        assert rec["design"]==bg.digest(path) and rec["predictions"]==bg.digest(receipt.parent/"predictions.npz")
        seals[f"{name}/fold-{fold}"]=bg.digest(receipt)
        elapsed=perf_counter()-start
        print(f"[{n}/30] {name}/{fold} elapsed {elapsed:.0f}s, remaining ~{elapsed*(30-n)/n:.0f}s",flush=True)
    bg.store(OUT/"cohort-sealed.json",seals)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=3)
    opts=parser.parse_args()
    if opts.worker:
        with threadpool_limits(limits=1):
            fit_one(opts.worker[0],int(opts.worker[1]),int(opts.worker[2]))
    else:
        seal()
