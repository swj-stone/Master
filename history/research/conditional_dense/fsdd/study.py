"""Reproduce both frozen selectors and their controls on held-out FSDD speakers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.grouped_solver_cost_audit import fit
from research.conditional_dense.confirmation import ROOT
from research.conditional_dense.fsdd.features import panels, read_corpus
from research.conditional_dense.shared_method import conditional_support
from research.conditional_dense.sketch_f.study import check_sampler, sketch_f
from research.conditional_dense.vote_width.method import vote_support
from research.feature_game_eval.audit_controls import effect_size
from sklearn.metrics import balanced_accuracy_score


HERE = Path(__file__).resolve().parent
METHODS = ("old", "new", "full", "half_F", "old_F", "old_sketch_F", "old_random",
           "new_F", "new_sketch_F", "new_random")
FIELDS = ("fold","seed","method","k","ba","repeat","position","select_ms","slice_ms",
          "fit_ms","predict_ms","total_ms")


def select(panel, method, widths):
    X, y, seed = panel.X, panel.y, panel.seed
    if method == "old":
        return conditional_support(X,y,seed=seed)[0]
    if method == "new":
        return vote_support(X,y,seed=seed)[0]
    if method == "full":
        return np.arange(X.shape[1])
    if method == "half_F":
        return full_f(X,y,(X.shape[1]+1)//2,seed)
    variant,control = method.split("_",1)
    width = widths[variant]
    if control == "F":
        return full_f(X,y,width,seed)
    if control == "sketch_F":
        return sketch_f(X,y,width,seed)
    if control == "random":
        return choose("random",X,y,"classification",seed,width)
    raise ValueError(method)


def full_f(X, y, width, seed):
    centered = np.array(X,dtype=np.float64,copy=True)
    centered -= centered.mean(axis=0)
    return choose("f",centered,y,"classification",seed,width)


def deploy(panel,method,widths):
    start = perf_counter_ns()
    cols = select(panel,method,widths)
    end_selection = perf_counter_ns()
    pred,decision,parts = fit(panel,cols,"cholesky")
    end = perf_counter_ns()
    return cols,pred,decision,dict(select_ms=(end_selection-start)/1e6,total_ms=(end-start)/1e6,**parts)


def speaker_scores(y, pred, group):
    return {person:balanced_accuracy_score(y[group==person],pred[group==person]) for person in np.unique(group)}


def run(out,pass_id,archive,expected_hash):
    fingerprint = hashlib.sha256(archive.read_bytes()).hexdigest()
    if fingerprint != expected_hash.lower():
        raise ValueError("Unexpected corpus hash")
    out.mkdir(parents=True,exist_ok=False)
    paths = [HERE/"protocol.md",HERE/"study.py",HERE/"features.py",HERE.parent/"shared_method.py",
             HERE.parent/"vote_width/method.py",HERE.parent/"sketch_f/study.py",
             ROOT/"experiments/grouped_solver_cost_audit.py",ROOT/"experiments/compare_predictive_minpi.py",
             ROOT/"research/feature_game_eval/audit_controls.py"]
    state = dict(process=pass_id,methods=METHODS,repeats=len(METHODS),threads=1,
                 corpus_sha256=fingerprint,archive=archive.name,numpy=np.__version__,sklearn=sklearn.__version__,
                 source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 folds=[])
    start = perf_counter_ns()
    X,y,units,names,state["corpus"] = read_corpus(archive)
    prep_ms = (perf_counter_ns()-start)/1e6
    (out/"preparation.json").write_text(json.dumps(dict(common_wave_to_matrix_ms=prep_ms),indent=2))
    np.savez_compressed(out/"index.npz",filename=names,speaker=units,digit=y)
    with (out/"trials.csv").open("x",newline="") as trials,(out/"speaker-scores.csv").open("x",newline="") as unitfile:
        writer = csv.DictWriter(trials,fieldnames=FIELDS); writer.writeheader()
        scores_by_unit = csv.writer(unitfile); scores_by_unit.writerow(("fold","seed","method","speaker","ba"))
        for panel,train,test in panels(X,y,units):
            widths = {v:len(select(panel,v,{})) for v in ("old","new")}
            samples,_,_ = check_sampler(panel,widths["old"])
            cache = dict(train_rows=train,test_rows=test,truth=panel.gold,units=panel.units,training_sketch=samples)
            effects = {"sketch":effect_size(panel.X[samples],panel.y[samples],"classification"),
                       "full":effect_size(panel.X,panel.y,"classification")}
            boundaries = {}
            scores = {}
            for arm in METHODS:
                cols,pred,decision,_ = deploy(panel,arm,widths)
                for label,value in (("cols",cols),("pred",pred),("decision",decision)):
                    cache[arm+"_"+label] = value
                if arm.endswith("_F"):
                    eta = effects["sketch" if "sketch" in arm else "full"]
                    exact = np.sort(np.argsort(-eta,kind="stable")[:len(cols)])
                    outside = np.setdiff1d(np.arange(len(eta)),cols)
                    gap = max(0.,float(eta[outside].max()-eta[cols].min())) if len(outside) else 0.
                    assert gap <= 1e-10, (panel.fold,arm,gap)
                    boundaries[arm] = dict(boundary_gap=gap,n_alternate_ties=len(np.setdiff1d(cols,exact)))
                by_person = speaker_scores(panel.gold,pred,panel.units)
                scores[arm] = np.mean(list(by_person.values()))
                for person,value in by_person.items():
                    scores_by_unit.writerow((panel.fold,panel.seed,arm,person,value))
            state["folds"].append(dict(fold=panel.fold,seed=panel.seed,widths=widths,p=panel.X.shape[1],
                train_people=np.unique(units[train]).tolist(),test_people=np.unique(units[test]).tolist(),
                F_boundaries=boundaries))
            order = np.random.default_rng([61347,panel.fold,panel.seed,pass_id]).permutation(METHODS)
            for rep in range(len(METHODS)):
                for pos,arm in enumerate(np.roll(order,rep)):
                    cols,pred,decision,timing = deploy(panel,arm,widths)
                    for suffix,value in (("cols",cols),("pred",pred),("decision",decision)):
                        np.testing.assert_array_equal(value,cache[arm+"_"+suffix])
                    writer.writerow(dict(fold=panel.fold,seed=panel.seed,method=arm,k=len(cols),ba=scores[arm],
                        repeat=rep,position=pos,**timing))
            for draw in range(20):
                order = np.random.default_rng([61341,panel.fold,panel.seed,draw]).permutation(panel.X.shape[1])
                for version,width in widths.items():
                    cols = np.sort(order[:width])
                    pred,_,_ = fit(panel,cols,"cholesky")
                    cache[f"{version}_random{draw}_cols"] = cols
                    cache[f"{version}_random{draw}_pred"] = pred
            np.savez_compressed(out/f"fold-{panel.fold}.npz",**cache)
            trials.flush(); unitfile.flush()
            print(f"fold {panel.fold} complete",flush=True)
    (out/"protocol.json").write_text(json.dumps(state,indent=2),encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--pass-id",type=int,required=True,choices=(1,2))
    args = p.parse_args()
    with threadpool_limits(1):
        run(args.out,args.pass_id,args.archive,args.sha256)
