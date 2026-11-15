"""Paired full-family and working-set schedules on frozen prospective PMLB folds."""
import argparse
import csv
import json
import os
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
from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.context_working_set import search_working_set
from minppi_mofs.piv3_compact_witness import _sample, _within
from minppi_mofs.utility import compute_utility


OUT = ROOT / "results/context_study/working-set"
DESIGN = ROOT / "configs/context_study/working_set.json"


def oracle(predictor, x, background, y, weights, anchor, rare):
    scorer = make_scorer(y, weights, "binary" if predictor.n_classes_==2 else "multiclass", "ovr_auc", rare)
    cache = {}
    def assess(mask):
        if mask not in cache:
            xx = np.broadcast_to(background, x.shape).copy()
            cols = sorted(mask)
            xx[:,cols] = x[:,cols]
            cache[mask] = scorer(predictor.predict_proba(xx))
        return cache[mask]
    def ballot(s, e):
        return tuple(_within(after,before) for after,before in zip(assess(s|e),assess(anchor|e),strict=True))
    return ballot, cache


def check_local(live, family, ballot):
    if not all(sum(ballot(live,e))>=3 for e in family):
        raise AssertionError("Final support fails the full family")
    if len(live)>1:
        for col in live:
            if all(sum(ballot(live-{col},e))>=3 for e in family):
                raise AssertionError("Final support permits a singleton deletion")


def one(name, fold):
    dst = OUT/name/f"fold-{fold}"
    src = PRIMARY/name/f"fold-{fold}"
    record = json.loads((src/"selection.json").read_bytes())
    receipt = json.loads((src/"completed.json").read_bytes())
    if digest(src/"selection.json") != receipt["selection_sha256"]:
        raise ValueError("The prospective frozen supports changed")
    data_path = DATA/name/f"{name}.tsv.gz"
    if digest(data_path)!=receipt["data_sha256"]:
        raise ValueError("The dataset differs from the primary split")
    frame = pd.read_csv(data_path,sep="\t")
    lookup = {v:i for i,v in enumerate(record["class_order"])}
    y = frame.pop("target").map(lookup).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train,rank,test = (np.asarray(record[k]) for k in ("train","rank","test"))
    dev = np.sort(np.r_[train,rank])
    seed = 20421+fold
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xr = imp.transform(x[train]),imp.transform(x[rank])
    predictor = fit(xt,y[train],seed)
    task = "binary" if predictor.n_classes_==2 else "multiclass"
    idx,weights = _sample(y[rank],task,np.random.SeedSequence([seed,659]))
    rare = task=="binary" and min(np.bincount(y[rank]))<=.05*len(rank)
    bg = np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    background = np.median(xt[bg],axis=0)
    k,ordered = record["anchor_k"],record["order"]
    order, family = ordered[:k], separating_contexts(ordered[k:])
    anchor = frozenset(order)
    arms = ["full_family","working_set"]
    if (fold+config()["datasets"].index(name))%2:
        arms.reverse()
    selections, stages, results = {}, {}, []
    for arm in arms:
        start=perf_counter()
        ballot, cache=oracle(predictor,xr[idx],background,y[rank][idx],weights,anchor,rare)
        if arm=="full_family":
            live, trace=search(order,family,ballot)
            stages[arm]=[dict(proposals=len(trace["trace"]),active=len(family))]
        else:
            live, stages[arm]=search_working_set(order,family,ballot)
        elapsed=perf_counter()-start
        count=len(cache)
        if arm=="full_family":
            if sorted(live)!=record["selected"]["antichain"] or count!=record["traces"]["antichain"]["mask_calls"]:
                raise ValueError(f"{name} fold {fold}: direct replay differs")
        # Reevaluate with a fresh cache independent of the search and its short circuits.
        verifier,_=oracle(predictor,xr[idx],background,y[rank][idx],weights,anchor,rare)
        check_local(live,family,verifier)
        selections[arm]=sorted(live)
        results.append(dict(dataset=name,fold=fold,arm=arm,width=x.shape[1],k=len(live),family=len(family),
                            stages=len(stages[arm]),masks=count,selector_s=elapsed))
    dst.mkdir(parents=True,exist_ok=True)
    store(dst/"selection.json",dict(source=receipt["selection_sha256"],selected=selections,stages=stages))
    final_imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe=final_imp.transform(x[dev]),final_imp.transform(x[test])
    predictions={}
    for row in results:
        arm=row["arm"]
        cols=selections[arm]
        start=perf_counter()
        refit=fit(xd[:,cols],y[dev],seed)
        pred=refit.predict_proba(xe[:,cols])
        row.update(refit_s=perf_counter()-start,auc=compute_utility(y[test],pred,metric="ovr_auc"),
                   accuracy=accuracy_score(y[test],pred.argmax(axis=1)))
        predictions[arm]=pred
    with np.load(src/"predictions.npz") as old:
        np.testing.assert_array_equal(predictions["full_family"],old["antichain"])
    np.savez_compressed(dst/"predictions.npz",row_id=test,y=y[test],**predictions)
    store(dst/"completed.json",dict(rows=results,frozen=digest(dst/"selection.json"),
                                    predictions=digest(dst/"predictions.npz")))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=config()["datasets"])
    parser.add_argument("--fold",type=int,choices=range(3))
    args=parser.parse_args()
    if args.dataset is not None:
        if args.fold is None:
            parser.error("--dataset requires --fold")
        return one(args.dataset,args.fold)
    OUT.mkdir(exist_ok=True,parents=True)
    code=[DESIGN,Path(__file__),ROOT/"src/minppi_mofs/context_working_set.py",PRIMARY/"protocol-lock.json"]
    stamps={p.name:digest(p) for p in code}
    protocol=OUT/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_bytes())!=stamps:
        raise ValueError("The supplementary protocol or code changed")
    store(protocol,stamps)
    rows=[]
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    for name in config()["datasets"]:
        for fold in range(3):
            folder=OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                subprocess.run([sys.executable,"-m","experiments.context_study.working_set",
                                "--dataset",name,"--fold",str(fold)],check=True,timeout=3600,env=env)
            record=json.loads((folder/"completed.json").read_bytes())
            if record["predictions"]!=digest(folder/"predictions.npz") or record["frozen"]!=digest(folder/"selection.json"):
                raise ValueError("An output changed")
            rows.extend(record["rows"])
        print(f"Verified all folds for {name}",flush=True)
    with (OUT/"scores.csv").open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
