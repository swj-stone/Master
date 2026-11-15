"""Compare frozen-model deletion votes with fresh reduced-model loss on common supports."""
import argparse
from itertools import combinations
import json
from math import ceil, comb
from pathlib import Path
from random import Random
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, ROOT, OUT as PRIMARY, config, digest
from experiments.context_study.working_set import oracle
from minppi_mofs.piv3_compact_witness import _sample

OUT = ROOT/"results/context_review/refit-target"
DESIGN = ROOT/"configs/context_review/refit_target.json"
RULES = ("point", "empty", "endpoints", "nested", "separating", "random", "pooled", "one_panel_min")


def candidates(record, index):
    order = sorted(record["order"][:record["anchor_k"]])
    anchor = frozenset(order)
    proposed = set()
    for name in ("point", "min", "antichain"):
        live = anchor
        for item in record["traces"][name]["trace"]:
            subset = live.difference(item["block"])
            if subset and subset != anchor:
                proposed.add(tuple(sorted(subset)))
            if item["accepted"]:
                live = subset
        assert sorted(live) == record["selected"][name]
    path_rng = Random(117034 + index)
    path = path_rng.sample(sorted(proposed), min(12, len(proposed)))
    rng = Random(47113 + index)
    sampled = []
    for size in sorted({1, ceil(len(order)/4), len(order)//2}):
        if not 0 < size < len(order):
            continue
        count = comb(len(order), size)
        if count <= 4:
            dropped = combinations(order, size)
        else:
            dropped = set()
            while len(dropped) < 4:
                dropped.add(tuple(sorted(rng.sample(order, size))))
        sampled.extend(tuple(sorted(anchor.difference(d))) for d in sorted(dropped))
    return {"path":sorted(path), "uniform_deletions":sorted(sampled)}


def votes(s, families, ballot):
    groups = {name:[tuple(map(bool, ballot(s,e))) for e in group] for name,group in families.items()}
    empty = groups["separating"][0]
    endpoint = (empty, groups["separating"][-1])
    def veto(rows):
        return any(sum(row)<3 for row in rows)
    coded = groups["separating"]
    decisions = dict(point=not empty[0], empty=veto([empty]), endpoints=veto(endpoint),
                     nested=veto(groups["nested"]), separating=veto(coded), random=veto(groups["random"]),
                     pooled=sum(map(sum,coded)) <= 2*len(coded), one_panel_min=any(not b[0] for b in coded))
    return decisions, groups


def metrics(labels, probabilities, rare):
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(labels, probabilities[:,1])
    else:
        auc = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    result = dict(auc=float(auc), accuracy=float(accuracy_score(labels, probabilities.argmax(axis=1))))
    if rare:
        result["ap"] = float(average_precision_score(labels, probabilities[:,1]))
    return result


def run_fold(name, fold):
    source = PRIMARY/name/f"fold-{fold}"
    folder = OUT/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    rec = json.loads((source/"selection.json").read_bytes())
    lock = json.loads((source/"completed.json").read_bytes())
    table = DATA/name/f"{name}.tsv.gz"
    assert digest(source/"selection.json") == lock["selection_sha256"] and digest(table) == lock["data_sha256"]
    frame = pd.read_csv(table,sep="\t")
    y = frame.pop("target").map({value:i for i,value in enumerate(rec["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train, rank, test = [np.asarray(rec[k]) for k in ("train","rank","test")]
    dev = np.sort(np.r_[train,rank])
    seed = 20421+fold
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt, xr = imp.transform(x[train]), imp.transform(x[rank])
    predictor = fit(xt,y[train],seed)
    task = "binary" if predictor.n_classes_ == 2 else "multiclass"
    ids, weights = _sample(y[rank],task,np.random.SeedSequence([seed,659]))
    bg = np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    anchor = frozenset(rec["selected"]["gain"])
    rare = task=="binary" and min(np.bincount(y[rank])) <= .05*len(rank)
    ballot, _ = oracle(predictor,xr[ids],np.median(xt[bg],axis=0),y[rank][ids],weights,anchor,rare)
    families = {label:tuple(map(frozenset,rec["traces"][arm]["context_extensions"]))
                for label,arm in (("nested","min"),("separating","antichain"),("random","random_contexts"))}
    cohorts = candidates(rec, 10*config()["datasets"].index(name)+fold)
    supports = sorted(set().union(*cohorts.values()))
    ratings = [votes(frozenset(s),families,ballot) for s in supports]
    store(folder/"selection.json",dict(donor=lock["selection_sha256"],anchor=sorted(anchor),supports=supports,
          cohorts=cohorts, ballots=[g for _,g in ratings], decisions=[d for d,_ in ratings],
          contexts={k:[sorted(e) for e in group] for k,group in families.items()}))
    imputer = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd, xe = imputer.transform(x[dev]), imputer.transform(x[test])
    def refit(cols):
        return fit(xd[:,cols],y[dev],seed).predict_proba(xe[:,cols])
    anchor_p = refit(sorted(anchor))
    with np.load(source/"predictions.npz") as frozen:
        np.testing.assert_array_equal(anchor_p,frozen["gain"])
    reference = metrics(y[test],anchor_p,rare)
    predictions = dict(row_id=test,y=y[test],anchor=anchor_p)
    rows = []
    for i,s in enumerate(supports):
        predictions[f"s{i}"] = refit(list(s))
        actual = metrics(y[test],predictions[f"s{i}"],rare)
        losses = {m:reference[m]-actual[m] for m in reference}
        fail = any(d > (.05 if m=="ap" else .03)*abs(reference[m])+1e-12 for m,d in losses.items())
        for cohort, members in cohorts.items():
            if s in members:
                rows.append(dict(dataset=name,fold=fold,cohort=cohort,subset=i,k=len(s),anchor_k=len(anchor),
                                 fail=fail,**ratings[i][0],**{f"loss_{k}":v for k,v in losses.items()}))
    np.savez_compressed(folder/"predictions.npz",**predictions)
    store(folder/"completed.json",dict(rows=rows,reference=reference,fold=fold,dataset=name,
          selection=digest(folder/"selection.json"),predictions=digest(folder/"predictions.npz"),
          protocol=digest(OUT/"protocol.json")))


def summarize(df):
    rows = []
    for (cohort,name),sub in df.groupby(["cohort","dataset"]):
        for rule in RULES:
            veto, fail = sub[rule],sub.fail
            rows.append(dict(dataset=name,cohort=cohort,rule=rule,n=len(sub),tp=int((veto&fail).sum()),
                fp=int((veto&~fail).sum()),fn=int((~veto&fail).sum()),tn=int((~veto&~fail).sum())))
    counts = pd.DataFrame(rows)
    counts.to_csv(OUT/"by-dataset.csv",index=False)
    pooled = counts.groupby(["cohort","rule"])[["n","tp","fp","fn","tn"]].sum()
    pooled.to_csv(OUT/"counts.csv")
    print(pooled.to_string(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args = parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            return run_fold(args.worker[0],int(args.worker[1]))
    OUT.mkdir(parents=True,exist_ok=True)
    sources = [DESIGN,Path(__file__),PRIMARY/"protocol-lock.json",ROOT/"experiments/context_study/working_set.py"]
    lock = {p.name:digest(p) for p in sources}
    path = OUT/"protocol.json"
    if path.exists() and json.loads(path.read_bytes())!=lock:
        raise ValueError("The frozen protocol changed")
    store(path,lock)
    rows, exposure = [], []
    for name in config()["datasets"]:
        for fold in range(3):
            folder = OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                subprocess.run([sys.executable,"-m","experiments.context_review.refit_target","--worker",
                                name,str(fold)],check=True,timeout=3600)
            rec = json.loads((folder/"completed.json").read_bytes())
            assert rec["protocol"]==digest(path) and rec["selection"]==digest(folder/"selection.json")
            assert rec["predictions"]==digest(folder/"predictions.npz")
            rows.extend(rec["rows"])
            design = json.loads((folder/"selection.json").read_bytes())
            exposure.append(dict(dataset=name,fold=fold,anchor_k=len(design["anchor"]),unique=len(design["supports"]),
                                 **{c:len(s) for c,s in design["cohorts"].items()}))
        print(f"Verified {name}, {len(rows)} proposal-source records",flush=True)
    pd.DataFrame(exposure).to_csv(OUT/"exposure.csv",index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"decisions.csv",index=False)
    summarize(frame)


if __name__=="__main__":
    main()
