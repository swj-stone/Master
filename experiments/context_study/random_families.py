"""Replicate the size- and budget-matched random context control on fixed PMLB folds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import matched_random_contexts, separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_witness import _sample

DESIGN = ROOT/"configs/context_study/random_families.json"
OUT = ROOT/"results/context_study/random-families"
METRICS = ["auc","accuracy","retained"]


def one(name, fold, offsets):
    src = PRIMARY/name/f"fold-{fold}"
    dst = OUT/name/f"fold-{fold}"
    dst.mkdir(exist_ok=True,parents=True)
    frozen = json.loads((src/"selection.json").read_bytes())
    receipt = json.loads((src/"completed.json").read_bytes())
    raw = DATA/name/f"{name}.tsv.gz"
    if digest(src/"selection.json")!=receipt["selection_sha256"] or digest(raw)!=receipt["data_sha256"]:
        raise ValueError("Primary experiment identity changed")
    table = pd.read_csv(raw,sep="\t")
    lookup = {v:i for i,v in enumerate(frozen["class_order"])}
    y = table.pop("target").map(lookup).to_numpy()
    x = table.to_numpy(dtype=np.float32)
    train,rank,test = (np.asarray(frozen[k]) for k in ["train","rank","test"])
    dev = np.sort(np.r_[train,rank])
    seed = 20421 + fold
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xr = imp.transform(x[train]),imp.transform(x[rank])
    predictor = fit(xt,y[train],seed)
    task = "binary" if predictor.n_classes_==2 else "multiclass"
    rows,weights = _sample(y[rank],task,np.random.SeedSequence([seed,659]))
    rare = task=="binary" and min(np.bincount(y[rank]))<=.05*len(rank)
    bg = np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    background = np.median(xt[bg],axis=0)
    order,k = frozen["order"],frozen["anchor_k"]
    tail = order[k:]
    anchor = frozenset(order[:k])
    template = separating_contexts(tail)
    supports, families, counts = {}, {}, {}
    for offset in [77531,*offsets]:
        family = matched_random_contexts(template,tail,seed=offset+fold)
        ballot,cache = oracle(predictor,xr[rows],background,y[rank][rows],weights,anchor,rare)
        chosen,trace = search(order[:k],family,ballot)
        if offset==77531:
            if sorted(chosen)!=frozen["selected"]["random_contexts"] or len(cache)!=frozen["traces"]["random_contexts"]["mask_calls"]:
                raise AssertionError("Original matched random control could not be reproduced")
            continue
        arm = str(offset)
        supports[arm] = sorted(chosen)
        families[arm] = [sorted(mask) for mask in family]
        counts[arm] = len(cache)
    store(dst/"selection.json",dict(primary_receipt=receipt["selection_sha256"],selected=supports,families=families))
    imputer = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe = imputer.transform(x[dev]),imputer.transform(x[test])
    predicted = {"y":y[test],"row_id":test}
    records = []
    for arm,cols in supports.items():
        probs = fit(xd[:,cols],y[dev],seed).predict_proba(xe[:,cols])
        auc = roc_auc_score(y[test],probs[:,1]) if probs.shape[1]==2 else roc_auc_score(y[test],probs,multi_class="ovr")
        acc = accuracy_score(y[test],probs.argmax(axis=1))
        records.append(dict(dataset=name,fold=fold,family_offset=int(arm),effective_seed=int(arm)+fold,k=len(cols),
                            width=x.shape[1],retained=len(cols)/x.shape[1],mask_calls=counts[arm],auc=auc,accuracy=acc))
        predicted[arm] = probs
    np.savez_compressed(dst/"predictions.npz",**predicted)
    store(dst/"completed.json",dict(rows=records,protocol=digest(OUT/"protocol.json"),
          selection=digest(dst/"selection.json"),predictions=digest(dst/"predictions.npz")))


def summarize(df, primary, offsets):
    per_family = df.groupby(["dataset","family_offset"])[METRICS].mean()
    per_family.to_csv(OUT/"per-dataset-family.csv")
    reference = primary[primary.arm=="antichain"].groupby("dataset")[METRICS].mean()
    primary_random = primary[primary.arm=="random_contexts"].groupby("dataset")[METRICS].mean()
    stats = pd.DataFrame(index=reference.index)
    for m in METRICS:
        values = per_family[m].unstack()
        stats[f"separating_{m}"] = reference[m]
        stats[f"primary_random_{m}"] = primary_random[m]
        stats[f"random_{m}_mean"] = values.mean(axis=1)
        stats[f"random_{m}_min"] = values.min(axis=1)
        stats[f"random_{m}_max"] = values.max(axis=1)
        stats[f"separating_minus_random_{m}"] = reference[m]-values.mean(axis=1)
    stats.to_csv(OUT/"per-dataset.csv")
    per_family.groupby("family_offset").mean().to_csv(OUT/"per-family.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", nargs=2)
    args = p.parse_args()
    offsets = json.loads(DESIGN.read_text(encoding="utf-8-sig"))["seed_offsets"]
    if args.worker:
        with threadpool_limits(limits=1):
            return one(args.worker[0],int(args.worker[1]),offsets)
    OUT.mkdir(exist_ok=True,parents=True)
    lock = {"primary_sha256":digest(PRIMARY/"protocol-lock.json"),"design":digest(DESIGN),
            "source":digest(Path(__file__)),"oracle":digest(ROOT/"experiments/context_study/working_set.py")}
    protocol = OUT/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_bytes())!=lock:
        raise ValueError("Frozen study changed")
    store(protocol,lock)
    records = []
    env = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    for name in config()["datasets"]:
        for fold in range(3):
            dst = OUT/name/f"fold-{fold}"
            done = dst/"completed.json"
            if not done.exists():
                subprocess.run([sys.executable,"-m","experiments.context_study.random_families","--worker",name,str(fold)],
                               env=env,cwd=ROOT,check=True,timeout=3600)
            result = json.loads(done.read_bytes())
            if (result["protocol"]!=digest(protocol) or result["selection"]!=digest(dst/"selection.json")
                    or result["predictions"]!=digest(dst/"predictions.npz")):
                raise ValueError(f"Invalid receipt at {dst}")
            records.extend(result["rows"])
        print(f"Completed {name}; {len(records)}/210 random-family refits",flush=True)
    df = pd.DataFrame(records)
    if len(df)!=210 or df.duplicated(["dataset","fold","family_offset"]).any():
        raise ValueError("Incomplete study")
    df.to_csv(OUT/"per-fold.csv",index=False)
    primary = pd.read_csv(PRIMARY/"per-fold.csv")
    primary["retained"] = primary.k/primary.width
    summarize(df,primary,offsets)


if __name__=="__main__":
    main()

