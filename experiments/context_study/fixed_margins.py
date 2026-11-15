"""Fixed row/column margins and codeword assignment controls on original PMLB folds."""
import argparse
import json
import os
from pathlib import Path
from random import Random
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
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_witness import _sample

DESIGN = ROOT/"configs/context_study/fixed_margins.json"
OUT = ROOT/"results/context_study/fixed-margins"
METRICS = ["auc", "accuracy", "retained", "coverage", "accepted", "mask_calls"]


def fixed_margin_contexts(template, tail, arm, seed, proposals):
    rng = Random(seed)
    if arm == "permute":
        labels = list(tail)
        rng.shuffle(labels)
        assignment = dict(zip(tail, labels, strict=True))
        return tuple(frozenset(assignment[col] for col in mask) for mask in template), 0
    if arm != "switch":
        raise ValueError(arm)
    interior = list(template[1:-1])
    if len(interior) < 2:
        return template, 0
    accepted = 0
    for _ in range(proposals):
        i, j = rng.sample(range(len(interior)), 2)
        a, b = interior[i], interior[j]
        left, right = sorted(a-b), sorted(b-a)
        if not left or not right:
            continue
        u, v = rng.choice(left), rng.choice(right)
        first, second = a.difference([u]).union([v]), b.difference([v]).union([u])
        used = set(template[:1] + template[-1:])
        used.update(mask for h,mask in enumerate(interior) if h not in (i,j))
        if first == second or first in used or second in used:
            continue
        interior[i], interior[j] = first, second
        accepted += 1
    return (template[0], *interior, template[-1]), accepted


def geometry(family, tail):
    signatures = [sum(1 << row for row,mask in enumerate(family) if col in mask) for col in tail]
    total = len(tail) * (len(tail)-1)
    covered = sum(bool(a & ~b) for a in signatures for b in signatures)
    return total, covered




def one(name, fold, offsets, proposals):
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
    supports, families, diagnostics = {}, {}, {}
    for arm, offset in [("coded", 0)] + [(a, i) for a in ("switch", "permute") for i in offsets]:
        family, accepted = (template, 0) if arm == "coded" else fixed_margin_contexts(template, tail, arm, offset+fold, proposals)
        if (len(set(family)) != len(family) or family[0] != template[0] or family[-1] != template[-1]
                or [len(mask) for mask in family] != [len(mask) for mask in template]
                or any(sum(col in m for m in family) != sum(col in m for m in template) for col in tail)):
            raise AssertionError("Fixed margins violated")
        total, covered = geometry(family, tail)
        if arm != "switch" and covered != total:
            raise AssertionError("Incomplete separating design")
        ballot, cache = oracle(predictor,xr[rows],background,y[rank][rows],weights,anchor,rare)
        chosen, _ = search(order[:k],family,ballot)
        if arm == "coded":
            if sorted(chosen) != frozen["selected"]["antichain"] or len(cache) != frozen["traces"]["antichain"]["mask_calls"]:
                raise AssertionError("Failed to replay original separating support")
            continue
        tag = f"{arm}_{offset}"
        supports[tag] = sorted(chosen)
        families[tag] = [sorted(mask) for mask in family]
        diagnostics[tag] = dict(arm=arm, family_offset=offset, effective_seed=offset+fold, eligible=total,
                                covered=covered, accepted=accepted, mask_calls=len(cache))
    store(dst/"selection.json",dict(primary_receipt=receipt["selection_sha256"],selected=supports,families=families))
    imputer = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe = imputer.transform(x[dev]),imputer.transform(x[test])
    predicted = {"y":y[test],"row_id":test}
    records = []
    for arm,cols in supports.items():
        probs = fit(xd[:,cols],y[dev],seed).predict_proba(xe[:,cols])
        auc = roc_auc_score(y[test],probs[:,1]) if probs.shape[1]==2 else roc_auc_score(y[test],probs,multi_class="ovr")
        acc = accuracy_score(y[test],probs.argmax(axis=1))
        records.append(dict(dataset=name,fold=fold,k=len(cols),width=x.shape[1],retained=len(cols)/x.shape[1],
                            auc=auc,accuracy=acc,**diagnostics[arm]))
        predicted[arm] = probs
    np.savez_compressed(dst/"predictions.npz",**predicted)
    store(dst/"completed.json",dict(rows=records,protocol=digest(OUT/"protocol.json"),
          selection=digest(dst/"selection.json"),predictions=digest(dst/"predictions.npz")))


def summarize(df, primary):
    reference = primary[primary.arm=="antichain"][["dataset","fold","auc","accuracy","retained"]].copy()
    pairs = df.groupby(["dataset","fold"]).eligible.first()
    reference = reference.join(pairs,on=["dataset","fold"])
    reference["covered"] = reference["eligible"]
    reference["arm"] = "coded"
    reference["family_offset"] = 0
    all_rows = pd.concat([reference,df],ignore_index=True)
    all_rows["coverage"] = all_rows.covered.div(all_rows.eligible.where(all_rows.eligible>0))
    per_family = all_rows.groupby(["dataset","arm","family_offset"])[METRICS].mean()
    per_family.to_csv(OUT/"per-dataset-family.csv")
    per_family.groupby(["dataset","arm"]).mean().to_csv(OUT/"per-dataset.csv")
    per_family.groupby(["arm","family_offset"]).mean().to_csv(OUT/"per-family.csv")
    by_arm = all_rows.groupby("arm")
    metrics = by_arm[METRICS].mean()
    metrics["pair_weighted"] = by_arm.covered.sum()/by_arm.eligible.sum()
    metrics["informative_fold_families"] = by_arm.coverage.count()
    metrics.to_csv(OUT/"summary.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", nargs=2)
    args = p.parse_args()
    spec = json.loads(DESIGN.read_text(encoding="utf-8-sig"))
    offsets = spec["offsets"]
    if args.worker:
        with threadpool_limits(limits=1):
            return one(args.worker[0],int(args.worker[1]),offsets,spec["proposals"])
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
                subprocess.run([sys.executable,"-m","experiments.context_study.fixed_margins","--worker",name,str(fold)],
                               env=env,cwd=ROOT,check=True,timeout=3600)
            result = json.loads(done.read_bytes())
            if (result["protocol"]!=digest(protocol) or result["selection"]!=digest(dst/"selection.json")
                    or result["predictions"]!=digest(dst/"predictions.npz")):
                raise ValueError(f"Invalid receipt at {dst}")
            records.extend(result["rows"])
        print(f"Completed {name}; {len(records)}/252 fixed-margin refits",flush=True)
    df = pd.DataFrame(records)
    if len(df)!=252 or df.duplicated(["dataset","fold","arm","family_offset"]).any():
        raise ValueError("Incomplete study")
    df.to_csv(OUT/"per-fold.csv",index=False)
    primary = pd.read_csv(PRIMARY/"per-fold.csv")
    primary["retained"] = primary.k/primary.width
    summarize(df,primary)


if __name__=="__main__":
    main()
