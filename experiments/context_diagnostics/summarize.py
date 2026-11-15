"""Recompute votes and held-out failure rates from stored, independent-row predictions."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.context_diagnostics.design import CONFIG, OUTPUT, ROOT, config, decide, digest
from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.experiment_protocol import atomic_write_json
from minppi_mofs.piv3_compact_witness import _measure, _within


def votes_for_rows(p, labels, weights, d, audit_ids):
    scorer=make_scorer(labels,weights,d["task"],d["utility"],d["rare"])
    scores=[scorer(pred) for pred in p]
    for index in audit_ids:
        for panel, fast in zip(weights,scores[index],strict=True):
            independent=_measure(labels,p[index],panel,d["task"],d["rare"])
            for key in fast:
                np.testing.assert_allclose(fast[key],independent[key],rtol=1e-8,atol=1e-10)
    lookup=d["lookup"]
    return np.asarray([[[_within(c,a) for c,a in zip(scores[i],scores[j],strict=True)]
                        for i,j in zip(row,lookup[0],strict=True)] for row in lookup[1:]],dtype=bool)


def readout(path, audit_all=False, portable=False):
    d=json.loads(path.read_bytes())
    values=path.parent/"predictions.npz"
    if digest(values)!=d["predictions_sha256"]:
        raise ValueError(f"Stored predictions changed in {path.parent}")
    if not portable:
        for filename,expected in d["source"].items():
            if digest(ROOT/filename)!=expected:
                raise ValueError(f"Archived source changed: {filename}")
    ms=list(map(frozenset,d["masks"]))
    contexts=list(map(frozenset,d["extensions"]))
    supports=list(map(frozenset,[d["anchor"],*d["candidates"]]))
    novel=d["novel"]
    rules=d["rules"]
    if set(novel)&set.union(*(set(f) for f in rules.values())):
        raise ValueError("A held-out restoration was exposed during assessment")
    if any(e&supports[0] for e in contexts) or len(set(contexts))!=len(contexts):
        raise ValueError("Invalid context family")
    for support,row in zip(supports,d["lookup"],strict=True):
        if [ms[i] for i in row]!=[support|e for e in contexts]:
            raise ValueError("Masked-prediction lookup disagrees with support")
    for left,right in (("nested","random_middle"),("separating","random_matched")):
        if [len(contexts[i]) for i in rules[left]] != [len(contexts[i]) for i in rules[right]]:
            raise ValueError("Unmatched comparator")
    with np.load(values,allow_pickle=False) as raw:
        preds=raw["predictions"]
        n=len(raw["rank_y"])
        if np.intersect1d(raw["rank_rows"],raw["guard_rows"]).size:
            raise ValueError("Voting and evaluation rows overlap")
        audit_ids=range(len(preds)) if audit_all else np.unique(np.linspace(0,len(preds)-1,
                                            min(4,len(preds)),dtype=int))
        ranked= votes_for_rows(preds[:,:n],raw["rank_y"],raw["rank_w"],d,audit_ids)
        guard= votes_for_rows(preds[:,n:],raw["guard_y"],raw["guard_w"],d,audit_ids)
    if len(ranked)!=len(d["original_votes"]):
        raise ValueError("Original candidate coverage differs")
    rows, flips, confirmations=[], [], []
    for idx,original in enumerate(d["original_votes"]):
        regenerated=ranked[idx,rules["nested"][:len(original)]]
        np.testing.assert_array_equal(regenerated,original,
                                      err_msg=f"Archived decision mismatch: {path.parent}, proposal {idx}")
        rank_n=ranked[idx].sum(axis=1)
        guard_n=guard[idx].sum(axis=1)
        identity=dict(dataset=d["dataset"],seed=d["seed"],proposal=idx,anchor_k=len(supports[0]),
                      tail_k=len(d["tail"]),proposed_k=len(supports[idx+1]))
        for role,values in (("ranking",rank_n),("guard",guard_n)):
            for size in sorted({len(contexts[i]) for i in novel}):
                group=[i for i in novel if len(contexts[i])==size]
                fails=int((values[group]<3).sum())
                flips.append(dict(**identity,role=role,tail_size=size,contexts=len(group),
                                  failing=fails,discordant=int(0<fails<len(group))))
        for size in sorted({len(contexts[i]) for i in novel}):
            group=[i for i in novel if len(contexts[i])==size]
            for i,j in combinations(group,2):
                dr=int(rank_n[i]>=3)-int(rank_n[j]>=3)
                dg=int(guard_n[i]>=3)-int(guard_n[j]>=3)
                confirmations.append(dict(**identity,tail_size=size,context_i=i,context_j=j,
                                          rank_difference=dr,guard_difference=dg))
        for name,family in rules.items():
            accepted, visits=decide(name,rank_n[family])
            confirm,_=decide(name,guard_n[family])
            rows.append(dict(**identity,rule=name,design_contexts=len(family),visits=visits,
                             accept=int(accepted),guard_design_accept=int(confirm),novel=len(novel),
                             rank_novel_failing=int((rank_n[novel]<3).sum()),
                             guard_novel_failing=int((guard_n[novel]<3).sum())))
    return rows,flips,confirmations


def summarize(data, groups):
    rows=[]
    for key,full in data.groupby(groups,sort=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(groups,key,strict=True))
        g=full[full.novel.gt(0)]
        wrong=g.guard_novel_failing.gt(0)
        accept=g.accept.astype(bool)
        wrong_rank=g.rank_novel_failing.gt(0)
        row.update(proposals=len(full),covered=len(g),datasets=g.dataset.nunique(),
                   splits=g[["dataset","seed"]].drop_duplicates().shape[0],
                   accepted_all=int(full.accept.sum()),accepted=int(accept.sum()),
                   failed_any=int(wrong.sum()),accepted_with_failure=int((accept&wrong).sum()),
                   rejected_without_failure=int((~accept&~wrong).sum()),
                   rank_accepted_with_failure=int((accept&wrong_rank).sum()),
                   rank_rejected_without_failure=int((~accept&~wrong_rank).sum()),
                   confirmed_on_guard=int((full.accept==full.guard_design_accept).sum()),
                   mean_visits=full.visits.mean(),
                   failure_among_accepted=wrong[accept].mean(),
                   accepted_failures_per_proposal=(accept&wrong).mean(),
                   rejected_without_failure_per_proposal=(~accept&~wrong).mean(),
                   context_violations_per_accepted_probe=(g.loc[accept,"guard_novel_failing"]/
                                                          g.loc[accept,"novel"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUTPUT/"trees")
    parser.add_argument("--audit-all",action="store_true",
                        help="Compare all mask scores with scikit-learn instead of four masks per fold")
    parser.add_argument("--portable",action="store_true",help="Audit the local predictions without original archives")
    args=parser.parse_args()
    schema=config()["archived_trees"]
    allrows, allflips, allpairs=[], [], []
    for seed in schema["seeds"]:
        for ds in schema["datasets"]:
            receipts=args.run/f"seed-{seed}"/ds/"receipt.json"
            rows,flips,pairs=readout(receipts,args.audit_all,args.portable)
            allrows.extend(rows)
            allflips.extend(flips)
            allpairs.extend(pairs)
    data=pd.DataFrame(allrows)
    data.to_csv(args.run/"decisions.csv",index=False)
    pd.DataFrame(allflips).to_csv(args.run/"same-size-disagreement.csv",index=False)
    pd.DataFrame(allpairs).to_csv(args.run/"same-size-pairs.csv",index=False)
    for name,groups in (("by-split",["dataset","seed","rule"]),("by-dataset",["dataset","rule"]),
                         ("overall",["rule"])):
        summarize(data,groups).to_csv(args.run/(name+".csv"),index=False)
    files=[args.run/"pins.json",CONFIG,Path(__file__),ROOT/"experiments/test/minpi_convergence_metrics.py",
           ROOT/"src/minppi_mofs/piv3_compact_witness.py",*sorted(args.run.glob("seed-*/*/receipt.json"))]
    atomic_write_json(args.run/"audit-sources.json",{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    print(summarize(data,["rule"]).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
