"""Use a held-out-task failure model to nominate a single cross-width candidate for refitting."""
import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from experiments.context_review.budgeted_guard import (
    DONOR, ROOT, RUN as POOLS, digest, inputs, metrics, predict_sets, read, store, _within
)

OUT = ROOT/"results/context_review/learned-guard"
DESIGN = ROOT/"configs/context_review/learned_guard.json"
MODELS = ROOT/"results/context_review/incremental-prediction/models.json"


def nominate(pool, anchor, order, records, method):
    if not pool:
        return anchor, []
    positions = {c:i for i,c in enumerate(order)}
    scored = []
    for item in pool:
        cols = tuple(item["cols"])
        deletion = 1-len(cols)/len(anchor)
        values = [deletion, *(max(deletion-knot, 0.) for knot in (.25,.5,.75))]
        if method == "size_vote":
            values.append(1-item["approvals"]/4)
        probs = []
        for model in records:
            standardized = (np.asarray(values) - model["means"])/model["scales"]
            probs.append(expit(model["intercept"] + standardized @ model["coef"]))
        probability = float(np.mean(probs))
        score = deletion*(1-probability)
        scored.append(dict(cols=cols, failure_probability=probability, value=score))
    winner = min(scored, key=lambda r:(-r["value"],sum(positions[c] for c in r["cols"]),r["cols"]))
    return winner["cols"], scored


def run(name,fold):
    dst = OUT/name/f"fold-{fold}"
    dst.mkdir(exist_ok=True,parents=True)
    src = POOLS/name/f"fold-{fold}"
    slate = read(src/"shortlists.json")
    old = read(src/"completed.json")
    assert digest(src/"selection.json") == old["selection"]
    assert digest(src/"shortlists.json") == read(src/"selection.json")["shortlists"]
    anchor = tuple(slate["anchor"])
    pool = slate["screens"]["three"]
    fit_models = read(MODELS)
    selected, traces, times = {}, {}, {}
    for source in ("uniform_deletions","path"):
        for method in ("size","size_vote"):
            start = perf_counter()
            models = [m for m in fit_models[source] if m["model"]==method and
                      (m["dataset"]==name or name=="optdigits")]
            assert len(models)==(14 if name=="optdigits" else 1)
            assert all(name not in m["training_tasks"] for m in models)
            arm = source+"-"+method
            selected[arm],traces[arm] = nominate(pool,anchor,slate["order"],models,method)
            times[arm] = perf_counter()-start
    store(dst/"nominations.json",dict(protocol=digest(OUT/"protocol.json"),pool=digest(src/"shortlists.json"),
          nominees=selected,traces=traces))
    x,y,train,prop,guard,dev,xt,target,donor,hash_data = inputs(name,fold)
    assert slate["source"]==hash_data
    inner = np.sort(np.r_[train,prop])
    all_inner = sorted({anchor,*selected.values()})
    pg,inner_s,prep = predict_sets(x,y,inner,x[guard],all_inner,20421+fold)
    rare=read(src/"selection.json")["rare"]
    ref = metrics(y[guard],pg[anchor],rare)
    passed={s:bool(_within(metrics(y[guard],p,rare),ref)) for s,p in pg.items()}
    final={arm:s if passed[s] else anchor for arm,s in selected.items()}
    np.savez_compressed(dst/"guard.npz",y=y[guard],**{f"s{i}":pg[s] for i,s in enumerate(all_inner)})
    store(dst/"selection.json",dict(nominations=digest(dst/"nominations.json"),supports=all_inner,
          guard_sha=digest(dst/"guard.npz"),passed=[passed[s] for s in all_inner],final=final))
    frozen=digest(dst/"selection.json")
    all_outer=sorted({anchor,*final.values()})
    pf,outer_s,final_prep=predict_sets(x,y,dev,xt,all_outer,20421+fold)
    np.savez_compressed(dst/"predictions.npz",**{f"s{i}":pf[s] for i,s in enumerate(all_outer)})
    y_test=target()
    scores={s:metrics(y_test,p,rare) for s,p in pf.items()}
    rows=[]
    for arm,s in final.items():
        source,method=arm.split("-")
        failed=not _within(scores[s],scores[anchor])
        vote=method=="size_vote"
        old_components=next(r for r in old["rows"] if r["budget"]=="three" and
                            r["method"]==("vote_gain" if vote else "fraction_gain"))
        q=selected[arm]
        measured=(old_components["anchor_pool_s"]+old_components["vote_s"]+times[arm]+prep+
                  sum(inner_s[u] for u in {anchor,q})+final_prep+outer_s[s])
        rows.append(dict(dataset=name,fold=fold,source=source,method=method,anchor_k=len(anchor),nominee_k=len(q),
                    nominee=all_inner.index(q),final=all_outer.index(s),k=len(s),accepted=q!=anchor and passed[q],
                    failure=failed,safe_saved=(1-len(s)/len(anchor))*(not failed),
                    guard_candidates=int(q!=anchor),vote_seconds=old_components["vote_s"],total_s=measured,**scores[s]))
    assert digest(dst/"selection.json")==frozen
    store(dst/"completed.json",dict(rows=rows,selection=frozen,predictions=digest(dst/"predictions.npz"),
          deployment_supports=all_outer))


def summarize(rows):
    full = pd.DataFrame(rows)
    full.to_csv(OUT/"per-fold.csv",index=False)
    x = full[full.dataset!="optdigits"]
    fields = ["safe_saved","failure","auc","accuracy","accepted","k","guard_candidates","total_s"]
    means = x.groupby(["source","method","dataset"])[fields].mean()
    means.groupby(["source","method"]).mean().to_csv(OUT/"by-source.csv")
    assert len(full)==42*4 and not full.duplicated(["source","method","dataset","fold"]).any()
    effect=[]
    for source,group in x.groupby("source"):
        arms = [group[group.method==m].set_index(["dataset","fold"]).sort_index() for m in ("size","size_vote")]
        left,right=arms
        assert len(left)==41 and np.array_equal(left.guard_candidates,right.guard_candidates)
        delta=right[fields].astype(float)-left[fields].astype(float)
        task=delta.groupby("dataset").mean()
        picks=np.random.default_rng(33119).integers(len(task),size=(30000,len(task)))
        for metric in fields:
            v=task[metric].to_numpy()
            lo,hi=np.quantile(v[picks].mean(axis=1),[.025,.975])
            effect.append(dict(source=source,metric=metric,delta=v.mean(),ci_low=lo,ci_high=hi,
                               changed_widths=int((left.nominee_k!=right.nominee_k).sum()),
                               changed_nominations=int((left.nominee!=right.nominee).sum()),
                               changed_finals=int((left.final!=right.final).sum()),
                               pos_folds=int((delta[metric]>0).sum()),neg_folds=int((delta[metric]<0).sum())))
    pd.DataFrame(effect).to_csv(OUT/"paired-effects.csv",index=False)
    print(pd.DataFrame(effect).query("metric in ('safe_saved','failure')").to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            run(args.worker[0],int(args.worker[1]))
        return
    OUT.mkdir(exist_ok=True,parents=True)
    paths=[DESIGN,MODELS,POOLS/"protocol.json",POOLS/"per-fold.csv",__file__]
    lock={str(Path(p).relative_to(ROOT).as_posix()):digest(Path(p)) for p in paths}
    binding=OUT/"protocol.json"
    if binding.exists() and read(binding)!=lock:
        raise ValueError("Frozen inputs changed")
    store(binding,lock)
    valid=pd.read_csv(DONOR/"eligibility.csv")
    jobs=[("optdigits",0),*((r.dataset,r.fold) for r in valid.itertuples() if r.eligible)]
    rows=[]
    start=perf_counter()
    for i,(name,fold) in enumerate(jobs,1):
        d=OUT/name/f"fold-{fold}"
        if not (d/"completed.json").exists():
            subprocess.run([sys.executable,"-m","experiments.context_review.learned_guard","--worker",name,str(fold)],
                           timeout=3600,check=True)
        rec=read(d/"completed.json")
        assert rec["selection"]==digest(d/"selection.json") and rec["predictions"]==digest(d/"predictions.npz")
        assert read(d/"nominations.json")["protocol"]==digest(binding)
        rows.extend(rec["rows"])
        print(f"[{i}/{len(jobs)}] {name} fold {fold}, elapsed {perf_counter()-start:.1f}s",flush=True)
    summarize(rows)


if __name__=="__main__":
    main()
