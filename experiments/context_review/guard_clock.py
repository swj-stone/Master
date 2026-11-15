"""Time stand-alone half-width policies, with a fresh reference, guard and final fit per arm."""
import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.timing_process import bind_logical_cpu
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_witness import _sample, _within

OUT = bg.ROOT/"results/context_review/guard-clock"
SPEC = bg.ROOT/"configs/context_review/guard_clock.json"


def vote_nominee(model, xt, xp, labels, order, k, w, seed, index, fold):
    task="binary" if model.n_classes_==2 else "multiclass"
    ids,weights=_sample(labels,task,np.random.SeedSequence([seed,659]))
    rare=task=="binary" and min(np.bincount(labels))<=.05*len(labels)
    rows=np.random.default_rng(seed).choice(len(xt),min(len(xt),512),replace=False)
    ballot,cache=oracle(model,xp[ids],np.median(xt[rows],axis=0),labels[ids],weights,frozenset(order[:k]),rare)
    family=separating_contexts(order[k:])
    candidates=bg.candidate_pool(order[:k],w,index,fold)[1]
    best=-1
    chosen=candidates[0]
    for candidate in candidates:
        running=4
        for context in family:
            running=min(running,sum(ballot(frozenset(candidate),context)))
            if running<=best or running==0:
                break
        if running>best:
            best,chosen=running,candidate
        if best==4:
            break
    return chosen,len(cache)


def policy(dataset, method, seed, index, fold):
    x,y,train,proposal,guard,development,outer,_,_,_ = dataset
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xp=imp.transform(x[train]),imp.transform(x[proposal])
    model=bg.fit(xt,y[train],seed)
    task="binary" if model.n_classes_==2 else "multiclass"
    cols=[f"c{i}" for i in range(x.shape[1])]
    _,info=bg.gain_anchor(model,pd.DataFrame(xt,columns=cols),y[train],pd.DataFrame(xp,columns=cols),
                          y[proposal],task=task,utility="ovr_auc",seed=seed,aggregate="gain")
    order=[int(c[1:]) for c in info["order"]]
    k=info["k"]
    w=(k+1)//2
    anchor=tuple(sorted(order[:k]))
    selected=nominee=anchor
    masks=0
    screen=0.
    if w<k:
        ts=perf_counter()
        if method=="vote":
            nominee,masks=vote_nominee(model,xt,xp,y[proposal],order,k,w,seed,index,fold)
        else:
            nominee=tuple(sorted(order[:w]))
        screen=perf_counter()-ts
        inner=np.sort(np.r_[train,proposal])
        p,_,_=bg.predict_sets(x,y,inner,x[guard],[anchor,nominee],seed)
        rare=task=="binary" and min(np.bincount(y[proposal]))<=.05*len(proposal)
        if _within(bg.metrics(y[guard],p[nominee],rare),bg.metrics(y[guard],p[anchor],rare)):
            selected=nominee
    final,_,_=bg.predict_sets(x,y,development,outer,[selected],seed)
    return nominee,selected,final[selected],masks,screen


def worker(name,fold,index):
    cpu=bind_logical_cpu()
    if cpu is None:
        raise RuntimeError("This timing design requires binding workers to a logical CPU")
    dataset=bg.inputs(name,fold)
    src=bg.RUN/name/f"fold-{fold}"
    slates=bg.read(src/"shortlists.json")
    selection=bg.read(src/"selection.json")
    ref_supports=sorted(set(tuple(v) for v in selection["selections"].values()))
    start_order=["prefix","vote"] if (index+fold)%2==0 else ["vote","prefix"]
    schedule=start_order+list(reversed(start_order))
    rows=[]
    with np.load(src/"predictions.npz") as reference:
        for position,arm in enumerate(schedule):
            ts,tc=perf_counter(),process_time()
            q,s,p,calls,screen=policy(dataset,arm,20421+fold,index,fold)
            wall,cpu_s=perf_counter()-ts,process_time()-tc
            key="one_vote_gain" if arm=="vote" else "one_fraction_gain"
            assert list(q)==next(iter(slates["shortlists"][key]),slates["anchor"])
            assert list(s)==selection["selections"][key]
            np.testing.assert_array_equal(p,reference[f"s{ref_supports.index(s)}"])
            rows.append(dict(dataset=name,fold=fold,position=position,arm=arm,wall_s=wall,cpu_s=cpu_s,
                             screen_s=screen,masks=calls,logical_cpu=cpu))
    dest=OUT/name/f"fold-{fold}"
    dest.mkdir(exist_ok=True,parents=True)
    bg.store(dest/"verified.json",dict(protocol=bg.digest(OUT/"protocol.json"),rows=rows))


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    lock={p.name:bg.digest(p) for p in (SPEC,Path(__file__),bg.RUN/"protocol.json",Path(bg.__file__))}
    binding=OUT/"protocol.json"
    if binding.exists() and bg.read(binding)!=lock:
        raise ValueError("Changed timing inputs")
    bg.store(binding,lock)
    tasks=bg.config()["datasets"]
    eligible=pd.read_csv(bg.DONOR/"eligibility.csv")
    jobs=[("optdigits",0,100),*((r.dataset,r.fold,tasks.index(r.dataset)) for r in eligible.itertuples() if r.eligible)]
    rows=[]
    start=perf_counter()
    for i,(name,fold,index) in enumerate(jobs,1):
        receipt=OUT/name/f"fold-{fold}"/"verified.json"
        if not receipt.exists():
            subprocess.run([sys.executable,"-m","experiments.context_review.guard_clock","--worker",name,str(fold),str(index)],
                           check=True,timeout=3600)
        info=bg.read(receipt)
        assert info["protocol"]==bg.digest(binding)
        rows.extend(info["rows"])
        print(f"[{i}/{len(jobs)}] {name} fold {fold}, elapsed {perf_counter()-start:.0f}s",flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==168 and not df.duplicated(["dataset","fold","position"]).any()
    df.to_csv(OUT/"runs.csv",index=False)
    fields=["wall_s","cpu_s","screen_s","masks"]
    means=df.groupby(["dataset","fold","arm"])[fields].mean().groupby(["dataset","arm"]).mean()
    means.to_csv(OUT/"by-task.csv")
    pmlb=means.drop(index="optdigits",level="dataset")
    base,vote=[pmlb.xs(a,level="arm") for a in ("prefix","vote")]
    diffs=vote-base
    ratio=vote.wall_s/base.wall_s
    rng=np.random.default_rng(33119)
    xs=diffs.wall_s.to_numpy()
    lo,hi=np.percentile(xs[rng.integers(len(xs),size=(30000,len(xs)))].mean(axis=1),[2.5,97.5])
    bg.store(OUT/"summary.json",dict(tasks=len(xs),complete_runs=164,external_runs=4,
             prefix=base.mean().to_dict(),vote=vote.mean().to_dict(),diff=diffs.mean().to_dict(),
             wall_interval=[lo,hi],median_wall_ratio=ratio.median(),vote_slower_tasks=int((ratio>1).sum())))
    print((OUT/"summary.json").read_text(encoding="utf8"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=3)
    args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            worker(args.worker[0],int(args.worker[1]),int(args.worker[2]))
    else:
        run()
