"""Forward test on UCI Gas Sensor Array Drift. Keep future labels out of the fit worker."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import ROOT, digest
from experiments.context_study.selection import select_vote
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.maximin_challenger import nominate
from minppi_mofs.piv3_compact_tree import select as compact_gain
from minppi_mofs.piv3_compact_witness import _sample, _within

SPEC=ROOT/"configs/context_review/drift.json"
OUT=ROOT/"results/context_review/drift-locked"
DATA=ROOT/"data/context_study/uci-gas-drift/uci224.zip"
URL="https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
SHA="91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"
SEED=20421


def say(msg):
    line=f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    with (OUT/"run.log").open("a",encoding="utf8") as log:
        print(line,file=log)
    print(line,flush=True)


def acquire():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True,exist_ok=True)
        temp=DATA.with_suffix(".part")
        with urlopen(URL,timeout=120) as response, temp.open("wb") as out:
            copyfileobj(response,out)
        if digest(temp)!=SHA:
            raise ValueError("UCI download digest differs")
        temp.replace(DATA)
    if digest(DATA)!=SHA:
        raise ValueError("The UCI archive changed")


def member(batch):
    return f"Dataset/batch{batch}.dat"


def load_features(archive,batch):
    rows=[]
    with archive.open(member(batch)) as handle:
        for line in handle:
            _,_,fields=line.partition(b" ")
            nums=fields.split()
            cols=np.fromiter((int(t.partition(b":")[0])-1 for t in nums),dtype=np.int64)
            if len(set(cols))!=len(cols) or min(cols)<0 or max(cols)>=128:
                raise ValueError("Invalid feature index")
            row=np.zeros(128,dtype=np.float32)
            row[cols]=[float(t.partition(b":")[2]) for t in nums]
            rows.append(row)
    x=np.stack(rows)
    if not np.isfinite(x).all():
        raise ValueError("Unexpected nonfinite measurements")
    return x


def load_labels(archive,batch):
    with archive.open(member(batch)) as f:
        return np.fromiter((int(line.split(maxsplit=1)[0])-1 for line in f),dtype=np.int64)


def score(y,p):
    present=np.unique(y)
    labels=np.argmax(p,axis=1)
    auc=[roc_auc_score(y==c,p[:,c]) for c in present if 0<(y==c).sum()<len(y)]
    return dict(accuracy=float(accuracy_score(y,labels)),
                auc=float(np.mean(auc)) if auc else None,
                recall=float(recall_score(y,labels,labels=present,average="macro",zero_division=0)))


def worker():
    with ZipFile(DATA) as data:
        xs={b:load_features(data,b) for b in range(1,11)}
        ys={b:load_labels(data,b) for b in range(1,8)}
    tr=np.vstack([xs[b] for b in range(1,6)])
    yt=np.concatenate([ys[b] for b in range(1,6)])
    if any(set(y)!=set(range(6)) for y in (yt,ys[6],ys[7])):
        raise ValueError("A training-side stage lacks classes")
    y_inner=np.r_[yt,ys[6]]
    y_all=np.r_[y_inner,ys[7]]
    x_inner=np.vstack([tr,xs[6]])
    x_all=np.vstack([x_inner,xs[7]])
    future=np.vstack([xs[b] for b in (8,9,10)])
    batch_ids=np.repeat([8,9,10],[len(xs[b]) for b in (8,9,10)])
    comp={}
    start=perf_counter()
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(tr)
    xt,xp=imp.transform(tr),imp.transform(xs[6])
    model=fit(xt,yt,SEED)
    names=[f"c{i}" for i in range(128)]
    _,info=compact_gain(model,pd.DataFrame(xt,columns=names),yt,pd.DataFrame(xp,columns=names),
                        ys[6],task="multiclass",utility="ovr_auc",seed=SEED,aggregate="gain")
    order=[int(s[1:]) for s in info["order"]]
    k=info["k"]
    comp["anchor"]=perf_counter()-start
    supports={"full":list(range(128)),"compact_gain":sorted(order[:k])}
    traces={}
    for arm,rule in (("vote","antichain"),("random_family","random_contexts")):
        start=perf_counter()
        supports[arm],traces[arm]=select_vote(rule,model,xt,xp,ys[6],order,k,0)
        comp[arm]=perf_counter()-start
    supports["gain_at_vote_width"]=sorted(order[:len(supports["vote"])])
    live=[i for i in order if i in supports["vote"]]
    family=separating_contexts(order[k:])
    start=perf_counter()
    ids,weights=_sample(ys[6],"multiclass",np.random.SeedSequence([SEED,659]))
    bg=np.random.default_rng(SEED).choice(len(tr),min(512,len(tr)),replace=False)
    args=(model,xp[ids],np.median(xt[bg],axis=0),ys[6][ids],weights,frozenset(order[:k]),False)
    vote,cache=oracle(*args)
    best,trace=nominate(live,family,vote)
    comp["maximin"]=perf_counter()-start
    eager,seen=oracle(*args)
    exact,brute=nominate(live,family,eager,prune=False)
    if best!=exact or trace["min_yes"]!=brute["min_yes"]:
        raise ValueError("Pruned maximum differs from enumeration")
    supports["maximin"]=sorted(best)
    supports["weak_gain"]=sorted(live[:-1] if len(live)>1 else live)
    say(f"Frozen proposals, anchor {k}, vote {len(live)}, challenger {len(best)}")
    start=perf_counter()
    prep=SimpleImputer(strategy="median",keep_empty_features=True).fit(x_inner)
    xi,xg=prep.transform(x_inner),prep.transform(xs[7])
    comp["guard_prep"]=perf_counter()-start
    guard={}
    gates={}
    inner_preds={}
    for arm in ("compact_gain","vote","maximin","weak_gain"):
        start=perf_counter()
        s=supports[arm]
        pg=fit(xi[:,s],y_inner,SEED).predict_proba(xg[:,s])
        inner_preds[arm]=pg
        m=score(ys[7],pg)
        guard[arm]={i:m[i] for i in ("auc","accuracy")}
        comp["guard_"+arm]=perf_counter()-start
    for arm in ("vote","maximin","weak_gain"):
        gates[arm]=_within(guard[arm],guard["compact_gain"])
        fallback="vote" if arm!="vote" and gates["vote"] else "compact_gain"
        supports[arm+"_guard"]=supports[arm] if gates[arm] else supports[fallback]
    start=perf_counter()
    imp_all=SimpleImputer(strategy="median",keep_empty_features=True).fit(x_all)
    xa,xe=imp_all.transform(x_all),imp_all.transform(future)
    comp["final_prep"]=perf_counter()-start
    for arm,n in (("rfe_at_vote_width",len(live)),("rfe_at_challenger_width",len(best)),
                  ("anova_at_10pct",13),("anova_at_25pct",32)):
        start=perf_counter()
        if arm.startswith("rfe"):
            estimator=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.1,tree_method="hist",
                                    device="cpu",random_state=SEED,n_jobs=1,importance_type="gain")
            selector=RFE(estimator,n_features_to_select=n,step=.2)
        else:
            selector=SelectKBest(f_classif,k=n)
        selector.fit(xa,y_all)
        supports[arm]=selector.get_support(indices=True).tolist()
        comp[arm]=perf_counter()-start
    arms=json.loads(SPEC.read_bytes())["arms"]
    assert set(supports)==set(arms)
    store(OUT/"supports.json",dict(supports=supports,rank_order=order,gate_pass=gates,guard=guard,
          lazy_visits=trace["visits"],eager_visits=brute["visits"],lazy_masks=len(cache),eager_masks=len(seen),
          mask_calls={a:v["mask_calls"] for a,v in traces.items()}))
    predictions={}
    for arm in arms:
        s=supports[arm]
        start=perf_counter()
        predictions[arm]=fit(xa[:,s],y_all,SEED).predict_proba(xe[:,s])
        comp["final_"+arm]=perf_counter()-start
    np.savez_compressed(OUT/"predictions.npz",batch=batch_ids,**predictions)
    np.savez_compressed(OUT/"guard-probabilities.npz",y=ys[7],**inner_preds)
    store(OUT/"frozen.json",dict(components_seconds=comp,rows={b:len(x) for b,x in xs.items()},
          supports=digest(OUT/"supports.json"),predictions=digest(OUT/"predictions.npz"),
          guard_predictions=digest(OUT/"guard-probabilities.npz"),lock=digest(OUT/"protocol.json")))
    say("All external predictions and supports frozen.")


def score_frozen():
    rec=json.loads((OUT/"frozen.json").read_bytes())
    for key,path in (("lock","protocol.json"),("supports","supports.json"),("predictions","predictions.npz"),
                     ("guard_predictions","guard-probabilities.npz")):
        if digest(OUT/path)!=rec[key]:
            raise ValueError(f"Frozen artifact changed: {path}")
    say("Opening future batch labels in scoring stage.")
    with ZipFile(DATA) as z:
        labels={b:load_labels(z,b) for b in (8,9,10)}
    support=json.loads((OUT/"supports.json").read_bytes())["supports"]
    rows=[]
    with np.load(OUT/"predictions.npz") as preds:
        for arm,s in support.items():
            for b,y in labels.items():
                p=preds[arm][preds["batch"]==b]
                rows.append(dict(arm=arm,batch=b,n=len(y),classes=len(np.unique(y)),k=len(s),**score(y,p)))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/"by-batch.csv",index=False)
    frame.groupby("arm")[["k","recall","accuracy","auc"]].mean().to_csv(OUT/"summary.csv")
    print(frame.groupby("arm")[["k","recall","accuracy","auc"]].mean(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    acquire()
    if args.worker:
        with threadpool_limits(limits=1):
            return worker()
    OUT.mkdir(exist_ok=True,parents=True)
    dependencies=[Path(__file__),SPEC,ROOT/"src/minppi_mofs/maximin_challenger.py",
        ROOT/"experiments/context_study/selection.py",ROOT/"experiments/context_study/benchmark.py",
        ROOT/"experiments/context_study/working_set.py",ROOT/"src/minppi_mofs/context_design.py",
        ROOT/"src/minppi_mofs/context_vote.py",ROOT/"src/minppi_mofs/piv3_compact_witness.py",
        ROOT/"src/minppi_mofs/piv3_compact_tree.py",ROOT/"experiments/test/minpi_convergence_metrics.py",
        ROOT/"src/minppi_mofs/utility.py"]
    lock=dict(archive_sha256=SHA,files={p.relative_to(ROOT).as_posix():digest(p) for p in dependencies})
    if (OUT/"protocol.json").exists() and json.loads((OUT/"protocol.json").read_bytes())!=lock:
        raise ValueError("Frozen protocol or implementation differs")
    store(OUT/"protocol.json",lock)
    if not (OUT/"frozen.json").exists():
        say("Selecting and refitting without future labels.")
        subprocess.run([sys.executable,"-m","experiments.context_review.drift","--worker"],timeout=3600,check=True)
    score_frozen()


if __name__=="__main__":
    main()
