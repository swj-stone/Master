"""Collect shared-mask rankings and matched nonlinear refits, then score the sealed selections."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from threading import Timer
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

from experiments.minpi_compression import legacy_gain
from experiments.minpi_rework import validation_panel
from experiments.refit_voting import games


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/"thesis-paper/evidence/refit_voting"
SEED = 42


def splits(panel,name):
    if panel=="prepared":
        directory, row = legacy_gain.jobs()[name,SEED]
        return legacy_gain.read_split(directory,row,SEED)
    return validation_panel.split(name,SEED)


def true_y(panel,name,idx):
    if panel=="prepared":
        directory,row=legacy_gain.jobs()[name,SEED]
        return pd.read_csv(directory/row["test_relpath"],usecols=[row["target"]])[row["target"]].to_numpy()
    return validation_panel._partition(name,SEED)[1][idx["test"]]


def metrics(y,pred,task,scale):
    if task=="classification":
        used=np.unique(y)
        auc=np.mean([roc_auc_score(y==c,pred[:,int(c)]) for c in used if (y!=c).any()])
        return dict(auc=auc,accuracy=accuracy_score(y,pred.argmax(axis=1)))
    mae=mean_absolute_error(y,pred)
    return dict(r2=r2_score(y,pred),mae=mae,scaled_mae=mae/scale)


def passes(score,full):
    if "auc" in full:
        return score["auc"]>=full["auc"]-.01 and score["accuracy"]>=full["accuracy"]-.01
    return score["r2"]>=full["r2"]-.02 and score["scaled_mae"]<=full["scaled_mae"]+.02


def fit(x,y,task):
    params=dict(n_estimators=64,max_depth=4,learning_rate=.12,n_jobs=1,tree_method="hist",
                random_state=SEED,verbosity=0)
    model=(XGBClassifier if task=="classification" else XGBRegressor)(**params)
    model.fit(x,y)
    return model


def predict(model,x,task):
    return model.predict_proba(x) if task=="classification" else model.predict(x)


def data_jobs():
    return [("prepared",name) for name,seed in legacy_gain.jobs() if seed==SEED]+[
        ("development",name) for name in validation_panel.SOURCES]


def collect_one(panel,name):
    x,y,task,idx=splits(panel,name)
    p=x.shape[1]
    train,vote,accept,test=(np.asarray(idx[k]) for k in ("fit","vote","width","test"))
    xf,yf=x[train],y[train]
    if task=="classification":
        yf=yf.astype(int)
    scale=max(1e-8,float(np.std(yf)))
    start=perf_counter()
    full=fit(xf,yf,task)
    acc=metrics(y[accept],predict(full,x[accept],task),task,scale)
    fit_s=perf_counter()-start
    start=perf_counter()
    full_predict=predict(full,x[test],task)
    full_predict_s=perf_counter()-start
    full_rec=(acc,fit_s,full_predict,full_predict_s)
    start=perf_counter()
    gain=np.lexsort((np.arange(p),-full.feature_importances_))
    orders={"gain":(gain,perf_counter()-start,None)}
    rng=np.random.default_rng(SEED+901)
    orders["random"]=(rng.permutation(p),0.,None)
    orders.update(games.ranks(xf,yf,x[vote],y[vote],task=="classification",SEED))
    widths=sorted(set([p,1,*((p+h-1)//h for h in (2,4,8))]))
    cache={tuple(range(p)):full_rec}
    rows=[]
    kept_predictions={}
    for m in rng.permutation(list(orders)):
        order,rank_s,_=orders[m]
        trials={}
        for k in widths:
            cols=tuple(sorted(map(int,order[:k])))
            if cols not in cache:
                start=perf_counter()
                candidate=fit(xf[:,cols],yf,task)
                validated=metrics(y[accept],predict(candidate,x[np.ix_(accept,cols)],task),task,scale)
                elapsed=perf_counter()-start
                start=perf_counter()
                pred=predict(candidate,x[np.ix_(test,cols)],task)
                cache[cols]=(validated,elapsed,pred,perf_counter()-start)
            trials[k]=(cols,cache[cols])
        for budget,candidates in (("grid",widths),("half",sorted({(p+1)//2,p}))):
            for k in candidates:
                cols,(native,_,held,final_s)=trials[k]
                if passes(native,acc):
                    break
            label=f"{m}-{budget}"
            total=rank_s+sum(trials[q][1][1] for q in candidates)+final_s
            rows.append(dict(panel=panel,dataset=name,method=m,budget=budget,k=k,p=p,
                             rank_s=rank_s,seconds=total,fits=len(candidates),columns=list(cols),
                             acceptance=native,scale=scale,task=task))
            kept_predictions[label]=held
    rows.append(dict(panel=panel,dataset=name,method="full",budget="full",k=p,p=p,
                     rank_s=0.,seconds=fit_s+full_predict_s,fits=1,columns=list(range(p)),
                     acceptance=acc,scale=scale,task=task))
    kept_predictions["full-full"]=full_predict
    return rows,kept_predictions,orders,cache,idx,task


def filepath(panel,name):
    return DEST/"runs"/f"{panel}-{name}"


def collect():
    rng=np.random.default_rng(4817)
    games.ranks(rng.normal(size=(64,4)),np.arange(64)%2,rng.normal(size=(32,4)),np.arange(32)%2,True,22)
    jobs=data_jobs()
    for i,(panel,name) in enumerate(jobs,1):
        folder=filepath(panel,name)
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"choices.json").is_file():
            continue
        start=perf_counter()
        timer=Timer(3600,lambda:os._exit(124))
        timer.daemon=True
        timer.start()
        try:
            records,forecast,orders,cache,idx,task=collect_one(panel,name)
            np.savez_compressed(folder/"forecasts.npz",**forecast)
            # Keep voting-stage scores only via their orders; individual validation records support later cost diagnostics.
            data=dict(records=records, orders={m:v[0].tolist() for m,v in orders.items()},
                      trials=[dict(cols=list(c),accept=r[0],time=r[1],predict_s=r[3]) for c,r in cache.items()],
                      forecast_sha=sha256((folder/"forecasts.npz").read_bytes()).hexdigest(),
                      input_rows={k:sha256(np.asarray(v,dtype="<i8").tobytes()).hexdigest() for k,v in idx.items()})
            (folder/"choices.json").write_text(json.dumps(data,indent=1),encoding="utf8")
        finally:
            timer.cancel()
        print(f"[{i}/{len(jobs)}] {panel}/{name} {perf_counter()-start:.1f}s",flush=True)
    hashes={f"{p}/{n}":sha256((filepath(p,n)/"choices.json").read_bytes()).hexdigest() for p,n in jobs}
    (DEST/"lock.json").write_text(json.dumps(hashes,indent=2),encoding="utf8")


def score():
    lock=json.loads((DEST/"lock.json").read_bytes())
    out=[]
    for p,n in data_jobs():
        fold=filepath(p,n)
        raw=(fold/"choices.json").read_bytes()
        assert sha256(raw).hexdigest()==lock[f"{p}/{n}"]
        data=json.loads(raw)
        assert sha256((fold/"forecasts.npz").read_bytes()).hexdigest()==data["forecast_sha"]
        _,_,_,idx=splits(p,n)
        truth=true_y(p,n,idx)
        with np.load(fold/"forecasts.npz") as forecasts:
            for row in data["records"]:
                pred=forecasts[row["method"]+"-"+row["budget"]]
                measured=metrics(truth,pred,row["task"],row["scale"])
                out.append({k:v for k,v in row.items() if k not in ("columns","acceptance")}|measured)
    pd.DataFrame(out).to_csv(DEST/"scores.csv",index=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=["collect","score"])
    with threadpool_limits(1):
        (collect if cli.parse_args().phase=="collect" else score)()
