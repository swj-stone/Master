"""Interrupt real candidate training on an unfavorable early vote."""
import argparse
import json
from hashlib import sha256
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits

from experiments.refit_voting import games, run


OUT=run.DEST/"race"
ARMS=("direct","intact","pooled","minimum","minimum0","minimum2")
CHECKPOINT=8


def contribution(booster,x,task,y,limit=None):
    probs=booster.inplace_predict(x,iteration_range=(0,limit or 0))
    if task=="regression":
        return (probs-y)**2
    if probs.ndim==1:
        return 2*(probs-y)**2
    return 1+np.square(probs).sum(axis=1)-2*probs[np.arange(len(y)),y.astype(int)]


class Voter:
    def __init__(self,boost,x,y,task,seed,only_intact):
        ix,w=games.sample(games.strata(y,task=="classification"),256,seed+375)
        self.x=np.array(x[ix],dtype=np.float32)
        self.y=np.asarray(y[ix])
        self.w=w/w.sum()
        self.task=task
        if task=="classification":
            freq=np.bincount(self.y.astype(int),weights=self.w)
            self.null=max(1e-8,1-np.sum(freq**2))
        else:
            self.null=max(1e-8,np.average((self.y-self.y@self.w)**2,weights=self.w))
        self.cases=[]
        rng=np.random.default_rng(seed+277)
        p=self.x.shape[1]
        for fraction in ([1.] if only_intact else (1.,.5,.25)):
            k=max(1,int(np.ceil(fraction*p)))
            items=[]
            for _ in range(1 if k==p else 4):
                masked=self.x.copy()
                if k<p:
                    available=rng.choice(p,k,replace=False)
                    masked[:,np.setdiff1d(np.arange(p),available)]=np.nan
                items.append(masked)
            batch=np.ascontiguousarray(np.concatenate(items))
            obs=contribution(boost,batch,task,np.tile(self.y,len(items)),CHECKPOINT)
            self.cases.append((batch,obs.reshape(len(items),-1).mean(axis=0)))

    def vote(self,boost,cols,arm):
        errors=[]
        for batch,ref in self.cases:
            n=len(batch)//len(self.y)
            loss=contribution(boost,np.ascontiguousarray(batch[:,cols]),self.task,np.tile(self.y,n))
            diff=loss.reshape(n,-1).mean(axis=0)-ref
            errors.append(diff)
            if arm=="intact":
                break
            if arm.startswith("minimum") and self.negative(diff,arm):
                return True
        if arm.startswith("minimum"):
            return False
        return self.negative(np.mean(errors,axis=0),arm)

    def negative(self,delta,arm):
        mean=delta@self.w
        se=np.sqrt(np.sum(self.w**2*(delta-mean)**2))
        multiplier={"minimum0":0.,"minimum2":2.}.get(arm,1.)
        return bool(mean > .01*self.null+multiplier*se)


class EarlyVeto(xgb.callback.TrainingCallback):
    def __init__(self,voter,cols,arm):
        self.voter,self.cols,self.arm=voter,cols,arm
        self.rejected=False

    def after_iteration(self,model,epoch,evals_log):
        if epoch==CHECKPOINT-1:
            self.rejected=self.voter.vote(model,self.cols,self.arm)
        return self.rejected


def pipeline(x,y,task,idx,arm):
    fit,vote,width,test=(idx[k] for k in ("fit","vote","width","test"))
    clock=perf_counter()
    full=run.fit(x[fit],y[fit],task)
    acc=run.metrics(y[width],run.predict(full,x[width],task),task,max(float(y[fit].std()),1e-8))
    p=x.shape[1]
    order=np.lexsort((np.arange(p),-full.feature_importances_))
    scale=max(float(y[fit].std()),1e-8)
    start=perf_counter()
    voter=Voter(full.get_booster(),x[vote],y[vote],task,run.SEED,arm=="intact") if arm!="direct" else None
    setup=perf_counter()-start
    rounds=64
    fits=1
    rejects=[]
    chosen=np.arange(p)
    selected=full
    for k in sorted(set([1,*((p+h-1)//h for h in (2,4,8))])-{p}):
        cols=np.sort(order[:k])
        family=xgb.XGBClassifier if task=="classification" else xgb.XGBRegressor
        callback=EarlyVeto(voter,cols,arm) if voter is not None else None
        candidate=family(n_estimators=64,max_depth=4,learning_rate=.12,n_jobs=1,tree_method="hist",
                         random_state=run.SEED,verbosity=0,callbacks=[callback] if voter is not None else None)
        candidate.fit(x[np.ix_(fit,cols)],y[fit])
        fits+=1
        rounds+=candidate.get_booster().num_boosted_rounds()
        if callback and callback.rejected:
            rejects.append(int(k))
            continue
        native=run.metrics(y[width],run.predict(candidate,x[np.ix_(width,cols)],task),task,scale)
        if run.passes(native,acc):
            chosen,selected=cols,candidate
            break
    pred=run.predict(selected,x[np.ix_(test,chosen)],task)
    elapsed=perf_counter()-clock
    return dict(k=len(chosen),p=p,setup_s=setup,seconds=elapsed,rounds=rounds, fits=fits, rejected=rejects,
                scale=scale, cols=chosen.tolist(), acceptance=acc),pred


def collect():
    for panel,name in run.data_jobs():
        folder=OUT/f"{panel}-{name}"
        folder.mkdir(parents=True,exist_ok=True)
        receipt=folder/"choices.json"
        if receipt.exists() and json.loads(receipt.read_bytes()).get("schema")==2:
            continue
        x,y,task,idx=run.splits(panel,name)
        start=perf_counter()
        out=[]
        preds={}
        rng=np.random.default_rng(3381)
        for r in range(2):
            for arm in rng.permutation(ARMS):
                data,pred=pipeline(x,y,task,idx,arm)
                out.append(dict(panel=panel,dataset=name,task=task,method=arm,repeat=r,**data))
                if r:
                    np.testing.assert_array_equal(pred,preds[arm])
                preds[arm]=pred
        # Direct narrowest-first scoring and the full-grid reference must agree exactly.
        original=run.filepath(panel,name)
        with np.load(original/"forecasts.npz") as old:
            np.testing.assert_allclose(preds["direct"],old["gain-grid"],atol=0,rtol=0)
        np.savez_compressed(folder/"forecasts.npz",**preds)
        result=dict(schema=2,rows=out,forecast=sha256((folder/"forecasts.npz").read_bytes()).hexdigest(),
                    input_rows={k:sha256(np.asarray(v,dtype="<i8").tobytes()).hexdigest() for k,v in idx.items()})
        receipt.write_text(json.dumps(result,indent=1),encoding="utf8")
        print(f"{name}: {perf_counter()-start:.2f}s",flush=True)
    hashes={f"{p}-{n}":sha256((OUT/f"{p}-{n}"/"choices.json").read_bytes()).hexdigest()
            for p,n in run.data_jobs()}
    (OUT/"lock.json").write_text(json.dumps(hashes,indent=2),encoding="utf8")


def score():
    lock=json.loads((OUT/"lock.json").read_bytes())
    rows=[]
    for panel,name in run.data_jobs():
        folder=OUT/f"{panel}-{name}"
        raw=(folder/"choices.json").read_bytes()
        assert sha256(raw).hexdigest()==lock[folder.name]
        data=json.loads(raw)
        assert sha256((folder/"forecasts.npz").read_bytes()).hexdigest()==data["forecast"]
        _,_,task,idx=run.splits(panel,name)
        y=run.true_y(panel,name,idx)
        with np.load(folder/"forecasts.npz") as forecasts:
            for rec in data["rows"]:
                m=rec["method"]
                vals=run.metrics(y,forecasts[m],task,rec["scale"])
                rows.append({k:v for k,v in rec.items() if k not in ("acceptance","cols","rejected")}
                            | dict(n_rejected=len(rec["rejected"])) | vals)
    pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=["collect","score"])
    with threadpool_limits(1):
        (collect if cli.parse_args().phase=="collect" else score)()
