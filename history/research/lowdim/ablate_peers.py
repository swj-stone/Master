"""Hold the row sketch and budget fixed, and remove the one-peer gain layer."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.feature_game_eval import panels as bench
from research.lowdim import formal_study as formal
from research.lowdim.method import fast_support, MAX_FEATURES

RANDOM_DRAWS = 32


def top(scores,k):
    return np.sort(np.argsort(-scores,kind="stable")[:k])


def check_saved(frame, core, paper, out):
    """Recompute scores of the 32 saved predictions and check sampled supports."""
    verified=0
    exhaustive=0
    for case in frame.itertuples(index=False):
        key=f"{case.cohort}-{case.dataset}-{case.fold}-{case.seed}.npz"
        src=(paper/f"{case.dataset}-{case.seed}.npz" if case.cohort=="formal" else core/key)
        with np.load(src) as original, np.load(out/key) as evidence:
            y=original["truth"]
            units=original.get("units")
            draws=evidence["random32_cols"]
            predictions=evidence["random32_pred"]
            decisions=evidence.get("random32_decision")
            values=evidence["random32_scores"]
            np.testing.assert_array_equal(evidence["new_cols"],original["new_cols"])
            assert draws.shape==(RANDOM_DRAWS,case.k) and np.all(np.diff(draws,axis=1)>0)
            assert draws.min()>=0 and draws.max()<case.p and len(values)==len(predictions)==RANDOM_DRAWS
            for i,pred in enumerate(predictions):
                decision=None if decisions is None else decisions[i]
                if case.cohort=="formal":
                    observed=formal.metrics(y,pred,decision)[case.metric]
                else:
                    observed=bench.deployment.score_predictions(case.dataset,y,pred,units,case.task,decision)[0]
                np.testing.assert_allclose(observed,values[i],atol=1e-12,rtol=0.)
                verified+=1
            np.testing.assert_allclose(np.quantile(values,[0,.25,.5,.75,1]),
                                       [case.random32_min,case.random32_q25,case.random32_median,
                                        case.random32_q75,case.random32_max],rtol=0.,atol=1e-12)
            if case.dataset=="abalone":
                supports=evidence["exhaustive_cols"]
                assert len(supports)==45==len(np.unique(supports,axis=0))
                np.testing.assert_array_equal(supports,np.asarray(list(combinations(range(10),8))))
                for pred,value in zip(evidence["exhaustive_pred"],evidence["exhaustive_scores"],strict=True):
                    observed=bench.deployment.score_predictions(case.dataset,y,pred,units,case.task,None)[0]
                    np.testing.assert_allclose(observed,value,atol=1e-12,rtol=0.)
                    exhaustive+=1
    assert verified==len(frame)*RANDOM_DRAWS
    assert exhaustive==9*45
    facts=dict(cases=len(frame),uniform_draws_per_case=RANDOM_DRAWS,verified_random_scores=verified,
               verified_exhaustive_abalone_scores=exhaustive)
    (out/"verification.json").write_text(json.dumps(facts,indent=2),encoding="utf-8")


def one_case(folder, reference, X, y, task, cohort, dataset, fold, seed, fit, score, metric):
    with np.load(reference) as baseline:
        new, info = fast_support(X,y,task=task,seed=seed)
        np.testing.assert_array_equal(new,baseline["new_cols"])
        k=len(new)
        empty,single=info["tier_scores"]
        np.testing.assert_array_equal(new,top(np.minimum(empty,single),k))
        empty_only=top(empty,k)
        new_score=score(baseline["new_pred"],baseline.get("new_decision"))
        pred, decision=fit(empty_only)
        empty_score=score(pred,decision)
        if np.array_equal(new,empty_only):
            np.testing.assert_array_equal(pred,baseline["new_pred"])
        record=dict(cohort=cohort,dataset=dataset,fold=fold,seed=seed,task=task,metric=metric,
                    rows=len(y),p=X.shape[1],k=k,score_new=new_score,
                    score_empty=empty_score,gap_empty=new_score-empty_score)
        payload=dict(empty_cols=empty_only,new_cols=new,empty_pred=pred,
                     zero_peer_votes=empty,mean_single_peer_votes=single)
        if decision is not None:
            payload["empty_decision"]=decision
        for name in ("empty","F","random"):
            cols=empty_only if name=="empty" else baseline[name+"_cols"]
            amount=len(np.setdiff1d(cols,new))
            record["replaced_"+name]=amount
            record["jaccard_"+name]=(k-amount)/(k+amount)
            if name!="empty":
                value=score(baseline[name+"_pred"],baseline.get(name+"_decision"))
                record["score_"+name]=value
                record["gap_"+name]=new_score-value
        dataset_id=int.from_bytes(hashlib.sha256(f"{cohort}/{dataset}".encode()).digest()[:4],"little")
        subsets, predictions, decisions, values=[],[],[],[]
        for draw in range(RANDOM_DRAWS):
            rng=np.random.default_rng([311912, dataset_id, fold, seed, draw])
            cols=np.sort(rng.permutation(X.shape[1])[:k])
            pred,decision=fit(cols)
            subsets.append(cols)
            predictions.append(pred)
            if decision is not None:
                decisions.append(decision)
            values.append(score(pred,decision))
        payload["random32_cols"]=np.stack(subsets)
        payload["random32_pred"]=np.stack(predictions)
        if decisions:
            payload["random32_decision"]=np.stack(decisions)
        payload["random32_scores"]=np.asarray(values)
        for quantile,name in ((0.,"min"),(.25,"q25"),(.5,"median"),(.75,"q75"),(1.,"max")):
            record["random32_"+name]=np.quantile(values,quantile)
        record["random32_beaten"]=np.sum(new_score>np.asarray(values)+1e-12)
        record["random32_tied"]=np.sum(np.abs(np.asarray(values)-new_score)<=1e-12)
        record["gap_random32_median"]=new_score-record["random32_median"]
        if dataset=="abalone":
            assert (X.shape[1],k)==(10,8)
            all_columns=np.asarray(list(combinations(range(10),8)),dtype=np.int64)
            all_pred=np.asarray([fit(cols)[0] for cols in all_columns])
            all_scores=np.asarray([score(pred,None) for pred in all_pred])
            match=np.flatnonzero(np.all(all_columns==new,axis=1))
            assert len(match)==1
            np.testing.assert_array_equal(all_pred[match[0]],baseline["new_pred"])
            record["exact_n"]=len(all_scores)
            record["exact_better"]=int((all_scores>new_score+1e-12).sum())
            record["exact_tied"]=int((np.abs(all_scores-new_score)<=1e-12).sum())
            record["exact_worse"]=int((all_scores<new_score-1e-12).sum())
            record["exact_median"]=np.median(all_scores)
            record["exact_best"]=all_scores.max()
            for name,array in (("cols",all_columns),("pred",all_pred),("scores",all_scores)):
                payload["exhaustive_"+name]=array
        np.savez_compressed(folder/f"{cohort}-{dataset}-{fold}-{seed}.npz",**payload)
        return record


def run(core, paper, out):
    out.mkdir(parents=True,exist_ok=False)
    outputs=[]
    for cohort,panel,truth,units,_ in bench.panels():
        if panel.X.shape[1]>MAX_FEATURES:
            continue
        key=f"{cohort}-{panel.name}-{panel.fold}-{panel.seed}.npz"
        def fit(cols):
            return bench.fit(panel,cols)
        def score(pred,decision):
            return bench.deployment.score_predictions(panel.name,truth,pred,units,panel.task,decision)[0]
        outputs.append(one_case(out,core/key,panel.X,panel.y,panel.task,cohort,panel.name,
                                panel.fold,panel.seed,fit,score,"BA" if panel.task=="classification" else "R2"))
    for name in ("breast_cancer","diabetes"):
        raw,target,task,_,order,_,_=formal.load_dataset(name)
        for fold,seed in enumerate(formal.P["formal_seeds"],1):
            X,y,Xt,truth,*_=formal.inputs(name,seed,raw,target,order,task)
            metric="auc" if task=="classification" else "r2"
            def fit(cols):
                return formal.train("ridge",X[:,cols],y,Xt[:,cols],task,seed)
            def score(pred,decision):
                return formal.metrics(truth,pred,decision)[metric]
            outputs.append(one_case(out,paper/f"{name}-{seed}.npz",X,y,task,"formal",name,fold,
                                    seed,fit,score,metric))
    frame=pd.DataFrame(outputs)
    assert frame.groupby("cohort").size().to_dict()=={"original":12,"extra":9,"formal":10}
    frame.to_csv(out/"cases.csv",index=False)
    abalone=frame.loc[frame.dataset=="abalone",["cohort","dataset","fold","seed","p","k","score_new",
                "exact_n","exact_better","exact_tied","exact_worse","exact_median","exact_best"]].copy()
    for col in ("exact_n","exact_better","exact_tied","exact_worse"):
        abalone[col]=abalone[col].astype(int)
    abalone["rank_min"]=1+abalone.exact_better
    abalone["rank_max"]=abalone.exact_better+abalone.exact_tied
    abalone["gap_to_best"]=abalone.score_new-abalone.exact_best
    abalone.to_csv(out/"abalone-all-subsets.csv",index=False)
    agg=dict(n=("k","size"),changed=("replaced_empty",lambda x:int((x>0).sum())),
             mean_gap=("gap_empty","mean"),min_gap=("gap_empty","min"),max_gap=("gap_empty","max"),
             wins=("gap_empty",lambda x:int((x>1e-12).sum())),losses=("gap_empty",lambda x:int((x < -1e-12).sum())),
             different_from_F=("replaced_F",lambda x:int((x>0).sum())),
             different_from_random=("replaced_random",lambda x:int((x>0).sum())),
             above_random_median=("gap_random32_median",lambda x:int((x>1e-12).sum())),
             equal_random_median=("gap_random32_median",lambda x:int((np.abs(x)<=1e-12).sum())),
             mean_gap_random_median=("gap_random32_median","mean"))
    frame.groupby(["cohort","dataset","task","metric"]).agg(**agg).to_csv(out/"sources.csv")
    check_saved(frame,core,paper,out)
    print(frame.groupby(["cohort","dataset"]).agg(**agg).to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("core","formal","out"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.core,args.formal,args.out)
