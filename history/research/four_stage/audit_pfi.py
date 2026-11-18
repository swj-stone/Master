"""Audit frozen-pilot permutation importance by an independent linear loss identity.

For column j and permutation pi, the prediction increment in all response
coordinates is d=(X[pi,j]-X[:,j])*B[j,:]. Each importance is the mean of
2*(prediction-y)*d + d*d over rows and outputs. No permuted matrix refit
or prediction is required to evaluate this identity.
"""
import argparse
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from research.algorithms.coalition_residual import _pilot, _sample, _standardize
from research.four_stage import benchmark, pfi


def increment_scores(model, X, targets, seed):
    n,p = X.shape
    prediction = model.predict(X).reshape(n,-1).astype(np.float64)
    target = targets.reshape(n,-1).astype(np.float64)
    coef = np.atleast_2d(model.coef_).T.astype(np.float64)
    residual_projection = (prediction-target) @ coef.T
    coef_norm = np.square(coef).sum(axis=1)
    eval_x = np.asarray(X,dtype=np.float64)
    # sklearn 1.9 draws a single job seed, passes it unchanged to each
    # column, and shuffles already-permuted values on subsequent repeats.
    job_seed = np.random.RandomState(seed).randint(np.iinfo(np.int32).max+1)
    rng = np.random.RandomState(job_seed)
    indices = np.arange(n)
    source = indices.copy()
    scores = []
    for _ in range(3):
        rng.shuffle(indices)
        source = source[indices]
        diff = eval_x[source] - eval_x
        twice_linear = 2*np.einsum("ij,ij->j",diff,residual_projection)
        quadratic = coef_norm*np.einsum("ij,ij->j",diff,diff)
        scores.append((twice_linear+quadratic)/target.size)
    return np.stack(scores,axis=1)


def sketch_check(X, y, task, seed, model, held_X, held_y):
    nfit=min(128,max(2,len(y)//3))
    subset=np.random.default_rng(seed).choice(len(y),min(len(y),nfit+512),replace=False)
    train,pool=subset[:nfit],subset[nfit:]
    labels=(np.eye(len(np.unique(y)))[np.unique(y,return_inverse=True)[1]]
            if task=="classification" else y.astype(float)[:,None])
    _,center,scale=_standardize(X[train])
    Z,_,_=_standardize(X[pool],center,scale)
    official=model.predict(Z).reshape(len(pool),-1)
    if official.shape[1]==1:
        priority=official.ravel()
    else:
        sorted_response=np.sort(official,axis=1)
        priority=sorted_response[:,-1]-sorted_response[:,-2]
    count=min(256,len(pool))
    pos=((np.arange(count)+.5)*len(pool)/count).astype(int)
    ids=np.argsort(priority,kind="stable")[pos]
    np.testing.assert_array_equal(held_X,Z[ids])
    np.testing.assert_array_equal(held_y,labels[pool[ids]])
    mf,mh,_=_sample(X,y,task,seed)
    np.testing.assert_array_equal(train,mf)
    minpi_pred=_pilot(X[train],labels[train],X[pool])
    return dict(held_rows=len(ids),pilot_fit_same=True,held_set_diff=int(np.setxor1d(pool[ids],mh).size),
                held_order_mismatches=int(np.count_nonzero(pool[ids]!=mh)),
                pilot_max_pred_diff=float(np.max(abs(official-minpi_pred))),
                x_dtype=str(held_X.dtype))


def case(X,y,task,seed,arr):
    tr=arr["train"]
    X,y=X[tr],y[tr]
    # Intercept the actual frozen-pilot call to get the pilot, evaluation
    # sample, and raw software result; use the unmodified sklearn scorer.
    captured={}
    def recorded(model,held_X,held_y,**kwargs):
        assert kwargs==dict(scoring="neg_mean_squared_error",n_repeats=3,random_state=seed,n_jobs=1)
        result=permutation_importance(model,held_X,held_y,**kwargs)
        captured.update(model=model,X=held_X,y=held_y,result=result)
        return result
    with patch.object(pfi,"permutation_importance",recorded):
        rerun = pfi.pfi_top(X,y,task,seed,len(arr["PFI80"]))
    np.testing.assert_array_equal(rerun,arr["PFI80"])
    obs=captured
    official=obs["result"].importances
    scores=increment_scores(obs["model"],obs["X"],obs["y"],seed)
    loss_scale=max(np.finfo(float).eps,float(np.max(abs(official))))
    error=float(np.max(abs(scores-official)))
    row_data=sketch_check(X,y,task,seed,obs["model"],obs["X"],obs["y"])
    output=[]
    for arm in ["PFI80","PFI_match"]:
        archived=arr[arm]
        k=len(archived)
        official_order=np.argsort(-official.mean(axis=1),kind="stable")
        np.testing.assert_array_equal(archived,np.sort(official_order[:k]))
        order=np.argsort(-scores.mean(axis=1),kind="stable")
        cols=np.sort(order[:k])
        replaced=np.setdiff1d(archived,cols)
        inserted=np.setdiff1d(cols,archived)
        band=(np.max(scores.mean(axis=1)[inserted])-np.min(scores.mean(axis=1)[replaced])) if len(replaced) else 0.
        cutoff=(official.mean(axis=1)[official_order[k-1]]-official.mean(axis=1)[official_order[k]]) if k<X.shape[1] else np.nan
        output.append(dict(arm=arm,width=k,p=X.shape[1],exact_support=np.array_equal(archived,cols),
                           replaced=len(replaced),cutoff_gap=cutoff,max_boundary_inversion=band,
                           maximum_absolute_error=error,maximum_error_relative_to_score_range=error/loss_scale,
                           official_replay_exact=True,**row_data))
    return output


def run(folder,output):
    reports=[]
    for name in benchmark.DATASETS:
        X,y,task,_=benchmark.load(name)
        for seed in benchmark.SEEDS:
            with np.load(folder/f"{name}-{seed}.npz") as data:
                reports.extend(dict(source=name,seed=seed,**row) for row in case(X,y,task,seed,data))
    with output.open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print("official replays:",sum(row["official_replay_exact"] for row in reports),"/",len(reports))
    print("linear identity exact supports:",sum(row["exact_support"] for row in reports),"/",len(reports))
    print("max absolute per-column per-permutation error:",max(row["maximum_absolute_error"] for row in reports))
    print("max relative error vs largest absolute importance:",max(row["maximum_error_relative_to_score_range"] for row in reports))
    print("pilot held sets with different membership:",sum(row["held_set_diff"]>0 for row in reports[::2]),"/30")
    print("max held set symmetric difference:",max(row["held_set_diff"] for row in reports))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.archive,args.output)
