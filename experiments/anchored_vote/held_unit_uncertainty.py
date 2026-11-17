"""Postrelease paired bootstrap of fixed predictions, preserving shared held units."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.anchored_vote import audit
from experiments.minpi_replication.second_panel import data


DRAWS = 1999
RNG_SEED = 20261117


def unit_weights(count, method, rng):
    block = 1
    indices = rng.integers(count, size=(DRAWS,count)) if method == "cluster" else None
    if method == "circular":
        block = math.ceil(np.cbrt(count))
        start = rng.integers(count, size=(DRAWS,math.ceil(count/block)))
        indices = ((start[:,:,None]+np.arange(block)) % count).reshape(DRAWS,-1)[:,:count]
    draws = np.zeros((DRAWS+1,count),dtype=int)
    draws[0] = 1
    np.add.at(draws[1:],(np.arange(DRAWS)[:,None],indices),1)
    np.testing.assert_array_equal(draws.sum(axis=1),np.full(DRAWS+1,count))
    return draws,block


def weighted_auc(binary,values,weights):
    """Weighted fraction of concordant positive-negative pairs, counting ties at one half."""
    order = np.argsort(values,kind="stable")
    starts = np.r_[0,1+np.flatnonzero(np.diff(values[order])!=0)]
    w = weights[:,order]
    pos = np.add.reduceat(w*binary[order],starts,axis=1)
    neg = np.add.reduceat(w*(~binary[order]),starts,axis=1)
    den = pos.sum(axis=1)*neg.sum(axis=1)
    pairs = (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)
    return np.divide(pairs,den,out=np.full(len(w),np.nan),where=den>0)


def weighted_deltas(y,a,b,w,task):
    if task == "classification":
        classes = [1] if a.shape[1] == 2 else range(a.shape[1])
        diffs = [weighted_auc(y==c,a[:,c],w)-weighted_auc(y==c,b[:,c],w) for c in classes]
        p = np.mean(diffs,axis=0)
        err = (a.argmax(axis=1)==y).astype(float) - (b.argmax(axis=1)==y)
        other = np.divide(w@err,w.sum(axis=1),out=np.full(len(w),np.nan),where=w.sum(axis=1)>0)
    else:
        # Center once to avoid a large squared mean when evaluating weighted total variance.
        z = y-y.mean()
        total = w.sum(axis=1)
        ss = w@(z*z)-(w@z)**2/total
        errors = (y-b)**2-(y-a)**2
        p = np.divide(w@errors,ss,out=np.full(len(w),np.nan),where=ss>0)
        other = w@(np.abs(y-a)-np.abs(y-b))/total
    # Check both the original sample and actual nonuniform draws against sklearn.
    for i in range(min(3,len(w))):
        if not np.isfinite(p[i]):
            continue
        if task == "classification":
            def primary(pred):
                score = pred[:,1] if a.shape[1]==2 else pred
                return roc_auc_score(y,score,sample_weight=w[i],multi_class="ovr",average="macro")
            def companion(pred):
                return accuracy_score(y,pred.argmax(axis=1),sample_weight=w[i])
        else:
            def primary(pred):
                return r2_score(y,pred,sample_weight=w[i])
            def companion(pred):
                return mean_absolute_error(y,pred,sample_weight=w[i])
        np.testing.assert_allclose([p[i],other[i]],[primary(a)-primary(b),companion(a)-companion(b)],
                                   rtol=0,atol=1e-9)
    return np.column_stack([p,other])


def main():
    # Validate incoming forecasts against all three repetitions and the released metric table.
    panel,_ = audit.load_panel()
    released=pd.read_csv(audit.OUT/"sources.csv")
    released=released[released.control=="exhaustive"].set_index("dataset")
    seeds=pd.read_csv(audit.OUT/"seeds.csv")
    rows=[]
    for name,sequence in zip(audit.SOURCES,np.random.SeedSequence(RNG_SEED).spawn(len(audit.SOURCES)),strict=True):
        test=[data.split(name,seed)[3]["test"] for seed in audit.SEEDS]
        all_ids=data.group_ids(name)
        groups=[all_ids[indices] for indices in test]
        units=np.unique(np.concatenate(groups))
        task=panel[name,42,"min"][2]
        metrics=("auc","accuracy") if task=="classification" else ("r2","mae")
        method=("circular" if name in ("tetouan","garment") else "cluster")
        if name=="shoppers":
            weights=np.ones((1,len(units)),dtype=int)
            method,block="single_held_month",None
        else:
            weights,block=unit_weights(len(units),method,np.random.default_rng(sequence))
        deltas=[]
        for seed,group in zip(audit.SEEDS,groups,strict=True):
            mapped=np.searchsorted(units,group)
            np.testing.assert_array_equal(units[mapped],group)
            truth=data.truth(name,seed)
            preds=[panel[name,seed,m][1]["primary"].astype(float) for m in ("min","exhaustive")]
            w=weights[:,mapped]
            deltas.append(weighted_deltas(truth,*preds,w,task))
        center=np.median(deltas,axis=0)
        for i,metric in enumerate(metrics):
            s=seeds[seeds.dataset==name].pivot(index="seed",columns="method",values=metric)
            np.testing.assert_allclose(center[0,i],(s["min"]-s["exhaustive"]).median(),rtol=0,atol=1e-9)
            if i==0:
                np.testing.assert_allclose(center[0,i],released.loc[name,"delta_primary"],rtol=0,atol=1e-9)
            valid=np.isfinite(center[1:,i])
            ci=np.percentile(center[1:,i][valid],[2.5,97.5]) if valid.any() else [np.nan,np.nan]
            rows.append(dict(dataset=name,metric=metric,paired_median_difference=center[0,i],
                             lower_95=ci[0],upper_95=ci[1],held_units=len(units),method=method,
                             block_length=block,draws=len(valid),valid_draws=int(valid.sum()),
                             skipped_draws=int((~valid).sum()),monte_carlo_seed=RNG_SEED))
    result=pd.DataFrame(rows)
    result.to_csv(audit.OUT/"held-unit-uncertainty.csv",index=False)
    print(result.to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
