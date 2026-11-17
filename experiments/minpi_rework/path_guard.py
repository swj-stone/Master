"""Shared all-width ridge guard evaluated along a nested feature order."""
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from experiments.minpi_compression import benchmark as b


def prefix_increments(z, y, query, ridge=1.):
    """Return the baseline and factors of each nested-prefix prediction increment."""
    mean, intercept = z.mean(axis=0), y.mean(axis=0)
    z, y = z - mean, y - intercept
    gram = z.T @ z
    gram.flat[::gram.shape[0]+1] += ridge
    lower = cholesky(gram, lower=True, check_finite=False)
    slopes = solve_triangular(lower,z.T@y,lower=True,check_finite=False)
    directions = solve_triangular(lower,(query-mean).T,lower=True,check_finite=False).T
    return intercept, directions, slopes


def select(method, x, y, task, idx, seed, rank, *, zero_tolerance=False):
    begin = perf_counter()
    p = x.shape[1]
    if method == "full":
        order, meta, k, checked, aucs = np.arange(p), {}, p, 0, 0
        rank_s = 0.
    else:
        order, meta = rank(method,x,y,task,idx,seed)
        rank_s = perf_counter()-begin
        np.testing.assert_array_equal(np.sort(order), np.arange(p))
        pool = np.sort(np.r_[idx["fit"],idx["vote"]])
        query = idx["width"]
        prep = b.transform(x,pool)
        z = prep.transform(x[pool])[:,order]
        h = prep.transform(x[query])[:,order]
        labels = y[pool]
        target = b.targets(labels,task,len(np.unique(labels)))
        scale = max(1e-12,np.std(labels))
        intercept, direction, slopes = prefix_increments(z,target,h)
        baseline = b.metrics(y[query],intercept + direction @ slopes,task,scale)
        prediction = np.broadcast_to(intercept,(len(query),*np.shape(intercept))).copy()
        checked, aucs, k = 0, 0, p
        for j in range(p-1):
            if task == "classification":
                prediction += direction[:,j,None] * slopes[j]
                checked += 1
                if np.mean(prediction.argmax(axis=1) == y[query]) < baseline["accuracy"] - (0. if zero_tolerance else .01):
                    continue
                aucs += 1
            else:
                prediction += direction[:,j] * slopes[j]
                checked += 1
            score = b.metrics(y[query],prediction,task,scale)
            if zero_tolerance:
                accepts = (score["accuracy"] >= baseline["accuracy"] and score["auc"] >= baseline["auc"]
                           if task == "classification" else
                           score["r2"] >= baseline["r2"] and score["scaled_mae"] <= baseline["scaled_mae"])
            else:
                accepts = b.eligible(score,baseline,task)
            if accepts:
                k = j+1
                break
    selection_s = perf_counter()-begin
    dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
    output = b.predict(x,y,task,dev,idx["test"],order,[k])[k]
    return dict(method=method,p=p,k=k,order=order.tolist(),checked=checked,
                auc_calls=aucs,rank_s=rank_s,selection_s=selection_s,
                deployment_s=perf_counter()-begin, **meta),output
