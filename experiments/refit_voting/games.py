"""Stratum-wise supervised loss changes for two lightweight subset predictors."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit


def strata(y, classification):
    if classification:
        return y.astype(int)
    cuts = np.unique(np.quantile(y, np.linspace(0., 1., 6)[1:-1]))
    return np.searchsorted(cuts, y)


def sample(y, cap, seed):
    rng = np.random.default_rng(seed)
    codes, counts = np.unique(y, return_counts=True)
    if len(y) <= cap:
        return np.arange(len(y)), np.ones(len(y))
    # Guarantee an observation from each available class, then sample proportionally.
    reserve = [rng.choice(np.flatnonzero(y == c)) for c in codes]
    others = np.setdiff1d(np.arange(len(y)), reserve)
    picks = np.sort(np.r_[reserve, rng.choice(others, cap-len(reserve), replace=False)])
    weights = np.ones(len(picks))
    for c, total in zip(codes, counts):
        mask = y[picks] == c
        weights[mask] = total / mask.sum()
    return picks, weights


def scaled(x, reference):
    x = np.asarray(x, float)
    median = np.nanmedian(reference, axis=0)
    lo, hi = np.nanquantile(reference,[.05,.95],axis=0)
    span = np.where(hi > lo, hi-lo, 1.)
    finite = np.where(np.isfinite(x),x,median)
    return np.nan_to_num(np.clip((finite-lo) / span,0.,1.)).astype(np.float32)


@dataclass
class Evidence:
    values: np.ndarray
    prior: np.ndarray
    labels: np.ndarray


def likelihood(x,y,weight,z,classification,scale):
    codes = strata(y,classification)
    labels = np.unique(codes)
    sizes = np.array([weight[codes==c].sum() for c in labels])
    # Each score is a log density ratio against the same unconditional bin.
    terms = np.empty((x.shape[1], len(z), len(labels)), np.float32)
    for j in range(x.shape[1]):
        cuts = np.unique(np.quantile(x[:,j], np.arange(1,8)/8))
        a, v = np.searchsorted(cuts,x[:,j]), np.searchsorted(cuts,z[:,j])
        n = len(cuts)+1
        freq = (np.bincount(a,weights=weight,minlength=n)+1)/(weight.sum()+n)
        for i,c in enumerate(labels):
            loc = codes == c
            counts = np.bincount(a[loc],weights=weight[loc],minlength=n)
            terms[j,:,i] = np.log((counts[v]+8*freq[v])/(sizes[i]+8)/freq[v])
    target = labels if classification else np.array([np.average(y[codes==c],weights=weight[codes==c]) for c in labels])/scale
    return Evidence(terms,np.log(sizes/sizes.sum()),target)


def landmarks(x,y,weight,z,classification,scale,seed):
    codes = strata(y,classification)
    chosen, _ = sample(codes,64,seed)
    labels = y[chosen]
    mass = np.empty(len(chosen))
    for c in np.unique(codes):
        loc = codes[chosen] == c
        mass[loc] = weight[codes==c].sum()/weight.sum()/loc.sum()
    # A kernel mixture over labeled observations permits different neighbors in each subset.
    distances = (z[:,None,:]-x[None,chosen,:])**2
    return Evidence(np.ascontiguousarray((-8*distances).transpose(2,0,1)),np.log(mass),
                    labels if classification else labels/scale)


@njit(cache=True)
def evaluate(base, extra, add, width, prior, targets, truth, labels, weights):
    total = 0.
    denom = np.sqrt(max(width,1))
    probs = np.empty(len(prior))
    predicted = np.zeros(max(labels,1))
    for row in range(len(truth)):
        maxlog = -np.inf
        for h in range(len(prior)):
            value = base[row,h]
            if add:
                value += extra[row,h]
            probs[h] = prior[h]+value/denom
            maxlog = max(maxlog,probs[h])
        tot = 0.
        for h in range(len(prior)):
            probs[h] = np.exp(probs[h]-maxlog)
            tot += probs[h]
        if labels:
            predicted[:] = 0.
            for h in range(len(prior)):
                predicted[int(targets[h])] += probs[h]/tot
            cost = 1.-2*predicted[int(truth[row])]
            for q in predicted:
                cost += q*q
        else:
            value = 0.
            for h in range(len(prior)):
                value += targets[h]*probs[h]/tot
            cost = (truth[row]-value)**2
        total += weights[row]*cost
    return total/weights.sum()


def evaluate_bank(e, truth, classes, weights, seed):
    p = len(e.values)
    rng = np.random.default_rng(seed)
    folds = min(p,4)
    sizes = list(range(1,folds)) if folds>1 else [0]
    scores = np.zeros((len(sizes),p))
    empty = np.zeros(e.values.shape[1:],np.float64)
    args = (e.prior,e.labels.astype(float),truth.astype(float),classes,weights)
    null = evaluate(empty,e.values[0],False,0,*args)
    only = [null-evaluate(empty,extra,True,1,*args) for extra in e.values]
    for _ in range(4):
        groups = np.array_split(rng.permutation(p),folds)
        for i, eligible in enumerate(groups):
            base = np.zeros_like(empty)
            k = 0
            for h, n in enumerate(sizes):
                if n:
                    ids = groups[(i+n)%folds]
                    base += e.values[ids].sum(axis=0,dtype=np.float64)
                    k += len(ids)
                old = evaluate(base,e.values[0],False,k,*args)
                for j in eligible:
                    scores[h,j] += old-evaluate(base,e.values[j],True,k+1,*args)
    scores /= 4.
    return {"min":scores.min(axis=0),"pooled":scores.mean(axis=0),"single":np.array(only)},scores


def ranks(x,y,z,v,classification,seed):
    out = {}
    for family in ("likelihood","landmark"):
        start = perf_counter()
        fit, weight = sample(strata(y,classification),384,seed+911)
        vote, importance = sample(strata(v,classification),192,seed+417)
        xf = scaled(x[fit],x[fit])
        zv = scaled(z[vote],x[fit])
        scale = max(float(np.std(y[fit])),1e-8) if not classification else 1.
        classes = int(np.max(y)+1) if classification else 0
        labels = v[vote]/scale
        args = (xf,y[fit],weight,zv,classification,scale)
        e = likelihood(*args) if family == "likelihood" else landmarks(*args,seed+921)
        scores, detailed = evaluate_bank(e,labels,classes,importance,seed+198)
        seconds = perf_counter()-start
        for rule, values in scores.items():
            order = np.lexsort((np.arange(len(values)),-values))
            out[family+"_"+rule] = (order, seconds, detailed)
    return out
