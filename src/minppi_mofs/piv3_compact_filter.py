"""A fit-free compact feature filter based on coalition-conditioned hit/miss triplet votes."""
from math import ceil, sqrt

import numpy as np
from scipy.special import expit

from minppi_mofs.piv2 import _folds


def _scaled(x, center, scale):
    out = np.asarray(x, dtype=np.float32).copy()
    np.subtract(out, center, out=out)
    np.divide(out, scale, out=out)
    np.nan_to_num(out, copy=False, posinf=0., neginf=0.)
    return np.clip(out, -4., 4., out=out)


def _reference_geometry(x, groups):
    """Supervised diagonal distance on the reference fold: Fisher relevance plus uniform weight."""
    p = x.shape[1]
    overall = x.mean(axis=0)
    var = x.var(axis=0)
    signal = np.zeros(p, dtype=np.float32)
    for g in np.unique(groups):
        rows = groups == g
        signal += rows.mean() * (x[rows].mean(axis=0) - overall)**2
    score = np.divide(signal,var,out=np.zeros_like(signal),where=var>0)
    score = np.maximum(0., score - (len(np.unique(groups))-1)/max(1,len(x)-1))
    return .2/p + .8*score/score.sum() if score.sum() > 0 else np.full(p, 1/p, np.float32)


def _triplets(ref, query, ref_group, group, weights):
    """Compare a nearby hit with a nearby miss, at two neighbor ranks, for each query."""
    # The query and reference partitions are disjoint. Arithmetic is quadratic only
    # in the bounded row sketch, and linear in the input width.
    d = ((ref**2) @ weights)[None,:] - 2.*((query*weights) @ ref.T)
    hits, misses = [], []
    for i,g in enumerate(group):
        same = ref_group == g
        different = ref_group != g
        if not same.any() or not different.any():
            return None
        for mask, dest in ((same,hits),(different,misses)):
            idx = np.flatnonzero(mask)
            depth = min(4,len(idx))
            near = idx[np.argpartition(d[i,idx],depth-1)[:depth]]
            near = near[np.lexsort((near,d[i,near]))]
            dest.extend((near[0],near[-1]))
    anchors = np.repeat(np.arange(len(query)),2)
    return (np.abs(query[anchors]-ref[misses]) - np.abs(query[anchors]-ref[hits])).astype(np.float32)


def _votes(delta, rng, aggregate):
    """Each column votes on triplets difficult for a sampled coalition excluding that column."""
    n,p = delta.shape
    signed = np.sign(delta)
    plain = signed.mean(axis=0)
    if p == 1:
        return np.stack([plain]*3), plain, [0]*3
    tiers = [1, min(p-1,round(sqrt(p))), min(p-1,max(1,round(5*(p-1)/6)))]
    rates = np.zeros((3,p),dtype=np.float64)
    for tier,k in enumerate(tiers):
        for _ in range(4):
            order = rng.permutation(p)
            members = order[:k]
            total = delta[:,members].sum(axis=1)
            # A prefix member's coalition swaps it with the next player. Each
            # coordinate thereby has a uniform k-subset of all the other coordinates.
            inside = total[:,None] + delta[:,order[k],None] - delta[:,members]
            # Lower half of margins means the coalition finds that triplet difficult.
            hard_out = total <= np.median(total)
            hard_in = inside <= np.median(inside,axis=0)
            v = signed[hard_out].mean(axis=0)
            v[members] = (signed[:,members]*hard_in).sum(axis=0)/hard_in.sum(axis=0)
            rates[tier] += v
    rates /= 4
    score = rates.min(axis=0) if aggregate == "min" else rates.mean(axis=0)
    return rates, plain if aggregate == "relief" else score, tiers


def _diverse_order(delta, score, classes):
    """A short greedy reranking diversifies the head; the entire tail stays eligible."""
    p = len(score)
    base = np.argsort(-score, kind="stable")
    window = min(p, max(32, ceil(2*sqrt(p)), 4*classes))
    pool = base[:window]
    margins = delta[:,pool]
    pref = np.sign(margins)
    centered = margins-margins.mean(axis=0)
    length = np.linalg.norm(centered,axis=0)
    centered /= np.where(length>0,length,1.)
    correlation = np.maximum(0.,centered.T@centered)
    redundancy = np.zeros(window)
    uncovered = np.ones(len(margins), dtype=np.float32)
    running = np.zeros(len(margins), dtype=np.float32)
    alive = np.ones(window,dtype=bool)
    path = []
    for step in range(window):
        conditional = (uncovered @ pref)/uncovered.sum()
        priority = (score[pool]+conditional)/2 - .25*redundancy
        best = int(np.argmax(np.where(alive,priority,-np.inf)))
        alive[best] = False
        path.append(pool[best])
        np.maximum(redundancy,correlation[best],out=redundancy)
        running += margins[:,best]
        # A triplet already well separated needs less subsequent coverage.
        scale = max(float(np.std(running)),1e-6)
        uncovered = expit(-running/scale)
    return np.concatenate([np.asarray(path),base[window:]])


def _size(delta, order, groups):
    """Smallest prefix close to the best independently measured triplet coverage."""
    # Keep one byte per query/prefix. Two triplets per query and two units per
    # win mean the paired contribution to the underlying accuracy is code/4.
    n, p = len(groups),len(order)
    coverage = np.empty((n,p),dtype=np.uint8)
    carry = np.zeros(len(delta),dtype=np.float32)
    for j in range(0,p,128):
        batch = np.cumsum(delta[:,order[j:j+128]],axis=1) + carry[:,None]
        wins = (batch>1e-6).astype(np.uint8) + (batch>=-1e-6)
        coverage[:,j:j+batch.shape[1]] = wins.reshape(n,2,-1).sum(axis=1)
        carry = batch[:,-1]
    avg = coverage.mean(axis=0)/4
    best = int(np.argmax(avg))
    feasible = np.ones(p,dtype=bool)
    # Within two percentage points plus one paired standard error, overall and
    # in each observed class/response band, relative to the same best prefix.
    for rows in [np.ones(n,dtype=bool),*(groups==v for v in np.unique(groups))]:
        a = coverage[rows].astype(np.float32)
        loss = (a[:,best,None]-a)/4
        se = loss.std(axis=0,ddof=1)/sqrt(len(a)) if len(a)>1 else 0.
        feasible &= loss.mean(axis=0) <= .02 + se
    k = int(np.flatnonzero(feasible)[0])+1
    return k,dict(best_coverage_k=best+1, best_coverage=float(avg[best]),
                  chosen_coverage=float(avg[k-1]),full_coverage=float(avg[-1]))


def select(train, yt, ranking, yr, *, task, utility, seed=42, aggregate="min"):
    """Learn an ordered support on training rows; pick its prefix on disjoint ranking rows.

    Inputs are named numeric DataFrames with the same schema. The supplied ranking
    partition is reserved for surrogate sizing; the downstream validation gate and
    test partition must be separate. No predictor is trained or queried.
    """
    if (not train.columns.is_unique or not train.columns.equals(ranking.columns)
            or not len(train.columns) or not len(train) or not len(ranking)):
        raise ValueError("Expected nonempty aligned numeric frames")
    if aggregate not in ("min","mean","relief") or task not in ("binary","multiclass","regression"):
        raise ValueError("Invalid task or vote aggregation")
    y, y_rank = np.asarray(yt),np.asarray(yr)
    if y.shape!=(len(train),) or y_rank.shape!=(len(ranking),) or not np.isfinite(y).all() or not np.isfinite(y_rank).all():
        raise ValueError("Expected finite aligned targets")
    names = train.columns.tolist()
    diag = dict(aggregate=aggregate,front_fits=0,order=names.copy(),task=task,utility=utility)
    rows_seed, rank_seed, game_seed = np.random.SeedSequence(seed).spawn(3)
    folds = _folds(y,task,np.random.default_rng(rows_seed))
    validation = _folds(y_rank,task,np.random.default_rng(rank_seed))
    if folds is None or validation is None or len(folds[0])<4 or len(validation[0])<4:
        return names,dict(diag,bypass="insufficient rows")
    reference_rows, query_rows = folds
    check_rows = validation[0]
    y_ref, y_query, y_check = y[reference_rows], y[query_rows], y_rank[check_rows]
    if task=="regression":
        cuts=np.unique(np.quantile(y_ref,[.2,.4,.6,.8]))
        y_ref,y_query,y_check=(np.searchsorted(cuts,v,side="left") for v in (y_ref,y_query,y_check))
    raw = train.iloc[reference_rows].to_numpy(dtype=np.float32)
    raw = np.where(np.isfinite(raw),raw,np.nan)
    finite = np.isfinite(raw)
    center = np.nansum(raw,axis=0)/np.maximum(1,finite.sum(axis=0))
    centered = np.where(finite,raw-center,0.)
    scale = np.sqrt((centered**2).sum(axis=0)/np.maximum(1,finite.sum(axis=0)))
    scale = np.where(scale>1e-7,scale,1.).astype(np.float32)
    ref = _scaled(raw, center, scale)
    query = _scaled(train.iloc[query_rows].to_numpy(),center,scale)
    check = _scaled(ranking.iloc[check_rows].to_numpy(),center,scale)
    weights = _reference_geometry(ref,y_ref)
    train_delta = _triplets(ref,query,y_ref,y_query,weights)
    valid_delta = _triplets(ref,check,y_ref,y_check,weights)
    if train_delta is None or valid_delta is None:
        return names,dict(diag,bypass="no hit or miss in reference")
    rates, scores, tiers = _votes(train_delta,np.random.default_rng(game_seed),aggregate)
    order = _diverse_order(train_delta,scores,len(np.unique(y_ref)))
    k, info = _size(valid_delta,order,y_check)
    diag.update(info,order=[names[j] for j in order],tier_sizes=tiers,tier_rates=rates.tolist(),aggregate_scores=scores.tolist(),
                reference_rows=len(ref),training_queries=len(query),ranking_queries=len(check),k=k,
                search_window=min(len(names),max(32,ceil(2*sqrt(len(names))),4*len(np.unique(y_ref)))))
    keep=set(order[:k])
    return [col for j,col in enumerate(names) if j in keep],diag

