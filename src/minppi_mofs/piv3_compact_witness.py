"""Local deletion votes across three contexts of the same frozen fitted predictor.

A vote is cast on predictive quality on a weighted ranking-only sample. The
minimum is a veto by the weakest observed coalition size. These witnesses are
about restricted predictions of the existing model, not retrained model utility.
"""
from collections import deque

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from minppi_mofs.piv3_compact_tree import select as gain_anchor


ROW_LIMIT = 256
BOOTSTRAPS = 3
MAX_BLOCKS = 5


def _sample(y, task, seed):
    """Enrich rare classes for precision, then restore ranking-pool class mass."""
    rng = np.random.default_rng(seed)
    if task == "regression":
        rows = np.sort(rng.choice(len(y), min(ROW_LIMIT,len(y)), replace=False))
        groups = [np.arange(len(rows))]
        masses = np.array([1.])
    else:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) > ROW_LIMIT:
            raise ValueError("More ranking classes than the voting row budget")
        budget = min(ROW_LIMIT, len(y))
        quotas = np.zeros(len(classes), dtype=int)
        for i,g in enumerate(np.argsort(counts,kind="stable")):
            quotas[g] = min(counts[g],budget//(len(classes)-i))
            budget -= quotas[g]
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y==g), quotas[j], replace=False)
                                        for j,g in enumerate(classes)]))
        groups = [np.flatnonzero(y[rows]==g) for g in classes]
        masses = counts/len(y)
    weights = np.zeros((1+BOOTSTRAPS,len(rows)))
    for ix,mass in zip(groups,masses,strict=True):
        weights[0,ix] = mass/len(ix)
        for draw in range(1,len(weights)):
            # A paired bootstrap: exactly these draws are reused for every mask.
            resampled = rng.integers(0,len(ix),size=len(ix))
            weights[draw,ix] = mass*np.bincount(resampled,minlength=len(ix))/len(ix)
    return rows, weights


def _measure(y, pred, weights, task, rare):
    if task == "regression":
        avg = np.average(y,weights=weights)
        residual = np.average((y-pred)**2,weights=weights)
        total = np.average((y-avg)**2,weights=weights)
        score = max(0.,1.-residual/total) if total else float(residual==0)
        return dict(r2=float(score), mae=float(np.average(np.abs(y-pred),weights=weights)))
    scores = dict(accuracy=float(np.average(pred.argmax(axis=1)==y,weights=weights)))
    if task == "binary":
        scores["auc"] = float(roc_auc_score(y,pred[:,1],sample_weight=weights))
        if rare:
            scores["ap"] = float(average_precision_score(y,pred[:,1],sample_weight=weights))
    else:
        # Macro one-vs-rest AUC, allowing the same stratification weights as accuracy.
        scores["auc"] = float(np.mean([roc_auc_score(y==i,pred[:,i],sample_weight=weights)
                                        for i in range(pred.shape[1])]))
    return scores


def _within(after,before):
    for metric,reference in before.items():
        loss = after[metric]-reference if metric=="mae" else reference-after[metric]
        tol = .05 if metric=="ap" else .03
        if loss > tol*abs(reference) + 1e-12:
            return False
    return True


def _decision(before,after,original,aggregate):
    votes = [[1 if _within(a,b) else -1 for a,b in zip(ar,br,strict=True)]
             for ar,br in zip(after,before,strict=True)]
    nets = np.mean(votes,axis=1)
    score = float(nets.min() if aggregate=="min" else nets.mean() if aggregate=="mean" else nets[0])
    anchor_pass = _within(after[0][0],original)
    return score > 0 and anchor_pass, dict(votes=votes,net=nets.tolist(),score=score,
                                          cumulative_anchor_pass=anchor_pass)


def select(model,train,yt,ranking,yr,*,task,utility,seed=42,aggregate="min"):
    """Take a Gain anchor, offer weak-tail blocks for deletion; at most five proposals."""
    if aggregate not in {"min","mean","small"}:
        raise ValueError("Unknown context aggregation")
    anchor,diag = gain_anchor(model,train,yt,ranking,yr,task=task,utility=utility,seed=seed,aggregate="gain")
    order = diag["order"]
    anchor_k = len(anchor)
    live = frozenset(order[:anchor_k])
    if set(anchor) != live:
        raise ValueError("Compact Gain anchor is not the declared prefix")
    initial = live
    diag.update(aggregate=aggregate,anchor=anchor,anchor_k=anchor_k,anchor_order=order,
                witnesses=[],witness_predictions=0,front_fits=0)
    if anchor_k<2:
        return anchor,diag
    yr = np.asarray(yr)
    rows,weights = _sample(yr,task,np.random.SeedSequence([seed,659]))
    labels = yr[rows]
    rare = task=="binary" and min(np.unique(yr,return_counts=True)[1]) <= .05*len(yr)
    x = np.ascontiguousarray(ranking.iloc[rows].to_numpy(),dtype=np.float32)
    rng = np.random.default_rng(seed)
    bg_rows = rng.choice(len(train),min(512,len(train)),replace=False)
    fill = np.nanmedian(train.iloc[bg_rows].to_numpy(dtype=np.float32),axis=0)
    masked = np.broadcast_to(fill,x.shape).copy()
    ix = {name:i for i,name in enumerate(train.columns)}
    outer = frozenset(order[anchor_k:])
    medium = frozenset(order[anchor_k:2*anchor_k])
    readout = model.predict if task=="regression" else model.predict_proba
    scores = {}

    def assess(mask):
        if mask not in scores:
            indices = sorted(ix[name] for name in mask)
            masked[:]=fill
            masked[:,indices]=x[:,indices]
            p=readout(masked)
            scores[mask]=[_measure(labels,p,w,task,rare) for w in weights]
        return scores[mask]

    original = assess(initial)[0]
    step=max(1,anchor_k//3)
    # Test each block from weakest toward strongest, holding the leading anchor feature.
    weak=list(reversed(order[1:anchor_k]))
    pending=deque(weak[i:i+step] for i in range(0,len(weak),step))
    witnesses=[]
    while pending and len(witnesses)<MAX_BLOCKS:
        block=pending.popleft()
        removal=frozenset(block)
        contexts=(live,live|medium,live|outer)
        before=[assess(s) for s in contexts]
        after=[assess(s-removal) for s in contexts]
        allow,log=_decision(before,after,original,aggregate)
        witnesses.append(dict(block=block,accepted=allow,context_widths=[len(s) for s in contexts],
                              before_point=[v[0] for v in before],after_point=[v[0] for v in after],**log))
        if allow:
            live=live-removal
        elif len(block)>1:
            half=(len(block)+1)//2
            pending.extendleft((block[half:],block[:half]))
    selected=[c for c in train.columns if c in live]
    diag.update(k=len(selected),order=[c for c in order if c in live]+[c for c in order if c not in live],
                witnesses=witnesses,witness_predictions=len(scores),witness_rows=rows.tolist(),
                ranking_class_prevalence_weighted=(task!="regression"),anchor_sample_scores=original,
                bootstraps=BOOTSTRAPS,quota_relative=.03,quota_ap_relative=.05,max_block_proposals=MAX_BLOCKS)
    return selected,diag

