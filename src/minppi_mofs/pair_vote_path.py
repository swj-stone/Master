"""Greedy block removal with exact coalition minima for additive squared loss."""
import numpy as np

from .additive_minpi import fit_additive, quadratic


def block_scores(gram, single, rule):
    """Row e describes one singleton or pair. Columns encode its contextual penalties."""
    p = len(single)
    i,j = np.triu_indices(p,k=1)
    head = np.arange(p)
    left,right = np.r_[head,i], np.r_[head,j]
    pair = left != right
    width = 1 + pair.astype(int)
    diagonal = gram.diagonal()
    if rule == "energy":
        norm = diagonal[left].copy()
        norm[pair] += diagonal[j] + 2*gram[i,j]
        return left,right,norm/width,None
    base = single[left].copy()
    base[pair] += single[j] - 2*gram[i,j]
    contextual = gram[left].copy()
    contextual[pair] += gram[j]
    row = np.arange(len(left))
    contextual[row,left] = 0.
    contextual[row,right] = 0.
    if rule == "min":
        weights = 2*np.maximum(contextual,0.)
    elif rule == "drop_min":
        # Minimizing the maximum addition benefit maximizes the minimum deletion vote.
        weights = 2*np.minimum(contextual,0.)
    elif rule == "pooled":
        weights = contextual
    elif rule in ("point","local"):
        weights = 2*contextual
    else:
        raise ValueError(rule)
    return left,right,(base-weights.sum(axis=1))/width,weights/width[:,None]


def pair_order(gram,single,rule, family="mixed"):
    """Choose the mixed singleton/pair with the lowest average marginal per column."""
    left,right,score,weights = block_scores(gram,single,rule)
    p = len(single)
    active = np.ones(p,dtype=bool)
    if rule == "local":
        # Adversary may remove one further feature before evaluating this proposed block.
        priority = np.argsort(-weights,axis=1)
        row_id = np.arange(len(left))
        ptr = np.zeros(len(left),dtype=int)
    deleted, blocks = [], []
    while len(deleted) < p:
        if rule == "local":
            stale = ~active[priority[row_id,ptr]]
            while stale.any():
                ptr[stale] += 1
                stale = ~active[priority[row_id,ptr]]
            decision = score + np.maximum(0.,weights[row_id,priority[row_id,ptr]])
        else:
            decision = score.copy()
        if family == "single":
            decision[p:] = np.inf
        elif family == "double" and len(deleted) < p-1:
            decision[:p] = np.inf
        row = int(np.argmin(decision))
        u,v = left[row],right[row]
        # The low-marginal endpoint goes first so odd widths are specified as well.
        if u != v and (score[v],v)<(score[u],u):
            u,v = v,u
        block = [int(u)] if u == v else [int(u),int(v)]
        blocks.append(block)
        deleted.extend(block)
        for j in block:
            active[j] = False
            if weights is not None:
                score += weights[:,j]
            score[(left==j)|(right==j)] = np.inf
    return np.asarray(deleted[::-1]),blocks


def rank_pair(x,y,z,truth,task,kind,rule):
    model = fit_additive(x,y,task,kind)
    center = model.parts(x).mean(axis=0)
    parts = model.parts(z)-center
    baseline = model.intercept+center.sum(axis=0)
    target = np.eye(len(baseline))[truth] if task=="classification" else truth[:,None]
    gram,single = quadratic(parts,target-baseline)
    label,rule = rule.split("_",1)
    family = {"pair":"mixed","double":"double","single":"single"}[label]
    order,blocks = pair_order(gram,single,rule,family=family)
    return order,dict(blocks=blocks)
