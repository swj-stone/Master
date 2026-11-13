"""Shadow voting followed by witnessed pair substitution on training rows."""
from math import sqrt

import numpy as np
from scipy.spatial import cKDTree

from minppi_mofs.piv2 import _folds
from minppi_mofs.piv3_lite import BYPASS_WIDTH, DRAWS, _signals, screen as shadow_screen


CORRELATION = .90
PROJECTIONS = 24
NEIGHBORS = 8


def _unit(frame):
    """Unit-norm centered columns. Incomplete or constant columns cannot be witnesses."""
    valid = np.isfinite(frame).all(axis=0)
    centered = frame.copy()
    centered[:, ~valid] = 0.
    centered -= centered.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    valid &= norm > 0
    centered /= np.where(valid, norm, 1.)
    return centered.T, valid


def _neighbors(propose, checks, rng):
    sketch, ok = _unit(propose)
    signed = [_unit(x) for x in checks]
    ids = np.flatnonzero(ok & signed[0][1] & signed[1][1])
    if len(ids) < 2:
        return np.empty((0, 2), int), np.empty((0, 2)), 0.
    projection = rng.normal(size=(sketch.shape[1], PROJECTIONS))
    vectors = sketch[ids] @ projection
    vectors /= np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True), 1e-20)
    # Query both directions so a sign-reversed copy is eligible.
    tree = cKDTree(vectors)
    k = min(NEIGHBORS+1,len(ids))
    candidates = []
    for sign in (1, -1):
        _, near = tree.query(sign*vectors, k=k)
        candidates.append(np.column_stack((np.repeat(ids,k),ids[near.ravel()])))
    pairs = np.unique(np.sort(np.vstack(candidates), axis=1), axis=0)
    pairs = pairs[pairs[:,0] != pairs[:,1]]
    # Verify on two disjoint halves, excluding every row used for the neighbor proposal.
    cor = np.empty((len(pairs),2))
    for f,(unit, _) in enumerate(signed):
        for j in range(0,len(pairs),256):
            a,b = pairs[j:j+256].T
            cor[j:j+len(a),f] = np.clip(np.einsum("ij,ij->i",unit[a],unit[b]), -1., 1.)
    magnitude = np.abs(cor).min(axis=1)
    keep = (magnitude >= CORRELATION) & (cor[:,0]*cor[:,1] > 0)
    return pairs[keep], cor[keep], float(magnitude.max(initial=0.))


def _comparison(signals,target,pairs,order,tiers,ab):
    """Batched MSE improvement of a over b on the same k-set excluding both."""
    a,b = pairs.T
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order))
    norm = np.einsum("ij,ij->i",signals,signals,dtype=np.float64)
    outcome = np.empty((3,len(a)),dtype=np.int8)
    for t,k in enumerate(tiers):
        first, second = order[k:k+2]
        common = (k+1)*target - signals[order[:k]].sum(axis=0,dtype=np.float64)
        dots = signals @ np.vstack((common,signals[first],signals[second])).T.astype(signals.dtype)
        ina,inb = pos[a]<k, pos[b]<k
        # Add back removed endpoints and subtract the first eligible substitutes.
        spread = (dots[a,0] - dots[b,0]).astype(np.float64)
        spread += ina*(norm[a]-ab) + inb*(ab-norm[b])
        use_first = (ina | inb) & (a != first) & (b != first)
        use_second = (ina & inb) | ((ina | inb) & ~use_first)
        spread -= use_first*(dots[a,1]-dots[b,1]) + use_second*(dots[a,2]-dots[b,2])
        contrast = 2*spread - norm[a] + norm[b]
        # Scale by the terms that were subtracted to avoid spurious signs on exact ties.
        eps = 16*np.finfo(signals.dtype).eps*(np.abs(dots[a,0])+np.abs(dots[b,0])+norm[a]+norm[b])
        outcome[t] = (contrast > eps).astype(np.int8)-(contrast < -eps).astype(np.int8)
    return outcome


def _votes(frames,labels,task,edges,seed):
    width = frames[0].shape[1]
    tiers = np.clip([1,round(sqrt(width)),round((width-1)*5/6)],0,width-2).tolist()
    votes = np.zeros((2,3,len(edges)),dtype=np.int16)
    alone = np.zeros(width)
    rng = np.random.default_rng(seed)
    classes = None if task == "regression" else np.unique(np.concatenate(labels))
    for f in range(2):
        fit,check = frames[f],frames[1-f]
        # Give exact sign-reversed copies the same bin-boundary convention.
        finite = np.isfinite(fit)
        with np.errstate(invalid="ignore"):
            sums = np.where(finite,fit,0.).sum(axis=0)
            centered = np.where(finite, fit-sums/np.maximum(finite.sum(axis=0),1),0.)
        largest = centered[np.abs(centered).argmax(axis=0),np.arange(width)]
        orient = np.where(largest < 0,-1.,1.)
        s,target,_ = _signals(fit*orient,check*orient,labels[f],labels[1-f],task,classes)
        target = target.ravel()
        norm = np.einsum("ij,ij->i",s,s,dtype=np.float64)
        alone += (2*np.einsum("ij,j->i",s,target,dtype=np.float64)-norm)/target.size
        ab = np.empty(len(edges))
        batch = max(1,250000//s.shape[1])
        for i in range(0,len(edges),batch):
            a,b = edges[i:i+batch].T
            ab[i:i+len(a)] = np.einsum("ij,ij->i",s[a],s[b],dtype=np.float64)
        for _ in range(DRAWS):
            votes[f] += _comparison(s,target,edges,rng.permutation(width),tiers,ab)
    return votes/DRAWS, alone, tiers


def _merge(names,edges,corr,rates,priority,aggregate):
    """Greedy stars: a witness is always a previously retained representative."""
    neighbors = [[] for _ in names]
    for i,(a,b) in enumerate(edges):
        neighbors[a].append((i,b,-1))
        neighbors[b].append((i,a,1))
    anchors, witnesses = set(), []
    for j in priority:
        alternatives = sorted(neighbors[j],key=lambda e:(-min(abs(corr[e[0]])),e[1]))
        for i,other,direction in alternatives:
            if other not in anchors:
                continue
            fold_rates = rates[:,:,i]*direction
            tier_rate = fold_rates.mean(axis=0)
            score = tier_rate.min() if aggregate == "min" else tier_rate.mean()
            if score >= 0 and np.all(fold_rates.mean(axis=1) >= 0):
                witnesses.append(dict(removed=names[j],representative=names[other],
                    correlations=corr[i].tolist(),fold_tier_rates=fold_rates.tolist(),
                    tier_rates=tier_rate.tolist(),score=float(score)))
                break
        else:
            anchors.add(j)
    return [name for j,name in enumerate(names) if j in anchors], witnesses


def screen(train,labels,*,task,seed=42,aggregate="min"):
    """Propose only witnessed within-support substitutions; return names in input order."""
    shadow,diag = shadow_screen(train,labels,task=task,seed=seed,aggregate=aggregate)
    shadow_bypass = diag["bypass"]
    diag.update(shadow_front_k=len(shadow),shadow_bypass=shadow_bypass,peer_removed_count=0,
                verified_edges=0,peak_abs_corr=0.,peer_tier_sizes=[],witnesses=[])
    if (train.shape[1] <= BYPASS_WIDTH or len(shadow)<2 or
            shadow_bypass in {"small input","insufficient observations per class"}):
        return shadow, diag
    labels = np.asarray(labels)
    sampling,proposal,contexts = np.random.SeedSequence([seed,4711]).spawn(3)
    folds = _folds(labels,task,np.random.default_rng(sampling))
    if folds is None or min(map(len,folds)) < 16:
        return shadow, diag
    rng = np.random.default_rng(proposal)
    sketch_rows,check_rows = [],[]
    for fold in folds:
        # Preserve both classes in the check even on imbalanced problems.
        chunks = [fold] if task == "regression" else [fold[labels[fold]==c] for c in np.unique(labels[fold])]
        samples, checks = [], []
        for chunk in chunks:
            perm = rng.permutation(chunk)
            n = len(perm)//4
            samples.extend(perm[:n])
            checks.extend(perm[n:])
        sketch_rows.extend(samples)
        check_rows.append(np.asarray(checks,dtype=int))
    if len(sketch_rows)<8:
        return shadow,diag
    ids = train.columns.get_indexer(shadow)
    frames = [train.iloc[r,ids].to_numpy(dtype=np.float64) for r in check_rows]
    proposal_frame = train.iloc[sketch_rows,ids].to_numpy(dtype=np.float64)
    pairs,cor,peak = _neighbors(proposal_frame,frames,rng)
    diag.update(peak_abs_corr=peak,verified_edges=len(pairs),peer_proposal_rows=len(sketch_rows),
                peer_check_rows=list(map(len,check_rows)),peer_corr_threshold=CORRELATION)
    if not len(pairs):
        return shadow,diag
    rates,marginal,tiers = _votes(frames,[labels[r] for r in check_rows],task,pairs,contexts)
    strength = np.asarray(diag["aggregate_scores"])[ids]
    order = np.lexsort((ids,-marginal,-strength))
    keep,witnesses = _merge(shadow,pairs,cor,rates,order,aggregate)
    diag.update(peer_tier_sizes=tiers,witnesses=witnesses,peer_removed_count=len(witnesses),
                shadow_removed_count=train.shape[1]-len(shadow),removed_count=train.shape[1]-len(keep))
    if witnesses:
        diag["bypass"] = None
    return keep,diag
