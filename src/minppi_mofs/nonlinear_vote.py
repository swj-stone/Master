"""Construct fixed anchor/hit/miss triplets and sample their coalition votes."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class TripletRanks:
    minimum: np.ndarray
    pooled: np.ndarray
    margin: np.ndarray
    strata: np.ndarray
    n_triplets: int


def _stratified_indices(labels, limit, rng):
    """Sample each label fairly, using leftover slots when a class is exhausted."""
    classes, encoded = np.unique(labels, return_inverse=True)
    if len(classes) > limit:
        raise ValueError("Sample limit does not cover every class")
    if len(labels) <= limit:
        return np.arange(len(labels))
    buckets = [rng.permutation(np.flatnonzero(encoded == i)) for i in range(len(classes))]
    positions = np.concatenate([np.arange(len(b)) for b in buckets])
    return np.concatenate(buckets)[np.argsort(positions, kind="stable")[:limit]]


def _triplets(x_fit, y_fit, x_vote, y_vote, seed, task):
    rng = np.random.default_rng(seed)
    # Quantile bins define local high/low target agreement for the regression surrogate.
    if task == "regression":
        cuts = np.unique(np.quantile(y_fit, [.25, .5, .75]))
        y_fit, y_vote = np.searchsorted(cuts, y_fit), np.searchsorted(cuts, y_vote)
    fi = _stratified_indices(y_fit, 2048, rng)
    vi = _stratified_indices(y_vote, 256, rng)
    ref, anchors = x_fit[fi], x_vote[vi]
    median = np.nanmedian(ref, axis=0)
    median = np.nan_to_num(median)
    ref = np.where(np.isfinite(ref), ref, median)
    anchors = np.where(np.isfinite(anchors), anchors, median)
    # Use ranks in the reference distribution so rare extremes cannot dominate distances.
    for j in range(ref.shape[1]):
        col = np.sort(ref[:, j])
        def midrank(values):
            return (np.searchsorted(col, values, side="left") +
                    np.searchsorted(col, values, side="right")) / (2 * len(col))
        anchors[:, j] = midrank(anchors[:, j])
        ref[:, j] = midrank(ref[:, j])
    distances = cdist(anchors, ref, metric="cityblock")
    diffs = []
    for anchor, d, label in zip(anchors, distances, y_vote[vi]):
        hits = np.flatnonzero(y_fit[fi] == label)
        misses = np.flatnonzero(y_fit[fi] != label)
        k = min(8, len(hits), len(misses))
        if k == 0:
            continue
        hits = hits[np.argsort(d[hits], kind="stable")[:k]]
        misses = misses[np.argsort(d[misses], kind="stable")[:k]]
        diffs.append(np.abs(anchor - ref[misses]) - np.abs(anchor - ref[hits]))
    if not diffs:
        raise ValueError("At least one triplet is required")
    return np.concatenate(diffs)


def rank_triplets(x_fit, y_fit, x_vote, y_vote, *, seed, task="classification", panels=8):
    """Triplet votes use a strict margin; ties receive zero.

    Draw a feature permutation and remove each target before taking the first k.
    One draw therefore gives a uniform k-subset excluding *each* target.
    The first k columns and the (k+1)-th column suffice for all those contexts.
    """
    diffs = _triplets(x_fit, y_fit, x_vote, y_vote, seed, task)
    p = diffs.shape[1]
    sizes = np.unique(np.rint(np.array([.125, .5, .875]) * (p-1)).astype(int))
    strata = np.zeros((len(sizes), p))
    rng = np.random.default_rng(seed + 1231)
    # Evaluate sums in integer midrank units; cancellation otherwise corrupts tied votes.
    # _triplets uses one common reference empirical distribution for all its columns.
    units = 2 * min(len(x_fit), 2048)
    delta = np.rint(diffs * units).astype(np.int32)
    for _ in range(panels):
        permutation = rng.permutation(p)
        for i, size in enumerate(sizes):
            prefix = permutation[:size]
            before = np.broadcast_to(delta[:, prefix].sum(axis=1, dtype=np.int32)[:, None],
                                     delta.shape).copy()
            before[:, prefix] += delta[:, permutation[size], None] - delta[:, prefix]
            strata[i] += ((before + delta) > 0).mean(axis=0) - (before > 0).mean(axis=0)
    strata /= panels
    margin = diffs.mean(axis=0)
    def ordered(score):
        return np.lexsort((np.arange(p), -margin, -score))
    return TripletRanks(ordered(strata.min(axis=0)), ordered(strata.mean(axis=0)),
                        ordered(margin), strata, len(diffs))
