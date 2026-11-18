"""Repeat sparse cross-fitted document votes before taking the context minimum."""
from numbers import Integral

import numpy as np

from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr, fast_document_vote_top_from_training
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def repeated_document_top_from_training(X, y, count=None, *, task="classification",
                                        random_state=1129, groups=None, ordered_groups=False,
                                        repeats=3, aggregation="median", backend="auto"):
    """Aggregate independently sampled zero/one-peer held-loss endpoints.

    Each repeat uses the original cross-fitted document game with at most
    512 sampled rows. A feature's endpoints are averaged or median pooled
    across repeats before taking their minimum. Strong peers remain chosen
    inside the fitting half of each repeat. No held outer labels are used.
    Automatic width uses the original positive-mass rule and bounds.

    The shrinkage ablation multiplies each endpoint mean mu by
    mu**2 / (mu**2 + variance_of_the_mean). That variance measures sketch
    variability conditional on the supplied training data; overlapping
    sketches do not establish independent-sample predictive uncertainty.
    The first repeat uses random_state exactly, preserving one-repeat
    compatibility with document_vote. Subsequent repeat seeds are fixed.
    Backend selection is forwarded to each repeated endpoint calculation.
    """
    if aggregation not in ("mean", "median", "shrinkage"):
        raise ValueError("aggregation must be mean, median or shrinkage")
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, native or numpy")
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    n, p = X.shape
    if p == 1:
        return np.array([0]), dict(scores=np.zeros(1), zero_peer_scores=np.zeros(1),
                                  one_peer_scores=np.zeros(1), budget=1, cap=1, folds=[],
                                  repeats=1, peer_window=(0, 0))
    seeds = [random_state]
    for repeat in range(1, repeats):
        seed = np.random.SeedSequence([7163977, random_state, repeat]).generate_state(1)[0]
        seeds.append(int(seed))
    endpoints, folds = [], []
    for seed in seeds:
        _, info = fast_document_vote_top_from_training(X, y, 1, task=task, random_state=seed,
                                                       groups=groups, ordered_groups=ordered_groups,
                                                       backend=backend)
        endpoints.append(np.stack((info["zero_peer_scores"], info["one_peer_scores"])))
        folds.extend(dict(fold, repeat_seed=seed) for fold in info["folds"])
    endpoints = np.asarray(endpoints)
    aggregate = np.mean(endpoints, axis=0)
    if aggregation == "median":
        aggregate = np.median(endpoints, axis=0)
    elif aggregation == "shrinkage" and repeats > 1:
        variance = np.var(endpoints, axis=0, ddof=1) / repeats
        squared = aggregate**2
        weight = np.divide(squared, squared + variance, out=np.zeros_like(squared),
                           where=squared + variance > 0)
        aggregate *= weight
    zero, one = aggregate
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    classes, labels = np.unique(y, return_inverse=True)
    counts = _counts(X, labels, len(classes))
    marginal = _row_dot(counts.rhs, counts.rhs) / (counts.square + 1.)
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(scores=scores, zero_peer_scores=zero,
                                      one_peer_scores=one, cap=cap, budget=int(count),
                                      folds=folds, repeats=repeats, replicate_endpoints=endpoints,
                                      aggregation=aggregation, peer_window=(0, 1))
