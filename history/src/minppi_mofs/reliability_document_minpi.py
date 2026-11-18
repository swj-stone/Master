"""Penalize directional disagreement using one cross-fitted document sketch."""
import numpy as np

from minppi_mofs.document_vote_minpi import _counts, _presence, _votes
from minppi_mofs.fast_document_vote_minpi import _checked_csr, load_native
from minppi_mofs.signal_vote_minpi import _row_dot, _split
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def reliability_document_top_from_training(
    X, y, count=None, *, task="classification", random_state=1129,
    groups=None, ordered_groups=False, backend="auto",
):
    """Use the smaller directional gain within each zero/one-peer context.

    The two cross-fit directions already exist in one document-vote sketch.
    For each context, their mean minus their usual two-value standard error
    equals their minimum. Apply this disagreement penalty before the minPI
    minimum over contexts. Overlapping cross-fit directions are dependent,
    so the penalty is a descriptive dispersion correction rather than a
    calibrated confidence bound. Ordered groups use their sole forward
    direction unchanged.

    Counts, sampling, strong-peer choice, tie breaking and automatic 99%
    positive-vote mass budget match the preceding single-sketch rule.
    Binary inputs can reuse its prebuilt kernel; other supported inputs
    use NumPy, with no compilation during fitting.
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, native or numpy")
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    n, p = X.shape
    if p == 1:
        return np.array([0]), dict(
            scores=np.zeros(1), zero_peer_scores=np.zeros(1), one_peer_scores=np.zeros(1),
            budget=1, cap=1, folds=[], peer_window=(0, 0), backend="numpy",
        )
    if n < 2:
        raise ValueError("Need at least two training documents")
    classes, labels = np.unique(y, return_inverse=True)
    left, right = _split(labels, task, groups, ordered_groups, np.random.default_rng(random_state))
    directions = 1 if ordered_groups else 2
    supported = (len(classes) == 2 and X.indices.dtype == np.int32 and
                 X.indptr.dtype == np.int32 and max(n, p) <= 2**31 - 1)
    kernel = None
    if backend != "numpy" and supported:
        try:
            kernel = load_native()
        except (OSError, AttributeError):
            if backend == "native":
                raise
    elif backend == "native":
        raise ValueError("Native backend requires binary labels and 32-bit CSR indices")
    if kernel is None:
        fit = _counts(X[left], labels[left], len(classes))
        held = _counts(X[right], labels[right], len(classes))
        endpoint, peer = _votes(fit, held)
        votes, peers = [endpoint], [peer]
        if not ordered_groups:
            endpoint, peer = _votes(held, fit)
            votes.append(endpoint)
            peers.append(peer)
        votes, peers = np.asarray(votes), np.asarray(peers)
        full = _counts(X, labels, len(classes))
        marginal = _row_dot(full.rhs, full.rhs) / (full.square + 1.)
        used = "numpy"
    else:
        left = np.ascontiguousarray(left, dtype=np.int64)
        right = np.ascontiguousarray(right, dtype=np.int64)
        encoded = np.ascontiguousarray(labels, dtype=np.uint8)
        pointers = np.ascontiguousarray(X.indptr)
        indices = np.ascontiguousarray(X.indices)
        peers = np.empty((directions, 2), dtype=np.int32)
        votes = np.empty((directions, 2, p), dtype=np.float64)
        marginal = np.empty(p, dtype=np.float64)
        status = kernel.document_votes(
            pointers.ctypes.data, indices.ctypes.data, encoded.ctypes.data, n, p,
            left.ctypes.data, len(left), right.ctypes.data, len(right), directions,
            peers.ctypes.data, votes.ctypes.data, marginal.ctypes.data,
        )
        if status:
            raise RuntimeError(f"Document vote kernel returned status {status}")
        used = "native"
    zero, one = votes.min(axis=0)
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    folds = [dict(fit_rows=left, held_rows=right, peers=tuple(map(int, peers[0])))]
    if not ordered_groups:
        folds.append(dict(fit_rows=right, held_rows=left, peers=tuple(map(int, peers[1]))))
    return np.sort(rank[:count]), dict(
        scores=scores, zero_peer_scores=zero, one_peer_scores=one, cap=cap, budget=int(count),
        folds=folds, peer_window=(0, 1), backend=used, directional_endpoints=votes,
        direction_penalty=np.ptp(votes, axis=0) / 2.,
    )
