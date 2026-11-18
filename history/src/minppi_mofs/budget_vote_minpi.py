"""Cross-fitted paired minPI with a width floor derived from agreement and vote concentration."""
import numpy as np

from minppi_mofs.moment_minpi import _moment_halves
from minppi_mofs.native_pivotal_minpi import _numpy_votes, load_native
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _endpoints(X, rows, n_left, mate, y, classification, directions, backend):
    target = y[rows]
    n_classes = 1
    if classification:
        classes, target = np.unique(target, return_inverse=True)
        n_classes = len(classes)
    target = np.ascontiguousarray(target, dtype=np.float64)
    library = None
    if backend != "numpy" and X.dtype in (np.dtype("float32"), np.dtype("float64")):
        try:
            library = load_native()
        except (OSError, AttributeError):
            if backend == "native":
                raise
    if library is None:
        if backend == "native":
            raise ValueError("Native votes require a float32 or float64 matrix")
        return _numpy_votes(X, rows[:n_left], rows[n_left:], mate, target, n_classes,
                            classification, directions), "numpy"
    votes = np.empty((directions, 2, X.shape[1]), dtype=np.float64)
    status = library.pivotal_endpoints(
        X.ctypes.data, X.itemsize, *X.strides, X.shape[1], rows.ctypes.data, n_left,
        len(rows) - n_left, mate.ctypes.data, target.ctypes.data, n_classes,
        int(classification), directions, votes.ctypes.data)
    if status:
        raise ValueError(f"Paired vote kernel returned status {status}")
    return votes, "native"


def _choose(votes, count, half_size):
    p = votes.shape[-1]
    scale = np.max(np.abs(votes))
    # Equivalent endpoint values can differ in the last few bits across BLAS
    # and native implementations. Define the ranking grid in dimensionless units.
    relative = np.round(votes / scale, decimals=12) if scale > 0 else votes
    zero, conditional = relative.mean(axis=0)
    scores = np.minimum(zero, conditional)
    order = np.argsort(-scores, kind="stable")
    positive = np.maximum(scores[order], 0.)
    total = positive.sum()
    ceiling = np.maximum(relative.max(axis=(0, 1)), 0.).sum()
    floor = np.maximum(relative.min(axis=(0, 1)), 0.).sum()
    agreement = floor / ceiling if ceiling else 0.
    effective = total**2 / (positive @ positive) if total else 0.
    if total:
        mass_width = min(p, 1 + np.searchsorted(positive.cumsum(), .98 * total))
        # With perfectly replicated positive gains, keep at least their
        # effective dimension. As agreement falls, protect weakly measured
        # coordinates, up to full width when no positive gain replicates.
        reliability = agreement * min(1., half_size / 32.)
        min_width = int(np.ceil(p - reliability * (p - effective) - 1e-12))
        adaptive_width = max(mass_width, min(p, max(1, min_width)))
    else:
        adaptive_width = p
    width = int(adaptive_width if count is None else count)
    return np.sort(order[:width]), dict(
        budget=width, scores=scores * scale, zero_peer_scores=zero * scale,
        one_peer_scores=conditional * scale, agreement=float(agreement),
        effective_dimension=float(effective))


def budget_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                  groups=None, ordered_groups=False, backend="auto"):
    """Select a support by low-cost held-row voting on seeded pairs.

    Pool the loss gains over permitted fitting directions; rank by the minimum
    over the zero- and one-peer contexts. The cumulative positive vote mass
    proposes a width. The ratio of total common positive gain to total possible
    positive gain across endpoints and directions controls an additional width
    floor, interpolating between all columns and their effective dimension.
    For very small held or fitting groups the floor increases. All quantities
    are training-only. A prescribed count overrides the learned width.
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, native or numpy")
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    if p == 1:
        cols, info = _choose(np.zeros((1, 2, 1)), count, len(y))
        return cols, dict(info, folds=[], peer_window=(0, 0), backend="none")
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    pair_order = rng.permutation(p)[:2 * (p // 2)].reshape(-1, 2)
    mate = np.arange(p, dtype=np.int64)
    mate[pair_order[:, 0]], mate[pair_order[:, 1]] = pair_order[:, 1], pair_order[:, 0]
    rows = np.ascontiguousarray(np.concatenate((left, right)), dtype=np.int64)
    directions = 1 if ordered_groups else 2
    votes, used = _endpoints(X, rows, len(left), mate, y, task == "classification", directions, backend)
    cols, info = _choose(votes, count, min(len(left), len(right)))
    folds = [dict(fit_rows=left, held_rows=right, mate=mate)]
    if directions == 2:
        folds.append(dict(fit_rows=right, held_rows=left, mate=mate))
    return cols, dict(info, folds=folds, peer_window=(0, 1), backend=used)
