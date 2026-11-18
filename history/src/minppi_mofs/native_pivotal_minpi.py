"""Compiled and NumPy implementations of the two-endpoint paired minPI game."""
import ctypes
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

from minppi_mofs.moment_minpi import _moment_gains, _moment_halves, _moments
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


SOURCE = Path(__file__).with_name("native_pivotal_kernel.c")
BUILD = Path(__file__).resolve().parents[2] / "tmp" / "pivotal_native"


def library_path():
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    return BUILD / ("pivotal-" + sha256(SOURCE.read_bytes()).hexdigest()[:16] + suffix)


@lru_cache(maxsize=1)
def load_native():
    """Load a previously compiled local library. This never invokes a compiler."""
    library = ctypes.CDLL(str(library_path()))
    ptr, integer, index = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    library.pivotal_endpoints.argtypes = [
        ptr, integer, index, index, index, ptr, integer, integer, ptr, ptr, integer, integer, integer, ptr]
    library.pivotal_endpoints.restype = integer
    return library


def _numpy_votes(X, left, right, mate, target, nc, classification, directions):
    if classification:
        target = (target[:, None] == np.arange(nc)).astype(float)
    else:
        target = target[:, None]
    fit = _moments(np.asarray(X[left], dtype=float), target[:len(left)], mate)
    held = _moments(np.asarray(X[right], dtype=float), target[len(left):], mate)
    votes = [_moment_gains(fit, held, mate)]
    if directions == 2:
        votes.append(_moment_gains(held, fit, mate))
    return np.asarray(votes)


def native_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                             groups=None, ordered_groups=False, backend="auto"):
    """Apply fixed-width moment minPI or delete a prefix with bounded observed vote mass.

    Both backends use the legacy moment sampling/pairing design. The native
    path accepts native-endian float32/64 arrays with arbitrary signed strides.
    Other dtypes and unavailable binaries use NumPy in automatic mode.
    Without a budget, order columns by the smaller of their zero- and
    full-peer mean votes. Each column's deletion cost is its largest positive
    vote across the directions and endpoints (zero if all are nonpositive).
    Drop the longest low-score prefix whose costs sum to at most 2% of all
    costs, capped at half the features. At least one is removed for p>1.
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, native or numpy")
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    info = {"folds": [], "peer_window": (0, p - 1), "backend": "none"}
    if p == 1 or count == p:
        return np.arange(p), dict(info, budget=p)
    library = None
    compatible = X.dtype in (np.dtype("float32"), np.dtype("float64"))
    if backend == "native" and not compatible:
        raise ValueError("The compiled kernel requires native-endian float32 or float64")
    if compatible and backend != "numpy":
        try:
            library = load_native()
        except (OSError, AttributeError):
            if backend == "native":
                raise
    split_seed, pair_seed = np.random.SeedSequence(random_state).spawn(2)
    left, right = _moment_halves(y, task, groups, ordered_groups, np.random.default_rng(split_seed))
    order = np.random.default_rng(pair_seed).permutation(p)
    pairs = order[:2 * (p // 2)].reshape(-1, 2)
    mate = np.arange(p, dtype=np.int64)
    mate[pairs[:, 0]], mate[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
    rows = np.asarray(np.concatenate((left, right)), dtype=np.int64)
    classification = task == "classification"
    if classification:
        classes, target = np.unique(y[rows], return_inverse=True)
        nc = len(classes)
    else:
        target = y[rows]
        nc = 1
    target = np.ascontiguousarray(target, dtype=np.float64)
    directions = 1 if ordered_groups else 2
    if library is None:
        votes = _numpy_votes(X, left, right, mate, target, nc, classification, directions)
    else:
        votes = np.empty((directions, 2, p), dtype=np.float64)
        rc = library.pivotal_endpoints(
            X.ctypes.data, X.itemsize, *X.strides, p, rows.ctypes.data, len(left), len(right),
            mate.ctypes.data, target.ctypes.data, nc, int(classification), directions, votes.ctypes.data)
        if rc:
            raise ValueError(f"Unsupported native sample or target encoding (kernel status {rc})")
    alone, conditional = votes.mean(axis=0)
    high = p - 1 if count is None else min(p - 1, max(1, count - 1))
    scores = np.minimum(alone, alone + (conditional - alone) * (high / (p - 1)))
    if count is None:
        cost = np.maximum(0., votes.max(axis=(0, 1)))
        weakest = np.lexsort((-np.arange(p), scores))
        num = np.searchsorted(cost[weakest].cumsum(), .02 * cost.sum(), side="right")
        drop = max(1, min(p // 2, num))
        support = np.sort(weakest[drop:])
        info["deletion_cost"] = cost
    else:
        support = np.sort(np.argsort(-scores, kind="stable")[:count])
    info.update(budget=len(support), scores=scores, peer_window=(0, high),
                backend="native" if library is not None else "numpy")
    info["folds"].append({"fit_rows": left, "held_rows": right, "mate": mate})
    if directions == 2:
        info["folds"].append({"fit_rows": right, "held_rows": left, "mate": mate})
    return support, info
