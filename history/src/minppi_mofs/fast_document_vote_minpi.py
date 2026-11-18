"""Optional precompiled binary CSR backend for presence-vote minPI."""
import ctypes
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from minppi_mofs.document_vote_minpi import _presence, document_vote_top_from_training
from minppi_mofs.signal_vote_minpi import _split
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


SOURCE = Path(__file__).with_name("document_vote_kernel.c")
BUILD = Path(__file__).resolve().parents[2] / "tmp" / "document_native"


def library_path():
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    return BUILD / ("document-" + sha256(SOURCE.read_bytes()).hexdigest()[:16] + suffix)


@lru_cache(maxsize=1)
def load_native():
    lib = ctypes.CDLL(str(library_path()))
    ptr, num = ctypes.c_void_p, ctypes.c_int
    lib.document_votes.argtypes = [ptr, ptr, ptr, num, num, ptr, num, ptr, num, num, ptr, ptr, ptr]
    lib.document_votes.restype = num
    return lib


def _checked_csr(X):
    """Validate raw offset and index buffers before crossing into compiled code."""
    if not sparse.issparse(X) or X.ndim != 2:
        raise ValueError("Expected a sparse document-by-term matrix")
    if X.format in ("csr", "csc"):
        ptr, idx = X.indptr, X.indices
        rows, cols = X.shape if X.format == "csr" else X.shape[::-1]
        if (ptr.ndim != 1 or idx.ndim != 1 or X.data.ndim != 1 or
                ptr.dtype.kind != "i" or idx.dtype.kind != "i" or len(ptr) != rows + 1 or
                len(idx) != len(X.data) or ptr[0] != 0 or ptr[-1] < 0 or ptr[-1] > len(idx) or
                np.any(ptr[1:] < ptr[:-1])):
            raise ValueError("Invalid sparse index pointers or buffer lengths")
        used = idx[:ptr[-1]]
        if len(used) and (used.min() < 0 or used.max() >= cols):
            raise ValueError("Sparse column or row index outside matrix bounds")
        if X.format == "csr" and len(used) == len(idx):
            return X
    X = X.tocsr(copy=True)
    X.check_format(full_check=True)
    return X


def fast_document_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                         groups=None, ordered_groups=False, backend="auto"):
    """Apply the original presence votes using an explicitly built binary CSR kernel.

    Sampling, tie breaking, rank normalization and adaptive width use the
    same NumPy operations and ordering as document_vote. Multiclass,
    non-int32 CSR indices, unsupported builds and singletons use the
    reference implementation in auto mode. No compiler runs in fit().
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, native or numpy")
    opts = dict(task=task, random_state=random_state, groups=groups, ordered_groups=ordered_groups)

    def reference():
        cols, info = document_vote_top_from_training(X, y, count, **opts)
        return cols, dict(info, backend="numpy")

    X = _checked_csr(X)
    if backend == "numpy" or task != "classification":
        return reference()
    X = _presence(X)
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    n, p = X.shape
    classes, labels = np.unique(y, return_inverse=True)
    if min(n, p) < 2 or len(classes) != 2 or X.indices.dtype != np.int32 or X.indptr.dtype != np.int32 or max(n,p) > 2**31-1:
        if backend == "native":
            raise ValueError("Native backend requires binary classification, n,p>=2 and 32-bit CSR indices")
        return reference()
    try:
        kernel = load_native()
    except (OSError, AttributeError):
        if backend == "native":
            raise
        return reference()
    left, right = _split(labels, task, groups, ordered_groups, np.random.default_rng(random_state))
    left = np.ascontiguousarray(left, dtype=np.int64)
    right = np.ascontiguousarray(right, dtype=np.int64)
    directions = 1 if ordered_groups else 2
    peers = np.empty((directions, 2), dtype=np.int32)
    votes = np.empty((directions, 2, p), dtype=np.float64)
    marginal = np.empty(p, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    pointers = np.ascontiguousarray(X.indptr)
    indices = np.ascontiguousarray(X.indices)
    rc = kernel.document_votes(pointers.ctypes.data, indices.ctypes.data, labels.ctypes.data, n, p,
                               left.ctypes.data, len(left), right.ctypes.data, len(right), directions,
                               peers.ctypes.data, votes.ctypes.data, marginal.ctypes.data)
    if rc:
        raise RuntimeError(f"Document vote kernel returned status {rc}")
    folds = [dict(fit_rows=left, held_rows=right, peers=tuple(map(int, peers[0])))]
    if not ordered_groups:
        folds.append(dict(fit_rows=right, held_rows=left, peers=tuple(map(int, peers[1]))))
    zero, one = votes.mean(axis=0)
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(scores=scores, zero_peer_scores=zero, one_peer_scores=one,
                                       cap=cap, budget=int(count), folds=folds, peer_window=(0, 1), backend="native")
