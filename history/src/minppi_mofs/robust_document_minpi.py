"""Repeated median-pooled document minPI with optional shared native statistics."""
import ctypes
from functools import lru_cache
from hashlib import sha256
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

from minppi_mofs.document_vote_minpi import _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr
from minppi_mofs.repeated_document_minpi import repeated_document_top_from_training
from minppi_mofs.signal_vote_minpi import _split
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


SOURCE = Path(__file__).with_name("document_repeated_kernel.c")
BASE_SOURCE = Path(__file__).with_name("document_vote_kernel.c")
BUILD = Path(__file__).resolve().parents[2] / "tmp" / "document_repeated_native"


def source_digest():
    return sha256(SOURCE.read_bytes() + b"\0" + BASE_SOURCE.read_bytes()).hexdigest()


def library_path():
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    return BUILD / ("document-repeated-" + source_digest()[:16] + suffix)


@lru_cache(maxsize=1)
def load_native():
    library = ctypes.CDLL(str(library_path()))
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    library.document_repeated_votes.argtypes = [
        pointer, pointer, pointer, integer, integer,
        pointer, pointer, pointer, pointer, integer, integer, pointer, pointer, pointer,
    ]
    library.document_repeated_votes.restype = integer
    return library


def robust_document_top_from_training(
    X, y, count=None, *, task="classification", random_state=1129,
    groups=None, ordered_groups=False, repeats=3, backend="auto",
):
    """Median-pool repeated zero/one-peer gains before their context minimum.

    This is numerically equivalent to repeated_document with median pooling.
    Each repeat samples its own fitting and held rows, using the caller's
    seed first and fixed derived seeds for subsequent repeats. The binary
    native implementation shares full-training marginal tie statistics and
    scratch storage across repeats. No compiler runs during fitting.

    Multiclass inputs, one-column inputs and unsupported CSR indices use
    the reference implementation in auto mode. A missing prebuilt library
    also falls back automatically. Group and ordered-group splits retain
    the same isolation as the reference implementation.
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, native or numpy")
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)

    def reference():
        cols, info = repeated_document_top_from_training(
            X, y, count, task=task, random_state=random_state, groups=groups,
            ordered_groups=ordered_groups, repeats=repeats, aggregation="median", backend="numpy",
        )
        return cols, dict(info, backend="numpy")

    if backend == "numpy":
        return reference()
    n, p = X.shape
    classes, labels = np.unique(y, return_inverse=True)
    supported = (n >= 2 and p >= 2 and len(classes) == 2 and
                 X.indices.dtype == np.int32 and X.indptr.dtype == np.int32 and
                 max(n, p, repeats) <= 2**31 - 1)
    if not supported:
        if backend == "native":
            raise ValueError("Native backend requires binary labels, n,p>=2 and 32-bit CSR indices")
        return reference()
    try:
        kernel = load_native()
    except (OSError, AttributeError):
        if backend == "native":
            raise
        return reference()
    seeds = [random_state]
    for repeat in range(1, repeats):
        seed = np.random.SeedSequence([7163977, random_state, repeat]).generate_state(1)[0]
        seeds.append(int(seed))
    splits = [
        _split(labels, task, groups, ordered_groups, np.random.default_rng(seed))
        for seed in seeds
    ]
    left = np.ascontiguousarray(np.concatenate([split[0] for split in splits]), dtype=np.int64)
    right = np.ascontiguousarray(np.concatenate([split[1] for split in splits]), dtype=np.int64)
    left_offsets = np.r_[0, np.cumsum([len(split[0]) for split in splits], dtype=np.int64)]
    right_offsets = np.r_[0, np.cumsum([len(split[1]) for split in splits], dtype=np.int64)]
    directions = 1 if ordered_groups else 2
    peers = np.empty((repeats, directions, 2), dtype=np.int32)
    votes = np.empty((repeats, directions, 2, p), dtype=np.float64)
    marginal = np.empty(p, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    pointers = np.ascontiguousarray(X.indptr)
    indices = np.ascontiguousarray(X.indices)
    status = kernel.document_repeated_votes(
        pointers.ctypes.data, indices.ctypes.data, labels.ctypes.data, n, p,
        left.ctypes.data, left_offsets.ctypes.data, right.ctypes.data, right_offsets.ctypes.data,
        repeats, directions, peers.ctypes.data, votes.ctypes.data, marginal.ctypes.data,
    )
    if status:
        raise RuntimeError(f"Repeated document vote kernel returned status {status}")
    endpoints = votes.mean(axis=1)
    zero, one = np.median(endpoints, axis=0)
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    folds = []
    for repeat, ((fit, held), seed) in enumerate(zip(splits, seeds)):
        folds.append(dict(fit_rows=fit, held_rows=held, peers=tuple(map(int, peers[repeat, 0])),
                          repeat_seed=seed))
        if not ordered_groups:
            folds.append(dict(fit_rows=held, held_rows=fit, peers=tuple(map(int, peers[repeat, 1])),
                              repeat_seed=seed))
    return np.sort(rank[:count]), dict(
        scores=scores, zero_peer_scores=zero, one_peer_scores=one, cap=cap, budget=int(count),
        folds=folds, repeats=repeats, replicate_endpoints=endpoints,
        aggregation="median", peer_window=(0, 1), backend="native",
    )
