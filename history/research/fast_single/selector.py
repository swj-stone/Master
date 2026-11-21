"""Sampled singleton ridge votes with noise-adjusted effective width."""
import ctypes
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
from scipy import sparse

from research.lowdim.method import MAX_FEATURES
from research.peer_crossover.single import prepare as prepare_sampler, select_single


ROWS = 384
PEERS = 8


@lru_cache(maxsize=1)
def prepare():
    """Build/load the experimental kernels ahead of timing and return their CDLL."""
    source = Path(__file__).with_name("panel_votes.c")
    flags = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off"]
    revision = sha256(source.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    folder = source.parents[2] / "results/minpi-study/fast-single-native"
    dll = folder / f"panel-{platform.machine()}-{sys.platform}-{revision}{ext}"
    if not dll.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("Building the sampled-peer reducer requires a C11 compiler")
        shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
        if sys.platform != "win32":
            shared.append("-fPIC")
        tmp = dll.with_name(dll.name + f".{os.getpid()}.tmp")
        try:
            subprocess.run([compiler, *flags, *shared, str(source), "-o", str(tmp), "-lm"], check=True)
            if not dll.exists():
                os.replace(tmp, dll)
        finally:
            tmp.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(dll))
    ptr, integer, stride = ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t
    tail = [ptr, integer, integer, ptr, integer, ptr, integer, ptr]
    lib.dense_moments.argtypes = [ptr, stride, stride, integer, *tail]
    lib.sparse_moments.argtypes = [ptr, integer, ptr, integer, ptr, integer, *tail]
    lib.panel_gains.argtypes = [ptr, integer, integer, ptr, integer, ptr, integer, integer, ptr]
    for fn in (lib.dense_moments, lib.sparse_moments, lib.panel_gains):
        fn.restype = integer
    prepare_sampler()
    return lib


def _response(y, rows, task):
    sample = y[rows]
    if task == "classification":
        labels, ids, counts = np.unique(sample, return_inverse=True, return_counts=True)
        if len(labels) < 2 and np.unique(y).size < 2:
            raise ValueError("Classification requires at least two training classes")
        sums = counts.astype(np.float64)
        energy = len(rows) - sums @ sums / len(rows)
        return ids.astype(np.float64), sums, len(labels), energy
    target = np.asarray(sample, dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite sampled targets")
    magnitude = np.max(np.abs(target))
    if magnitude:
        target = target / magnitude
    target = target - target[0]
    target -= target.mean()
    deviation = np.sqrt(target @ target / len(target))
    if deviation:
        target /= deviation
    return target, np.array([target.sum()]), 0, target @ target


def _native_csr(X):
    """Normalize CSR buffers before native code or SciPy's sparse row indexing."""
    X = X.tocsr()
    arrays = (X.data, X.indices, X.indptr)
    if all(a.flags.c_contiguous and a.flags.aligned and a.dtype.isnative for a in arrays):
        return X
    buffers = tuple(np.require(a, dtype=a.dtype.newbyteorder("="), requirements=("C", "A"))
                    for a in arrays)
    return X.__class__(buffers, shape=X.shape, copy=False)


def _panel_scores(X, rows, peers, target, sums, nclass):
    m, p, h = len(rows), X.shape[1], len(peers)
    dense = not sparse.issparse(X)
    if X.dtype not in (np.dtype("float32"), np.dtype("float64")):
        # All moments need only sampled rows, even for non-native input dtypes.
        X = X[rows].astype(np.float64)
        rows = np.arange(m, dtype=np.int64)
    work = np.zeros((2+len(sums)+h, p), dtype=np.float64)
    kernel = prepare()
    operands = (rows.ctypes.data, m, p, target.ctypes.data, nclass, peers.ctypes.data, h, work.ctypes.data)
    if dense:
        status = kernel.dense_moments(X.ctypes.data, *X.strides, X.itemsize, *operands)
    else:
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        status = kernel.sparse_moments(
            X.data.ctypes.data, X.data.itemsize, X.indices.ctypes.data, X.indices.itemsize,
            X.indptr.ctypes.data, X.indptr.itemsize, *operands)
    if status:
        raise MemoryError("Unable to allocate sampled-peer scratch space")
    tiers = np.empty((2, p), dtype=np.float64)
    status = kernel.panel_gains(work.ctypes.data, p, m, sums.ctypes.data, len(sums),
                                peers.ctypes.data, h, int(not dense), tiers.ctypes.data)
    if status == 1:
        return _panel_scores(X[rows].toarray(), np.arange(m, dtype=np.int64), peers, target, sums, nclass)
    if status:
        raise ValueError("Sampled features have nonfinite second moments")
    return tiers


def select_fast(X, y, *, task, seed=1129):
    """Select sorted column indices using only training data; all peers at p≤32."""
    y = np.asarray(y)
    X = _native_csr(X) if sparse.issparse(X) else np.asarray(X)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (X.shape[0],):
        raise ValueError("Expected a nonempty feature matrix and aligned response")
    if task not in ("classification", "regression"):
        raise ValueError("Expected classification or regression")
    n, p = X.shape
    if p <= MAX_FEATURES:
        cols, info = select_single(X, y, task=task, seed=seed)
        info["null_scale"] = 0.
        return cols, info

    m = min(ROWS, n)
    rows = np.arange(m, dtype=np.int64)
    if m < n:
        prepare_sampler().sample_rows(n, m, seed, rows.ctypes.data)
    rng = np.random.default_rng([int(seed), 963221])
    peers = rng.choice(p, size=PEERS, replace=False).astype(np.int64, copy=False)
    target, sums, nclass, energy = _response(y, rows, task)
    layers = _panel_scores(X, rows, peers, target, sums, nclass)
    scores = np.minimum(layers[0], layers[1])
    # For standardized nonconstant z, E||z'Y_perm||²=m||Y||_F²/(m-1).
    null = m*energy/((m-1)*(m+np.sqrt(m))) if m>1 else 0.
    high = scores.max()
    if high > 0.:
        scaled = scores / high
        reliability = np.divide(scores, scores + null, out=np.zeros_like(scores), where=scores > 0)
        weights = scaled*reliability
        width = min(p, int(np.ceil(weights.sum()**2 / (weights @ weights))))
        # Select the top K, resolving equal scores by increasing input index.
        cutoff = np.partition(scores, p-width)[p-width]
        keep = scores > cutoff
        tied = np.flatnonzero(scores==cutoff)
        keep[tied[:width-np.count_nonzero(keep)]] = True
        cols = np.flatnonzero(keep)
    else:
        width = p
        cols = np.arange(p)
    return cols, dict(rank=scores, tier_scores=layers, tiers=(0, 1), width=width,
                      selected_width=width, sketch_size=m, peers=peers, null_scale=null)
