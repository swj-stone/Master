"""Exact all-single-peer selector for testing both peer rules at all input widths."""
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

from research.lowdim.method import MAX_FEATURES, fast_support


@lru_cache(maxsize=1)
def prepare():
    """Load the tile reducer, compiling the experiment-specific library if necessary."""
    source = Path(__file__).with_name("all_peers.c")
    flags = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off"]
    revision = sha256(source.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    path = (source.parents[2] / "results/minpi-study/peer-crossover-native"
            / f"single-{platform.machine()}-{sys.platform}-{revision}{ext}")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
        if sys.platform != "win32":
            shared.append("-fPIC")
        compiler = shutil.which("gcc") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("A C11 compiler is required to build the peer-crossover tile reducer")
        try:
            subprocess.run([compiler, *flags, *shared, str(source), "-o", str(temp), "-lm"], check=True)
            if not path.is_file():
                os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(path))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.sample_rows.argtypes = [ctypes.c_int64, integer, ctypes.c_uint64, ptr]
    lib.sum_tile.argtypes = [ptr, integer, integer, integer, integer, ptr, integer, ptr, ptr,
                             ctypes.c_double, ctypes.c_double, ptr]
    lib.sample_rows.restype = lib.sum_tile.restype = None
    return lib


def _target(y, rows, task):
    if task == "classification":
        classes, ids = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two classes")
        classes, sample = np.unique(ids[rows], return_inverse=True)
        target = (sample[:, None] == np.arange(len(classes))).astype(np.float64)
    else:
        target = np.asarray(y[rows], dtype=np.float64)[:, None]
        if not np.isfinite(target).all():
            raise ValueError("Expected finite sampled labels")
        amplitude = np.max(np.abs(target))
        if amplitude:
            target = target / amplitude
        target = target - target[0]
        deviation = target.std()
        if deviation:
            target /= deviation
    return target - target.mean(axis=0)


def _sparse_groups(sample):
    """Merge exact copies and proportional one-entry columns, keeping multiplicities."""
    x = sample.tocsc(copy=True).astype(np.float64, copy=False)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if not np.isfinite(x.data).all():
        raise ValueError("Expected finite sampled features")
    group = np.full(x.shape[1], -1, dtype=np.int64)
    representatives, seen = [], {}
    for j, (begin, end) in enumerate(zip(x.indptr[:-1], x.indptr[1:])):
        if end == begin:
            continue
        indices, data = x.indices[begin:end], x.data[begin:end]
        if len(data) == x.shape[0] and np.all(data == data[0]):
            continue
        # A nonzero at one row standardizes to the same vector up to sign.
        # Reversing a column sign leaves every squared ridge gain invariant.
        key = int(indices[0]) if len(data) == 1 else (indices.tobytes(), data.tobytes())
        gid = seen.get(key)
        if gid is None:
            gid = len(representatives)
            seen[key] = gid
            representatives.append(j)
        group[j] = gid
    return x[:, representatives], group


def _votes(sample, target, p):
    m = sample.shape[0]
    penalty = np.sqrt(m)
    diagonal = m + penalty
    # Some archived text-like matrices arrive dense. This affects storage only.
    if not sparse.issparse(sample) and p >= 1024:
        probe = sample[:min(m, 24)]
        if np.count_nonzero(probe) < .04 * probe.size:
            sample = sparse.csr_matrix(sample)
    if sparse.issparse(sample):
        x, group = _sparse_groups(sample)
        sums = np.asarray(x.sum(axis=0)).ravel()
        ss = np.asarray(x.power(2).sum(axis=0)).ravel() - sums*sums/m
        if np.any(ss <= 0):
            # Cancellation can occur for a varying, nearly constant dense column.
            return _votes_dense(sample.toarray(), target, p)
        scale = np.sqrt(m / ss)
        z = x.multiply(scale).tocsc()
        center = sums * scale / np.sqrt(m)
        rhs = np.asarray(z.T @ target)
    else:
        return _votes_dense(sample, target, p)
    return _accumulate(z, center, rhs, group, p, penalty, diagonal)


def _votes_dense(sample, target, p):
    m = sample.shape[0]
    z = np.array(sample, dtype=np.float64, order="F", copy=True)
    if not np.isfinite(z).all():
        raise ValueError("Expected finite sampled features")
    z -= z[0].copy()
    z -= z.mean(axis=0)
    ss = np.einsum("ij,ij->j", z, z)
    active = ss > 0.
    if not np.isfinite(ss).all():
        raise ValueError("Invalid column second moments")
    group = np.full(p, -1, dtype=np.int64)
    group[active] = np.arange(active.sum())
    if not active.all():
        z = z[:, active]
    z *= np.sqrt(m / ss[active])
    center = np.zeros(z.shape[1])
    rhs = z.T @ target
    return _accumulate(z, center, rhs, group, p, np.sqrt(m), m+np.sqrt(m))


def _accumulate(z, center, rhs, group, p, penalty, diagonal):
    active = group >= 0
    weights = np.bincount(group[active], minlength=z.shape[1]).astype(np.int64, copy=False)
    rhs = np.ascontiguousarray(rhs, dtype=np.float64)
    empty = np.einsum("ij,ij->i", rhs, rhs) / diagonal
    # Each constant peer leaves the marginal gain unchanged. Other copies in
    # the same group use the conditional gain with |cross| = sample size.
    m = z.shape[0]
    proj = m/diagonal
    self_gain = (1.-proj)**2 * diagonal / max(penalty, diagonal-m*proj) * empty
    totals = (p - active.sum())*empty + (weights-1)*self_gain
    size = z.shape[1]
    tile_size = 512
    kernel = prepare().sum_tile
    tiles = [z[:, j:j+tile_size] for j in range(0, size, tile_size)]
    for a, left in enumerate(tiles):
        for b in range(a, len(tiles)):
            right = tiles[b]
            gram = left.T @ right
            if sparse.issparse(gram):
                gram = gram.toarray()
            gram = np.ascontiguousarray(gram, dtype=np.float64)
            kernel(gram.ctypes.data, a*tile_size, b*tile_size, *gram.shape,
                   rhs.ctypes.data, rhs.shape[1], weights.ctypes.data,
                   center.ctypes.data, penalty, diagonal, totals.ctypes.data)
    layers = np.zeros((2, p))
    layers[0, active] = empty[group[active]]
    layers[1, active] = (totals / (p-1) if p > 1 else empty)[group[active]]
    if not np.isfinite(layers).all():
        raise ValueError("Invalid singleton gain")
    return layers


def select_single(X, y, *, task, seed=1129):
    """Return original-order columns and both vote tiers at any input width."""
    y = np.asarray(y)
    if not sparse.issparse(X):
        X = np.asarray(X)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (X.shape[0],):
        raise ValueError("Expected a nonempty matrix and aligned response")
    if task not in ("regression", "classification"):
        raise ValueError("Expected classification or regression")
    n, p = X.shape
    if p <= MAX_FEATURES:
        return fast_support(X.toarray() if sparse.issparse(X) else X, y, task=task, seed=seed)
    m = min(n, 768)
    rows = np.arange(m, dtype=np.int64)
    if n > m:
        prepare().sample_rows(n, m, seed, rows.ctypes.data)
    sample = X[rows] if n > m else X
    target = _target(y, rows, task)
    layers = _votes(sample, target, p)
    ranks = np.minimum(layers[0], layers[1])
    width = p
    if ranks.max() > 0:
        scaled = ranks / ranks.max()
        width = min(p, int(np.ceil(scaled.sum()**2 / (scaled @ scaled))))
    cols = np.sort(np.argsort(-ranks, kind="stable")[:width])
    return cols, dict(rank=ranks, tier_scores=layers, tiers=(0, 1), width=width,
                      selected_width=width, sketch_size=m)
