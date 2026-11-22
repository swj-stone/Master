"""Two execution plans for the paper's sampled single-peer vote."""
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

from research.fast_single import selector as paper


@lru_cache(maxsize=1)
def prepare():
    source = Path(__file__).with_suffix(".c")
    original = Path(paper.__file__).with_name("panel_votes.c")
    flags = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off"]
    revision = sha256(source.read_bytes() + original.read_bytes() + str(flags).encode()).hexdigest()[:16]
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    cache = source.parents[2] / "results/minpi-study/vote-frontier-native"
    dll = cache / f"active-{platform.machine()}-{revision}{ext}"
    if not dll.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        compiler = shutil.which("gcc") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("A C11 compiler is required for the compact CSR reducer")
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
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.compact_votes.argtypes = [ptr, integer, ptr, integer, ptr, integer, ptr, integer,
                                  integer, ptr, integer, ptr, ptr, integer, ptr]
    lib.compact_votes.restype = integer
    paper.prepare()
    return lib


def compact_scores(X, rows, peers, target, sums, nclass):
    X = paper._native_csr(X)
    if X.dtype not in (np.dtype("float32"), np.dtype("float64")):
        X = X[rows].astype(np.float64)
        rows = np.arange(len(rows), dtype=np.int64)
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    p = X.shape[1]
    tiers = np.zeros((2, p))
    status = prepare().compact_votes(X.data.ctypes.data, X.data.itemsize, X.indices.ctypes.data,
                                     X.indices.itemsize, X.indptr.ctypes.data, X.indptr.itemsize,
                                     rows.ctypes.data, len(rows), p, target.ctypes.data, nclass,
                                     sums.ctypes.data, peers.ctypes.data, len(peers), tiers.ctypes.data)
    if status == 1:
        return paper._panel_scores(X[rows].toarray(), np.arange(len(rows), dtype=np.int64),
                                   peers, target, sums, nclass)
    if status == -1:
        raise MemoryError("Unable to allocate compact vote workspace")
    if status:
        raise ValueError("Sampled features have nonfinite second moments")
    return tiers


def gemm_scores(X, rows, peers, target, sums, nclass):
    """Compute shifted sufficient statistics with a dense matrix multiplication."""
    sample = np.asarray(X[rows], dtype=np.float64)
    sample -= sample[0].copy()
    m, p = sample.shape
    c, h = len(sums), len(peers)
    loadings = np.empty((c + h, m))
    if nclass:
        loadings[:c] = np.arange(c)[:, None] == target[None, :]
    else:
        loadings[0] = target
    loadings[c:] = sample[:, peers].T
    work = np.empty((2 + c + h, p))
    work[0] = sample.sum(axis=0)
    np.einsum("ij,ij->j", sample, sample, out=work[1], optimize=False)
    np.matmul(loadings, sample, out=work[2:])
    tiers = np.empty((2, p))
    status = paper.prepare().panel_gains(work.ctypes.data, p, m, sums.ctypes.data, c,
                                         peers.ctypes.data, h, 0, tiers.ctypes.data)
    if status:
        raise ValueError("Sampled features have nonfinite second moments")
    return tiers


def _select(X, y, task, seed, choose_peers=None):
    X = paper._native_csr(X) if sparse.issparse(X) else np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (X.shape[0],):
        raise ValueError("Expected a nonempty feature matrix and aligned response")
    if task not in ("classification", "regression"):
        raise ValueError("Expected classification or regression")
    n, p = X.shape
    if p <= paper.MAX_FEATURES:
        return paper.select_fast(X, y, task=task, seed=seed)
    m = min(n, paper.ROWS)
    rows = np.arange(m, dtype=np.int64)
    if n > m:
        paper.prepare_sampler().sample_rows(n, m, seed, rows.ctypes.data)
    rng = np.random.default_rng([int(seed), 963221])
    peers = rng.choice(p, paper.PEERS, replace=False).astype(np.int64, copy=False)
    if choose_peers is not None:
        peers = choose_peers(X, rows, peers, seed)
    target, sums, nclass, energy = paper._response(y, rows, task)
    score = paper._panel_scores
    if sparse.issparse(X) and int((X.indptr[rows+1]-X.indptr[rows]).sum()) < p:
        score = compact_scores
    layers = score(X, rows, peers, target, sums, nclass)
    ranks = np.minimum(*layers)
    null = m*energy/((m-1)*(m+np.sqrt(m))) if m>1 else 0.
    high = ranks.max()
    if high > 0.:
        w = (ranks/high)*np.divide(ranks, ranks + null, out=np.zeros_like(ranks), where=ranks>0.)
        k = min(p, int(np.ceil(w.sum()**2 / (w @ w))))
        cutoff = np.partition(ranks, p-k)[p-k]
        keep = ranks > cutoff
        ties = np.flatnonzero(ranks == cutoff)
        keep[ties[:k - np.count_nonzero(keep)]] = True
        cols = np.flatnonzero(keep)
    else:
        cols = np.arange(p)
    return cols, dict(rank=ranks, tier_scores=layers, tiers=(0, 1), width=len(cols),
                      selected_width=len(cols), sketch_size=m, peers=peers, null_scale=null)


def select_arch(X, y, *, task, seed=1129):
    """Use compact scratch when sampled CSR postings occupy fewer slots than p."""
    if not sparse.issparse(X):
        return paper.select_fast(X, y, task=task, seed=seed)
    return _select(X, y, task, seed)
