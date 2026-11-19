"""Complementary singleton/pair ridge votes on a training-row sketch."""
import ctypes
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis

from research.lowdim.method import block_support
from research.wide_blocks.build_native import library_path


@lru_cache(maxsize=1)
def _kernel():
    lib = ctypes.CDLL(str(library_path()))
    ptr, size, integer = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    lib.wide_rows.argtypes = [size, size, ctypes.c_uint64, ptr]
    lib.wide_rows.restype = None
    lib.wide_path.argtypes = [ptr, ptr, integer, integer, size, ctypes.c_double,
                              integer, ptr, ptr, ptr, ptr]
    lib.wide_path.restype = integer
    return lib


def _sample(n, seed):
    if n <= 768:
        return slice(None)
    rows = np.empty(768, dtype=np.int64)
    _kernel().wide_rows(n, len(rows), int(seed), rows.ctypes.data)
    return rows


def _response(y, rows, task):
    values = y[rows]
    if task == "classification":
        classes, ids, counts = np.unique(values, return_inverse=True, return_counts=True)
        if len(classes) == 1 and np.unique(y).size < 2:
            raise ValueError("Classification requires at least two classes")
        targets = np.zeros((len(ids), len(classes)))
        targets[np.arange(len(ids)), ids] = 1.
        targets -= counts / len(ids)
        return targets, len(ids) - float(counts @ counts) / len(ids)
    values = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Regression sketch must have finite targets")
    magnitude = np.max(np.abs(values))
    if magnitude:
        values /= magnitude
    values -= values[0]
    values -= values.mean()
    norm = np.linalg.norm(values)
    if norm:
        values *= np.sqrt(len(values)) / norm
    return values[:, None], float(len(values)) if norm else 0.


class _Sketch:
    """Keep sparse sketches compressed; gather only needed dense columns."""

    def __init__(self, X, rows):
        self.rows = rows
        self.m = min(X.shape[0], 768)
        self.p = X.shape[1]
        self.sparse = sparse.issparse(X)
        self.X = X
        if self.sparse:
            self.X = X.tocsr()[rows].tocsc().astype(np.float64, copy=False)
            if not self.X.has_canonical_format:
                self.X = self.X.copy()
                self.X.sum_duplicates()
            if not np.isfinite(self.X.data).all():
                raise ValueError("Feature sketch must have finite values")
            self.means, var = mean_variance_axis(self.X, axis=0)
            self.scales = np.divide(1., np.sqrt(var), out=np.zeros_like(var), where=var>0.)

    def standardize(self, cols):
        if self.sparse:
            raw = self.X[:, cols].toarray()
        elif isinstance(self.rows, slice) or isinstance(cols, slice):
            raw = self.X[self.rows, cols]
        else:
            raw = self.X[np.ix_(self.rows, cols)]
        Z = np.array(raw, dtype=np.float64, order="C", copy=True)
        if not np.isfinite(Z).all():
            raise ValueError("Feature sketch must have finite values")
        # Bound the scale of finite features before taking differences or squares.
        magnitude = np.max(np.abs(Z), axis=0)
        np.divide(Z, magnitude, out=Z, where=magnitude>0.)
        Z -= Z[0].copy()
        Z -= Z.mean(axis=0)
        norm = np.einsum("ij,ij->j", Z, Z)
        Z *= np.sqrt(np.divide(self.m, norm, out=np.zeros_like(norm), where=norm>0.))
        return Z, norm > 0.

    def project(self, cols, targets):
        if not self.sparse:
            Z, _ = self.standardize(cols)
            return Z.T @ targets
        raw = self.X[:, cols].T @ targets
        raw -= self.means[cols, None] * targets.sum(axis=0)
        raw *= self.scales[cols, None]
        return raw


def _candidates(sketch, Y, limit):
    """Half marginal, half conditional on the strongest marginal anchor."""
    p, c = sketch.p, Y.shape[1]
    if p <= limit:
        return np.arange(p, dtype=np.int64)
    chunk = min(16384 if sketch.sparse else 256, max(256, 262144 // c))
    spans = [slice(i, min(i + chunk, p)) for i in range(0, p, chunk)]
    # Retain RHS only when small. Otherwise recompute by chunks on the second pass.
    cache = np.empty((p, c)) if p * c <= 1_000_000 else None
    marginal = np.empty(p)
    diag = sketch.m + np.sqrt(sketch.m)
    for cols in spans:
        B = sketch.project(cols, Y)
        marginal[cols] = np.einsum("ij,ij->i", B, B) / diag
        if cache is not None:
            cache[cols] = B
    if not np.isfinite(marginal).all():
        raise ValueError("Screening requires representable feature moments")
    half = limit // 2
    first = np.argsort(-marginal, kind="stable")[:half]
    anchor = first[:1]
    if marginal[anchor[0]] == 0.:
        return np.arange(limit, dtype=np.int64)
    Z, _ = sketch.standardize(anchor)
    anchor_rhs = cache[anchor[0]] if cache is not None else (Z.T @ Y)[0]
    conditional = np.empty(p)
    for cols in spans:
        if cache is not None:
            B = cache[cols]
            cross = sketch.project(cols, Z)[:, 0]
        else:
            combined = sketch.project(cols, np.column_stack((Y, Z)))
            B, cross = combined[:, :-1], combined[:, -1]
        factor = cross / diag
        delta = B - factor[:, None] * anchor_rhs
        denom = np.maximum(np.sqrt(sketch.m), diag - factor * cross)
        conditional[cols] = np.einsum("ij,ij->i", delta, delta) / denom
    conditional[first] = -np.inf
    second = np.argsort(-conditional, kind="stable")[:half]
    return np.sort(np.concatenate((first, second))).astype(np.int64, copy=False)


def _path(Z, variable, Y, loss, fill):
    m, p = Z.shape
    gram = np.ascontiguousarray(Z.T @ Z)
    gram.flat[::p+1] = m * variable + np.sqrt(m)
    rhs = np.ascontiguousarray(Z.T @ Y)
    votes = np.empty((3, p))
    order = np.empty(p, dtype=np.int64)
    sizes = np.zeros(p, dtype=np.uint8)
    cap = np.empty(1, dtype=np.int64)
    width = _kernel().wide_path(gram.ctypes.data, rhs.ctypes.data, p, Y.shape[1], m,
                                loss, fill, votes.ctypes.data, order.ctypes.data,
                                sizes.ctypes.data, cap.ctypes.data)
    if width < 0:
        raise ValueError("Voting requires representable feature moments")
    return order[:width], sizes[sizes>0], int(cap[0]), votes


def wide_support(X, y, *, task, seed=1129, audit=False, fill_to_cap=False,
                 candidate_limit=128):
    """Return selected original column indices and the actual and capped widths.

    Up to 32 columns dispatch to the original block selector. Otherwise evaluate
    all columns up to candidate_limit (128 by default, or 256 in sensitivity
    analyses). Wider data use a two-part shortlist on the same training sketch.
    fill_to_cap disables early stopping only for the wider extension. No signal
    retains every original column. Audit arrays use original IDs for candidate
    columns and ordered selections, and candidate-space positions for vote tiers.
    """
    if task not in ("classification", "regression") or candidate_limit not in (128, 256):
        raise ValueError("Expected classification/regression and candidate limit 128/256")
    if not sparse.issparse(X):
        X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (X.shape[0],):
        raise ValueError("Expected a nonempty feature matrix and aligned response")
    n, p = X.shape
    if p <= 32:
        dense = X.toarray() if sparse.issparse(X) else X
        cols, info = block_support(dense, y, task=task, seed=seed, audit=audit)
        info.update(candidate_count=p, screened=False, no_signal=not np.any(info["rank"]))
        if audit:
            info["candidate_columns"] = np.arange(p)
        return cols, info

    rows = _sample(n, seed)
    Y, loss = _response(y, rows, task)
    sketch = _Sketch(X, rows)
    candidates = _candidates(sketch, Y, candidate_limit)
    Z, variable = sketch.standardize(candidates)
    order, sizes, cap, votes = _path(Z, variable, Y, loss, fill_to_cap)
    signal = bool(len(order))
    cols = np.sort(candidates[order]) if signal else np.arange(p, dtype=np.int64)
    info = dict(width=len(cols), selected_width=len(cols), width_cap=cap if signal else p,
                screened=len(candidates)<p, candidate_count=len(candidates),
                sketch_size=sketch.m, no_signal=not signal)
    if audit:
        info.update(candidate_columns=candidates, selection_order=candidates[order],
                    block_sizes=sizes, tier_scores=votes[:2], rank=votes[2], tiers=(0, 1))
    return cols, info
