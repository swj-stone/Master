"""minPI with exact empty/singleton peer layers on low-dimensional matrices."""
import ctypes
from functools import lru_cache
import numpy as np

from research.lowdim.build_native import library_path

MAX_FEATURES = 32


@lru_cache(maxsize=1)
def _kernel():
    path = library_path()
    if not path.is_file():
        raise FileNotFoundError("Build low-dimensional minPI with python -m research.lowdim.build_native")
    function = ctypes.CDLL(str(path)).lowdim_votes
    ptr = ctypes.c_void_p
    stride = ctypes.c_ssize_t
    function.argtypes = [ptr, stride, stride, ctypes.c_int, ptr, stride, ctypes.c_size_t,
                         ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ptr, ptr, ptr, ptr]
    function.restype = ctypes.c_int
    return function


def fast_support(X, y, *, task, seed=1129):
    """Select columns with conditional ridge loss votes using only training data.

    For p<=32 the two tiers are no peers and the mean over every single peer.
    The top prefix length is the ceiling of the effective number of votes.
    Larger widths retain the previously frozen selector.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),):
        raise ValueError("Expected a nonempty feature matrix and aligned response")
    if task not in ("regression", "classification"):
        raise ValueError("Expected classification or regression")
    n, p = X.shape
    if p > MAX_FEATURES:
        if task == "regression":
            from research.conditional_dense.original_tasks.regression import regression_support
            return regression_support(X, y, seed=seed)
        from research.conditional_dense.vote_width.method import vote_support
        return vote_support(X, y, seed=seed)

    if X.dtype not in (np.dtype("float64"), np.dtype("float32")):
        X = X.astype(np.float64)
    if task == "classification":
        classes, ids = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two classes")
        response = ids.astype(np.float64)
        class_count = len(classes)
    else:
        response = np.asarray(y, dtype=np.float64)
        class_count = 0
    votes = np.empty((3, p), dtype=np.float64)
    columns = np.empty(p, dtype=np.int64)
    base = votes.ctypes.data
    width = _kernel()(X.ctypes.data, *X.strides, X.itemsize,
                      response.ctypes.data, response.strides[0], n, p, class_count,
                      int(seed), base, base+p*8, base+p*16, columns.ctypes.data)
    if width <= 0:
        raise ValueError("Sampled input must be finite with representable second moments")
    return columns[:width], dict(width=width, selected_width=width, rank=votes[0],
                                  tier_scores=votes[1:], tiers=(0,1) if p>1 else (0,),
                                  sketch_size=min(n,768))
