"""minPI with exact empty/singleton peer layers on low-dimensional matrices."""
import ctypes
from functools import lru_cache
import numpy as np

from research.lowdim.build_native import library_path

MAX_FEATURES = 32
VOTES, PAIR_PATH, BLOCK_PATH = range(3)


@lru_cache(maxsize=3)
def _kernel(mode=VOTES):
    path = library_path()
    if not path.is_file():
        raise FileNotFoundError("Build low-dimensional minPI with python -m research.lowdim.build_native")
    library = ctypes.CDLL(str(path))
    entry = ("lowdim_votes", "lowdim_pairs", "lowdim_blocks")[mode]
    function = getattr(library, entry)
    ptr = ctypes.c_void_p
    stride = ctypes.c_ssize_t
    arguments = [ptr, stride, stride, ctypes.c_int, ptr, stride, ctypes.c_size_t,
                 ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ptr, ptr, ptr, ptr]
    if mode != VOTES:
        arguments.append(ptr)  # optional column order
    if mode == BLOCK_PATH:
        arguments.append(ptr)  # optional block sizes
    function.argtypes = arguments
    function.restype = ctypes.c_int
    return function


def fast_support(X, y, *, task, seed=1129):
    """Select columns with conditional ridge loss votes using only training data.

    For p<=32 the two tiers are no peers and the mean over every single peer.
    The top prefix length is the ceiling of the effective number of votes.
    Larger widths retain the previously frozen selector.
    """
    return _select_support(X, y, task, seed, mode=VOTES, audit=False)


def pair_support(X, y, *, task, seed=1129, audit=False):
    """Keep the voting-based width and select a complementary subset.

    At up to 32 columns with K >= 2, seed the subset with the pair having
    the greatest joint ridge gain. Fill K by the conditional gain given all
    columns already chosen. K=1 uses the empty-peer gain. Full width returns
    all columns. With audit=True, selection_order records the low-dimensional
    search path, or is empty for the full-width shortcut. rank and tier_scores
    still describe the original width rule. Larger inputs use fast_support's
    existing wide rule. Returned column indices are always in original order
    for the low-dimensional rules.
    """
    return _select_support(X, y, task, seed, mode=PAIR_PATH, audit=audit)


def block_support(X, y, *, task, seed=1129, audit=False):
    """Use the vote width as a cap, then grow by joint gains of one or two columns.

    With a positive signal and room for two columns, start from the best
    pair. At each later step compare all feasible singletons and pairs by
    -m*log(1-gain/loss) - block_size*log(m), using the current minimized
    penalized training loss. Stop when no score is positive or the cap is
    reached. A one-column cap uses the best marginal gain. No signal returns
    the full input. On low-dimensional inputs audit=True records the ordered
    columns and the block sizes; the no-signal shortcut has an empty path.
    Wider data use the same fallback as fast_support.
    """
    return _select_support(X, y, task, seed, mode=BLOCK_PATH, audit=audit)


def _select_support(X, y, task, seed, *, mode, audit):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),):
        raise ValueError("Expected a nonempty feature matrix and aligned response")
    if task not in ("regression", "classification"):
        raise ValueError("Expected classification or regression")
    n, p = X.shape
    if p > MAX_FEATURES:
        if task == "regression":
            from research.conditional_dense.original_tasks.regression import regression_support
            result = regression_support(X, y, seed=seed)
        else:
            from research.conditional_dense.vote_width.method import vote_support
            result = vote_support(X, y, seed=seed)
        if mode == BLOCK_PATH:
            result[1]["width_cap"] = len(result[0])
        return result

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
    columns = np.empty(p + (mode == BLOCK_PATH), dtype=np.int64)
    base = votes.ctypes.data
    order = np.empty(p, dtype=np.int64) if audit else None
    extra = (order.ctypes.data if audit else None,) if mode != VOTES else ()
    if mode == BLOCK_PATH:
        sizes = np.zeros(p, dtype=np.uint8) if audit else None
        extra += (sizes.ctypes.data if audit else None,)
    width = _kernel(mode)(X.ctypes.data, *X.strides, X.itemsize,
                      response.ctypes.data, response.strides[0], n, p, class_count,
                      int(seed), base, base+p*8, base+p*16, columns.ctypes.data, *extra)
    if width <= 0:
        raise ValueError("Sampled input must be finite with representable second moments")
    info = dict(width=width, selected_width=width, rank=votes[0],
                tier_scores=votes[1:], tiers=(0,1) if p>1 else (0,),
                sketch_size=min(n,768))
    if mode == BLOCK_PATH:
        info["width_cap"] = int(columns[p])
        if audit:
            info["block_sizes"] = sizes[sizes > 0]
            info["selection_order"] = order[:width] if sizes[0] else order[:0]
    elif order is not None:
        info["selection_order"] = order[:width] if width < p else order[:0]
    return columns[:width], info
