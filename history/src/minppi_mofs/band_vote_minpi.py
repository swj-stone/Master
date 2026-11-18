"""Local coverage and paired loss votes for wavelength-ordered feature matrices."""
import numpy as np

from minppi_mofs.budget_vote_minpi import _endpoints
from minppi_mofs.moment_minpi import _draw, _moment_halves
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _partition(X, rows):
    # Center on the first observation to handle exactly constant decimal
    # channels. Missing and constant channels cannot certify a wider band.
    sample = np.asarray(X[rows], dtype=np.float64)
    valid = np.isfinite(sample).all(axis=0)
    sample[:, ~valid] = 0
    sample -= sample[0]
    sample -= sample.mean(axis=0)
    norm = np.sqrt(np.einsum("ij,ij->j", sample, sample))
    sample /= np.where(norm > 0, norm, 1.)
    p = X.shape[1]
    is_start = np.ones(p, dtype=bool)
    for width in (2, 4, 8):
        n = p // width
        if not n:
            break
        span = n * width
        block = sample[:, :span].reshape(len(sample), n, width)
        center = block[:, :, (width - 1) // 2]
        corr = np.einsum("nbc,nb->bc", block, center)
        accepted = np.min(corr, axis=1) >= .95
        is_start[:span].reshape(n, width)[accepted, 1:] = False
    starts = np.flatnonzero(is_start)
    stops = np.r_[starts[1:], p]
    return starts, stops


def _geometry(X, y, count, task, random_state, groups, ordered_groups):
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    if p == 1:
        return X, y, np.array([], dtype=np.intp), np.array([], dtype=np.intp), {
            "band_starts": np.array([0]), "band_stops": np.array([1]), "folds": []}
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    left, right = _draw(left, 48, rng), _draw(right, 48, rng)
    # Use the fit-side sketch alone, also on forward chronological splits.
    if len(left) < 4:
        starts, stops = np.arange(p), np.arange(1, p + 1)
    else:
        starts, stops = _partition(X, left)
    folds = [dict(fit_rows=left, held_rows=right)]
    if not ordered_groups:
        folds.append(dict(fit_rows=right, held_rows=left))
    return X, y, left, right, dict(band_starts=starts, band_stops=stops, folds=folds)


def band_center_from_training(X, y, *, task="classification", random_state=1129,
                              groups=None, ordered_groups=False):
    """Choose the central observed channel of each training-correlation band."""
    _, _, _, _, info = _geometry(X, y, None, task, random_state, groups, ordered_groups)
    cols = (info["band_starts"] + info["band_stops"] - 1) // 2
    return cols, dict(info, budget=len(cols))


def band_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                groups=None, ordered_groups=False, backend="auto"):
    """Return one voted representative per supported local spectral band.

    Input columns must follow their acquisition axis. Bin widths are dyadic,
    at most eight. A bin can span multiple channels only when every channel
    in the fit-side sketch correlates at least .95 with its geometric center.
    Singletons preserve coverage where local smoothness cannot be established.
    Representatives maximize the lesser of held squared-loss reductions for
    adding the channel to the intercept and to a model containing its adjacent
    paired channel. When p is odd the last unpaired channel has identical
    endpoint gains. Ties favor proximity to the middle, then the lower index.
    Fixed widths prioritize the same representatives, then other high scores.
    """
    if backend not in ("auto", "numpy", "native"):
        raise ValueError("backend must be auto, numpy or native")
    X, y, left, right, info = _geometry(X, y, count, task, random_state, groups, ordered_groups)
    p = X.shape[1]
    indices = np.arange(p)
    starts, stops = info["band_starts"], info["band_stops"]
    if p == 1:
        zero, one = np.zeros(1), np.zeros(1)
        used = "none"
    else:
        mate = np.arange(p, dtype=np.int64) ^ 1
        if p % 2:
            mate[-1] = p - 1
        rows = np.ascontiguousarray(np.concatenate((left, right)), dtype=np.int64)
        votes, used = _endpoints(X, rows, len(left), mate, y, task == "classification",
                                 len(info["folds"]), backend)
        zero, one = votes.mean(axis=0)
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale else scores
    block = np.repeat(np.arange(len(starts)), stops - starts)
    distance = np.abs(2 * indices - (starts + stops - 1)[block])
    ranked = np.lexsort((indices, distance, -relative, block))
    cols = ranked[starts]
    if count is not None and count != len(cols):
        other = np.ones(p, dtype=bool)
        other[cols] = False
        cols = np.lexsort((indices, distance, -relative, other))[:count]
    return np.sort(cols), dict(info, budget=len(cols), scores=scores, zero_peer_scores=zero,
                               one_peer_scores=one, peer_window=(0, min(p - 1, 1)), backend=used)
