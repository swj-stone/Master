"""Conditional ridge improvement votes on a pilot-stratified training sketch."""
import operator
import numpy as np
from scipy.linalg import solve


def _standardize(x, center=None, scale=None):
    observed = np.isfinite(x)
    x = np.where(observed, x, 0.)
    if center is None:
        center = x.sum(axis=0) / np.maximum(1, observed.sum(axis=0))
    x = np.where(observed, x - center, 0.)
    if scale is None:
        scale = np.sqrt(np.einsum("ij,ij->j", x, x) / len(x))
        scale[scale < 1e-12] = 1.
    return x / scale, center, scale


def _pilot(x, target, pool):
    x, center, scale = _standardize(x)
    pool, _, _ = _standardize(pool, center, scale)
    ybar = target.mean(axis=0)
    centered = target - ybar
    if x.shape[1] < len(x):
        gram = x.T @ x
        gram.flat[::len(gram)+1] += 1.
        beta = solve(gram, x.T @ centered, assume_a="pos", check_finite=False)
    else:
        gram = x @ x.T
        gram.flat[::len(gram)+1] += 1.
        beta = x.T @ solve(gram, centered, assume_a="pos", check_finite=False)
    return pool @ beta + ybar


def _sample(X, y, task, seed):
    rng = np.random.default_rng(seed)
    nfit = min(128, max(2, len(y)//3))
    subset = rng.choice(len(y), size=min(len(y), nfit+512), replace=False)
    fit, pool = subset[:nfit], subset[nfit:]
    if task == "classification":
        classes, labels = np.unique(y, return_inverse=True)
        target = np.eye(len(classes))[labels[subset]]
    else:
        target = np.asarray(y[subset], dtype=float)[:, None]
    prediction = _pilot(X[fit], target[:nfit], X[pool])
    if prediction.shape[1] > 1:
        two = np.partition(prediction, -2, axis=1)[:, -2:]
        difficulty = two[:, 1] - two[:, 0]
    else:
        difficulty = prediction.ravel()
    order = np.argsort(difficulty, kind="stable")
    q = min(256, len(pool))
    positions = ((np.arange(q) + .5) * len(pool) / q).astype(int)
    selected = order[positions]
    return fit, pool[selected], target[nfit:][selected]


def select_features(X, y, *, task="classification", random_state=1129, count=None, mode="conditional"):
    """Select finite original column IDs; fit and vote only on supplied training rows.

    Gain is the sum of individual row reductions in squared residual loss,
    less the change in the unit Ridge penalty. Two coalition sizes (empty
    and current selected prefix) are compared. The current context and its
    regularized least squares gain are updated by a Schur complement.
    """
    if mode not in {"marginal","conditional"} or task not in {"classification","regression"}:
        raise ValueError("Unsupported mode or task")
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or len(X) < 5 or X.shape[1] < 1 or y.shape != (len(X),):
        raise ValueError("Expected a numeric matrix with at least five rows and aligned labels")
    if task == "regression" and not np.isfinite(y).all():
        raise ValueError("Regression targets must be finite")
    p = X.shape[1]
    if count is not None and (isinstance(count, (bool,np.bool_)) or not 1 <= operator.index(count) <= p):
        raise ValueError("Invalid width")
    fit, rows, target = _sample(X, y, task, random_state)
    data, _, _ = _standardize(X[rows])
    target = target - target.mean(axis=0)
    rhs = data.T @ target
    diag = np.einsum("ij,ij->j", data, data) + 1.
    initial = np.einsum("ij,ij->i", rhs, rhs) / diag
    order = np.argsort(-initial, kind="stable")
    if mode == "marginal" and count is not None:
        return np.sort(order[:count]), {"width":int(count)}
    candidates = np.sort(order[:max(256,2*len(rows))])
    data = np.asfortranarray(data[:, candidates])
    rhs = rhs[candidates].copy()
    diag = diag[candidates].copy()
    marginal = initial[candidates]
    active = np.ones(len(candidates), dtype=bool)
    limit = min(64, len(candidates), len(rows)-1, count if count is not None else p)
    factors = np.empty((limit,len(candidates)))
    path = []
    loss = float(np.square(target).sum())
    gains = []
    for step in range(limit):
        gain = np.einsum("ij,ij->i", rhs, rhs) / np.maximum(diag,1.)
        score = marginal if mode == "marginal" else np.minimum(marginal, gain)
        winner = int(np.argmax(np.where(active, score, -np.inf)))
        # Multiple-candidate search costs ~2 log(p) residual variance units.
        # This is a training criterion, not a test loss bound.
        threshold = loss * (1. + 2.*np.log(max(2, len(candidates)-step))) / max(1, len(rows)-step-1)
        if count is None and score[winner] <= threshold:
            break
        path.append(candidates[winner])
        gains.append(float(score[winner]))
        active[winner] = False
        root = np.sqrt(max(1.,diag[winner]))
        residual_cross = data[:,winner] @ data
        residual_cross[winner] += 1.
        if step:
            residual_cross -= factors[:step,winner] @ factors[:step]
        projection = residual_cross / root
        factors[step] = projection
        rhs -= projection[:,None] * (rhs[winner].copy()/root)
        diag -= projection**2
        loss = max(0.,loss - gain[winner])
    # A sketch with no detectable signal yields a full-width fallback.
    if not path and count is None:
        chosen = np.arange(p)
    else:
        chosen = np.array(path,dtype=int)
        if count is not None and len(chosen) < count:
            unused = np.ones(p,dtype=bool)
            unused[chosen] = False
            last = candidates[np.argsort(-np.minimum(marginal, np.einsum("ij,ij->i",rhs,rhs)/np.maximum(diag,1.)),kind="stable")]
            tail = np.r_[last[unused[last]],order[~np.isin(order,candidates)]]
            chosen = np.r_[chosen,tail[:count-len(chosen)]]
    return np.sort(chosen),dict(width=len(chosen), fit_rows=fit, vote_rows=rows, path=path, gains=gains)
