"""Two-column minPI tournaments with a bounded training-row sample."""
import numpy as np

from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _take(rows, cap, rng):
    return rows if len(rows) <= cap else rng.choice(rows, cap, replace=False)


def _halves(y, task, groups, ordered, cap, rng):
    if len(y) < 2:
        raise ValueError("The tournament needs at least two training rows")
    if ordered:
        last = groups == groups[groups.argmax()]
        if last.all():
            raise ValueError("At least two distinct training groups are required")
        return _take(np.flatnonzero(~last), cap, rng), _take(np.flatnonzero(last), cap, rng)
    sample = rng.choice(len(y), min(len(y), 2 * cap), replace=False)
    if groups is None:
        if task == "classification":
            sample = sample[np.argsort(y[sample], kind="stable")]
        return sample[::2], sample[1::2]
    unique = np.unique(groups[sample])
    if len(unique) == 1:
        rest = np.flatnonzero(groups != unique[0])
        if not len(rest):
            raise ValueError("At least two distinct training groups are required")
        return sample[:cap], _take(rest, cap, rng)
    left = np.isin(groups[sample], rng.permutation(unique)[::2])
    return sample[left][:cap], sample[~left][:cap]


def _center_unit(values):
    """Center and scale each column to unit Euclidean length within this sample."""
    finite = np.isfinite(values)
    if finite.all():
        # Anchor before summing so repeated constants center to exactly zero.
        values -= values[0].copy()
        values -= values.mean(axis=0)
    else:
        anchor = values[np.argmax(finite, axis=0), np.arange(values.shape[1])]
        anchor = np.where(np.isfinite(anchor), anchor, 0.)
        np.subtract(values, anchor, out=values, where=finite)
        values[~finite] = 0.
        avg = values.sum(axis=0) / np.maximum(1, finite.sum(axis=0))
        np.subtract(values, avg, out=values, where=finite)
    length = np.sqrt(np.einsum("ij,ij->j", values, values))
    values /= np.where(length > 0., length, 1.)
    return values


def _effects(X, y, rows, active, classes):
    # Advanced indexing creates a private matrix for in-place centering.
    values = np.asarray(X[np.ix_(rows, active)], dtype=np.float64)
    if classes is None:
        target = np.asarray(y[rows, None], dtype=np.float64)
    else:
        target = (y[rows, None] == classes).astype(float)
    return _center_unit(values), _center_unit(target)


def _pair_scores(train, check, target, query):
    """Exact empty- and present-mate margins of tr(A_S.T inv(R_S) B_S)."""
    a, b = train.T @ target, check.T @ query
    alone = np.einsum("ij,ij->i", a, b)
    n_pairs = len(alone) // 2
    even, odd = slice(0, 2 * n_pairs, 2), slice(1, 2 * n_pairs, 2)
    # Unit ridge after standardizing the training columns to variance one.
    rho = np.einsum("ij,ij->j", train[:, even], train[:, odd]) * (len(train) / (len(train) + 1.))
    cross = np.einsum("ij,ij->i", a[even], b[odd]) + np.einsum("ij,ij->i", a[odd], b[even])
    joint = (alone[even] + alone[odd] - rho * cross) / (1. - rho**2)
    scores = alone.copy()
    scores[even] = np.minimum(alone[even], joint - alone[odd])
    scores[odd] = np.minimum(alone[odd], joint - alone[even])
    return scores


def tournament_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                 groups=None, ordered_groups=False):
    """Select columns by the weaker of the empty- and full-peer signed votes.

    Within each independently paired block, let A and B be the correlations
    with target coordinates on two disjoint training samples. The block game
    sums the held correlations along the fitted regularized directions,
    v(S) = tr(A_S.T inv(R_S) B_S). R has diagonal one and off-diagonal
    correlations from the fitting sample shrunk by n/(n+1). This game uses
    label association on held training rows, not held prediction accuracy.
    Both sides center and scale within their own sample. Classification uses
    one normalized indicator per class. Each original row contributes its
    signed feature-target product to the held vote, linearly.

    For h >= p/2, an explicit budget compares 2(p-h) randomly permuted
    columns in pairs and retains every unexamined column. The loser in each
    pair is deleted. Lower budgets keep the highest-scoring pair winners.
    Without a count, compare all pairs, delete nonpositive losers and spend
    at most 2% of total positive score on additional positive-score losers.
    This budget is a surrogate mass threshold, with no held-score guarantee.
    At least one column is dropped when p > 1 under automatic selection.
    """
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    strategy = "forward" if ordered_groups else "group" if groups is not None else "row"
    info = {"strategy": strategy, "peer_window": (0, p - 1), "folds": [],
            "active_columns": np.empty(0, dtype=int), "pairs": np.empty((0, 2), dtype=int),
            "scores": np.empty(0, dtype=float)}
    if p == 1 or count == p:
        return np.arange(p), dict(info, budget=p)

    rng = np.random.default_rng(random_state)
    size = p if count is None else min(p, 2 * (p - count))
    active = rng.permutation(p)[:size]
    cap = min(64, max(16, 65_536 // size))
    fit, held = _halves(y, task, groups, ordered_groups, cap, rng)
    classes = np.unique(np.concatenate((y[fit], y[held]))) if task == "classification" else None
    train, target = _effects(X, y, fit, active, classes)
    check, query = _effects(X, y, held, active, classes)
    scores = _pair_scores(train, check, target, query)

    pairs = np.arange(size // 2 * 2).reshape(-1, 2)
    left, right = pairs.T
    left_loses = (scores[left] < scores[right]) | ((scores[left] == scores[right]) &
                                                   (active[left] > active[right]))
    losers = np.where(left_loses, left, right)
    keep = np.ones(p, dtype=bool)
    if count is None:
        order = losers[np.lexsort((-active[losers], scores[losers]))]
        mass = np.maximum(0., scores)
        deletions = np.searchsorted(np.cumsum(mass[order]), .02 * mass.sum(), side="right")
        keep[active[order[:max(1, deletions)]]] = False
    else:
        keep[active[losers]] = False
        extra = int(keep.sum()) - count
        if extra > 0:
            potential = np.flatnonzero(keep[active])
            weakest = np.lexsort((-active[potential], scores[potential]))[:extra]
            keep[active[potential[weakest]]] = False

    cols = np.flatnonzero(keep)
    info.update(budget=len(cols), active_columns=active, pairs=active[pairs], scores=scores,
                folds=[{"fit_rows": fit, "held_rows": held}])
    return cols, info
