"""Feature-level quota MinPI for refitted ridge models on sampled coalitions.

All columns use the scale supplied by the caller. For comparable coefficient
tie scores, fit a standardizer on the fitting rows before calling rank_features.
The intercept is always fitted and unpenalized. Ridge uses summed squared
residuals plus ridge * squared coefficient norm, for each response coordinate.
"""
from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class Ranking:
    order: np.ndarray
    minimum: np.ndarray | None
    scores: np.ndarray
    tie_scores: np.ndarray
    work: dict
    witness: list


def coalition_batches(n_features, rounds, seed):
    """Yield disjoint base/target arrays. Every target addition is individual."""
    p, rounds = index(n_features), index(rounds)
    if p < 1 or rounds < 1:
        raise ValueError("At least one feature and one sampling round are required")
    b = min(4, p)
    widths = range(1, 4) if b == 4 else range(b)
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        groups = [np.sort(g) for g in np.array_split(rng.permutation(p), b)]
        for target, candidates in enumerate(groups):
            for width in widths:
                blocks = [groups[(target + r + 1) % b] for r in range(width)]
                base = np.sort(np.concatenate(blocks)) if blocks else np.empty(0, dtype=int)
                yield base, candidates


def _targets(train, vote, task):
    train, vote = np.asarray(train), np.asarray(vote)
    if train.ndim != 1 or vote.ndim != 1:
        raise ValueError("Supply one target per example")
    if task == "regression":
        y, z = train.astype(float)[:, None], vote.astype(float)[:, None]
        if not (np.isfinite(y).all() and np.isfinite(z).all()):
            raise ValueError("Regression targets must be finite")
    elif task in {"classification", "binary", "multiclass"}:
        classes, codes = np.unique(train, return_inverse=True)
        if not np.isin(vote, classes).all():
            raise ValueError("A voting class is absent from the fitting set")
        y = np.eye(len(classes))[codes]
        z = (vote[:, None] == classes).astype(float)
    else:
        raise ValueError(f"Unknown task: {task}")
    return y, z


def _utility(loss, scale):
    return 1.0 / (1.0 + np.maximum(loss, 0.0) / scale)


def rank_features(X_fit, y_fit, X_vote, y_vote, *, task, seed=42,
                  method="minpi", rounds=4, alpha=.95, ridge=1.):
    """Rank columns descending by sampled MinPI, mean utility gain, or coefficient.

    minpi returns an upper bound on the all-coalition minimum. Only its -1
    certifies the exact minimum. mean uses every scheduled continuous marginal.
    coefficient uses the same full ridge fit as the MinPI tie-break.
    """
    if method not in {"minpi", "mean", "coefficient"}:
        raise ValueError(f"Unknown ranking method: {method}")
    if not (np.isfinite(ridge) and ridge > 0 and 0 < alpha <= 1 and index(rounds) >= 1):
        raise ValueError("Require ridge > 0, 0 < alpha <= 1, and at least one round")
    x, z = np.asarray(X_fit, dtype=float), np.asarray(X_vote, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or not x.shape[1] or x.shape[1] != z.shape[1]:
        raise ValueError("Fitting and voting matrices must have the same positive width")
    if len(x) != len(y_fit) or len(z) != len(y_vote) or not min(len(x), len(z)):
        raise ValueError("Supply matching nonempty fitting and voting sets")
    if not (np.isfinite(x).all() and np.isfinite(z).all()):
        raise ValueError("Impute missing values on the fitting split first")
    y, truth = _targets(y_fit, y_vote, task)
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    x, z, y, truth = x - mean_x, z - mean_x, y - mean_y, truth - mean_y
    p = x.shape[1]
    ids = np.arange(p)
    gram = x.T @ x
    gram.flat[::p + 1] += ridge
    cross = x.T @ y
    full = cho_solve(cho_factor(gram, check_finite=False), cross, check_finite=False)
    tie = np.linalg.norm(full, axis=1)
    work = dict(full_fits=1, base_fits=0, factorizations=1, scheduled_marginals=0,
                evaluated_marginals=0, absorbed_skips=0, losing_base_skips=0, boundary_refits=0)
    witnesses = [None] * p
    if method == "coefficient":
        return Ranking(np.lexsort((ids, -tie)), None, tie.copy(), tie, work, witnesses)

    scale = np.einsum("ij,ij->", truth, truth) / len(z)
    if scale == 0:
        scale = 1.
    residual = truth - z @ full
    full_utility = _utility(np.einsum("ij,ij->", residual, residual) / len(z), scale)
    quota = alpha * full_utility
    minima = np.ones(p, dtype=np.int8)
    gain_sum = np.zeros(p)
    exposures = np.zeros(p, dtype=int)
    for base, targets in coalition_batches(p, rounds, seed):
        work["scheduled_marginals"] += len(targets)
        active = targets if method == "mean" else targets[minima[targets] != -1]
        work["absorbed_skips"] += len(targets) - len(active)
        if not len(active):
            continue
        work["base_fits"] += 1
        if len(base):
            factor = cho_factor(gram[np.ix_(base, base)], check_finite=False)
            beta = cho_solve(factor, cross[base], check_finite=False)
            work["factorizations"] += 1
            zbase = z[:, base]
            residual = truth - zbase @ beta
        else:
            residual = truth
        loss = np.einsum("ij,ij->", residual, residual) / len(z)
        utility = _utility(loss, scale)
        before = utility >= quota
        if method == "minpi" and not before:
            previous_count = len(active)
            active = active[minima[active] != 0]
            work["losing_base_skips"] += previous_count - len(active)
        if not len(active):
            continue
        work["evaluated_marginals"] += len(active)
        denom = gram[active, active]
        slope_rhs = cross[active]
        direction = z[:, active]
        if len(base):
            links = gram[np.ix_(base, active)]
            project = cho_solve(factor, links, check_finite=False)
            denom = denom - np.einsum("ij,ij->j", links, project)
            slope_rhs = slope_rhs - links.T @ beta
            direction = direction - zbase @ project
        slopes = slope_rhs / denom[:, None]
        dot = np.einsum("ij,ij->i", direction.T @ residual, slopes)
        squared = np.einsum("ij,ij->j", direction, direction) * np.einsum("ij,ij->i", slopes, slopes)
        after = _utility(loss + (squared - 2. * dot) / len(z), scale)
        if len(base) == p - 1:
            after[:] = full_utility
        elif method == "minpi":
            # The expanded SSE can round to the other side of an exact quota
            # tie. Recompute close endpoints directly with the same loss
            # convention as the full reference; the voting threshold is fixed.
            near = np.flatnonzero(abs(after - quota) <= 64 * np.finfo(float).eps)
            for j in near:
                support = np.sort(np.append(base, active[j]))
                factor_j = cho_factor(gram[np.ix_(support, support)], check_finite=False)
                coef_j = cho_solve(factor_j, cross[support], check_finite=False)
                error = truth - z[:, support] @ coef_j
                after[j] = _utility(np.einsum("ij,ij->", error, error) / len(z), scale)
                work["factorizations"] += 1
                work["boundary_refits"] += 1
        if method == "mean":
            gain_sum[active] += after - utility
            exposures[active] += 1
        else:
            deltas = (after >= quota).astype(np.int8) - int(before)
            minima[active] = np.minimum(minima[active], deltas)
            for feature in active[deltas == -1]:
                witnesses[feature] = tuple(map(int, base))
    scores = minima if method == "minpi" else gain_sum / exposures
    return Ranking(np.lexsort((ids, -tie, -scores)), minima if method == "minpi" else None,
                   scores, tie, work, witnesses)
