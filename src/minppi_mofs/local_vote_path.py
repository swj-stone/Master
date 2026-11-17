"""Backward feature paths from exact ridge-deletion quota votes.

Each feature is tested inside one small, medium and full active coalition.
The small/medium coalitions are nested random partitions of the active set.
The fitted intercept and feature scaling stay fixed throughout a path.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .sampled_linear_minpi import _targets, _utility


def quota_integral(with_feature, without_feature):
    """Uniform-quota integral of the minimum addition vote across contexts."""
    a, b = np.asarray(with_feature), np.asarray(without_feature)
    positive = np.maximum(0., a.min(axis=0) - b.max(axis=0))
    starts = np.where(b > a, a, 0.)
    ends = np.where(b > a, b, 0.)
    order = np.argsort(starts, axis=0)
    starts = np.take_along_axis(starts, order, axis=0)
    ends = np.take_along_axis(ends, order, axis=0)
    previous = np.vstack([np.zeros_like(ends[:1]), np.maximum.accumulate(ends, axis=0)[:-1]])
    return positive - np.maximum(ends - np.maximum(previous, starts), 0.).sum(axis=0)


class DeletionGame:
    """Exact utilities of every single-feature deletion in one factorization."""

    def __init__(self, x, y, z, truth, ridge=1.):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        x, z, y, truth = (np.asarray(a, dtype=float) for a in (x, z, y, truth))
        if x.ndim != 2 or z.ndim != 2 or z.shape[1] != x.shape[1] or not x.shape[1]:
            raise ValueError("Matching positive feature dimensions required")
        if not len(z) or not len(x) or len(truth) != len(z) or len(y) != len(x):
            raise ValueError("Matching nonempty sample dimensions required")
        y, truth = y.reshape(len(y), -1), truth.reshape(len(z), -1)
        if truth.shape[1] != y.shape[1]:
            raise ValueError("Response dimensions differ")
        mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
        x, y = x - mean_x, y - mean_y
        z, truth = z - mean_x, truth - mean_y
        p = x.shape[1]
        self.gram = x.T @ x
        self.gram.flat[::p + 1] += ridge
        self.cross = x.T @ y
        self.vote_gram = z.T @ z / len(z)
        self.vote_cross = z.T @ truth / len(z)
        self.loss_null = np.sum(truth**2) / len(z)
        self.scale = self.loss_null or 1.
        self.factors = self.endpoints = 0

    def edges(self, cols, keep=None):
        gram = self.gram[np.ix_(cols, cols)]
        factor = cho_factor(gram, check_finite=False)
        if keep is None or np.array_equal(keep,np.arange(len(cols))):
            keep = np.arange(len(cols))
            inv = cho_solve(factor, np.eye(len(cols)), check_finite=False)
            beta = inv @ self.cross[cols]
        else:
            unit = np.zeros((len(cols),len(keep)))
            unit[keep,np.arange(len(keep))] = 1.
            inv = cho_solve(factor, unit, check_finite=False)
            beta = cho_solve(factor, self.cross[cols], check_finite=False)
        slope = beta[keep] / inv[keep,np.arange(len(keep))][:,None]
        q = self.vote_gram[np.ix_(cols, cols)]
        h = self.vote_cross[cols]
        qb = q @ beta
        loss = self.loss_null + np.sum(beta * (qb - 2 * h))
        cross_term = np.sum(slope * (inv.T @ (h - qb)), axis=1)
        direction_norm = np.sum(inv * (q @ inv), axis=0)
        after = loss + 2 * cross_term + direction_norm * np.sum(slope**2, axis=1)
        self.factors += 1
        self.endpoints += len(keep)
        # Increase in the penalized training objective, conditional on the other columns.
        train_cost = np.sum(beta[keep] * slope, axis=1)
        return _utility(loss, self.scale), _utility(after, self.scale), train_cost


class GCVGame:
    """Generalized cross-validation utilities for all one-column deletions."""

    def __init__(self, x, y, ridge=1.):
        self.n, p = x.shape
        if self.n < 2 or p < 1 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("GCV needs two or more rows, at least one column and positive ridge")
        self.ridge = ridge
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
        self.gram = x.T @ x
        self.gram.flat[::p+1] += ridge
        self.cross = x.T @ y
        self.total = np.sum(y*y)
        self.scale = self.total / self.n or 1.
        self.factors = self.endpoints = 0

    def edges(self, cols, keep=None):
        gram = self.gram[np.ix_(cols, cols)]
        inv = cho_solve(cho_factor(gram, check_finite=False), np.eye(len(cols)), check_finite=False)
        cross = self.cross[cols]
        beta = inv @ cross
        trace = np.trace(inv)
        keep = np.arange(len(cols)) if keep is None else keep
        inv = inv[:,keep]
        diagonal = inv[keep,np.arange(len(keep))]
        slope = beta[keep] / diagonal[:, None]
        cost = np.sum(beta[keep] * slope, axis=1)
        norm = np.sum(beta * beta)
        inv_norm = np.sum(inv * inv, axis=0)
        reduced_norm = norm - 2*np.sum((inv.T @ beta) * slope, axis=1) + inv_norm*np.sum(slope*slope, axis=1)
        penalized_loss = self.total - np.sum(cross * beta)
        rss = penalized_loss - self.ridge * norm
        reduced_rss = penalized_loss + cost - self.ridge*reduced_norm
        df = 1 + len(cols) - self.ridge*trace
        reduced_df = df - 1 + self.ridge*inv_norm/diagonal
        gcv = self.n*rss / (self.n-df)**2
        reduced_gcv = self.n*reduced_rss / (self.n-reduced_df)**2
        self.factors += 1
        self.endpoints += len(keep)
        return _utility(gcv, self.scale), _utility(reduced_gcv, self.scale), cost


def path_order(x, y, z, truth, *, task, seed=42, mode="integral", ridge=1., schedule="quarter"):
    """Return most useful columns first, with less important batches removed en route."""
    modes = {"integral", "point", "minimum", "mean", "full_marginal", "train_cost",
             "signed", "signed_eager", "signed_train", "negative_only"}
    use_gcv = mode.endswith("_gcv")
    rule = mode.removesuffix("_gcv")
    if rule not in modes:
        raise ValueError(f"Unknown path mode {mode}")
    if schedule not in {"quarter","halve","once"}:
        raise ValueError(f"Unknown backward schedule {schedule}")
    y, truth = _targets(y, truth, task)
    game = (GCVGame(np.vstack([x,z]), np.vstack([y,truth]), ridge) if use_gcv
            else DeletionGame(x, y, z, truth, ridge))
    rng = np.random.default_rng(seed)
    active = np.arange(x.shape[1])
    removal, stages = [], []
    quota = None
    skipped = 0
    signed = rule.startswith("signed") or rule == "negative_only"
    while len(active) > 1:
        size = len(active)
        full, deleted, cost = game.edges(active)
        if quota is None:
            quota = .95 * full
        a, b = [np.full(size, full)], [deleted]
        negative = deleted > full
        left, right = deleted.copy(), np.full(size,full)
        if rule not in {"train_cost", "full_marginal"}:
            perm = rng.permutation(size)
            for count in (2, 4):
                if count > size:
                    continue
                wide, thin = np.empty(size), np.empty(size)
                for subset in np.array_split(perm, count):
                    if signed and rule != "signed_eager":
                        keep = np.flatnonzero(~negative[subset])
                        skipped += len(subset)-len(keep)
                    else:
                        keep = np.arange(len(subset))
                    if len(keep):
                        ids = subset[keep]
                        u, v, _ = game.edges(active[subset], keep)
                        if signed:
                            negative[ids] |= v > u
                            left[ids] = np.maximum(left[ids], v)
                            right[ids] = np.minimum(right[ids], u)
                        else:
                            wide[ids], thin[ids] = u, v
                if not signed:
                    a.append(wide)
                    b.append(thin)
        a, b = np.array(a), np.array(b)
        if signed:
            score = np.where(negative, -1, (right > left).astype(int) * (rule != "negative_only"))
        elif rule == "integral":
            score = quota_integral(a, b)
        elif rule == "point":
            score = ((a >= quota).astype(int) - (b >= quota)).min(axis=0)
        elif rule == "minimum":
            score = (a - b).min(axis=0)
        elif rule == "mean":
            score = (a - b).mean(axis=0)
        elif rule == "full_marginal":
            score = a[0] - b[0]
        else:
            score = cost
        # Re-score surviving features only at the boundaries of the chosen batches.
        if signed and rule != "signed_train":
            ranked = np.lexsort((active, cost, full-deleted, score))
        else:
            ranked = np.lexsort((active, cost, score))
        count = size-1 if schedule == "once" else max(1,size // (2 if schedule == "halve" else 4))
        drop = ranked[:count]
        removal.extend(active[drop].tolist())
        stages.append(dict(size=size, negative=int((score < 0).sum()), zero=int((score == 0).sum()),
                           positive=int((score > 0).sum()), dropped=active[drop].tolist()))
        active = np.delete(active, drop)
    return np.array(list(reversed(removal + active.tolist())), dtype=int), dict(
        factorizations=game.factors, deletion_edges=game.endpoints, skipped_edges=skipped, stages=stages)
