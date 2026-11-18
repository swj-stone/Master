"""Independent dense refits for checking train-only complementary feature selection."""
from importlib import import_module
from itertools import combinations

import numpy as np


# The preceding swap experiment independently reproduced the native row sampler.
sampled_rows = import_module(
    "example-paper.Strategic Feature Selection.real-data-check.swap_check"
).sampled_rows


def standardize(X, y, task, seed):
    rows = sampled_rows(len(y), seed)
    z = np.asarray(X[rows], dtype=np.float64)
    z = z - z.mean(axis=0)
    scale = z.std(axis=0)
    z /= np.where(scale > 0, scale, 1)
    response = y[rows]
    if task == "classification":
        _, labels = np.unique(response, return_inverse=True)
        response = np.eye(labels.max() + 1)[labels]
    else:
        response = response.astype(np.float64)[:, None]
        amplitude = np.abs(response).max()
        if amplitude:
            response /= amplitude
        response -= response.mean(axis=0)
        if response.std() > 0:
            response /= response.std()
    response -= response.mean(axis=0)
    return z, response


class Refits:
    """Score each candidate by solving its complete ridge system from scratch."""

    def __init__(self, z, response):
        self.p = z.shape[1]
        self.gram = z.T @ z + np.sqrt(len(z)) * np.eye(self.p)
        self.rhs = z.T @ response
        self.cache = {(): 0.0}

    def utility(self, columns):
        ids = tuple(sorted(columns))
        if ids not in self.cache:
            rhs = self.rhs[list(ids)]
            coefficients = np.linalg.solve(self.gram[np.ix_(ids, ids)], rhs)
            self.cache[ids] = float(np.sum(rhs * coefficients))
        return self.cache[ids]

    def best(self, prefix, choices):
        """Lexicographic tie breaking. Compare improvements over the same prefix."""
        base = self.utility(prefix)
        high = -np.inf
        winner = None
        for suffix in choices:
            gain = self.utility(tuple(prefix) + suffix) - base
            if winner is None or gain - high > 32 * np.finfo(float).eps * max(abs(gain), abs(high)):
                high, winner = gain, suffix
        return winner

    def forward(self, initial, width):
        path = list(initial)
        while len(path) < width:
            remaining = ((j,) for j in range(self.p) if j not in path)
            path.extend(self.best(path, remaining))
        return path

    def best_pair(self):
        return self.best((), combinations(range(self.p), 2))


def mechanism_supports(ref, old_columns, old_ranks):
    k = len(old_columns)
    if k == 1:
        pair = ref.best((), ((j,) for j in range(ref.p)))
    else:
        pair = ref.best_pair()
    ranking = np.argsort(-old_ranks, kind="stable")
    pair_marginal = list(pair) + [j for j in ranking if j not in pair][:k-len(pair)]
    anchor = ref.best((), ((j,) for j in range(ref.p)))
    paths = {
        "old": old_columns.tolist(),
        "pair_old_fill": pair_marginal,
        "anchor_forward": ref.forward(anchor, k),
        "pair_forward": ref.forward(pair, k),
    }
    return {arm: np.asarray(path, dtype=np.int64) for arm, path in paths.items()}
