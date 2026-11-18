"""Compiled paired minPI with a NumPy reference check at ambiguous cutoffs."""
import numpy as np

from minppi_mofs.native_pivotal_minpi import native_top_from_training


def _cutoff_reason(columns, scores, cost=None):
    """Flag small ranking gaps and cumulative pruning costs close to their threshold."""
    p = len(scores)
    removed = np.ones(p, dtype=bool)
    removed[columns] = False
    # An absolute floor also covers near-zero gains after cancellation. This
    # is a floating-point sensitivity check, not an error bound on ridge fits.
    scale = max(1., np.max(np.abs(scores)))
    tol = 64 * np.finfo(np.float64).eps * scale
    weakest_kept = np.min(scores[columns])
    if weakest_kept - np.max(scores, where=removed, initial=-np.inf) <= tol:
        return "score_boundary"
    if cost is None:
        return None
    count = p - len(columns)
    mass = cost.sum()
    available = .02 * mass
    used = np.sum(cost, where=removed)
    # Cumulative addition and 2% of the total can each move under reordering.
    # At a forced lower or upper width only the feasible direction matters.
    cost_tol = 64 * p * np.finfo(np.float64).eps * max(1., mass)
    if count > 1 and available - used <= cost_tol:
        return "budget_boundary"
    if count < p // 2:
        # Near-tied first retained columns can have different positive costs.
        tied = (scores <= weakest_kept + tol) & ~removed
        next_cost = np.min(cost, where=tied, initial=np.inf)
        if used + next_cost - available <= cost_tol:
            return "budget_boundary"
    return None


def stable_native_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                    groups=None, ordered_groups=False, backend="auto"):
    """Return the support, recomputing ambiguous native selections with NumPy.

    Both paths use identical training rows, column matching, peer counts, ranked
    gain and width rule. A possible cutoff tie triggers a NumPy recomputation
    with the original seed. For automatic width the guard also checks whether
    rounding could move the cumulative cost through the 2% pruning threshold.
    Backend auto uses NumPy for inputs unsupported by the native library.
    """
    params = dict(task=task, random_state=random_state, groups=groups, ordered_groups=ordered_groups)
    columns, info = native_top_from_training(X, y, count, backend=backend, **params)
    reason = None
    if info["backend"] == "native":
        reason = _cutoff_reason(columns, info["scores"], info.get("deletion_cost"))
        if reason is not None:
            columns, info = native_top_from_training(X, y, count, backend="numpy", **params)
    info.update(selection_backend=info["backend"], guard_reason=reason or "none")
    return columns, info
