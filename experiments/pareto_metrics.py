"""
Pareto-front quality indicators for the MOFS main table (ICOMP 2026 §6).

Adds the indicators the reviewers asked for (IGD, IGD+, Spacing) on top of the
hypervolume already computed in ``run_main_table.py``.  Every indicator operates
on a *fitness* array with the three minimisation objectives used project-wide::

    column 0 : |F|         cardinality              (minimise)
    column 1 : 1 - quality 1-AUC (classification) or 1-R^2 (regression)
    column 2 : time (s)    training time            (minimise)

Because the three objectives live on very different scales (|F| up to 200,
1-quality in [0,1], time in seconds), they are normalised to [0,1] per dataset
using the reference front's per-objective range before any distance metric is
computed; otherwise |F| and time would dominate.

Conventions
-----------
* Minimisation on every objective.
* IGD / IGD+ / Spacing: **lower is better**.
* The reference (approximate true Pareto) front is the non-dominated subset of
  the union of every method's front across all seeds on a dataset -- the
  standard construction when the true front is unknown (Li & Yao, 2019).

This module is task-agnostic: it only sees the fitness columns, so it works
unchanged for the regression datasets (where column 1 is 1-R^2).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "non_dominated",
    "build_reference_front",
    "normalizer",
    "igd",
    "spacing",
]


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of *points* (minimisation, all objectives)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        dominated_by = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        dominated_by[i] = False
        if dominated_by.any():
            keep[i] = False
    return pts[keep]


def build_reference_front(fronts: list[np.ndarray]) -> np.ndarray:
    """Approximate true Pareto front = non-dominated subset of the union of fronts."""
    valid = [np.asarray(f, float) for f in fronts if f is not None and len(f)]
    if not valid:
        return np.empty((0, 3))
    union = np.unique(np.vstack(valid), axis=0)
    return non_dominated(union)


def normalizer(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-objective (lo, span) from the reference front; span>0 guaranteed."""
    ref = np.asarray(reference, float)
    lo = ref.min(axis=0)
    hi = ref.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo, span


def _norm(points: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (np.asarray(points, float) - lo) / span


def igd(
    front: np.ndarray,
    reference: np.ndarray,
    *,
    plus: bool = False,
    lo: np.ndarray | None = None,
    span: np.ndarray | None = None,
) -> float:
    """(Inverted) Generational Distance from *reference* to *front*.

    Parameters
    ----------
    plus:
        If True, compute IGD+ (weakly Pareto-compliant): the distance from a
        reference point ``z`` to a solution ``a`` counts only the objectives on
        which ``a`` is *worse* than ``z`` (``max(a-z, 0)``).
    lo, span:
        Normalisation constants from :func:`normalizer`.  Strongly recommended.

    Returns lower-is-better; ``inf`` if the front is empty.
    """
    A = np.asarray(front, float)
    Z = np.asarray(reference, float)
    if len(A) == 0 or len(Z) == 0:
        return float("inf")
    if lo is not None:
        A = _norm(A, lo, span)
        Z = _norm(Z, lo, span)
    total = 0.0
    for z in Z:
        diff = np.maximum(A - z, 0.0) if plus else (A - z)
        total += np.sqrt((diff ** 2).sum(axis=1)).min()
    return float(total / len(Z))


def spacing(
    front: np.ndarray,
    *,
    lo: np.ndarray | None = None,
    span: np.ndarray | None = None,
) -> float:
    """Schott's spacing -- uniformity of the front (lower = more uniform).

    Well-defined for any number of objectives, unlike Deb's 2-objective spread.
    """
    A = np.asarray(front, float)
    if len(A) < 2:
        return 0.0
    if lo is not None:
        A = _norm(A, lo, span)
    nn = np.empty(len(A))
    for i in range(len(A)):
        dist = np.sqrt(((A - A[i]) ** 2).sum(axis=1))
        dist[i] = np.inf
        nn[i] = dist.min()
    dbar = nn.mean()
    return float(np.sqrt(((dbar - nn) ** 2).sum() / (len(A) - 1)))
