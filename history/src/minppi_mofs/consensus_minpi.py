"""Consensus ranking of perturbed minPI paths with group-aware training calibration."""
from math import ceil
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from minppi_mofs.environment_minpi import InsufficientEnvironmentsError, _environment_states
from minppi_mofs.stable_minpi import _calibrate_supports, _partition
from minppi_mofs.utility_minpi import _utility_inputs


def _options(y, groups, ordered_groups, environment_groups, random_state, n_paths):
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
        raise ValueError("Use an integer seed in [0, 2**32)")
    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, Integral) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if not isinstance(environment_groups, (bool, np.bool_)):
        raise ValueError("environment_groups must be boolean")
    if ordered_groups and (groups is None or not environment_groups):
        raise ValueError("ordered_groups requires genuine training environments")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd.isna(groups).any():
            raise ValueError("Groups must be complete and aligned with training rows")
    return groups


def _rank_path(X, y, task, groups, ordered_groups, seed, genuine):
    p = X.shape[1]
    sizes = [0] + [1 << k for k in range((p - 1).bit_length())]
    states, folds = _environment_states(X, y, task, groups, ordered_groups, seed, sizes[-1])

    def gains():
        ballots = [state.scores() for state in states]
        return np.minimum.reduce(ballots) if genuine else np.mean(ballots, axis=0)

    available = np.ones(p, dtype=bool)
    current = gains()
    order = []
    for size, end in zip(sizes, sizes[1:] + [p]):
        if size:
            if len(block) > 1:
                for state in states:
                    state.append(block[:-1])
                previous = gains()
            else:
                previous = current
            for state in states:
                state.append(block[-1:])
            current = gains()
            score = np.minimum(current, previous)
        else:
            score = current
        pool = np.flatnonzero(available)
        block = pool[np.argsort(-score[pool], kind="stable")[:end-size]]
        order.extend(block)
        available[block] = False
    return np.asarray(order, dtype=int), folds


def consensus_path_from_training(X, y, counts, *, task="classification", random_state=1129,
                                 groups=None, ordered_groups=False, environment_groups=True, n_paths=3):
    """Choose nested top widths by the average feature rank across seeded minPI paths.

    Each path uses a deterministic child of random_state. Genuine deployment
    environments use their minimum normalized held gain. For iid replicates
    and groups used solely to keep duplicate records together, the path
    averages normalized held gains instead. Both variants take the minimum
    across adjacent selected support sizes. Groups always constrain splits.
    Integer rank sums give exactly the same ordering as mean percentile rank;
    exact ties retain input order. Scores are recomputed for the whole
    feature order independently of the requested widths.
    """
    counts = tuple(counts)
    if not counts:
        raise ValueError("Provide at least one feature budget")
    X, y = _utility_inputs(X, y, counts[0], task)
    p = X.shape[1]
    if any(isinstance(h, (bool, np.bool_)) or not isinstance(h, Integral) or not 1 <= h <= p for h in counts):
        raise ValueError("Budgets must be integers in [1,p]")
    groups = _options(y, groups, ordered_groups, environment_groups, random_state, n_paths)
    genuine = groups is not None and environment_groups
    info = {"aggregation": "min" if genuine else "mean", "paths": [], "distinct_paths": 0}
    if min(counts) == p:
        return {p: np.arange(p)}, info
    children = np.random.SeedSequence(int(random_state)).spawn(n_paths)
    ranks = np.arange(p, dtype=np.int64)
    total = np.zeros(p, dtype=np.int64)
    seen = set()
    for child in children:
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        order, folds = _rank_path(X, y, task, groups, ordered_groups, seed, genuine)
        total[order] += ranks
        seen.add(order.tobytes())
        info["paths"].append({"seed": seed, "order": order.tolist(), "folds": [
            {k: idx.tolist() for k, idx in fold.items()} for fold in folds]})
    info["distinct_paths"] = len(seen)
    order = np.argsort(total, kind="stable")
    return {int(h): np.sort(order[:h]) for h in counts}, info


def consensus_top_from_training(X, y, count, **kwargs):
    """Return one width from the same consensus order as the multi-width API."""
    supports, info = consensus_path_from_training(X, y, (count,), **kwargs)
    return supports[count], info


class ConsensusMinPISelector(SelectorMixin, BaseEstimator):
    """Calibrate the width of a frozen consensus support on separate training environments."""

    def __init__(self, *, task="classification", random_state=1129, tolerance=.02,
                 fractions=(.10, .50, .75), n_paths=3):
        self.task = task
        self.random_state = random_state
        self.tolerance = tolerance
        self.fractions = fractions
        self.n_paths = n_paths

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y, *, groups=None, ordered_groups=False, environment_groups=True):
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("Feature names must be unique")
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        X, y = _utility_inputs(X, y, 1, self.task)
        groups = _options(y, groups, ordered_groups, environment_groups, self.random_state, self.n_paths)
        if isinstance(self.tolerance, (bool, np.bool_)) or not isinstance(self.tolerance, Real) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be a nonnegative finite number")
        if not isinstance(self.fractions, (list, tuple, np.ndarray)) or len(self.fractions) < 1:
            raise ValueError("fractions must be a nonempty sequence")
        if any(isinstance(f, (bool, np.bool_)) or not isinstance(f, Real) or not 0 < f <= 1 for f in self.fractions):
            raise ValueError("Each fraction must be in (0, 1]")
        p = X.shape[1]
        widths = sorted({ceil(f * p) for f in self.fractions} - {p})
        chosen = np.arange(p)
        info = {"rule": "full", "width": p, "reason": "no_compressed_width",
                "split": "ordered" if ordered_groups else ("iid" if groups is None else "group"),
                "tolerance": float(self.tolerance), "calibration_mode": "deployment",
                "discovery_rows": [], "validation_rows": [], "full_scores": [], "validation_candidates": [],
                "path_row_index_basis": "relative_to_discovery_rows"}
        if widths:
            try:
                disc, held = _partition(y, groups, ordered_groups, self.task, self.random_state)
                info.update(discovery_rows=disc.tolist(), validation_rows=[v.tolist() for v in held])
                supports, info["consensus_path"] = consensus_path_from_training(
                    X[disc], y[disc], widths, task=self.task, random_state=self.random_state,
                    groups=None if groups is None else groups[disc], ordered_groups=ordered_groups,
                    environment_groups=environment_groups, n_paths=self.n_paths)
                chosen, calibrated = _calibrate_supports(
                    X, y, disc, held, supports, self.task, self.random_state, self.tolerance, rule="consensus")
                info.update(calibrated)
            except InsufficientEnvironmentsError as exc:
                info.update(reason="insufficient_environments", detail=str(exc))
        self.support_ = np.zeros(p, dtype=bool)
        self.support_[chosen] = True
        self.diagnostics_ = info
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_
