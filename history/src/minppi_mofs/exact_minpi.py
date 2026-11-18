"""Exact size-stratified pivotal votes on supervised hit/miss triplets."""

from collections.abc import Iterator
from math import ceil

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from minppi_mofs.piv2 import _folds
from minppi_mofs.piv3_compact_filter import _reference_geometry, _scaled, _triplets


def witness_margins(X, y, task, seed):
    """Return reference-fold standardized miss-minus-hit distances for training queries."""
    if task not in {'classification', 'regression'}:
        raise ValueError('task must be classification or regression')
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] == 0 or y.shape != (len(X),) or pd.isna(y).any():
        raise ValueError('Expected a nonempty matrix and complete aligned one-dimensional targets')
    if task == 'regression' and not np.isfinite(y.astype(float)).all():
        raise ValueError('Regression targets must be finite')
    folds = _folds(y, task, np.random.default_rng(seed), row_limit=256)
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError('Insufficient training observations for hit/miss witnesses')
    return _witness_margins_from_folds(X, y, task, *folds)


def _witness_margins_from_folds(X, y, task, ref, query, *, columns=None):
    """Build votes' distance differences on given rows and optional input columns."""
    ref_y, query_y = y[ref], y[query]
    if task == 'regression':
        cuts = np.unique(np.quantile(ref_y, [.2, .4, .6, .8]))
        ref_y, query_y = (np.searchsorted(cuts, part) for part in (ref_y, query_y))
    ref_idx = ref if columns is None else np.ix_(ref, columns)
    query_idx = query if columns is None else np.ix_(query, columns)
    raw = np.asarray(X[ref_idx], dtype=np.float32)
    valid = np.isfinite(raw)
    observed = np.maximum(valid.sum(axis=0), 1)
    center = np.where(valid, raw, 0.).sum(axis=0) / observed
    residual = np.where(valid, raw - center, 0.)
    scale = np.sqrt((residual * residual).sum(axis=0) / observed)
    scale = np.where(scale > 1e-7, scale, 1.).astype(np.float32)
    ref_x = _scaled(raw, center, scale)
    query_x = _scaled(X[query_idx], center, scale)
    weights = _reference_geometry(ref_x, ref_y)
    margins = _triplets(ref_x, query_x, ref_y, query_y, weights)
    if margins is None:
        raise ValueError('A witness lacks a reference hit or miss')
    return margins


def _log_choose_table(width, k):
    # A triangular recurrence avoids cancellation between large log-factorials.
    table = np.full((width + 1, k + 1), -np.inf)
    table[:, 0] = 0.
    for r in range(1, k + 1):
        counts = np.arange(r, width + 1)
        table[counts, r] = table[counts, r-1] + np.log((counts-r+1) / r)
    return table


def _margin_mass(plus, minus, zero, k, margin, log_choose, normalizer):
    """Three-color hypergeometric probability that sum equals margin."""
    mass = np.zeros(len(plus))
    for pairs in range((k - abs(margin)) // 2 + 1):
        pos = pairs + max(0, margin)
        neg = pairs + max(0, -margin)
        abstain = k - pos - neg
        log_mass = log_choose[plus, pos] + log_choose[minus, neg] + log_choose[zero, abstain]
        mass += np.exp(log_mass - normalizer)
    return mass


def _peer_counts(tiers, p):
    """Validate explicit peer counts, preserving their order for the layer scorer."""
    if tiers is None:
        return tuple(dict.fromkeys(min(k, p-1) for k in (0, 4, 16)))
    if isinstance(tiers, (int, np.integer)):
        tiers = (tiers,)
    try:
        counts = tuple(tiers)
    except TypeError as exc:
        raise ValueError("Peer counts must be integers in [0, p)") from exc
    if not counts or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                         or not 0 <= k < p for k in counts):
        raise ValueError("Peer counts must be a nonempty collection of integers in [0, p)")
    return tuple(map(int, counts))


def _selection_tiers(tiers, p):
    """Canonical set for a reusable selector; the largest layer supplies its first bound."""
    if isinstance(tiers, Iterator):
        raise ValueError("Use a reusable collection of peer counts, such as a tuple")
    return sorted(set(_peer_counts(tiers, p)), reverse=True)


def pivotal_rates(votes, tiers):
    """Mean signed marginal utility at each finite-population coalition size."""
    votes = np.asarray(votes)
    if votes.ndim != 2 or min(votes.shape) == 0 or not np.isin(votes, [-1, 0, 1]).all():
        raise ValueError('Expected a nonempty ternary vote matrix')
    n, p = votes.shape
    tiers = _peer_counts(tiers, p)
    positive = votes > 0
    negative = votes < 0
    plus, minus = positive.sum(axis=1), negative.sum(axis=1)
    zero = p - plus - minus
    log_choose = _log_choose_table(p, max(tiers))
    weights = np.zeros((2, len(tiers), n))
    for t, k in enumerate(tiers):
        norm = log_choose[p-1, k]
        for index, diff in ((0, -1), (1, 1)):
            # Rows lacking the candidate sign have no such feature to receive a weight.
            a = np.maximum(0, plus - int(index == 0))
            b = np.maximum(0, minus - int(index == 1))
            weight = _margin_mass(a, b, zero, k, 0, log_choose, norm)
            weight += _margin_mass(a, b, zero, k, diff, log_choose, norm)
            weights[index, t] = weight
    return (weights[0] @ positive - weights[1] @ negative) / n


class ExactMinPI(TransformerMixin, BaseEstimator):
    """Fixed-budget dense numeric feature screen based on worst-tier pivotality."""

    tiers = None  # Default for estimators serialized before configurable layers.

    def __init__(self, *, task='classification', random_state=42, fraction=.1, tiers=None):
        self.task = task
        self.random_state = random_state
        self.fraction = fraction
        self.tiers = tiers

    def fit(self, X, y):
        if not 0 < self.fraction <= 1:
            raise ValueError('fraction must be in (0, 1]')
        if isinstance(X, pd.DataFrame) and not X.columns.is_unique:
            raise ValueError('Feature names must be unique')
        margins = witness_margins(X, y, self.task, self.random_state)
        votes = np.sign(margins).astype(np.int8)
        self.n_witnesses_, self.n_features_in_ = votes.shape
        self.feature_names_in_ = np.asarray(X.columns) if isinstance(X, pd.DataFrame) else None
        p = self.n_features_in_
        self.tier_sizes_ = np.asarray(_selection_tiers(self.tiers, p)[::-1])
        self.tier_rates_ = pivotal_rates(votes, self.tier_sizes_)
        self.scores_ = self.tier_rates_.min(axis=0)
        self.ranking_ = np.argsort(-self.scores_, kind='stable')
        self.support_ = np.zeros(p, dtype=bool)
        self.support_[self.ranking_[:ceil(self.fraction*p)]] = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'support_')
        if np.ndim(X) != 2 or np.shape(X)[1] != self.n_features_in_:
            raise ValueError('Feature count differs from fit')
        if isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is not None and not np.array_equal(X.columns, self.feature_names_in_):
                raise ValueError('Feature schema differs from fit')
            return X.iloc[:, self.support_]
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, 'support_')
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

