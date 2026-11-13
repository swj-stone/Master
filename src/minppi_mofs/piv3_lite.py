"""Cross-fitted shadow comparisons for a training-only min-vote feature screen."""

from math import ceil, sqrt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from minppi_mofs.piv2 import N_BINS, PRIOR_WEIGHT, _folds


BYPASS_WIDTH = 64
SHADOWS = 64
DRAWS = 4


def _inputs(train, labels, task, aggregate):
    if not isinstance(train, pd.DataFrame) or not train.columns.is_unique or not train.shape[1]:
        raise ValueError("Expected a nonempty feature schema with unique column names")
    if any(not isinstance(name, str) for name in train.columns):
        raise ValueError("screen expects named string columns")
    y = np.asarray(labels)
    if task not in {"binary", "multiclass", "regression"} or aggregate not in {"min", "mean"}:
        raise ValueError("Unknown prediction task or coalition aggregation")
    if y.shape != (len(train),) or not len(y) or pd.isna(y).any():
        raise ValueError("Expected complete, aligned one-dimensional training labels")
    if task == "regression":
        y = y.astype(np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Regression labels must be finite")
    return y


def _signals(fit, check, fit_y, check_y, task, classes):
    """Learn binned feature response means on the fit half, then gather on the check half."""
    if task == "regression":
        mean, scale = fit_y.mean(), fit_y.std() or 1.
        yf = ((fit_y - mean) / scale)[:, None]
        yc = ((check_y - mean) / scale)[:, None]
    else:
        # Binary Brier losses on the positive class equal mean two-class Brier losses.
        outputs = classes[1:] if task == "binary" else classes
        yf = (fit_y[:, None] == outputs).astype(np.float32)
        yc = (check_y[:, None] == outputs).astype(np.float32)
        prior = yf.mean(axis=0)
        yf -= prior
        yc -= prior
    yf = np.ascontiguousarray(yf, dtype=np.float32)
    yc = np.ascontiguousarray(yc, dtype=np.float32)
    width = fit.shape[1]
    signals = np.empty((width, len(check), yf.shape[1]), dtype=np.float32)
    train_gain = np.empty(width, dtype=np.float64)
    for start in range(0, width, 128):
        stop = min(width, start + 128)
        a, b = fit[:, start:stop].T, check[:, start:stop].T
        # Equal-frequency cuts trained on the fit half only. Missing/nonfinite has its own bin.
        valid_a, valid_b = np.isfinite(a), np.isfinite(b)
        if valid_a.all():
            q = np.quantile(a, np.arange(1, N_BINS)/N_BINS, axis=1, method="inverted_cdf").T
        else:
            q = np.full((len(a), N_BINS-1), np.inf)
            for j in range(len(a)):
                if valid_a[j].any():
                    q[j] = np.quantile(a[j, valid_a[j]], np.arange(1, N_BINS)/N_BINS, method="inverted_cdf")
        # Repeated cuts create empty bins; assigning equal values consistently is sufficient.
        ia = (a[:, :, None] > q[:, None, :]).sum(axis=2, dtype=np.uint8)
        ib = (b[:, :, None] > q[:, None, :]).sum(axis=2, dtype=np.uint8)
        ia[~valid_a] = ib[~valid_b] = N_BINS
        offsets = np.arange(len(a))[:, None] * (N_BINS+1)
        ia, ib = (ia + offsets).ravel(), ib + offsets
        cells = len(a) * (N_BINS+1)
        counts = np.bincount(ia, minlength=cells)
        totals = np.column_stack([np.bincount(ia, weights=np.tile(col, len(a)), minlength=cells) for col in yf.T])
        means = totals / (counts[:, None] + PRIOR_WEIGHT)
        signals[start:stop] = means[ib]
        gain = (2*means*totals - counts[:, None]*means**2).sum(axis=1)
        train_gain[start:stop] = gain.reshape(len(a), -1).sum(axis=1) / yf.size
    return signals.reshape(width, -1), yc, train_gain


def _contexts(signals, target, order, tiers):
    """k-set not containing j: replace j by the next player when j is in the permutation prefix."""
    positions = np.empty_like(order)
    positions[order] = np.arange(len(order))
    residuals = []
    for k in tiers:
        total = signals[order[:k]].sum(axis=0, dtype=np.float64)
        # Multiplying the utility contrast by (k+1)^2 removes its scale with k.
        outside = (k+1)*target-total
        residuals.extend((outside, outside-signals[order[k]]))
    return positions, residuals


def _paired_contrasts(dots, norms, positions, k, shadows, tier_offset):
    """Scaled difference in squared-error reduction, adding real j vs its shadow to the same coalition."""
    p = len(positions)
    j = np.arange(p)
    # dots matrices are inner products of the heldout signals with the common residual.
    outside = 2*(dots[:p, tier_offset]-dots[shadows, tier_offset])
    inside = 2*(dots[:p, tier_offset+1]-dots[shadows, tier_offset+1])
    # In the inside case the coalition excludes j: the common residual must add s_j back.
    inside += 2*(norms[:p]-dots[j, shadows-p+dots.shape[1]-(len(norms)-p)])
    return np.where(positions < k, inside, outside) - norms[:p] + norms[shadows]


def _decide(net, marginal, null_limits, informative, aggregate):
    pooled = net.mean(axis=0)
    scores = pooled.min(axis=0) if aggregate == "min" else pooled.mean(axis=0)
    # A nomination needs nonpositive evidence on both halves for the same coalition tier.
    corroborated = np.any(np.all(net <= 0, axis=0), axis=0)
    protected = np.any(marginal > null_limits[:, None], axis=0)
    remove = (scores < 0) & corroborated & ~protected
    if not informative or remove.all():
        remove[:] = False
    return ~remove, scores, protected


def screen(train, labels, *, task, seed=42, aggregate="min"):
    """Return retained original column names in input order and reproducible screening diagnostics."""
    y = _inputs(train, labels, task, aggregate)
    names = train.columns.tolist()
    p = len(names)
    diag = dict(aggregate=aggregate, front_fits=0, sample_rows=0, input_count=p, bypass=None,
                tier_sizes=[], tier_rates=[], fold_tier_rates=[], aggregate_scores=[])
    if p <= BYPASS_WIDTH or len(y) < 32:
        diag["bypass"] = "small input"
        return names, diag
    row_seed, shadow_seed, vote_seed = np.random.SeedSequence(seed).spawn(3)
    folds = _folds(y, task, np.random.default_rng(row_seed))
    if folds is None:
        diag["bypass"] = "insufficient observations per class"
        return names, diag
    classes = None if task == "regression" else np.unique(y[folds[0]])
    if task == "binary" and len(classes) != 2:
        raise ValueError("Expected exactly two binary classes")
    shadow_rng = np.random.default_rng(shadow_seed)
    source = shadow_rng.choice(p, SHADOWS, replace=False)
    frames = []
    for indices in folds:
        real = train.iloc[indices].to_numpy(dtype=np.float64)
        shadow = np.column_stack([real[shadow_rng.permutation(len(real)), j] for j in source])
        frames.append(np.column_stack((real, shadow)))
    tiers = [1, round(sqrt(p)), round((p-1)*5/6)]
    net = np.zeros((2, 3, p), dtype=np.int16)
    marginal, limits, signal_checks = [], [], []
    rng = np.random.default_rng(vote_seed)
    for f in range(2):
        sig, target, train_gain = _signals(frames[f], frames[1-f], y[folds[f]], y[folds[1-f]], task, classes)
        n = target.size
        norms = np.einsum("ij,ij->i", sig, sig, dtype=np.float64) / n
        target_dot = np.einsum("ij,j->i", sig, target.ravel(), dtype=np.float64) / n
        single = 2*target_dot-norms
        marginal.append(single[:p])
        limits.append(max(0., float(np.quantile(single[p:], .95))))
        anchor = int(train_gain[:p].argmax())
        anchor_prediction = sig[anchor].reshape(target.shape)
        per_row = (2*anchor_prediction*target - anchor_prediction**2).mean(axis=1)
        mean = float(np.mean(per_row, dtype=np.float64))
        se = float(np.std(per_row, ddof=1, dtype=np.float64) / sqrt(len(per_row)))
        signal_checks.append(dict(column=names[anchor], mean_gain=mean, standard_error=se, passed=mean > 3*se))
        residuals, orders = [], []
        for _ in range(DRAWS):
            pos, res = _contexts(sig, target.ravel(), rng.permutation(p), tiers)
            orders.append(pos)
            residuals.extend(res)
        # Batched products share coalition residuals and the 64 shadow cross terms across all players.
        rhs = np.vstack([*residuals, sig[p:]]).astype(np.float32)
        dots = (sig @ rhs.T) / n
        for rep, positions in enumerate(orders):
            for h, k in enumerate(tiers):
                control = p + rng.integers(SHADOWS, size=p)
                contrasts = _paired_contrasts(dots, norms, positions, k, control, rep*6+2*h)
                # Ties contribute zero, preserving a three-way vote for numerically identical signals.
                eps = 2*np.finfo(np.float32).eps * np.maximum(np.abs(dots[:p, rep*6+2*h]), np.abs(dots[control, rep*6+2*h]))
                net[f, h] += (contrasts > eps).astype(np.int16) - (contrasts < -eps).astype(np.int16)
    rates = net.astype(np.float64) / DRAWS
    informative = any(record["passed"] for record in signal_checks)
    keep, scores, protected = _decide(rates, np.asarray(marginal), np.asarray(limits), informative, aggregate)
    diag.update(sample_rows=sum(map(len, folds)), shadow_count=SHADOWS, draws=DRAWS, tier_sizes=tiers,
                fold_tier_rates=rates.tolist(), tier_rates=rates.mean(axis=0).tolist(),
                aggregate_scores=scores.tolist(), shadow_gain_q95=limits, signal_checks=signal_checks,
                protected_count=int(protected.sum()), removed_count=int((~keep).sum()),
                bypass=None if informative else "weak global signal")
    return [name for name, retained in zip(names, keep, strict=True) if retained], diag


class MinPIScreen(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer for dense numeric data, preserving pandas names or array order."""

    def __init__(self, *, task="binary", seed=42, aggregate="min"):
        self.task = task
        self.seed = seed
        self.aggregate = aggregate

    def fit(self, X, y):
        named = isinstance(X, pd.DataFrame)
        frame = X if named else pd.DataFrame(X).rename(columns=str)
        names, self.diagnostics_ = screen(frame, y, task=self.task, seed=self.seed, aggregate=self.aggregate)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.support_ = np.asarray(frame.columns.isin(names))
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != self.feature_names_in_.tolist():
                raise ValueError("Input feature names or order differ from fit")
            return X.iloc[:, self.support_]
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("Input width differs from fit")
        return array[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        if input_features is not None and list(input_features) != self.feature_names_in_.tolist():
            raise ValueError("Input features differ from fit")
        return self.feature_names_in_[self.support_]

