"""The eleven matched rankers for raw-width PIv3 experiments."""

import numpy as np
from numba import njit

from experiments.test.piv2_conventional_filters import (
    N_BINS, ROW_LIMIT, _association, _quantile_codes, _sample_rows, _spearman,
)
from experiments.test.piv2_extra_selectors import masked_banzhaf_importance
from experiments.test.piv2_prefix_rankings import METHODS, rank_features
from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance


@njit(cache=True)
def _add_redundancy(codes, counts, target, available, redundancy):
    """Accumulate feature/target MI in nats from the non-empty cells of each histogram."""
    n = codes.shape[1]
    target_bins = counts[target]
    for j in range(len(codes)):
        if not available[j]:
            continue
        cells = np.zeros((N_BINS, N_BINS), dtype=np.int64)
        for i in range(n):
            cells[codes[j, i], codes[target, i]] += 1
        acc = 0.
        for a in range(N_BINS):
            for b in range(N_BINS):
                seen = cells[a, b]
                if seen:
                    expected = counts[j, a] * target_bins[b] / n
                    acc += seen * np.log(seen / expected)
        redundancy[j] += acc / n


def _lazy_mrmr(train, yt, policy, progress):
    y = np.asarray(yt)
    regression = policy["task"] == "regression"
    rows = _sample_rows(y, regression, policy["seed"])
    y = y[rows]
    target = _quantile_codes(y) if regression else np.unique(y, return_inverse=True)[1]
    x = train.iloc[rows].to_numpy()
    width = x.shape[1]
    codes = np.empty((width, len(rows)), dtype=np.uint8)
    counts = np.empty((width, N_BINS), dtype=np.int64)
    relevance = np.empty(width)
    for j, feature in enumerate(x.T):
        encoded = _quantile_codes(feature)
        codes[j] = encoded
        counts[j] = np.bincount(encoded, minlength=N_BINS)
        relevance[j] = _association(encoded, target, "mi")

    diagnostics = dict(ranking_sample_rows=len(rows), row_limit=ROW_LIMIT, max_bins=N_BINS,
                       target_binned=regression, mid_by_step=[], ranking_scores=relevance.tolist())

    def ordered():
        available = np.ones(width, dtype=bool)
        redundancy = np.zeros(width)
        scores = relevance.copy()
        for step in range(width):
            best = int(np.argmax(scores))
            available[best] = False
            diagnostics["mid_by_step"].append(float(scores[best]))
            if step == 0 or (step + 1) % 50 == 0:
                progress("mrmr", checked=step+1, candidates=width)
            yield train.columns[best]
            if step + 1 == width:
                return
            if width <= 500:
                for j in np.flatnonzero(available):
                    # Use signed integers for the same histogram index arithmetic as the reference.
                    redundancy[j] += _association(codes[j].astype(np.intp), codes[best], "mi")
            else:
                _add_redundancy(codes, counts, best, available, redundancy)
            scores = relevance - redundancy / (step+1)
            scores[~available] = -np.inf

    return ordered(), 0, diagnostics


def _tree_scores(model, frame, y, policy):
    rows = np.arange(len(frame))
    if len(rows) > 1024:
        rows = np.sort(np.random.default_rng(policy["seed"]).choice(len(frame), 1024, replace=False))
    columns = model.get_booster().feature_names or [f"f{i}" for i in range(frame.shape[1])]
    sample = frame.iloc[rows].set_axis(columns, axis=1)
    classes = len(np.unique(y)) if policy["task"] == "multiclass" else 1
    # Predicted contributions have shape (rows, classes, width+1).
    batch_rows = max(1, 4_000_000 // (classes * (frame.shape[1]+1)))
    sums = np.zeros(frame.shape[1])
    for start in range(0, len(sample), batch_rows):
        batch = sample.iloc[start:start+batch_rows]
        means = fitted_tree_shap_importance(model, batch, seed=policy["seed"], max_rows=len(batch))
        sums += means * len(batch)
    return sums / len(sample), dict(ranking_sample_rows=len(sample))


def rank(backend, train, yt, val, yv, *, policy, metric, support, input_model, progress):
    """Return an iterator of names, additional selector fits, and mutable ranking diagnostics.

    train and val retain their original width. input_model() returns a model already fitted
    on support in that order. Its training cost belongs to the caller's fit ledger.
    """
    names = train.columns
    support = list(support)
    if (backend not in METHODS or not names.is_unique or not names.equals(val.columns)
            or not support or len(set(support)) != len(support) or not set(support).issubset(names)):
        raise ValueError("Inconsistent ranking method, support or feature names")
    full = support == names.tolist()
    frame = train if full else train.loc[:, support]
    if backend == "mrmr":
        return _lazy_mrmr(frame, yt, policy, progress)
    valid = val if full else val.loc[:, support]
    if backend == "tree_shap":
        scores, info = _tree_scores(input_model(), frame, yt, policy)
    elif backend == "banzhaf":
        _, scores, info = masked_banzhaf_importance(frame, yt, valid, yv, policy, metric, progress,
                                                    names, model=input_model(), max_batch_elements=4_000_000)
    elif backend == "spearman" and len(support) > 500:
        y = np.asarray(yt)
        regression = policy["task"] == "regression"
        rows = _sample_rows(y, regression, policy["seed"])
        # Ranking features in blocks avoids materialising multiple dense rank matrices at full width.
        scores = np.concatenate([_spearman(frame.iloc[rows, i:i+128].to_numpy(dtype=float),
                                           y[rows], regression)
                                 for i in range(0, len(support), 128)])
        info = dict(ranking_sample_rows=len(rows), row_limit=ROW_LIMIT, ranking_scores=scores.tolist())
    else:
        order, fits, _, info = rank_features(backend, frame, yt, valid, yv, policy, metric,
                                             progress, original_columns=names)
        return iter(frame.columns[order]), fits, info
    scores = np.asarray(scores, dtype=float).copy()
    scores[np.isnan(scores)] = -np.inf
    return iter(frame.columns[np.argsort(-scores, kind="stable")]), 0, info
