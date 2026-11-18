"""Paired feature selectors for the seed-42 PIv2 benchmark."""

from math import ceil, sqrt

import numpy as np
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split

from experiments.test.run_sampled_permutation_seed42 import predict_with, sampled_drops
from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance
from minppi_mofs.modeling import _empty_subset_predictions, fit_predict, make_model
from minppi_mofs.utility import compute_utility


BACKENDS = ("gain", "pfi", "f_score", "mutual_info", "tree_shap", "exact", "mc32", "banzhaf", "l1")
MI_ROWS = 4096
MC_PATHS = 32


def _top_indices(scores, k):
    # Undefined univariate F statistics for constant features rank last.
    scores = np.asarray(scores, dtype=float).copy()
    scores[np.isnan(scores)] = -np.inf
    return np.sort(np.argsort(-scores, kind="stable")[:k])


def _univariate(backend, train, yt, policy):
    regression = policy["task"] == "regression"
    if backend == "f_score":
        scorer = f_regression if regression else f_classif
        scores, _ = scorer(train, yt)
        return scores, len(train)
    indices = np.arange(len(train))
    if len(indices) > MI_ROWS:
        indices, _ = train_test_split(
            indices, train_size=MI_ROWS, random_state=policy["seed"],
            stratify=None if regression else yt,
        )
        indices.sort()
    scorer = mutual_info_regression if regression else mutual_info_classif
    scores = scorer(train.iloc[indices], np.asarray(yt)[indices], random_state=policy["seed"])
    return scores, len(indices)


def _gain_scan(train, yt, val, yv, model, full_prediction, policy, metric, progress):
    width = train.shape[1]
    gains = model.get_booster().get_score(importance_type="gain")
    order = sorted(range(width), key=lambda j: (gains.get(f"f{j}", 0.), j), reverse=True)
    best = -np.inf
    curve = []
    for k in range(1, width + 1):
        columns = train.columns[sorted(order[:k])].tolist()
        prediction = full_prediction if k == width else fit_predict(train[columns], yt, val[columns], **policy)
        score = compute_utility(yv, prediction, metric=metric)
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation utility")
        curve.append(float(score))
        if score > best + 1e-14:
            best, selected, selected_prediction = score, columns, prediction
        progress("gain_scan", checked=k, candidates=width)
    return selected, selected_prediction, width, dict(gain_order=train.columns[order].tolist(), curve=curve)


def _coalition_search(backend, train, yt, val, yv, policy, metric, progress, original_columns):
    width = train.shape[1]
    if len(original_columns) > 8:
        raise ValueError("This pilot runs real-coalition search only on the small original feature space")
    names = train.columns.tolist()
    positions = {name: i for i, name in enumerate(names)}
    rng = np.random.default_rng(policy["seed"])
    paths = []
    masks = set()
    if backend == "exact":
        masks.update(range(1, 1 << width))
    else:
        # Restrict the same underlying permutations in both arms to the available columns.
        for _ in range(MC_PATHS):
            ordered_names = [original_columns[j] for j in rng.permutation(len(original_columns))]
            path = [positions[name] for name in ordered_names if name in positions]
            paths.append(path)
            mask = 0
            for i in path:
                mask |= 1 << i
                masks.add(mask)
    empty = _empty_subset_predictions(yt, task=policy["task"], row_count=len(val))
    predictions = {0: empty}
    utilities = {0: compute_utility(yv, empty, metric=metric)}

    def evaluate(mask):
        indices = [i for i in range(width) if mask & (1 << i)]
        pred = fit_predict(train.iloc[:, indices], yt, val.iloc[:, indices], **policy)
        predictions[mask] = pred
        utilities[mask] = compute_utility(yv, pred, metric=metric)

    for n, mask in enumerate(sorted(masks), 1):
        evaluate(mask)
        if n == 1 or n % 5 == 0 or n == len(masks):
            progress(backend, checked=n, candidates=len(masks))
    diagnostics = dict(coalition_count=len(masks))
    if backend == "exact":
        winner = min(masks, key=lambda m: (-utilities[m], m.bit_count(), m))
    else:
        marginal = np.zeros(width)
        for path in paths:
            parent = 0
            for i in path:
                child = parent | (1 << i)
                marginal[i] += utilities[child] - utilities[parent]
                parent = child
        marginal /= MC_PATHS
        chosen = np.flatnonzero(marginal > 0).tolist() or [int(marginal.argmax())]
        winner = sum(1 << i for i in chosen)
        if winner not in predictions:
            evaluate(winner)
        diagnostics.update(paths=MC_PATHS, mean_marginal=marginal.tolist())
    selected = [name for i, name in enumerate(names) if winner & (1 << i)]
    return selected, predictions[winner], len(predictions) - 1, diagnostics


def select(backend, train, yt, val, yv, policy, metric, progress, *, original_width, original_columns=None):
    """Select using train/validation only; return names, validation predictions, fits and diagnostics."""
    width = train.shape[1]
    if backend not in BACKENDS or width < 1 or width > original_width:
        raise ValueError(f"Invalid backend or feature width ({backend}, {width}, {original_width})")
    if not train.columns.equals(val.columns) or not train.columns.is_unique:
        raise ValueError("Misaligned selector inputs")
    if backend in ("banzhaf", "l1"):
        from experiments.test.piv2_extra_selectors import select_extra

        return select_extra(backend, train, yt, val, yv, policy, metric, progress,
                            original_width=original_width, original_columns=original_columns)
    if backend in ("exact", "mc32"):
        if original_columns is None:
            original_columns = train.columns.tolist()
        if len(original_columns) != original_width or not set(train.columns).issubset(original_columns):
            raise ValueError("Missing original column positions for matched coalition paths")
        return _coalition_search(backend, train, yt, val, yv, policy, metric, progress, original_columns)

    k = min(width, ceil(sqrt(original_width)))
    if backend in ("f_score", "mutual_info"):
        scores, sample_rows = _univariate(backend, train, yt, policy)
        columns = train.columns[_top_indices(scores, k)].tolist()
        prediction = fit_predict(train[columns], yt, val[columns], **policy)
        return columns, prediction, 1, dict(k=k, ranking_sample_rows=sample_rows)

    safe = [f"f{i}" for i in range(width)]
    safe_train = train.set_axis(safe, axis=1)
    safe_val = val.set_axis(safe, axis=1)
    model = make_model(**policy)
    model.fit(safe_train, yt)
    # TreeSHAP only needs a training model, and predicts on validation after selecting k.
    if backend == "tree_shap":
        scores = fitted_tree_shap_importance(model, safe_train, seed=policy["seed"], max_rows=1024)
        indices = _top_indices(scores, k)
        diagnostics = dict(k=k, ranking_sample_rows=min(len(train), 1024))
    else:
        full_prediction = predict_with(model, safe_val, policy)
        if backend == "gain":
            return _gain_scan(train, yt, val, yv, model, full_prediction, policy, metric, progress)
        rows, base, drops = sampled_drops(model, safe_val, np.asarray(yv), full_prediction, metric, policy, progress)
        means = drops.mean(axis=1)
        indices = np.flatnonzero(means > 0)
        if len(indices) == 0:
            indices = [int(means.argmax())]
        diagnostics = dict(sample_size=len(rows), repeats=drops.shape[1], base_utility=base,
                           mean_drops=means.tolist())
    columns = train.columns[indices].tolist()
    if len(columns) == width:
        prediction = predict_with(model, safe_val, policy) if backend == "tree_shap" else full_prediction
        return columns, prediction, 1, diagnostics
    return columns, fit_predict(train[columns], yt, val[columns], **policy), 2, diagnostics
