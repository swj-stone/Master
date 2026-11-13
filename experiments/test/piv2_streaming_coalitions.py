"""Lazy real-refit exact and 32-permutation Shapley selectors for timed trials."""

import numpy as np

from minppi_mofs.modeling import _empty_subset_predictions, fit_predict
from minppi_mofs.utility import compute_utility


PATHS = 32
PREDICTION_CACHE_BYTES = 16 * 1024**2


def _permutation_plan(names, originals, seed):
    positions = {name: i for i, name in enumerate(names)}
    rng = np.random.default_rng(seed)
    paths = []
    distinct = set()
    for _ in range(PATHS):
        ordered = [originals[j] for j in rng.permutation(len(originals))]
        path = [positions[name] for name in ordered if name in positions]
        paths.append(path)
        mask = 0
        for index in path:
            mask |= 1 << index
            distinct.add(mask)
    return paths, len(distinct)


def select_coalitions(backend, train, yt, val, yv, policy, metric, progress, *,
                      original_width, original_columns):
    names = train.columns.tolist()
    width = len(names)
    if (backend not in {"exact", "mc32"} or not width or not train.columns.equals(val.columns)
            or not train.columns.is_unique or len(original_columns) != original_width
            or names != [c for c in original_columns if c in set(names)]):
        raise ValueError("Invalid coalition method or inconsistent input features")

    def columns(mask):
        return [c for i, c in enumerate(names) if mask & (1 << i)]

    def fit_mask(mask):
        cols = columns(mask)
        prediction = fit_predict(train[cols], yt, val[cols], **policy)
        utility = float(compute_utility(yv, prediction, metric=metric))
        if not np.isfinite(utility):
            raise ValueError(f"Non-finite validation utility on mask {mask:#x}")
        return utility, prediction

    full = (1 << width) - 1
    if backend == "exact":
        progress(backend, checked=0, candidates=full, fits=0)
        best_key = None
        best_prediction = None
        for mask in range(1, full + 1):
            score, prediction = fit_mask(mask)
            key = (-score, mask.bit_count(), mask)
            if best_key is None or key < best_key:
                best_key, best_prediction = key, prediction
            progress(backend, checked=mask, candidates=full, fits=mask, last_mask=hex(mask),
                     best_validation_utility=-best_key[0], best_mask=hex(best_key[2]))
        return columns(best_key[2]), best_prediction, full, dict(coalition_count=full, unique_scored=full)

    paths, planned = _permutation_plan(names, original_columns, policy["seed"])
    empty = _empty_subset_predictions(yt, task=policy["task"], row_count=len(val))
    utilities = {0: float(compute_utility(yv, empty, metric=metric))}
    saved_predictions = {}
    cached_bytes = 0
    fits = 0
    marginal = np.zeros(width)
    progress(backend, checked=0, candidates=planned, fits=0, paths_completed=0, paths_planned=PATHS)
    for path_index, path in enumerate(paths):
        parent = 0
        for position, i in enumerate(path):
            child = parent | (1 << i)
            fresh = child not in utilities
            if fresh:
                score, prediction = fit_mask(child)
                utilities[child] = score
                fits += 1
                # Always keep the complete model; cap other predictions while retaining all scalar utilities.
                if child == full or cached_bytes + prediction.nbytes <= PREDICTION_CACHE_BYTES:
                    saved_predictions[child] = prediction
                    if child != full:
                        cached_bytes += prediction.nbytes
            marginal[i] += utilities[child] - utilities[parent]
            finished = position + 1 == width
            if fresh or finished:
                progress(backend, checked=fits, candidates=planned, fits=fits, last_mask=hex(child),
                         paths_completed=path_index + int(finished), paths_planned=PATHS)
            parent = child
    if len(utilities) - 1 != planned:
        raise ValueError("Unexpected coalition count in permutation schedule")
    marginal /= PATHS
    selected = np.flatnonzero(marginal > 0).tolist() or [int(marginal.argmax())]
    winner = sum(1 << i for i in selected)
    cached_winner = winner in saved_predictions
    winner_was_scored = winner in utilities
    if cached_winner:
        prediction = saved_predictions[winner]
    else:
        progress(backend, checked=fits, candidates=planned+1, fits=fits, pending_winner_mask=hex(winner),
                 paths_completed=PATHS, paths_planned=PATHS)
        score, prediction = fit_mask(winner)
        if winner_was_scored and abs(score-utilities[winner]) > 1e-12:
            raise ValueError("Winner validation score changed on refit")
        fits += 1
        progress(backend, checked=fits, candidates=planned+1, fits=fits, last_mask=hex(winner),
                 paths_completed=PATHS, paths_planned=PATHS)
    diagnostics = dict(coalition_count=planned, unique_scored=planned+int(not winner_was_scored),
                       winner_prediction_fit=int(not cached_winner), paths=PATHS,
                       mean_marginal=marginal.tolist())
    return columns(winner), prediction, fits, diagnostics
