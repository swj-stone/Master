"""Choose a compressed forest on independent fitting, voting and acceptance rows."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from experiments.minpi_compression.benchmark import eligible, metrics
from experiments.minpi_rework import quota_coalition_rf as forest
from experiments.minpi_rework.quota_compress_tree import Game
from experiments.minpi_rework.predictive_guard import margin_path
from experiments.minpi_rework.predictive_mask_study import centered
from minppi_mofs.grouped_paths import collapse_path_groups
from minppi_mofs.predictive_quota import PredictiveGame
from minppi_mofs.size_extrema import rank_minimum


@dataclass
class AcceptancePath:
    order: np.ndarray
    margins: np.ndarray
    worst: np.ndarray
    intact: np.ndarray


def prepare_paths(model, x_vote, y_vote, x_width, y_width, task, include_min=True, *,
                  rank_backend="native", unit_ids=None):
    """Materialize the original path predictions and vote curves exactly once."""
    if rank_backend not in ("native", "interval"):
        raise ValueError("Expected native or interval ranking")
    costs = {}
    def path_game(x):
        raw = model.game(x)
        if unit_ids is None:
            return raw
        return Game(*collapse_path_groups(raw.a, raw.incidence, unit_ids))

    start = perf_counter()
    calibration = path_game(x_width)
    game = PredictiveGame.from_paths(calibration.a, calibration.incidence,
                                     centered(model, y_width, task))
    costs["calibration"] = perf_counter() - start

    start = perf_counter()
    importance = model.model.feature_importances_
    if unit_ids is not None:
        importance = np.bincount(unit_ids, weights=importance, minlength=calibration.p)
    orders = {"mdi": np.argsort(importance, kind="stable")}
    costs["mdi_rank"] = perf_counter() - start
    if include_min:
        start = perf_counter()
        voting = path_game(x_vote)
        voting_game = PredictiveGame.from_paths(voting.a, voting.incidence,
                                                centered(model, y_vote, task))
        if rank_backend == "interval":
            orders["min"], _ = rank_minimum(voting_game)
        else:
            orders["min"], _ = voting_game.ranked_path("min")
        costs["min_rank"] = perf_counter() - start

    paths = {}
    for name, order in orders.items():
        start = perf_counter()
        margins = margin_path(model, calibration, order)
        sizes = np.arange(1, len(order) + 1)
        curves = game.path_curves(order)[:, 1:] * (len(order) / sizes)
        paths[name] = AcceptancePath(order, margins, curves.max(axis=1), curves[:, -1])
        costs[name + "_curve"] = perf_counter() - start
    return paths, game.null, costs


def select_width(paths, null, y, task, scale, epsilon=.02, policy="min_first"):
    """Minimize (width, worst loss, path name) over jointly feasible candidates."""
    if policy not in ("eager", "intact_first", "min_first"):
        raise ValueError(f"Unknown width policy {policy}")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("The loss quota must be finite and nonnegative")
    first = next(iter(paths.values()))
    p = len(first.order)
    full = metrics(y, first.margins[0], task, scale)
    threshold = epsilon * null + 1e-12 * max(1., null)
    candidates = sorted((p - dropped, path.worst[dropped], name, dropped)
                        for name, path in paths.items() for dropped in range(p + 1))
    calls = 0
    best = None
    for width, worst, name, dropped in candidates:
        path = paths[name]
        if policy == "min_first" and worst > threshold:
            continue
        if policy == "intact_first" and path.intact[dropped] > threshold:
            continue
        if dropped:
            score = metrics(y, path.margins[dropped], task, scale)
            calls += 1
            if not eligible(score, full, task):
                continue
        if worst <= threshold and best is None:
            best = (width, worst, name, dropped)
            if policy != "eager":
                break
    if best is None:
        raise ArithmeticError("The full model failed its own acceptance checks")
    width, worst, name, dropped = best
    return np.sort(paths[name].order[dropped:]), {
        "path": name, "width": width, "worst": float(worst / max(null, 1e-100)),
        "native_calls": calls + 1,
    }


def select_forest(x_fit, y_fit, x_vote, y_vote, x_width, y_width, task, seed=42,
                  epsilon=.02, *, rank_backend="interval", unit_ids=None):
    """Fit the original two-path MinPI forest. Classification uses contiguous class IDs."""
    if task not in ("classification", "regression"):
        raise ValueError("Expected classification or regression")
    y_fit, y_vote, y_width = (np.asarray(y) for y in (y_fit, y_vote, y_width))
    if task == "classification":
        classes = np.unique(y_fit)
        if (len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes)))
                or not np.isin(y_vote, classes).all()
                or not np.array_equal(np.unique(y_width), classes)):
            raise ValueError("Fit and acceptance samples must contain every encoded class")
        y_fit, y_vote, y_width = (y.astype(int) for y in (y_fit, y_vote, y_width))
    started = perf_counter()
    model = forest.fit(x_fit, y_fit, task, seed)
    paths, null, _ = prepare_paths(model, x_vote, y_vote, x_width, y_width, task,
                                   rank_backend=rank_backend, unit_ids=unit_ids)
    columns, trace = select_width(paths, null, y_width, task, max(float(np.std(y_fit)), 1e-12),
                                  epsilon)
    trace["elapsed_s"] = perf_counter() - started
    if unit_ids is not None:
        trace["selected_unit_ids"] = columns.tolist()
        columns = np.flatnonzero(np.isin(unit_ids, columns))
    return model, columns, trace
