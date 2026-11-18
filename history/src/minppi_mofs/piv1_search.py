"""Validation search for PIv1. A negative vote can change priority without banning a column."""

from math import ceil, sqrt

import numpy as np
from scipy.stats import rankdata


VOTE_WEIGHTS = {"anchor": 0., "vote": .30, "vote_strong": .65}


def _order(strength):
    return tuple(np.argsort(-strength, kind="stable").tolist())


def _prefixes(order):
    return [tuple(sorted(order[:k])) for k in range(len(order) + 1)]


def _repair_candidates(mask, decision_order, reference_order):
    """Both sides can propose replacements, including re-entry of a negative voter."""
    selected = set(mask)
    for outside_order, inside_order in (
        (decision_order, reference_order), (reference_order, decision_order),
        (decision_order, decision_order), (reference_order, reference_order),
    ):
        entering = [i for i in outside_order if i not in selected]
        leaving = [i for i in reversed(inside_order) if i in selected]
        for count in (1, 2):
            if len(entering) >= count and len(leaving) >= count:
                yield selected.difference(leaving[:count]).union(entering[:count]), "exchange"
        if entering:
            yield selected.union(entering[:1]), "add"
        if len(leaving) > 1:
            yield selected.difference(leaving[:1]), "drop"


def search_subsets(vote_scores, shap_importance, gain_importance, evaluate, *,
                   recipe="vote", consistent_negative=None):
    """Evaluate real subsets and return the best validated mask under one fixed recipe.

    Indices are in input-column order. The callback returns a scalar or the pooled
    utility followed by optional diagnostics. Higher utility wins, ties prefer fewer
    columns. The budget counts distinct masks, including the full-model reference.
    """
    if recipe not in VOTE_WEIGHTS:
        raise ValueError(f"Unknown PIv1 search recipe: {recipe}")
    shap = np.asarray(shap_importance, dtype=float)
    gain = np.asarray(gain_importance, dtype=float)
    if (shap.ndim != 1 or len(shap) == 0 or gain.shape != shap.shape
            or not np.isfinite(shap).all() or not np.isfinite(gain).all()):
        raise ValueError("Expected aligned, finite, nonempty importance vectors")
    width = len(shap)
    votes = np.zeros(width) if recipe == "anchor" else np.asarray(vote_scores, dtype=float)
    if votes.shape != shap.shape or not np.isfinite(votes).all() or (np.abs(votes) > 1).any():
        raise ValueError("Expected one signed net vote in [-1, 1] per feature")
    negative = votes < 0
    if consistent_negative is not None and recipe != "anchor":
        consistent = np.asarray(consistent_negative, dtype=bool)
        if consistent.shape != negative.shape:
            raise ValueError("Inconsistent split-vote dimensions")
        negative &= consistent
    # Midranks share credit across tied importances, especially zero Gain values.
    reference = (rankdata(shap) + rankdata(gain) - 2.) / (2 * max(width - 1, 1))
    decision_order = _order(reference + VOTE_WEIGHTS[recipe] * votes)
    shap_order = _order(shap)
    gain_order = _order(gain)
    orders = [decision_order, shap_order, gain_order]
    paths = [_prefixes(order) for order in orders]
    base_budget = min(24, max(6, ceil(2.4 * sqrt(width))), 2**min(width, 5) - 1)
    negative_count = int(negative.sum())
    budget = base_budget
    if recipe == "vote_strong":
        budget = ceil(base_budget * (1. - .5 * negative_count / width))
    trials = {}

    def assess(mask, source):
        mask = tuple(sorted(mask))
        if not mask or mask in trials or len(trials) >= budget:
            return
        observed = np.asarray(evaluate(mask), dtype=float).reshape(-1)
        if not len(observed) or not np.isfinite(observed[0]):
            raise ValueError("Missing or nonfinite pooled validation utility")
        trials[mask] = dict(indices=mask, utility=float(observed[0]), source=source)

    def best():
        high = max(t["utility"] for t in trials.values())
        return min((t for t in trials.values() if t["utility"] >= high - 1e-14),
                   key=lambda t: (len(t["indices"]), t["indices"]))

    # Absolute small sizes and fractional large sizes cover both sparse and dense tasks.
    sizes = [width, round(sqrt(width)), round(width / 2), round(7 * width / 8),
             round(sqrt(width) / 2), round(width / 4), round(3 * width / 4), 1]
    sizes = list(dict.fromkeys(max(1, k) for k in sizes))[:max(3, budget // 2 + 1)]
    for k in sizes:
        assess(paths[0][k], "full" if k == width else recipe + "_prefix")
    ranked_sizes = sorted((k for k in sizes if k < width),
                          key=lambda k: (-trials[paths[0][k]]["utility"], k))
    alternate_limit = min(budget, len(trials) + max(1, budget // 5))
    for k in ranked_sizes:
        for name, path in zip(("shap", "gain"), paths[1:]):
            if len(trials) < alternate_limit:
                assess(path[k], name + "_prefix")

    def refine(limit):
        # Bisect the largest untested interval next to the strongest validated endpoint.
        while len(trials) < limit:
            proposals = []
            for index, path in enumerate(paths):
                previous_k, previous_score = 0, -np.inf
                for k, mask in enumerate(path[1:], start=1):
                    if mask not in trials:
                        continue
                    endpoint = trials[mask]["utility"]
                    if k - previous_k > 1:
                        mid = (previous_k + k) // 2
                        proposals.append((max(previous_score, endpoint), k-previous_k, -index, -mid))
                    previous_k, previous_score = k, endpoint
            if not proposals:
                return
            _, _, neg_index, neg_k = max(proposals)
            path_name = (recipe, "shap", "gain")[-neg_index]
            assess(paths[-neg_index][-neg_k], path_name + "_refine")

    refine(max(len(trials), budget - max(1, round(budget / 4))))
    incumbent = best()["indices"]
    for mask, operation in _repair_candidates(incumbent, decision_order, shap_order):
        assess(mask, recipe + "_" + operation)
    refine(budget)
    # With tiny widths or identical rankings, further one-column perturbations
    # can still give the control the same number of distinct candidates.
    for center in list(trials):
        for i in decision_order:
            if len(trials) == budget:
                break
            assess(set(center).symmetric_difference([i]), "neighbor")
        if len(trials) == budget:
            break
    winner = best()
    return dict(selected=winner["indices"], utility=winner["utility"], winner_source=winner["source"],
                trials=list(trials.values()), budget=budget, base_budget=base_budget,
                negative_count=negative_count, recipe=recipe)
