"""Compact feature supports from coalition votes over one fitted XGBoost ensemble.

The structural game awards each split's training gain if all features from the root
through that split are present. Votes compare a player's marginal structural gain
to the mean marginal in the same draw. Neither the game nor the frozen-predictor
prefix check fits a model. A caller can subsequently verify a proposal independently.
"""

import json
from math import ceil

import numpy as np
from sklearn.metrics import average_precision_score

from minppi_mofs.utility import compute_utility


DRAWS = 24
PREFIX_CHECKS = 12
MASK_TOLERANCE = .08


def _model_columns(model, columns):
    booster = model.get_booster()
    if booster.num_features() != len(columns):
        raise ValueError("The fitted model and the input schema have different widths")
    keys = booster.feature_names
    if keys is None:
        keys = [f"f{i}" for i in range(len(columns))]
    elif list(keys) != columns:
        raise ValueError("The fitted model has a different feature order")
    return booster, dict(zip(keys, columns, strict=True))


def _path_game(booster):
    """Return positive-weight unanimous paths and all physically used split columns."""
    tree_model = json.loads(booster.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]
    by_output = {}
    used = set()
    for tree, output in zip(tree_model["trees"], tree_model["tree_info"], strict=True):
        paths = by_output.setdefault(output, {})
        children = tree["left_children"]
        stack = [(0, frozenset())]
        while stack:
            node, prefix = stack.pop()
            if children[node] < 0:
                continue
            feature = tree["split_indices"][node]
            used.add(feature)
            path = prefix | {feature}
            weight = max(0., tree["loss_changes"][node])
            if weight > 0:
                paths[path] = paths.get(path, 0.) + weight
            stack.extend(((children[node], path), (tree["right_children"][node], path)))
    # A multiclass ensemble supplies one tree per class per round. Equal total gain
    # per output prevents a high-gain class from overwhelming the other classes.
    mixed = {}
    for paths in by_output.values():
        total = sum(paths.values())
        for key, weight in paths.items():
            mixed[key] = mixed.get(key, 0.) + weight / total
    return mixed, sorted(used)


def _encoded_paths(paths, used):
    lookup = {feature: pos for pos, feature in enumerate(used)}
    grouped = {}
    for key, weight in paths.items():
        entries, weights = grouped.setdefault(len(key), ([], []))
        entries.append([lookup[j] for j in sorted(key)])
        weights.append(weight)
    return [(np.asarray(entries, dtype=np.intp), np.asarray(weights))
            for entries, weights in grouped.values()]


def _marginals(groups, positions, k):
    """All gain differences in exactly k-member coalitions, excluding the focal player.

    In each draw all players share one permutation. An inside player is replaced
    by the first outside member when constructing its focal-player-excluding k-set.
    """
    n = len(positions)
    result = np.zeros(n)
    for members, weights in groups:
        ranks = positions[members]
        high_at = ranks.argmax(axis=1)
        rows = np.arange(len(ranks))
        highest = ranks[rows, high_at]
        # A path entirely among the first k+1 can be unlocked by any path member.
        complete = highest <= k
        if np.any(complete):
            result += np.bincount(members[complete].ravel(),
                                  weights=np.repeat(weights[complete], members.shape[1]), minlength=n)
        # For a path extending beyond k+1, only the last member can unlock it.
        # This succeeds precisely when all remaining path members are in the first k.
        one_missing = (~complete) & (np.sum(ranks >= k, axis=1) == 1)
        if np.any(one_missing):
            last = members[rows[one_missing], high_at[one_missing]]
            result += np.bincount(last, weights=weights[one_missing], minlength=n)
    return result


def _rank_paths(paths, used, seed, aggregate):
    size = len(used)
    tiers = np.clip(np.rint(np.array([1/6, 1/2, 5/6]) * (size-1)), 0, size-1).astype(int)
    groups = _encoded_paths(paths, used)
    rng = np.random.default_rng(seed)
    votes = np.zeros((3, size))
    magnitudes = np.zeros_like(votes)
    for _ in range(DRAWS):
        positions = rng.permutation(size)
        for h,k in enumerate(tiers):
            gains = _marginals(groups, positions, int(k))
            level = gains.mean()
            if level:
                relative = gains / level
                votes[h] += np.where(relative > 1+1e-12, 1, np.where(relative < 1-1e-12, -1, 0))
                magnitudes[h] += relative
    votes /= DRAWS
    magnitudes /= DRAWS
    reduce = np.min if aggregate == "min" else np.mean
    score, magnitude = reduce(votes, axis=0), reduce(magnitudes, axis=0)
    order = np.lexsort((np.arange(size), -magnitude, -score))
    return order, dict(tier_sizes=tiers.tolist(), draws=DRAWS, tier_rates=votes.tolist(),
                       relative_gains=magnitudes.tolist(), vote_score=score.tolist())


def _scores(y, predictions, task, utility, rare):
    if task == "regression":
        return dict(r2=float(compute_utility(y, predictions, metric=utility)),
                    mae=float(np.mean(np.abs(y-predictions))))
    scores = dict(auc=float(compute_utility(y, predictions, metric=utility)),
                  accuracy=float(np.mean(predictions.argmax(axis=1)==y)))
    if rare:
        scores["ap"] = float(average_precision_score(y, predictions[:, 1]))
    return scores


def choose_prefix(model, train, ranking, yr, *, task, utility, order, seed=42):
    """Use a frozen predictor to size a ranked prefix, with at most 12 masked predictions.

    The effective number of split-gain contributors supplies an initial width. Bracket
    by doubling or halving, then bisect; return the smallest *observed* feasible prefix.
    Actual prediction curves need not be monotone.
    """
    columns = list(train.columns)
    order = list(order)
    if (not columns or not train.columns.is_unique or list(ranking.columns) != columns
            or not order or len(set(order)) != len(order) or not set(order).issubset(columns)):
        raise ValueError("Expected aligned data and a nonempty ranking of unique input columns")
    booster, model_names = _model_columns(model, columns)
    active = {model_names[f] for f in booster.get_score(importance_type="weight")}
    if not active.issubset(order):
        raise ValueError("Ranking must include every column used by a fitted split")
    yr = np.asarray(yr)
    if yr.shape != (len(ranking),) or len(yr) < 1:
        raise ValueError("Ranking labels are not aligned")
    readout = model.predict if task == "regression" else model.predict_proba
    x = np.ascontiguousarray(ranking.to_numpy(), dtype=np.float32)
    _, counts = np.unique(yr, return_counts=True)
    rare = task == "binary" and len(counts) == 2 and counts.min() <= .05*len(yr)
    reference = _scores(yr, readout(x), task, utility, rare)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(train), min(len(train), 512), replace=False)
    background = np.nanmedian(train.iloc[rows].to_numpy(dtype=np.float32), axis=0)
    masked = np.broadcast_to(background, x.shape).copy()
    feature_ids = train.columns.get_indexer(order)
    # Once all actually used features are included, the masked and full predictions
    # agree identically, even if the tail contains unused external-ranking features.
    last = max((i+1 for i,name in enumerate(order) if name in active), default=1)
    widths = {last: reference}
    feasible = {last}
    tested = []
    max_loss = MASK_TOLERANCE

    def check(k):
        if k not in widths:
            masked[:] = background
            ids = feature_ids[:k]
            masked[:, ids] = x[:, ids]
            scores = _scores(yr, readout(masked), task, utility, rare)
            widths[k] = scores
            good = all((v <= reference[m]*(1+max_loss)+1e-12 if m=="mae"
                        else v+1e-12 >= reference[m]*(1-max_loss)) for m,v in scores.items())
            if good:
                feasible.add(k)
            tested.append(dict(k=k, scores=scores, passes=good))
        return k in feasible

    total_gain = np.asarray(list(booster.get_score(importance_type="total_gain").values()))
    effective = (total_gain.sum()**2 / (total_gain @ total_gain)
                 if np.any(total_gain > 0) else 1.)
    k = min(last, max(1, ceil(effective)))
    initial = k
    lo, hi = 0, last
    if check(k):
        hi = k
        while hi > 1 and len(tested) < PREFIX_CHECKS:
            k = max(1, hi//2)
            if not check(k):
                lo = k
                break
            hi = k
    else:
        lo = k
        while lo < last and len(tested) < PREFIX_CHECKS:
            k = min(last, 2*lo)
            if check(k):
                hi = k
                break
            lo = k
    while hi-lo > 1 and len(tested) < PREFIX_CHECKS:
        mid = (lo+hi)//2
        if check(mid):
            hi = mid
        else:
            lo = mid
    best = min(feasible)
    subset = set(order[:best])
    return [name for name in columns if name in subset], dict(
        order=order, k=best, model_used_count=len(active), initial_width=initial,
        mask_tolerance=max_loss, ranking_rows=len(yr), baseline=reference,
        selected_mask_scores=widths[best], masked_proposals=tested, exact_endpoint_width=last,
        background_rows=len(rows), front_fits=0)


def select(model, train, yt, ranking, yr, *, task, utility, seed=42, aggregate="min"):
    """Return a compact frozen-tree feature support and inspectable intermediate evidence."""
    if aggregate not in {"min", "mean", "gain"}:
        raise ValueError("aggregate must be min, mean or gain")
    if np.asarray(yt).shape != (len(train),):
        raise ValueError("Training rows and target are not aligned")
    columns = list(train.columns)
    booster, model_names = _model_columns(model, columns)
    details = dict(aggregate=aggregate)
    if aggregate == "gain":
        values = booster.get_score(importance_type="gain")
        active = {model_names[f]: gain for f,gain in values.items()}
        order = sorted(active, key=lambda f: (-active[f], train.columns.get_loc(f)))
    else:
        paths, used = _path_game(booster)
        order = []
        if used:
            indices, vote = _rank_paths(paths, used, seed, aggregate)
            order = [columns[used[i]] for i in indices]
            details.update(vote, structural_path_count=len(paths), vote_columns=[columns[i] for i in used])
    if not order:
        order = columns[:1]
    selected, sizing = choose_prefix(model, train, ranking, yr, task=task, utility=utility,
                                     order=order, seed=seed)
    return selected, dict(**details, **sizing)

