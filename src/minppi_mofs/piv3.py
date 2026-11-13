"""Budgeted feature pruning with replicated voting, weak-tier priority and reversible proposals."""

from math import ceil, sqrt

import numpy as np
from scipy.stats import chi2, rankdata

from minppi_mofs.piv2 import _bins, prefilter


def _training_order(train, y, seed, task="binary"):
    """Build a vote consensus and a second starting model entirely on training rows."""
    width = train.shape[1]
    vote_seeds = [int(s.generate_state(1)[0]) for s in seed.spawn(3)]
    votes, nominations = [], []
    for vote_seed in vote_seeds:
        _, diag = prefilter(train, y, task=task, seed=vote_seed)
        votes.append(diag["tier_rates"] or np.zeros((3, width)).tolist())
        nominations.append(len(diag["removed"]))
    tiers = np.mean(votes, axis=0)
    weakest = tiers.min(axis=0)

    # Pearson association on training-only quantile bins. The BH threshold
    # supplies an alternative proposal; column inclusion still needs validation.
    scores = np.zeros(width)
    p = np.ones(width)
    if task == "binary":
        prevalence = y.mean()
    else:
        # Quantiles discretize the regression response for this proposal only.
        target = _bins(y, y)[0] if task == "regression" else y
        _, codes, target_counts = np.unique(target, return_inverse=True, return_counts=True)
        levels = len(target_counts)
    for j in range(width):
        values = train.iloc[:, j].to_numpy(dtype=float)
        bins, _ = _bins(values, values)
        counts = np.bincount(bins)
        occupied = counts > 0
        counts = counts[occupied]
        if len(counts) < 2:
            continue
        if task == "binary":
            observed = np.bincount(bins, weights=y, minlength=len(occupied))[occupied]
            expected = counts * prevalence
            scores[j] = np.sum((observed - expected)**2 / (counts*prevalence*(1-prevalence)))
            p[j] = chi2.sf(scores[j], len(counts)-1)
        elif levels > 1:
            observed = np.bincount(bins * levels + codes, minlength=len(occupied) * levels)
            observed = observed.reshape(-1, levels)[occupied]
            expected = np.outer(counts, target_counts / len(y))
            scores[j] = np.sum((observed - expected)**2 / expected)
            p[j] = chi2.sf(scores[j], (len(counts)-1) * (levels-1))
    by_p = np.argsort(p, kind="stable")
    passing = np.flatnonzero(p[by_p] <= .05 * np.arange(1, width+1)/width)
    core = by_p[:int(passing[-1])+1] if len(passing) else by_p[:1]
    is_core = np.isin(np.arange(width), core)
    strength = rankdata(scores) + rankdata(weakest)
    order = np.lexsort((np.arange(width), -scores, -strength, ~is_core)).tolist()
    metadata = dict(vote_seeds=vote_seeds, tier_rates=tiers.tolist(), min_vote=weakest.tolist(),
                    nominations=nominations, association=scores.tolist(), association_p=p.tolist(),
                    bh_q=.05, fallback_core=not len(passing))
    return order, frozenset(core.tolist()), max(1, int(np.median(nominations))), metadata


def _boundary_pick(pool, count, rng):
    """Take the highest-ranked count, with random variation only around the cutoff."""
    count = min(count, len(pool))
    if count == len(pool):
        return frozenset(pool)
    radius = min(count, len(pool)-count, max(1, int(sqrt(count))))
    boundary = pool[count-radius:count+radius]
    return frozenset(pool[:count-radius]).union(rng.choice(boundary, radius, replace=False).tolist())


def search(train, labels, *, search_seed, tolerance, reference_scores, evaluate, budget=12, task="binary"):
    """Select a feasible mask in at most budget distinct calls to evaluate(column_names).

    Only the supplied callback evaluates validation rows. Classification scores
    are (AUC, accuracy), using macro OvR AUC for multiclass. Regression scores
    are (clipped R², MAE). Each score must meet its relative loss allowance
    against the full model. A zero reference allows no deterioration.
    Candidate IDs are one-based; zero denotes the original full model.
    """
    reference = np.asarray(reference_scores, dtype=float)
    labels = np.asarray(labels)
    columns = list(train.columns)
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError("Expected binary, multiclass, or regression")
    def valid_scores(scores):
        return (scores.shape == (2,) and np.isfinite(scores).all() and np.all(scores >= 0)
                and scores[0] <= 1 and (task == "regression" or scores[1] <= 1))

    if (not valid_scores(reference) or not 0 < tolerance < 1 or not columns or not train.columns.is_unique
            or not isinstance(budget, int) or budget < 0 or labels.shape != (len(train),)
            or len(labels) == 0):
        raise ValueError("Expected aligned training data, two valid reference scores and valid search limits")
    if task == "binary" and not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Binary labels must cover 0 and 1")
    if task == "multiclass" and np.unique(labels).size < 3:
        raise ValueError("Multiclass labels must cover at least three classes")
    metric_names = ("r2", "mae") if task == "regression" else ("auc", "accuracy")
    allowance = tolerance * reference
    seeds = np.random.SeedSequence(search_seed).spawn(2)
    rng = np.random.default_rng(seeds[1])
    order, core, initial_step, metadata = _training_order(train, labels, seeds[0], task=task)
    width = len(columns)
    full = frozenset(range(width))
    masks, risks, valid = [full], [0.], [True]
    seen = {full}
    trace = []

    def names(indices):
        return [columns[i] for i in sorted(indices)]

    def best():
        return min((i for i in range(len(masks)) if valid[i]),
                   key=lambda i: (len(masks[i]), risks[i], names(masks[i])))

    def check(mask, parent, operation):
        if not mask or mask in seen or len(trace) >= budget:
            return None
        candidate = names(mask)
        scores = np.asarray(evaluate(candidate), dtype=float)
        if not valid_scores(scores):
            raise ValueError(f"Invalid candidate {metric_names} scores")
        losses = reference - scores
        if task == "regression":
            losses[1] *= -1
        # A zero allowance maps a violation to a finite signal for step control.
        ratios = np.where(losses > 0, 2., 0.)
        np.divide(losses, allowance, out=ratios, where=allowance > 0)
        loss_ratio = float(np.max(ratios))
        feasible = loss_ratio <= 1 + 1e-12
        index = len(masks)
        trace.append(dict(id=index, parent=parent, operation=operation, proposed=candidate, k=len(mask),
                          added=names(mask-masks[parent]), removed=names(masks[parent]-mask),
                          **dict(zip(metric_names, map(float, scores))), accepted=feasible,
                          tightest_relative_budget_used=loss_ratio))
        seen.add(mask)
        masks.append(mask)
        risks.append(loss_ratio)
        valid.append(feasible)
        return index

    def prune(parent, step):
        current = masks[parent]
        if len(current) <= 1:
            return None
        pool = [i for i in reversed(order) if i in current]
        drop = min(len(current)-1, max(1, step))
        for _ in range(8):
            idx = check(current - _boundary_pick(pool, drop, rng), parent, "prune")
            if idx is not None:
                return idx
        return None

    def feedback(idx, previous):
        if idx is None:
            return previous
        ratio = risks[idx]
        if valid[idx]:
            return max(1, round(previous * float(np.clip(2-1.5*max(0, ratio), .5, 2))))
        return max(1, int(previous * float(np.clip(.8 / ratio, .2, .75))))

    def refill(parent, ceiling):
        current = masks[parent]
        outsiders = [i for i in order if i not in current]
        # Increase a failed sparse proposal according to the tightest metric's deficit.
        # The smallest feasible model stays available throughout.
        limit = max(1, (ceiling-len(current))//2)
        add = min(limit, max(1, ceil(sqrt(len(current))*min(4, risks[parent]))))
        for _ in range(8):
            idx = check(current | _boundary_pick(outsiders, add, rng), parent, "refill")
            if idx is not None:
                return idx
        return None

    def exchange(parent):
        current = masks[parent]
        inside = [i for i in reversed(order) if i in current]
        outside = [i for i in order if i not in current]
        if not outside:
            return None
        radius = max(1, ceil(sqrt(len(current))))
        for _ in range(8):
            leave = int(rng.choice(inside[:radius]))
            enter = int(rng.choice(outside[:radius]))
            idx = check((current-{leave})|{enter}, parent, "exchange")
            if idx is not None:
                return idx
        return None

    step = min(initial_step, max(1,width-1))
    step = feedback(prune(0, step), step)
    check(core, 0, "training_core")
    # Repair first if the independent compact anchor failed. Otherwise start
    # refining the accepted compact model. Alternate repair, deletion, exchange.
    modes = ("refill", "prune", "exchange")
    while len(trace) < budget:
        count_before = len(trace)
        for mode in modes:
            parent = best()
            current = masks[parent]
            failed = [i for i in range(1, len(masks)) if not valid[i] and len(masks[i]) < len(current)]
            frontier = min(failed, key=lambda i: (risks[i], len(masks[i]), i)) if failed else None
            if mode == "refill" and frontier is not None:
                refill(frontier, len(current))
            elif mode == "prune":
                tried = min(step, max(1,len(current)-1))
                step = feedback(prune(parent, tried), tried)
            elif mode == "exchange":
                exchange(parent if current != full or frontier is None else frontier)
            if len(trace) == budget:
                break
        if len(trace) == count_before:
            break
    winner = best()
    metadata.update(features=columns, order=[columns[i] for i in order], training_core=names(core),
                    initial_step=initial_step, winner_id=winner, steps=trace, budget=budget)
    return names(masks[winner]), metadata
