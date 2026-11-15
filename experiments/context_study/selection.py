"""Panel and context ablations with a common fitted ensemble, anchor, and proposal order."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.context_design import matched_random_contexts, separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_witness import ROW_LIMIT, _sample, _within
from minppi_mofs.utility import compute_utility


def contexts(arm, tail, k, fold):
    if arm in {"point", "empty_majority"}:
        return (frozenset(),)
    if arm in {"antichain", "random_contexts"}:
        code = separating_contexts(tail)
        if arm == "antichain":
            return code
        return matched_random_contexts(code, tail, seed=77531 + fold)
    return tuple(dict.fromkeys((frozenset(), frozenset(tail[:k]), frozenset(tail))))


def single_panel(y, seed):
    """Reproduce the first paired panel without allocating/resampling additional panels."""
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    quota = np.zeros_like(counts)
    remaining = min(ROW_LIMIT, len(y))
    for i, group in enumerate(np.argsort(counts, kind="stable")):
        quota[group] = min(counts[group], remaining // (len(labels)-i))
        remaining -= quota[group]
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == label), n, replace=False)
                                   for label, n in zip(labels, quota, strict=True)]))
    mass = (counts / (len(y) * quota))[np.searchsorted(labels, y[rows])]
    return rows, mass[None, :]


def select_vote(arm, model, train, rank, y, ordered, k, fold):
    seed = 20421 + fold
    groups = contexts(arm, ordered[k:], k, fold)
    pair_seed = np.random.SeedSequence([seed, 659])
    is_single = arm in {"point", "all_point"}
    task = "binary" if len(np.unique(y)) == 2 else "multiclass"
    rows, weights = single_panel(y, pair_seed) if is_single else _sample(y, task, pair_seed)
    rare = task == "binary" and min(np.bincount(y)) <= .05*len(y)
    score = make_scorer(y[rows], weights, task, "ovr_auc", rare)
    x = rank[rows]
    bg_idx = np.random.default_rng(seed).choice(len(train), min(512, len(train)), replace=False)
    background = np.median(train[bg_idx], axis=0)
    masked = np.empty_like(x)
    anchor = frozenset(ordered[:k])
    cache = {}

    def assess(support):
        if support not in cache:
            columns = sorted(support)
            masked[:] = background
            masked[:, columns] = x[:, columns]
            cache[support] = score(model.predict_proba(masked))
        return cache[support]

    def ballot(subset, extension):
        after = assess(subset | extension)
        before = assess(anchor | extension)
        decisions = tuple(_within(a, b) for a, b in zip(after, before, strict=True))
        return decisions*4 if is_single else decisions

    rule = "mean" if arm == "pooled" else "min"
    selected, diagnostics = search(ordered[:k], groups, ballot, rule=rule)
    diagnostics.update(rows=rows.tolist(), n_panels=len(weights), mask_calls=len(cache))
    return sorted(selected), diagnostics


def select_l1(train, y, rank, yr, seed):
    scale = StandardScaler().fit(train)
    x = scale.transform(train)
    v = scale.transform(rank)
    candidates = []
    for strength in (.01, .1, 1., 10.):
        clf = OneVsRestClassifier(LogisticRegression(l1_ratio=1, solver="liblinear",
                                                     C=strength, random_state=seed, max_iter=2000))
        clf.fit(x, y)
        coef = np.concatenate([est.coef_ for est in clf.estimators_])
        selected = np.flatnonzero(np.any(coef != 0, axis=0)).tolist()
        if not selected:
            selected = [int(train.var(axis=0).argmax())]
        score = compute_utility(yr, clf.predict_proba(v), metric="ovr_auc")
        candidates.append((score, -len(selected), -strength, selected))
    best = max(candidates)
    return best[-1], {"chosen_C": -best[2], "ranking_auc": best[0]}


