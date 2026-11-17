"""Conditional expected marginal votes for coalitions on sampled tree paths."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from minppi_mofs.nonlinear_vote import _stratified_indices


@dataclass
class TreeRanks:
    minimum: np.ndarray
    pooled: np.ndarray
    margin: np.ndarray
    impurity: np.ndarray
    strata: np.ndarray


def path_basis(tree, p):
    """Row n identifies the distinct features needed to reach node n."""
    parents = np.zeros(tree.node_count, dtype=int)
    path = np.zeros((tree.node_count, p), dtype=bool)
    for node in range(tree.node_count):
        left, right = tree.children_left[node], tree.children_right[node]
        if left >= 0:
            children = [left, right]
            parents[children] = node
            path[children] = path[node]
            path[children, tree.feature[node]] = True
    return parents, path


def containment_probabilities(p, sizes, depth):
    """q[d,k] is P(a uniform k-of-(p-1) coalition contains d fixed players)."""
    q = np.ones((depth + 1, len(sizes)))
    for d in range(1, depth + 1):
        q[d] = q[d-1] * np.maximum(sizes-d+1, 0) / (p-d)
    return q


def path_contributions(tree, x_vote, one_hot, sizes):
    membership, vote, continuous = path_evidence(tree, x_vote, one_hot)
    d = np.maximum(membership.sum(axis=1) - 1, 0)
    q = containment_probabilities(x_vote.shape[1], sizes, int(d.max()))[d]
    return ((q * vote[:, None]).T @ membership,
            (q * continuous[:, None]).T @ membership)


def path_evidence(tree, x_vote, one_hot):
    parents, membership = path_basis(tree.tree_, x_vote.shape[1])
    counts = tree.decision_path(x_vote).T @ one_hot
    node = tree.tree_.value[:, 0, :]
    probability = node / node.sum(axis=1, keepdims=True)
    # Average pairwise true-class wins (half credit for ties).
    wins = (rankdata(probability, axis=1)-1) / (one_hot.shape[1]-1)
    vote = (counts * (wins - wins[parents])).sum(axis=1)
    continuous = (counts * (probability - probability[parents])).sum(axis=1)
    return membership, vote, continuous


def conditional_order(membership, weights, impurity, aggregation):
    """Greedy additions under the game conditioned on already selected features."""
    p = membership.shape[1]
    active = weights != 0
    basis = membership[active]
    weighted = np.ascontiguousarray(basis.T * weights[active])
    degree = basis.sum(axis=1)
    order = []
    chosen = np.zeros(p, dtype=bool)
    for remaining in range(p, 0, -1):
        sizes = np.unique(np.rint(np.array([.125, .5, .875]) * (remaining-1)).astype(int))
        d = np.maximum(degree - 1, 0)
        q = containment_probabilities(remaining, sizes, int(d.max(initial=0)))[d]
        strata = weighted @ q
        scores = strata.min(axis=1) if aggregation == "minimum" else strata.mean(axis=1)
        scores[chosen] = -np.inf
        selected = np.lexsort((np.arange(p), -impurity, -scores))[0]
        chosen[selected] = True
        order.append(selected)
        degree -= basis[:, selected]
    return np.array(order)


def rank_trees(x_fit, y_fit, x_vote, y_vote, *, seed, task="classification", trees=32, conditional=False):
    if task == "regression":
        cuts = np.unique(np.quantile(y_fit, [.25, .5, .75]))
        y_fit, y_vote = np.searchsorted(cuts, y_fit), np.searchsorted(cuts, y_vote)
    rng = np.random.default_rng(seed)
    fi = _stratified_indices(y_fit, 2048, rng)
    vi = _stratified_indices(y_vote, 512, rng)
    prep = SimpleImputer(strategy="median", keep_empty_features=True)
    fit = prep.fit_transform(x_fit[fi])
    vote = prep.transform(x_vote[vi])
    model = RandomForestClassifier(n_estimators=trees, max_depth=7, min_samples_leaf=3,
                                   n_jobs=1, random_state=seed).fit(fit, y_fit[fi])
    one_hot = (y_vote[vi, None] == model.classes_).astype(float)
    if conditional:
        one_hot /= np.maximum(one_hot.sum(axis=0), 1)
    p = fit.shape[1]
    sizes = np.unique(np.rint(np.array([.125, .5, .875]) * (p-1)).astype(int))
    swings = np.zeros((len(sizes), p))
    continuous = np.zeros_like(swings)
    if conditional:
        rows = [path_evidence(t, vote, one_hot) for t in model.estimators_]
        membership, net, change = (np.concatenate(a) for a in zip(*rows))
        impurity = model.feature_importances_
        d = np.maximum(membership.sum(axis=1)-1, 0)
        q = containment_probabilities(p, sizes, int(d.max()))[d]
        swings = (q * net[:, None]).T @ membership / (len(model.classes_) * trees)
        return TreeRanks(conditional_order(membership, net, impurity, "minimum"),
                         conditional_order(membership, net, impurity, "mean"),
                         conditional_order(membership, change, impurity, "minimum"),
                         np.lexsort((np.arange(p), -impurity)), swings)
    for tree in model.estimators_:
        discrete, margin = path_contributions(tree, vote, one_hot, sizes)
        swings += discrete
        continuous += margin
    swings /= len(vote) * trees
    continuous /= len(vote) * trees
    impurity = model.feature_importances_
    def ordered(score):
        return np.lexsort((np.arange(p), -impurity, -score))
    return TreeRanks(ordered(swings.min(axis=0)), ordered(swings.mean(axis=0)),
                     ordered(continuous.min(axis=0)), ordered(impurity), swings)
