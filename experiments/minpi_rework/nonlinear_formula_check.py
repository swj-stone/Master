"""Exhaustive finite check of stratified marginal votes on a tiny tree."""
from itertools import combinations

import numpy as np
from scipy.stats import rankdata
from sklearn.tree import DecisionTreeClassifier

from experiments.minpi_rework.nonlinear_vote_tree import path_contributions


def brute_vote(tree, rows, labels, subset):
    total = 0.0
    for x, y in zip(rows, labels):
        node = 0
        while tree.children_left[node] >= 0 and tree.feature[node] in subset:
            j = tree.feature[node]
            node = tree.children_left[node] if x[j] <= tree.threshold[node] else tree.children_right[node]
        values = tree.value[node, 0]
        total += (rankdata(values)[y]-1)/(len(values)-1)
    return total


def check():
    rng = np.random.default_rng(3217)
    fit, vote = rng.normal(size=(128, 4)), rng.normal(size=(17, 4))
    y_fit = np.digitize(fit[:, 0]*fit[:, 1]+fit[:, 2], [-.4, .4])
    y_vote = np.digitize(vote[:, 0]*vote[:, 1]+vote[:, 2], [-.4, .4])
    model = DecisionTreeClassifier(max_depth=6, random_state=11).fit(fit, y_fit)
    sizes = np.arange(4)
    analytic, _ = path_contributions(model, vote, np.eye(3)[y_vote], sizes)
    brute = np.empty_like(analytic)
    for size in sizes:
        for j in range(4):
            backgrounds = combinations(set(range(4)) - {j}, size)
            brute[size, j] = np.mean([brute_vote(model.tree_, vote, y_vote, (*s, j)) -
                                      brute_vote(model.tree_, vote, y_vote, s) for s in backgrounds])
    np.testing.assert_allclose(analytic, brute, atol=1e-12)
    print("All 16 feature / stratum combinations agree with exhaustive tree traversal.")


if __name__ == "__main__":
    check()
