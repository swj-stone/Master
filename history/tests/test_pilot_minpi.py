from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.pilot_minpi import _coverage_scores, pilot_top_from_training


def test_weakest_vote_agrees_with_enumerated_coverage_game():
    parent = np.array([0, 0, 0, 3, 3, 5])
    evidence = np.array([.8, .3, -.2, .5, .5, -.1])
    def value(subset):
        return sum(max([0.] + [evidence[i] for i in subset if parent[i] == g])
                   for g in np.unique(parent))
    actual = []
    for j in range(len(parent)):
        others = set(range(len(parent))) - {j}
        by_size = [np.mean([value(set(s) | {j}) - value(s) for s in combinations(others, k)])
                   for k in range(len(parent))]
        actual.append(min(by_size))
    np.testing.assert_allclose(_coverage_scores(evidence, parent), actual, atol=1e-15)


@pytest.mark.parametrize("ordered", [True, False])
def test_signed_copies_are_replaced_and_groups_do_not_cross(ordered):
    rng = np.random.default_rng(812)
    x = rng.choice([-1., 1.], size=(400, 4)) + .05 * rng.normal(size=(400, 4))
    X = np.column_stack([x, 2*x[:, 0]+3., -x[:, 1], x[:, 2]*.5])
    groups = np.arange(len(x)) // 20
    y = x[:, 0] - x[:, 1] + rng.normal(size=len(x)) * .2
    kwargs = dict(task="regression", groups=groups, ordered_groups=ordered)
    cols, info = pilot_top_from_training(X, y, **kwargs)
    assert len(cols) == info["groups_found"] == 4
    for duplicates in ([0, 4], [1, 5], [2, 6], [3]):
        assert len(np.intersect1d(cols, duplicates)) == 1
    a, b = (info["folds"][0][k] for k in ("fit_rows", "held_rows"))
    assert not np.intersect1d(groups[a], groups[b]).size
    if ordered:
        assert groups[a].max() < groups[b].min()
    X[np.setdiff1d(np.arange(len(X)), np.r_[a, b])] = np.nan
    again, _ = pilot_top_from_training(X, y, **kwargs)
    np.testing.assert_array_equal(cols, again)


def test_no_proxy_adapts_to_one_deletion_and_honors_fixed_width():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(250, 9))
    X[::25, :4] = np.nan
    y = np.arange(len(X)) % 3
    cols, info = pilot_top_from_training(X, y)
    assert len(cols) == 8 and info["groups_found"] == 9
    assert len(pilot_top_from_training(X, y, 3)[0]) == 3
    np.testing.assert_array_equal(pilot_top_from_training(X[:, :1], y)[0], [0])
