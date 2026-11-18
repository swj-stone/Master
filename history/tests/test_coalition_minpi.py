from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.coalition_minpi import _greedy_coalition_order, coalition_top_from_training
from minppi_mofs.utility_minpi import _fitted_halves


def _enumerated_order(folds, count, tiers=None):
    """Build actual masked predictions for every peer set on each held row."""
    p = folds[0][0].shape[1]
    fixed, history = [], []
    for step in range(count):
        candidates = [j for j in range(p) if j not in fixed]
        levels = (range(count - step) if tiers is None else
                  {min(k, len(candidates) - 1) for k in tiers})
        minima = []
        for feature in candidates:
            others = [j for j in candidates if j != feature]
            scores = []
            for k in levels:
                set_means = []
                for peers in combinations(others, k):
                    cols = fixed + list(peers)
                    votes = []
                    for check, target, coef in folds:
                        residual = target - check[:, cols] @ coef[cols]
                        new_residual = residual - check[:, [feature]] * coef[feature]
                        votes.extend((residual ** 2 - new_residual ** 2).sum(axis=1))
                    set_means.append(np.mean(votes))
                scores.append(np.mean(set_means))
            minima.append(min(scores))
        winner = np.argmax(minima)
        fixed.append(candidates[winner])
        history.append(minima[winner])
    return fixed, history


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("tiers", [None, (1, 3, 5)])
def test_greedy_matches_exhaustive_masked_coalitions(task, tiers):
    rng = np.random.default_rng(923)
    X = rng.normal(size=(83, 6))
    X[::4, 0] = np.nan
    y = rng.permutation(np.arange(len(X)) % 3) if task == "classification" else rng.normal(size=len(X))
    folds = list(_fitted_halves(X, y, task, 1129))
    order, expected = _enumerated_order(folds, 4, tiers)
    support, info = coalition_top_from_training(X, y, 4, task=task, random_state=1129, tiers=tiers)
    assert order == info["selection_order"]
    np.testing.assert_array_equal(support, sorted(order))
    np.testing.assert_allclose(info["worst_gains"], expected, rtol=1e-10, atol=1e-12)


def test_fixing_winner_changes_next_choice_for_redundant_effects():
    check = np.array([[1., 1., 0.], [0., 0., 1.]])
    folds = [(check, np.array([[2.], [1.25]]), np.ones((3, 1)))]
    # Both first-round duplicates score 1.0, above the independent 0.75.
    # Once feature 0 is present, its duplicate adds only 0.5.
    order, scores = _greedy_coalition_order(folds, 2, None)
    assert order == [0, 2]
    np.testing.assert_allclose(scores, [1., .75])


def test_ties_and_full_budget_shortcut():
    X = np.ones((21, 5))
    y = np.arange(len(X), dtype=float)
    for count in (1, 3, 5):
        cols, info = coalition_top_from_training(X, y, count, task="regression")
        np.testing.assert_array_equal(cols, np.arange(count))
        assert info["ballot_rows"] == (0 if count == 5 else len(X))
    with pytest.raises(ValueError, match="count"):
        coalition_top_from_training(X, y, False)
    with pytest.raises(ValueError, match="Peer counts"):
        coalition_top_from_training(X, y, 5, tiers=[5])
