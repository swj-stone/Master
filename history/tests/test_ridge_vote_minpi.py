from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.ridge_vote_minpi import ridge_vote_top_from_training


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ridge_votes_equal_explicit_uniform_coalition_losses(task):
    rng = np.random.default_rng(70)
    X = rng.normal(size=(81, 5))
    y = X[:, 0] + X[:, 1] - X[:, 3]
    if task == "classification":
        y = np.digitize(y, [-1., .3])
        target = np.eye(3)[y]
    else:
        target = y[:, None]
    X[rng.random(X.shape) < .16] = np.nan
    X[:, 4] = .1
    cols, info = ridge_vote_top_from_training(X, y, 3, task=task)
    gains = np.zeros((3, X.shape[1]))
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        x, z = X[fit].copy(), X[held].copy()
        center = np.nanmean(x, axis=0)
        x = np.nan_to_num(x - center)
        z = np.nan_to_num(z - center)
        x[:, 4] = z[:, 4] = 0.
        scale = np.sqrt((x**2).mean(axis=0))
        scale[scale == 0.] = 1.
        x, z = x / scale, z / scale
        mean = target[fit].mean(axis=0)
        beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ (target[fit] - mean))
        def loss(active):
            pred = z[:, active] @ beta[active, :] + mean
            return ((target[held] - pred)**2).sum(axis=1).mean()
        for i in range(X.shape[1]):
            rest = [j for j in range(X.shape[1]) if j != i]
            for tier, k in enumerate([0, 2, 4]):
                gains[tier, i] += np.mean([loss(list(S)) - loss([*S, i])
                                           for S in combinations(rest, k)]) / 2.
    np.testing.assert_allclose(info["zero_peer_scores"], gains[0], atol=1e-12)
    np.testing.assert_allclose(info["full_peer_scores"], gains[2], atol=1e-12)
    np.testing.assert_allclose(info["scores"], gains[1:].min(axis=0), atol=1e-12)
    np.testing.assert_array_equal(cols, np.sort(np.argsort(-info["scores"], kind="stable")[:3]))


def test_adaptive_guard_keeps_signal_and_respects_time_order():
    rng = np.random.default_rng(83)
    X = rng.normal(size=(202, 24))
    y = 3 * X[:, 4] - 2.3 * X[:, 10] + .06 * rng.normal(size=len(X))
    groups = np.arange(len(X)) // 52
    args = dict(task="regression", groups=groups, ordered_groups=True)
    selected, info = ridge_vote_top_from_training(X, y, **args)
    assert {4, 10}.issubset(selected) and len(selected) < X.shape[1]
    [fold] = info["folds"]
    assert max(groups[fold["fit_rows"]]) < min(groups[fold["held_rows"]])
    assert len(fold["fit_rows"]) <= 96 and len(fold["held_rows"]) <= 96
    np.testing.assert_array_equal(ridge_vote_top_from_training(X, y, **args)[0], selected)


def test_no_target_evidence_keeps_all_and_fixed_width_breaks_ties_by_index():
    X = np.ones((9, 12))
    y = np.zeros(9)
    all_cols, info = ridge_vote_top_from_training(X, y)
    np.testing.assert_array_equal(all_cols, np.arange(12))
    assert not info["scores"].any()
    np.testing.assert_array_equal(ridge_vote_top_from_training(X, y, 3)[0], np.arange(3))
    one, _ = ridge_vote_top_from_training(X[:, :1], y)
    np.testing.assert_array_equal(one, [0])


def test_wide_dual_ridge_matches_an_explicit_primal_loss_of_deleting_one_column():
    rng = np.random.default_rng(3407)
    X = rng.normal(size=(16, 31))
    y = 1.3 * X[:, 10] - .9 * X[:, 19] + rng.normal(size=len(X))
    group = np.repeat([0, 1], 8)
    _, info = ridge_vote_top_from_training(X, y, 4, task="regression", groups=group, ordered_groups=True)
    fit, held = X[:8], X[8:]
    x = (fit - fit.mean(axis=0)) / fit.std(axis=0)
    z = (held - fit.mean(axis=0)) / fit.std(axis=0)
    beta = np.linalg.solve(x.T @ x + (31 / 8) * np.eye(31), x.T @ (y[:8] - y[:8].mean()))
    pred = y[:8].mean() + z @ beta
    full_loss = np.square(y[8:] - pred).mean()
    deleted_loss = np.square(y[8:, None] - pred[:, None] + z * beta).mean(axis=0)
    np.testing.assert_allclose(info["full_peer_scores"], deleted_loss - full_loss, atol=1e-12)
