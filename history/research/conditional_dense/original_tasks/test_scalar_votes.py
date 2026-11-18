import numpy as np

from research.conditional_dense.original_tasks.regression import (
    regression_geometric_support, regression_support,
)


def test_scalar_votes_match_direct_refits_and_the_two_budget_formulas():
    rng = np.random.default_rng(915)
    X = rng.normal(size=(31, 12))
    X[:, -1] = X[:, 0]
    X[:, -2] = 4.
    y = X[:, 2] + 0.5 * X[:, 5] + rng.normal(0., 0.1, len(X))
    seed = 81
    cols, info = regression_support(X, y, seed=seed)

    Z = X - X.mean(axis=0)
    Z /= np.where(X.std(axis=0) > 0, X.std(axis=0), 1.)
    target = (y - y.mean()) / y.std()
    order = np.random.default_rng(seed).permutation(X.shape[1])
    penalty = np.sqrt(len(X))

    def loss(support):
        matrix = Z[:, support]
        beta = np.linalg.solve(matrix.T @ matrix + penalty * np.eye(len(support)),
                               matrix.T @ target)
        return np.sum((target - matrix @ beta) ** 2) + penalty * np.sum(beta ** 2)

    expected = []
    for size in (1, 3):
        gains = []
        for j in range(X.shape[1]):
            peers = order[order != j][:size]
            gains.append(loss(peers) - loss(np.append(peers, j)))
        expected.append(gains)
    expected = np.asarray(expected)
    np.testing.assert_allclose(info["tier_scores"], expected, rtol=1e-4, atol=4e-6)
    scores = expected.min(axis=0)
    np.testing.assert_allclose(info["rank"], scores, rtol=1e-4, atol=4e-6)

    peers, rest = order[:3], order[3:]
    design = Z[:, peers]
    cross = design.T @ Z[:, rest]
    projected = np.linalg.solve(design.T @ design + penalty * np.eye(3), cross)
    variance = np.sum(Z[:, rest] ** 2, axis=0) - (projected * cross).sum(axis=0)
    width = 3 + int(np.ceil(np.sqrt(np.clip(variance / len(y), 0., 1.)).sum()))
    expected_k = min(width, int(np.ceil(scores.sum() ** 2 / (scores @ scores))))
    assert width == info["width"]
    assert expected_k == info["selected_width"] == len(cols)
    np.testing.assert_array_equal(cols, np.sort(np.argsort(-scores, kind="stable")[:expected_k]))


def test_large_sketch_is_reproducible_uses_only_sampled_rows_and_is_unit_invariant():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(901, 20))
    y = 2 * X[:, 1] - X[:, 17] + rng.normal(size=len(X))
    seed = 102
    cols, info = regression_support(X, y, seed=seed)
    old_cols, old_info = regression_geometric_support(X, y, seed=seed)
    assert info["sketch_size"] == 768
    assert len(old_cols) == info["width"] == old_info["selected_width"]
    np.testing.assert_array_equal(old_info["rank"], info["rank"])
    assert np.isin(cols, old_cols).all()

    sample = np.random.default_rng(seed).choice(len(X), 768, replace=False)
    outside = np.setdiff1d(np.arange(len(X)), sample)
    y[outside] = 1.e8
    X[outside] = 123.
    units_changed, changed = regression_support(X, -7.25 * y + 19., seed=seed)
    np.testing.assert_array_equal(units_changed, cols)
    np.testing.assert_allclose(changed["rank"], info["rank"], rtol=3e-6, atol=2e-6)


def test_zero_votes_use_geometric_width_and_stable_column_indices():
    X = np.zeros((21, 8))
    selected, info = regression_support(X, np.full(len(X), 3.1))
    assert info["selected_width"] == info["width"] == 2
    np.testing.assert_array_equal(info["rank"], np.zeros(8))
    np.testing.assert_array_equal(selected, [0, 1])
    singleton, single_info = regression_support(np.ones((1, 1)), [2.])
    assert single_info["selected_width"] == 1
    np.testing.assert_array_equal(singleton, [0])
