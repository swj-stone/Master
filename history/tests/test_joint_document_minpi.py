from itertools import combinations

import numpy as np
import pytest
from scipy import sparse

from minppi_mofs.joint_document_minpi import joint_document_top_from_training


@pytest.mark.parametrize("ordered", [False, True])
def test_joint_votes_equal_all_masked_coalition_size_expectations(ordered):
    rng = np.random.default_rng(619)
    X = (rng.random((97, 6)) < .35).astype(float)
    X[:, 0] = 1
    y = np.digitize(X[:, 2] + .8 * X[:, 4] + rng.normal(size=len(X)) * .3, [.2, 1.1])
    groups = np.arange(len(X)) // 21
    _, info = joint_document_top_from_training(sparse.csr_matrix(X), y, groups=groups,
                                               ordered_groups=ordered, max_iter=8)
    expected = np.zeros((X.shape[1], X.shape[1]))
    held_count = sum(len(fold["held_rows"]) for fold in info["folds"])
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(groups[fit], groups[held]).size
        assert len(np.union1d(fit, held)) == len(X)
        if ordered:
            assert groups[fit].max() < groups[held].min()
        contributions = (X[held] - fold["means"])[:, :, None] * fold["coefficients"][None]
        truth = np.eye(3)[y[held]]

        def loss(columns):
            pred = fold["target_mean"] + contributions[:, columns].sum(axis=1)
            return np.square(truth - pred).sum() / len(held)

        for j in range(X.shape[1]):
            others = [k for k in range(X.shape[1]) if k != j]
            for size in range(X.shape[1]):
                gains = [loss(list(cols)) - loss([*cols, j]) for cols in combinations(others, size)]
                expected[size, j] += len(held) / held_count * np.mean(gains)
    np.testing.assert_allclose(expected[0], info["zero_peer_scores"], atol=2e-12)
    np.testing.assert_allclose(expected[-1], info["full_peer_scores"], atol=2e-12)
    np.testing.assert_allclose(expected.min(axis=0), info["scores"], atol=2e-12)
    for size in range(X.shape[1]):
        interpolation = expected[0] + size / (X.shape[1] - 1) * (expected[-1] - expected[0])
        np.testing.assert_allclose(expected[size], interpolation, atol=2e-12)


def test_joint_presence_counts_replay_and_fixed_support():
    rng = np.random.default_rng(719)
    X = sparse.csr_matrix((rng.random((810, 37)) < .13).astype(float))
    y = rng.integers(0, 2, X.shape[0])
    first, a = joint_document_top_from_training(X, y)
    second, b = joint_document_top_from_training(X * 4, y)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(a["scores"], b["scores"])
    assert sum(len(f["held_rows"]) for f in a["folds"]) == len(y)
    all_columns, _ = joint_document_top_from_training(X, y, X.shape[1])
    np.testing.assert_array_equal(all_columns, np.arange(X.shape[1]))
    with pytest.raises(ValueError, match="positive integer"):
        joint_document_top_from_training(X, y, max_iter=0)
