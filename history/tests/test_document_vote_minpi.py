import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from minppi_mofs.document_vote_minpi import document_vote_top_from_training


@pytest.mark.parametrize("ordered", [False, True])
def test_sparse_contingencies_equal_dense_signed_held_ridge_losses(ordered):
    rng = np.random.default_rng(836)
    X = (rng.random((840, 29)) < .23).astype(float)
    label = np.digitize(X[:, 7] + .6*X[:, 2] + .3*rng.normal(size=len(X)), [-.1, .7])
    X[:, 13] = X[:, 2]
    X[:, 0] = 1
    groups = np.repeat(np.arange(6), 140)
    counts = X * rng.integers(1, 6, size=X.shape)
    selected, info = document_vote_top_from_training(sparse.csr_matrix(counts), label, groups=groups,
                                                     ordered_groups=ordered)
    targets = np.eye(3)[label]
    expected = np.zeros((2, X.shape[1]))
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(groups[fit], groups[held]).size
        if ordered:
            assert groups[fit].max() < groups[held].min()

        def loss(cols):
            if cols:
                pred = Ridge(alpha=1., solver="svd").fit(X[fit][:, cols], targets[fit]).predict(X[held][:, cols])
            else:
                pred = targets[fit].mean(axis=0)
            return np.square(targets[held] - pred).sum() / len(held)

        anchor, alternate = fold["peers"]
        for col in range(X.shape[1]):
            peer = alternate if col == anchor else anchor
            expected[0, col] += loss([]) - loss([col])
            expected[1, col] += loss([peer]) - loss([peer, col])
    expected /= len(info["folds"])
    np.testing.assert_allclose(expected[0], info["zero_peer_scores"], atol=1e-12)
    np.testing.assert_allclose(expected[1], info["one_peer_scores"], atol=1e-12)
    assert len(selected) <= int(np.ceil(X.shape[1]/2))
    binary, _ = document_vote_top_from_training(sparse.csr_array(X), label, groups=groups, ordered_groups=ordered)
    np.testing.assert_array_equal(selected, binary)


def test_stored_zero_duplicates_and_invalid_entries():
    cols = np.array([1, 1, 2, 3, 2, 1], dtype=np.int32)
    rows = np.array([0, 4, 5, 6], dtype=np.int32)
    mat = sparse.csr_matrix((np.array([1, 1, 0, 2, 1, 0]), cols, rows), shape=(3, 7))
    labels = [0, 1, 0]
    expected = sparse.csr_matrix(mat.toarray() > 0)
    _, a = document_vote_top_from_training(mat, labels)
    fixed, b = document_vote_top_from_training(expected, labels, 7)
    np.testing.assert_array_equal(a["scores"], b["scores"])
    np.testing.assert_array_equal(fixed, np.arange(7))
    assert mat.nnz == 6
    with pytest.raises(ValueError, match="finite nonnegative"):
        document_vote_top_from_training(-mat, labels)
