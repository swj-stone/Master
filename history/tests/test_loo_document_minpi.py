import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from minppi_mofs.document_vote_minpi import document_vote_top_from_training
from minppi_mofs.loo_document_minpi import loo_document_top_from_training


@pytest.mark.parametrize("n", [2, 3, 23])
def test_loo_votes_match_explicit_refits_for_fixed_contexts(n):
    rng = np.random.default_rng(971)
    X = (rng.random((n, 9)) < .3).astype(float)
    X[:, 0] = 0
    X[:, 1] = 1
    X[:, 2] = X[:, 3]
    y = np.arange(n) % 2
    _, info = loo_document_top_from_training(sparse.csr_matrix(X), y)

    def loss(cols):
        errors = []
        for held in range(n):
            train = np.arange(n) != held
            if cols:
                model = Ridge(alpha=1., solver="cholesky").fit(X[train][:, cols], y[train])
                prediction = model.predict(X[[held]][:, cols])[0]
            else:
                prediction = y[train].mean()
            errors.append(2 * (prediction - y[held])**2)
        return np.mean(errors)

    first, alternate = info["peers"]
    baseline = loss([])
    for j in range(X.shape[1]):
        peer = alternate if j == first else first
        assert info["zero_peer_scores"][j] == pytest.approx(baseline - loss([j]), abs=2e-12)
        assert info["one_peer_scores"][j] == pytest.approx(loss([peer]) - loss([peer, j]), abs=2e-12)
    np.testing.assert_allclose(info["scores"],
                               np.minimum(info["zero_peer_scores"], info["one_peer_scores"]))


def test_seed_and_sparse_count_storage_leave_support_and_scores_identical():
    rng = np.random.default_rng(191)
    X = sparse.csr_matrix((rng.random((301, 65)) < .1).astype(float))
    y = (X[:, 7].toarray().ravel() > 0).astype(int)
    first, info = loo_document_top_from_training(X, y, random_state=1)
    X.data *= 3
    second, repeated = loo_document_top_from_training(X, y, random_state=199)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(info["scores"], repeated["scores"])
    assert 1 <= len(first) <= 33
    assert info["rows_used"] == 301


def test_grouped_rows_keep_original_crossfit_semantics():
    rng = np.random.default_rng(981)
    X = sparse.csr_matrix((rng.random((80, 9)) < .25).astype(float))
    y = np.arange(80) % 2
    groups = np.repeat([0, 1, 2, 3], 20)
    args = dict(groups=groups, ordered_groups=True, random_state=31)
    got, info = loo_document_top_from_training(X, y, **args)
    expected, old = document_vote_top_from_training(X, y, **args)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(info["scores"], old["scores"])
    assert groups[info["folds"][0]["fit_rows"]].max() < groups[info["folds"][0]["held_rows"]].min()
