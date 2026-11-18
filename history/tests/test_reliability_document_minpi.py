import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from minppi_mofs.document_vote_minpi import document_vote_top_from_training
from minppi_mofs.fast_document_vote_minpi import library_path
from minppi_mofs.reliability_document_minpi import reliability_document_top_from_training


def test_each_direction_and_context_matches_explicit_ridge_refits():
    rng = np.random.default_rng(519)
    X = (rng.random((53, 7)) < .3).astype(float)
    X[:, 0] = 0
    X[:, 1] = X[:, 2]
    y = (np.arange(len(X)) % 3).astype(int)
    _, info = reliability_document_top_from_training(sparse.csr_matrix(X), y, backend="numpy")
    target = np.eye(3)[y]
    endpoints = []
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]

        def loss(cols):
            if cols:
                estimator = Ridge(alpha=1., solver="cholesky")
                prediction = estimator.fit(X[fit][:, cols], target[fit]).predict(X[held][:, cols])
            else:
                prediction = target[fit].mean(axis=0)
            return np.square(target[held] - prediction).sum() / len(held)

        first, alternate = fold["peers"]
        scores = np.empty((2, X.shape[1]))
        for j in range(X.shape[1]):
            peer = alternate if j == first else first
            scores[:, j] = (loss([]) - loss([j]), loss([peer]) - loss([peer, j]))
        endpoints.append(scores)
    endpoints = np.asarray(endpoints)
    np.testing.assert_allclose(info["directional_endpoints"], endpoints, atol=2e-12)
    standard_error = np.std(endpoints, axis=0, ddof=1) / np.sqrt(2.)
    reduced = endpoints.mean(axis=0) - standard_error
    np.testing.assert_allclose(np.stack((info["zero_peer_scores"], info["one_peer_scores"])),
                               reduced, atol=2e-12)
    np.testing.assert_allclose(info["scores"], reduced.min(axis=0), atol=2e-12)


@pytest.mark.skipif(not library_path().exists(), reason="Optional document kernel not built")
@pytest.mark.parametrize("ordered", [False, True])
def test_native_and_numpy_reliability_votes_preserve_group_isolation(ordered):
    rng = np.random.default_rng(710)
    X = sparse.csr_matrix(rng.binomial(1, .11, size=(701, 73)) * 3)
    y = np.arange(X.shape[0]) % 2
    groups = np.arange(X.shape[0]) // 101
    kwargs = dict(groups=groups, ordered_groups=ordered, random_state=847)
    support, native = reliability_document_top_from_training(X, y, backend="native", **kwargs)
    expected, plain = reliability_document_top_from_training(X, y, backend="numpy", **kwargs)
    np.testing.assert_array_equal(support, expected)
    for field in ("scores", "zero_peer_scores", "one_peer_scores", "directional_endpoints"):
        np.testing.assert_array_equal(native[field], plain[field])
    for fold in native["folds"]:
        fit, held = groups[fold["fit_rows"]], groups[fold["held_rows"]]
        assert not np.intersect1d(fit, held).size
        if ordered:
            assert fit.max() < held.min()
    if ordered:
        previous, info = document_vote_top_from_training(X, y, **kwargs)
        np.testing.assert_array_equal(support, previous)
        np.testing.assert_array_equal(native["scores"], info["scores"])
