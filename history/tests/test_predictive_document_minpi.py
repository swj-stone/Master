import numpy as np
import pytest
from scipy import sparse

from minppi_mofs.document_vote_minpi import document_vote_top_from_training
from minppi_mofs.predictive_document_minpi import predictive_document_top_from_training


@pytest.mark.parametrize("n,class_count", [(2, 2), (3, 3), (23, 2), (27, 4), (8, 1)])
def test_posterior_votes_match_explicit_leave_one_out_cell_probabilities(n, class_count):
    rng = np.random.default_rng(192)
    X = (rng.random((n, 11)) < .3).astype(float)
    X[:, 0] = 0
    X[:, 1] = 1
    X[:, 2] = X[:, 3]
    y = np.arange(n) % class_count
    _, info = predictive_document_top_from_training(sparse.csr_matrix(X), y)

    def loss(cols):
        losses = np.zeros(n)
        for held in range(n):
            matched = np.all(X[:, cols] == X[held, cols], axis=1)
            matched[held] = False
            numerator = np.sum(y[matched] == y[held]) + .5
            denominator = matched.sum() + .5 * class_count
            losses[held] = -np.log(numerator / denominator)
        return np.mean([losses[y == c].mean() for c in range(class_count)])

    first, alternate = info["peers"]
    baseline = loss([])
    assert info["intercept_loss"] == pytest.approx(baseline, abs=2e-12)
    for j in range(X.shape[1]):
        peer = alternate if j == first else first
        assert info["zero_peer_scores"][j] == pytest.approx(baseline - loss([j]), abs=2e-12)
        assert info["one_peer_scores"][j] == pytest.approx(loss([peer]) - loss([peer, j]), abs=2e-12)
    np.testing.assert_allclose(info["scores"],
                               np.minimum(info["zero_peer_scores"], info["one_peer_scores"]))


def test_posterior_votes_use_all_rows_independent_of_seed_and_positive_count_magnitude():
    rng = np.random.default_rng(391)
    X = sparse.csr_matrix((rng.random((801, 55)) < .15).astype(float))
    y = np.arange(801) % 3
    support, first = predictive_document_top_from_training(X, y, random_state=31)
    X.data *= 7
    other, repeated = predictive_document_top_from_training(X, y, random_state=79)
    np.testing.assert_array_equal(support, other)
    np.testing.assert_array_equal(first["scores"], repeated["scores"])
    assert first["rows_used"] == 801
    assert len(support) <= 28


def test_grouped_posterior_inputs_preserve_document_crossfit_fallback():
    rng = np.random.default_rng(371)
    X = sparse.csr_matrix((rng.random((40, 9)) < .25).astype(float))
    y = np.arange(40) % 2
    kwargs = dict(groups=np.repeat(np.arange(4), 10), ordered_groups=True, random_state=8)
    got, info = predictive_document_top_from_training(X, y, **kwargs)
    want, old = document_vote_top_from_training(X, y, **kwargs)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(info["scores"], old["scores"])
    assert info["estimator"] == "group_crossfit"
