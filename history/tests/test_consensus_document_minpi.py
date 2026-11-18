import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from minppi_mofs.consensus_document_minpi import consensus_document_top_from_training
from minppi_mofs.standard_document_minpi import standard_document_top_from_training


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("max_peers", [1, 4])
@pytest.mark.parametrize("standardized", [False, True])
def test_all_row_context_expectations_match_explicit_ridge_losses(ordered, max_peers, standardized):
    rng = np.random.default_rng(359)
    X = (rng.random((617, 7)) < .3).astype(float)
    X[:, 1] = X[:, 2]
    X[:, 0] = 1
    X[:, 6] = 0
    X[:3, 6] = 1
    y = np.digitize(X[:, 2] + X[:, 4] + rng.normal(size=len(X)) * .3, [.2, 1.2])
    groups = np.arange(len(y)) // 73
    selector = standard_document_top_from_training if standardized else consensus_document_top_from_training
    _, info = selector(sparse.csr_matrix(X), y, max_peers=max_peers,
                       groups=groups, ordered_groups=ordered)
    target = np.eye(3)[y]
    expected = np.zeros((2, X.shape[1]))
    total_held = sum(len(f["held_rows"]) for f in info["folds"])
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert len(np.union1d(fit, held)) == len(X)
        assert np.intersect1d(groups[fit], groups[held]).size == 0
        if ordered:
            assert groups[fit].max() < groups[held].min()

        def loss(cols):
            if cols:
                model = Ridge(alpha=1., solver="svd")
                if standardized:
                    model = make_pipeline(StandardScaler(), model)
                model.fit(X[fit][:, cols], target[fit])
                pred = model.predict(X[held][:, cols])
            else:
                pred = target[fit].mean(axis=0)
            return np.square(target[held] - pred).sum() / len(held)

        for j in range(X.shape[1]):
            peers = [k for k in fold["anchors"] if k != j][:info["peer_count"]]
            one = np.mean([loss([k]) - loss([k, j]) for k in peers])
            expected[:, j] += len(held) / total_held * np.array([loss([]) - loss([j]), one])
    np.testing.assert_allclose(info["zero_peer_scores"], expected[0], atol=1e-12)
    np.testing.assert_allclose(info["one_peer_scores"], expected[1], atol=1e-12)
    np.testing.assert_allclose(info["scores"], expected.min(axis=0), atol=1e-12)


def test_all_rows_used_input_untouched_seed_replay_and_fixed_count():
    rng = np.random.default_rng(851)
    X = sparse.csr_matrix((rng.random((921, 23)) < .16).astype(int) * 3)
    y = rng.integers(0, 2, X.shape[0])
    original = X.copy()
    first, info = consensus_document_top_from_training(X, y, random_state=171)
    second, replay = consensus_document_top_from_training(X, y, random_state=171)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(info["scores"], replay["scores"])
    np.testing.assert_array_equal(X.data, original.data)
    assert sum(len(f["held_rows"]) for f in info["folds"]) == len(y)
    assert 1 <= len(first) <= (X.shape[1] + 1) // 2
    fixed, _ = consensus_document_top_from_training(X, y, 23)
    np.testing.assert_array_equal(fixed, np.arange(23))
    with pytest.raises(ValueError, match="positive integer"):
        consensus_document_top_from_training(X, y, max_peers=0)
    with pytest.raises(ValueError, match="classification"):
        consensus_document_top_from_training(X, y, task="regression")
