import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.repeated_document_minpi import repeated_document_top_from_training


@pytest.mark.parametrize("ordered", [False, True])
def test_repeated_endpoints_equal_explicit_held_ridge_and_then_take_minimum(ordered):
    rng = np.random.default_rng(491)
    X = (rng.random((125, 6)) < .3).astype(float)
    y = (X[:, 1] + X[:, 2] + .2 * rng.normal(size=len(X)) > .5).astype(int)
    groups = np.arange(len(X)) // 26
    _, info = repeated_document_top_from_training(sparse.csr_matrix(X), y, groups=groups,
                                                  ordered_groups=ordered, aggregation="median")
    expected = {}
    target = np.eye(2)[y]
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(groups[fit], groups[held]).size
        if ordered:
            assert groups[fit].max() < groups[held].min()

        def loss(cols):
            if cols:
                model = Ridge(alpha=1., solver="svd").fit(X[fit][:, cols], target[fit])
                pred = model.predict(X[held][:, cols])
            else:
                pred = target[fit].mean(axis=0)
            return np.square(target[held] - pred).sum() / len(held)

        endpoint = np.zeros((2, X.shape[1]))
        first, second = fold["peers"]
        for j in range(X.shape[1]):
            peer = second if j == first else first
            endpoint[:, j] = loss([]) - loss([j]), loss([peer]) - loss([peer, j])
        expected.setdefault(fold["repeat_seed"], []).append(endpoint)
    expected = np.array([np.mean(directions, axis=0) for directions in expected.values()])
    np.testing.assert_allclose(info["replicate_endpoints"], expected, atol=1e-12)
    pooled = np.median(expected, axis=0)
    np.testing.assert_allclose(info["zero_peer_scores"], pooled[0], atol=1e-12)
    np.testing.assert_allclose(info["one_peer_scores"], pooled[1], atol=1e-12)
    np.testing.assert_allclose(info["scores"], pooled.min(axis=0), atol=1e-12)


def test_one_repeat_preserves_original_seed_scores_and_support():
    rng = np.random.default_rng(881)
    X = sparse.csr_matrix((rng.random((900, 33)) < .2).astype(float))
    y = rng.integers(0, 2, X.shape[0])
    old, a = fast_document_vote_top_from_training(X, y, random_state=717)
    one, b = repeated_document_top_from_training(X, y, random_state=717, repeats=1)
    np.testing.assert_array_equal(one, old)
    np.testing.assert_array_equal(b["scores"], a["scores"])
    assert b["folds"][0]["repeat_seed"] == 717
    _, c = repeated_document_top_from_training(X, y, random_state=717, aggregation="shrinkage")
    for score, raw in zip((c["zero_peer_scores"], c["one_peer_scores"]),
                          c["replicate_endpoints"].mean(axis=0)):
        assert np.all(np.abs(score) <= np.abs(raw) + 1e-15)
        assert np.all(np.sign(score) == np.sign(raw))
    repeated, d = repeated_document_top_from_training(X, y, random_state=717)
    again, _ = repeated_document_top_from_training(X, y, random_state=717)
    np.testing.assert_array_equal(repeated, again)
    assert len(set(f["repeat_seed"] for f in d["folds"])) == 3
    with pytest.raises(ValueError, match="positive integer"):
        repeated_document_top_from_training(X, y, repeats=0)


def test_numpy_backend_never_loads_the_native_kernel(monkeypatch):
    def unexpected_native_load():
        raise AssertionError("Pure NumPy replay must bypass the native kernel")

    monkeypatch.setattr("minppi_mofs.fast_document_vote_minpi.load_native", unexpected_native_load)
    X = sparse.csr_matrix(np.arange(200).reshape(20, 10) % 3 == 0)
    y = np.arange(20) % 2
    columns, info = repeated_document_top_from_training(X, y, backend="numpy")
    assert len(columns) == info["budget"]
