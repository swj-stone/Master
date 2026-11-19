"""Deterministic checks for the dense, CSR, capped and screened wide paths."""
import numpy as np
import pytest
from scipy import sparse

from research.lowdim.method import block_support
from research.wide_blocks.method import _path, _response, _sample, _Sketch, wide_support


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("n", [128, 819])
def test_exact_kernel_matches_lowdim(task, n):
    rng = np.random.default_rng(427)
    X = rng.normal(size=(n, 27))
    signal = X[:, :12] @ rng.normal(size=12)
    y = np.digitize(signal, [-1, 1]) if task == "classification" else signal
    expected, audit = block_support(X, y, task=task, seed=917, audit=True)
    rows = _sample(n, 917)
    targets, loss = _response(y, rows, task)
    z, variable = _Sketch(X, rows).standardize(np.arange(X.shape[1]))
    order, blocks, cap, votes = _path(z, variable, targets, loss, False)
    np.testing.assert_allclose(votes[:2], audit["tier_scores"], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(order, audit["selection_order"])
    np.testing.assert_array_equal(np.sort(order), expected)
    np.testing.assert_array_equal(blocks, audit["block_sizes"])
    assert cap == audit["width_cap"]


@pytest.mark.parametrize(("p", "limit"), [(33, 128), (128, 128), (129, 128),
                                         (237, 256), (279, 256), (1027, 128)])
def test_sparse_dense_and_fill(p, limit):
    rng = np.random.default_rng(909)
    X = rng.normal(size=(169, p)) * (rng.random((169, p)) < .12)
    y = np.digitize(X[:, :20].sum(axis=1) + rng.normal(size=len(X)), [-1, 1])
    kw = dict(task="classification", seed=1337, audit=True, candidate_limit=limit)
    dense, d = wide_support(X, y, **kw)
    csr, s = wide_support(sparse.csr_matrix(X), y, **kw)
    np.testing.assert_array_equal(dense, csr)
    np.testing.assert_array_equal(d["candidate_columns"], s["candidate_columns"])
    assert d["screened"] == (p > limit)
    assert d["candidate_count"] == min(limit, p)
    assert set(dense) <= set(d["candidate_columns"])
    filled, f = wide_support(X, y, fill_to_cap=True, **kw)
    assert f["width_cap"] == d["width_cap"] == len(filled)
    np.testing.assert_array_equal(f["selection_order"][:len(dense)], d["selection_order"])


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("limit", [128, 256])
def test_no_signal_full_outside_pool(sparse_input, limit):
    X = np.ones((17, 283))
    y = np.arange(len(X))
    if sparse_input:
        X = sparse.csr_matrix(X)
    for fill in [True, False]:
        result, meta = wide_support(X, y, task="regression", audit=True,
                                    candidate_limit=limit, fill_to_cap=fill)
        assert meta["no_signal"] and meta["width_cap"] == 283
        assert meta["candidate_count"] == limit
        assert not len(meta["selection_order"])
        np.testing.assert_array_equal(result, np.arange(283))


def test_screen_retains_complementary_zero_marginal():
    rng = np.random.default_rng(317)
    q, _ = np.linalg.qr(rng.normal(size=(275, 215)))
    # Center all column vectors, then obtain a zero-mean orthonormal basis.
    q, _ = np.linalg.qr(q - q.mean(axis=0))
    y, suppressor = q[:, 0], q[:, 1]
    X = q[:, 10:210].copy()
    X += .04 * y[:, None]
    X[:, 0] = y + .8 * suppressor
    X[:, 199] = suppressor
    chosen, info = wide_support(X, y, task="regression", audit=True)
    assert 199 in info["candidate_columns"] and 199 in chosen
    assert np.abs(y @ X[:, 199]) < 1e-13


def test_exact_full_filled_when_signal_is_tied():
    x = np.arange(40, dtype=float)
    X = np.tile(x[:, None], (1, 37))
    chosen, info = wide_support(X, x, task="regression", audit=True, fill_to_cap=True)
    assert info["width_cap"] == len(chosen) == 37 and not info["no_signal"]
    assert info["selection_order"].tolist()[:2] == [0, 1]
    np.testing.assert_array_equal(chosen, np.arange(37))
