import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from minppi_mofs.exchange_vote_minpi import exchange_vote_top_from_training


def _predict_column(X, y, fit, held, j):
    x, q = X[fit, j], X[held, j]
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros((len(held), y.shape[1]))
    avg = x[valid].mean()
    x = np.where(valid, x, avg)
    q = np.where(np.isfinite(q), q, avg)
    std = np.sqrt(np.mean((x - avg)**2))
    if not std:
        return np.zeros((len(held), y.shape[1]))
    x, q = (x - avg) / std, (q - avg) / std
    model = Ridge(alpha=1., fit_intercept=False, solver="svd").fit(
        x[:, None], y[fit] - y[fit].mean(axis=0))
    return model.predict(q[:, None]).reshape(len(held), -1)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("h,ordered", [(1, False), (12, True), (None, False)])
def test_votes_agree_with_direct_refits_and_support_exchanges(task, h, ordered):
    rng = np.random.default_rng(78)
    X = rng.normal(size=(160, 18))
    y = X[:, 3] - .3 * X[:, 2] + .5 * rng.normal(size=160)
    if task == "classification":
        y = np.digitize(y, [-.5, .3, 2.])
        target = (y[:, None] == np.unique(y)).astype(float)
    else:
        target = y[:, None]
    X[::11, 2:9] = np.nan
    X[::7, :2] = np.inf
    X[:, 10] = 1
    groups = np.repeat(np.arange(8), 20)
    cols, info = exchange_vote_top_from_training(X, y, h, task=task, random_state=139,
                                                groups=groups, ordered_groups=ordered)
    active, anchors = info["active_columns"], info["anchors"]
    endpoints = []
    direct = []
    for part in info["folds"]:
        fit, held = part["fit_rows"], part["held_rows"]
        assert not np.intersect1d(groups[fit], groups[held]).size
        if ordered:
            assert np.max(groups[fit]) < np.min(groups[held])
        t = target[held] - target[fit].mean(axis=0)
        low, high = [], []
        for j, anchor in zip(active, anchors, strict=True):
            obs, ctx = [_predict_column(X, target, fit, held, v) for v in (j, anchor)]
            low.append(np.sum(t**2 - (t - obs)**2, axis=1).mean())
            high.append(np.sum((t - ctx)**2 - (t - ctx - obs)**2, axis=1).mean())
        low, high = np.array(low), np.array(high)
        endpoints.append(np.maximum(low, high))
        k = min(X.shape[1] - 1, max(1, len(cols) - 1))
        direct.append(np.minimum(low, low + k / (X.shape[1] - 1) * (high - low)))
    np.testing.assert_allclose(info["scores"], direct, atol=1e-12, rtol=1e-12)
    if h is None:
        reproducibly_weak = np.all(np.asarray(endpoints) <= 1.e-13, axis=0).sum()
        drop = min(.5, (1 + reproducibly_weak) / (2 + len(active)))
        assert len(cols) == X.shape[1] - max(1, int(drop * X.shape[1]))
    m = len(active) // 2
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[np.random.default_rng(139).permutation(X.shape[1])[:len(cols)]] = True
    for i, j, yes in zip(active[:m], active[m:],
                         np.all(info["scores"][:, m:] > info["scores"][:, :m], axis=0), strict=True):
        if yes and not mask[j]:
            mask[i], mask[j] = False, True
    np.testing.assert_array_equal(cols, np.flatnonzero(mask))


def test_dense_verified_constants_preserve_full_standardized_ridge_on_unseen_rows():
    rng = np.random.default_rng(100)
    X = rng.normal(size=(400, 24))
    y = X[:, :3].sum(axis=1) + rng.normal(size=len(X))
    X[:, 10:] = np.arange(14)
    X[100, 10] = -1  # A value missed on the cheap eight-row probe.
    cols, info = exchange_vote_top_from_training(X, y, task="regression")
    assert info["branch"] == "constant" and cols.tolist() == list(range(11))
    Xt = rng.normal(size=(50, 24))
    pred = []
    for subset in (np.arange(24), cols):
        scale = StandardScaler().fit(X[:, subset])
        model = Ridge(solver="svd").fit(scale.transform(X[:, subset]), y)
        pred.append(model.predict(scale.transform(Xt[:, subset])))
    np.testing.assert_allclose(*pred, atol=1e-12, rtol=1e-12)
    fixed, _ = exchange_vote_top_from_training(X, y, count=13, task="regression")
    assert len(fixed) == 13 and set(cols) <= set(fixed)
    fewer, info = exchange_vote_top_from_training(X, y, count=6, task="regression")
    assert len(fewer) == 6 and set(fewer) <= set(cols) and info["branch"] == "constant_then_exchange"
    X[2, 12], X[:, 13] = np.inf, np.nan
    more, _ = exchange_vote_top_from_training(X, y, task="regression")
    assert 12 in more and 13 in more


def test_csr_duplicates_explicit_zero_and_missingness():
    # The first column cancels in every row, the second is explicitly zero,
    # the third has a NaN. A constant nonzero fourth column is preserved.
    indices = np.tile([0, 0, 1, 2, 3], 28)
    values = np.tile([2., -2., 0., 0., 1.], 28)
    values[3] = np.nan
    X = sparse.csr_matrix((values, indices, np.arange(0, 141, 5)), shape=(28, 32))
    assert not X.has_canonical_format
    cols, info = exchange_vote_top_from_training(X, np.arange(28), task="regression")
    assert cols.tolist() == [2, 3] and info["branch"] == "constant"
    assert not X.has_canonical_format and np.array_equal(X.indices, indices)


def test_csr_exchange_densifies_only_smaller_audit(monkeypatch):
    rng = np.random.default_rng(190)
    X = rng.normal(size=(200, 130))
    X[rng.random(X.shape) < .7] = 0.
    y = rng.choice([0, 1, 2], len(X))
    wanted, ref = exchange_vote_top_from_training(X, y)
    original = sparse.csr_matrix.toarray
    calls = []

    def check_size(self, *args, **kwargs):
        calls.append(self.shape)
        assert self.shape[0] <= 96 and self.shape[1] <= 36
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", check_size)
    cols, info = exchange_vote_top_from_training(sparse.csr_matrix(X), y)
    assert calls and info["branch"] == "exchange"
    np.testing.assert_array_equal(wanted, cols)
    np.testing.assert_array_equal(ref["scores"], info["scores"])


def test_degenerate_cases_are_deterministic_and_remain_compressed():
    X = np.zeros((10, 21))
    y = np.arange(len(X)) % 3
    for A in (X, sparse.csr_matrix(X)):
        c, info = exchange_vote_top_from_training(A, y)
        assert len(c) == 1 and info["branch"] == "constant"
        singleton, _ = exchange_vote_top_from_training(A[:, :1], y)
        assert singleton.tolist() == [0]
        full, _ = exchange_vote_top_from_training(A, y, count=21)
        np.testing.assert_array_equal(full, np.arange(21))
    with pytest.raises(ValueError):
        exchange_vote_top_from_training(sparse.csr_matrix(X), y, 22)
