"""Checks for the exact wide extension used in the two-rule crossover."""
import numpy as np
import pytest
from scipy import sparse

from research.lowdim.method import fast_support
from research.peer_crossover.single import _target, _votes, prepare, select_single


def refit_layers(x, y):
    z = x - x.mean(axis=0)
    scales = z.std(axis=0)
    z /= np.where(scales > 0., scales, 1.)
    m, p = x.shape

    def loss(columns):
        if not columns:
            return (y*y).sum()
        a = z[:, columns]
        ridge = np.sqrt(m) * np.eye(len(columns))
        beta = np.linalg.solve(a.T @ a + ridge, a.T @ y)
        return ((y-a@beta)**2).sum() + np.sqrt(m)*(beta**2).sum()

    one = np.array([loss([j]) for j in range(p)])
    single = np.zeros(p)
    for j in range(p):
        for k in range(j):
            pair = loss([j, k])
            single[j] += one[k]-pair
            single[k] += one[j]-pair
    return np.array([loss([])-one, single/(p-1)])


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_wide_against_independent_ridge_refits(task):
    rng = np.random.default_rng(1839)
    x = rng.normal(size=(73, 36))
    x[:, :7] = 0
    x[2, :3] = [1, -2, 3]
    x[:, 5] = 4
    x[:, 6] = x[:, 7]
    y = rng.normal(size=len(x))
    if task == "classification":
        y = np.digitize(y, [-.6, .5])
        target = np.eye(3)[y]
    else:
        target = y[:, None] / np.std(y)
    reference = refit_layers(x, target-target.mean(axis=0))
    for matrix in (x, sparse.csr_matrix(x)):
        cols, info = select_single(matrix, y, task=task, seed=7)
        np.testing.assert_allclose(info["tier_scores"], reference, rtol=3e-12, atol=1e-12)
        score = np.minimum(*reference)
        assert len(cols) == np.ceil(score.sum()**2 / (score @ score))
        assert score[cols].min() >= np.sort(score)[-len(cols)]-1e-12


def test_same_stratified_sample_as_native():
    rng = np.random.default_rng(1818)
    x = rng.normal(size=(2375, 17))
    y = rng.normal(size=len(x))
    rows = np.empty(768, dtype=np.int64)
    prepare().sample_rows(len(x), len(rows), 1129, rows.ctypes.data)
    for task, response in (("regression", y), ("classification", np.digitize(y, [-1., .6]))):
        _, original = fast_support(x, response, task=task, seed=1129)
        target = _target(response, rows, task)
        wide_layers = _votes(x[rows], target, x.shape[1])
        np.testing.assert_allclose(wide_layers, original["tier_scores"], rtol=3e-12, atol=1e-12)


def test_multitile_restores_constant_peers_and_copies():
    rng = np.random.default_rng(793)
    x = rng.integers(0, 3, size=(21, 564)).astype(float)
    x[:, 1] = x[:, 0]
    x[:, 2] = 0.
    x[:, 3] = 5.
    y = rng.normal(size=len(x))
    cols, info = select_single(x, y, task="regression")
    sparse_cols, sparse_info = select_single(sparse.csr_matrix(x), y, task="regression")
    np.testing.assert_array_equal(cols, sparse_cols)
    np.testing.assert_allclose(info["tier_scores"], sparse_info["tier_scores"], atol=1e-12)
    for matrix in [x, sparse.csr_matrix(x), sparse.csr_matrix(x.shape)]:
        all_cols, zero = select_single(matrix, y*0, task="regression")
        np.testing.assert_array_equal(all_cols, np.arange(x.shape[1]))
        np.testing.assert_array_equal(zero["rank"], 0.)
