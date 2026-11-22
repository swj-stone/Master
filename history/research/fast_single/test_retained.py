import numpy as np
import pytest
from scipy import sparse

from research.fast_single import retained
from research.fast_single.selector import prepare_sampler, select_fast


@pytest.mark.parametrize("p,m", [(29,768), (72,384)])
def test_repair_uses_training_labels_at_fixed_row_budget(p,m):
    rng = np.random.default_rng(9191)
    n, seed = 1024, 1129
    X = rng.normal(size=(n,p))
    rows = np.arange(m, dtype=np.int64)
    prepare_sampler().sample_rows(n,m,seed,rows.ctypes.data)
    positives = np.setdiff1d(np.arange(n), rows)[:7]
    y = np.zeros(n, dtype=int)
    y[positives] = 1
    cols, trace = retained.select_retained(X, y, task="classification", seed=seed)
    fresh = trace["repaired_rows"]
    assert len(np.unique(fresh)) == m
    np.testing.assert_array_equal(np.intersect1d(fresh, positives), positives)
    check, info = select_fast(X[fresh], y[fresh], task="classification", seed=seed)
    np.testing.assert_array_equal(cols, check)
    np.testing.assert_array_equal(info["rank"], trace["rank"])
    again, repeated = retained.select_retained(X, y, task="classification", seed=seed)
    np.testing.assert_array_equal(again, cols)
    np.testing.assert_array_equal(repeated["repaired_rows"], fresh)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_sparse_votes_equal_paper(task):
    rng = np.random.default_rng(3232)
    X = sparse.random(641, 2000, density=.001, format="csr", random_state=rng)
    y = rng.normal(size=X.shape[0])
    if task == "classification":
        y = y > 0
    old, a = select_fast(X, y, task=task, seed=704)
    new, b = retained.select_retained(X, y, task=task, seed=704)
    np.testing.assert_array_equal(a["tier_scores"], b["tier_scores"])
    np.testing.assert_array_equal(old, new)
