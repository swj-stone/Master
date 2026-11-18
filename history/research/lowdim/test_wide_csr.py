import numpy as np
import pytest
from scipy import sparse
from threadpoolctl import threadpool_limits

from research.conditional_dense.vote_width.method import vote_support
from research.lowdim.wide_csr import csr_vote_support


@pytest.mark.parametrize("n,p,classes,dtype", [(132, 74, 2, np.float32), (903, 91, 4, np.float64)])
def test_frozen_dense_equivalence(n, p, classes, dtype):
    rng = np.random.default_rng(41677)
    X = rng.choice([-2, 0, 0, 0, 0, 3], size=(n, p)).astype(dtype)
    X[:, 2:7] = 0
    X[:, 7] = X[:, 8]
    y = rng.integers(classes, size=n)
    with threadpool_limits(1):
        cols, dense = vote_support(X, y, seed=423)
        chosen, csr = csr_vote_support(sparse.csr_matrix(X), y, seed=423)
    np.testing.assert_array_equal(cols, chosen)
    for key in dense:
        np.testing.assert_array_equal(csr[key], dense[key])
