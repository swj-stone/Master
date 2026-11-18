import numpy as np
import pytest
from scipy.sparse import csr_matrix

from minppi_mofs.exchange_vote_minpi import exchange_vote_top_from_training
from minppi_mofs.sparse_exchange_minpi import sparse_exchange_top_from_training


@pytest.mark.parametrize("situation", ["empty", "many_zero", "few_zero", "noncanonical"])
@pytest.mark.parametrize("count", [None, 12])
def test_sparse_fast_path_keeps_reference_support(situation, count):
    rng = np.random.default_rng(9652)
    rows, p = 150, 40
    matrix = rng.normal(size=(rows,p))
    matrix[:, :{"empty":40, "many_zero":23, "few_zero":3, "noncanonical":23}[situation]] = 0.
    X = csr_matrix(matrix)
    if situation == "noncanonical":
        # An unsorted row includes explicit zero, NaN and duplicate entries that cancel.
        X = csr_matrix((np.r_[7., -7., 0., np.nan, X.data], np.r_[8, 8, 3, 7, X.indices],
                        np.r_[0, X.indptr[1:]+4]), shape=X.shape)
    y = rng.normal(size=rows)
    opts = dict(task="regression", random_state=599)
    baseline, old = exchange_vote_top_from_training(X, y, count, **opts)
    new, info = sparse_exchange_top_from_training(X, y, count, **opts)
    np.testing.assert_array_equal(new, baseline)
    assert old["branch"] == info["branch"]
    if "constant_columns" in old:
        np.testing.assert_array_equal(old["constant_columns"], info["constant_columns"])
