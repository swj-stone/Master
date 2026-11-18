import numpy as np
import pytest
from scipy import sparse

from minppi_mofs.repeated_document_minpi import repeated_document_top_from_training
from minppi_mofs.robust_document_minpi import robust_document_top_from_training, library_path
from minppi_mofs import fast_document_vote_minpi, robust_document_minpi


@pytest.mark.skipif(not library_path().exists(), reason="Optional repeated document kernel not built")
@pytest.mark.parametrize("grouped,ordered", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("repeats", [1, 3, 4])
def test_native_repeated_endpoints_and_selected_columns_match_reference(grouped, ordered, repeats):
    rng = np.random.default_rng(903)
    dense = rng.binomial(1, .09, size=(809, 127))
    dense[:, 1] = dense[:, 0]
    dense[:, 40] = 0
    groups = np.arange(len(dense)) // 134
    y = np.where(groups == groups.max(), 1, dense[:, 17])
    csr = sparse.csr_matrix(dense)
    values = np.c_[csr.data * 3, np.zeros(csr.nnz, dtype=csr.dtype)].ravel()
    indices = np.repeat(csr.indices, 2)
    X = sparse.csr_matrix((values, indices, csr.indptr * 2), shape=csr.shape)
    kwargs = dict(random_state=61059, groups=groups if grouped else None,
                  ordered_groups=ordered, repeats=repeats)
    for count in (None, 1, 113):
        support, info = robust_document_top_from_training(X, y, count, backend="native", **kwargs)
        expected, old = repeated_document_top_from_training(
            X, y, count, aggregation="median", backend="numpy", **kwargs)
        np.testing.assert_array_equal(support, expected)
        for field in ("scores", "zero_peer_scores", "one_peer_scores", "replicate_endpoints"):
            np.testing.assert_array_equal(info[field], old[field])
        assert len(info["folds"]) == len(old["folds"])
        for a, b in zip(info["folds"], old["folds"]):
            for field in ("fit_rows", "held_rows"):
                np.testing.assert_array_equal(a[field], b[field])
            assert a["peers"] == b["peers"] and a["repeat_seed"] == b["repeat_seed"]
    assert X.nnz == 2 * csr.nnz


@pytest.mark.parametrize("shape,classes", [((18, 9), 3), ((18, 1), 2)])
def test_reference_fallback_for_multiclass_and_one_column(shape, classes):
    X = sparse.csr_array(np.arange(np.prod(shape)).reshape(shape) % 3)
    y = np.arange(shape[0]) % classes
    support, info = robust_document_top_from_training(X, y, random_state=79)
    expected, old = repeated_document_top_from_training(X, y, random_state=79)
    np.testing.assert_array_equal(support, expected)
    np.testing.assert_array_equal(info["scores"], old["scores"])
    assert info["backend"] == "numpy"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_invalid_term_values_are_rejected_before_native_execution(bad):
    X = sparse.csr_matrix(np.eye(5))
    X.data[0] = bad
    with pytest.raises(ValueError, match="finite nonnegative"):
        robust_document_top_from_training(X, np.arange(5) % 2)


def test_invalid_sparse_indices_are_rejected_before_native_execution():
    X = sparse.csr_matrix(np.eye(5))
    assert X.has_canonical_format
    X.indices[0] = 5
    with pytest.raises(ValueError, match="outside matrix bounds"):
        robust_document_top_from_training(X, np.arange(5) % 2)


def test_explicit_numpy_backend_never_loads_either_native_library(monkeypatch):
    def forbidden():
        raise AssertionError("NumPy execution loaded a native library")

    monkeypatch.setattr(robust_document_minpi, "load_native", forbidden)
    monkeypatch.setattr(fast_document_vote_minpi, "load_native", forbidden)
    X = sparse.csr_matrix(np.eye(17))
    support, info = robust_document_top_from_training(X, np.arange(17) % 2, backend="numpy")
    assert info["backend"] == "numpy" and len(support) == info["budget"]
