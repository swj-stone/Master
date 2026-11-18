import numpy as np
import pytest
from scipy import sparse

from minppi_mofs.document_vote_minpi import document_vote_top_from_training
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training, library_path


@pytest.mark.skipif(not library_path().exists(), reason="Optional binary CSR backend not built")
@pytest.mark.parametrize("ordered,labels_type", [(True, "numeric"), (False, "string")])
def test_native_ties_missing_class_per_fold_and_noncanonical_counts(ordered, labels_type):
    rng = np.random.default_rng(410)
    dense = rng.binomial(1, .09, (804, 301))
    dense[:, 1] = dense[:, 0]
    dense[:, 100] = 0
    dense = dense[:, ::-1] * rng.integers(1, 5, size=dense.shape)
    groups = np.repeat(np.arange(6), 134)
    y = np.where(groups == 5, 1, (dense[:, 43] > 0).astype(int))
    if labels_type == "string":
        y = np.where(y, "yes", "no")
    csr = sparse.csr_matrix(dense)
    # Duplicate each column index, including stored zeros, then reverse the
    # entries within each row. Input canonicalization must not mutate X.
    data = np.c_[csr.data, np.zeros(csr.nnz, dtype=csr.dtype)].ravel()
    indices = np.repeat(csr.indices, 2)
    X = sparse.csr_matrix((data, indices, csr.indptr * 2), shape=csr.shape)
    first, last = X.indptr[5:7]
    X.indices[first:last] = X.indices[first:last][::-1]
    X.data[first:last] = X.data[first:last][::-1]
    kwargs = dict(groups=groups, ordered_groups=ordered, random_state=61059)
    for count in (None, 273, 1):
        native, info = fast_document_vote_top_from_training(X, y, count, backend="native", **kwargs)
        expected, ref = document_vote_top_from_training(X, y, count, **kwargs)
        np.testing.assert_array_equal(native, expected)
        assert info["backend"] == "native"
        for field in ("zero_peer_scores", "one_peer_scores", "scores"):
            np.testing.assert_allclose(info[field], ref[field], atol=1e-15, rtol=1e-14)
        for a, b in zip(info["folds"], ref["folds"]):
            assert a["peers"] == b["peers"]
            np.testing.assert_array_equal(a["fit_rows"], b["fit_rows"])
            np.testing.assert_array_equal(a["held_rows"], b["held_rows"])
    assert X.nnz == 2 * csr.nnz


def test_multiclass_reference_fallback():
    X = sparse.csr_array(np.eye(12))
    y = np.arange(12) % 3
    cols, info = fast_document_vote_top_from_training(X, y)
    original, _ = document_vote_top_from_training(X, y)
    np.testing.assert_array_equal(cols, original)
    assert info["backend"] == "numpy"
    with pytest.raises(ValueError, match="binary classification"):
        fast_document_vote_top_from_training(X, y, backend="native")


@pytest.mark.parametrize("damage", ["negative_col", "past_col", "negative_offset", "backward", "past_end"])
def test_invalid_csr_buffers_are_rejected_before_any_native_read(damage):
    X = sparse.csr_matrix(np.eye(5))
    assert X.has_canonical_format
    if damage == "negative_col":
        X.indices[0] = -1
    elif damage == "past_col":
        X.indices[0] = X.shape[1]
    elif damage == "negative_offset":
        X.indptr[2] = -1
    elif damage == "backward":
        X.indptr[3] = 1
    else:
        X.indptr[-1] = len(X.indices) + 1
    with pytest.raises(ValueError, match="sparse|Sparse"):
        fast_document_vote_top_from_training(X, np.arange(5) % 2, backend="native")
