import numpy as np
import pytest

from minppi_mofs.native_pivotal_minpi import load_native, native_top_from_training
from minppi_mofs.stable_native_minpi import _cutoff_reason, stable_native_top_from_training


def test_rank_and_budget_guards_cover_distinct_sources_of_instability():
    score = np.array([0., 1., 2., 3., 4., 5.])
    assert _cutoff_reason(np.arange(2, 6), score) is None
    tied = score.copy()
    tied[2] = np.nextafter(1., 2.)
    assert _cutoff_reason(np.arange(2, 6), tied) == "score_boundary"
    # Dropping two features consumes exactly the allowed two units.
    cost = np.array([1., 1., 10., 20., 30., 38.])
    assert _cutoff_reason(np.arange(2, 6), score, cost) == "budget_boundary"
    # The cap prevents removing more than three features, even with budget left.
    assert _cutoff_reason(np.arange(3, 6), score, np.array([0., 0., 0., 1., 1., 1.])) is None


@pytest.mark.parametrize("count", [None, 6])
def test_exact_tied_columns_use_reference_and_match_its_indices(count):
    try:
        load_native()
    except OSError:
        pytest.skip("Native library has not been built")
    X = np.zeros((210, 8))
    y = np.arange(len(X)) % 2
    cols, info = stable_native_top_from_training(X, y, count)
    reference, ref = native_top_from_training(X, y, count, backend="numpy")
    np.testing.assert_array_equal(cols, reference)
    np.testing.assert_array_equal(info["scores"], ref["scores"])
    assert info["selection_backend"] == "numpy" and info["guard_reason"] == "score_boundary"


def test_uncompiled_fallback_remains_available():
    X = np.arange(300, dtype=np.int32).reshape(50, 6)
    y = np.arange(50) % 3
    cols, info = stable_native_top_from_training(X, y, 3)
    np.testing.assert_array_equal(cols, native_top_from_training(X, y, 3, backend="numpy")[0])
    assert info["selection_backend"] == "numpy" and info["guard_reason"] == "none"
