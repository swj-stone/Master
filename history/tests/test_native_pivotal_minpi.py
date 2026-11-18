import numpy as np
import pytest

from minppi_mofs import native_pivotal_minpi as implementation
from minppi_mofs.moment_minpi import moment_top_from_training


@pytest.mark.skipif(not implementation.library_path().exists(), reason="Optional AOT library not built")
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["C", "F", "negative"])
def test_compiled_and_reference_match_moment_support_with_missing_values(task, precision, layout):
    rng = np.random.default_rng(1001)
    X = rng.normal(size=(360, 23)).astype(precision)
    X[::7, ::3] = np.nan
    X[:, 3] = .125
    X[2:15, 1] = np.inf
    y = rng.normal(size=len(X))
    if task == "classification":
        y = np.digitize(y, [-.45, .55, 1.2])
    g = np.repeat(np.arange(6), 60)
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "negative":
        X, y, g = X[::-1, ::-1], y[::-1], g[::-1]
    opts = dict(task=task, groups=g, ordered_groups=layout == "F")
    old, legacy = moment_top_from_training(X, y, 17, **opts)
    for count in (17, None):
        native, info = implementation.native_top_from_training(X, y, count, backend="native", **opts)
        reference, other = implementation.native_top_from_training(X, y, count, backend="numpy", **opts)
        assert info["backend"] == "native" and other["backend"] == "numpy"
        np.testing.assert_allclose(info["scores"], other["scores"], rtol=1e-11, atol=1e-14)
        np.testing.assert_array_equal(native, reference)
        if count is not None:
            np.testing.assert_array_equal(native, old)
            np.testing.assert_allclose(info["scores"], legacy["scores"], rtol=1e-11, atol=1e-14)
        for part in info["folds"]:
            fit, held = part["fit_rows"], part["held_rows"]
            assert not np.intersect1d(g[fit], g[held]).size
            if opts["ordered_groups"]:
                assert g[fit].max() < g[held].min()


def test_fallback_never_compiles_and_respects_score_mass(monkeypatch):
    def unavailable():
        raise OSError("library unavailable")

    monkeypatch.setattr(implementation, "load_native", unavailable)
    rng = np.random.default_rng(30)
    X = rng.normal(size=(300, 24))
    y = X[:, :6].sum(axis=1) + .4 * rng.normal(size=300)
    cols, info = implementation.native_top_from_training(X, y, task="regression")
    assert info["backend"] == "numpy" and 12 <= len(cols) <= 23
    ref, _ = implementation.native_top_from_training(X, y, task="regression", backend="numpy")
    np.testing.assert_array_equal(cols, ref)
    cost, scores = info["deletion_cost"], info["scores"]
    weakest = sorted(range(24), key=lambda j: (scores[j], -j))
    eligible = [k for k in range(1, 13) if sum(cost[weakest][:k]) <= .02 * sum(cost)]
    assert set(cols) == set(weakest[max(eligible, default=1):])
    with pytest.raises(OSError, match="unavailable"):
        implementation.native_top_from_training(X, y, task="regression", backend="native")


def test_single_column_or_unsupported_numeric_storage():
    X = np.ones((20, 5), dtype=np.int32)
    y = np.arange(20)
    cols, info = implementation.native_top_from_training(X, y)
    assert info["backend"] == "numpy" and cols.tolist() == [0, 1, 2]
    only, _ = implementation.native_top_from_training(X[:, :1], y)
    np.testing.assert_array_equal(only, [0])
    with pytest.raises(ValueError, match="float32 or float64"):
        implementation.native_top_from_training(X, y, backend="native")
