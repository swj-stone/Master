"""Independent checks for the paper comparator (run via history/thesis-python.ps1)."""
import numpy as np
import pytest
from pathlib import Path
from sklearn.metrics import mutual_info_score

from experiments.test.piv2_conventional_filters import _association, _mrmr
from research.comparators.mrmr_mid import _information, mrmr_mid_order


def check_oracle(X, y, count):
    order, scores = mrmr_mid_order(X, y, count)
    # Independent reference: sklearn cell-based MI and the explicit Eq. (7),
    # with each candidate rescored from scratch for the previously chosen set.
    pairwise = np.array([[mutual_info_score(a, b) for b in X.T] for a in X.T])
    relevance = np.array([mutual_info_score(a, y) for a in X.T])
    for step, chosen in enumerate(order):
        candidates = relevance.copy()
        if step:
            candidates -= pairwise[:, order[:step]].mean(axis=1)
            candidates[order[:step]] = -np.inf
        np.testing.assert_allclose(scores[step], candidates[chosen], rtol=0, atol=2e-12)
        assert abs(candidates.max() - candidates[chosen]) <= 2e-12
    return order


def test_hand_computed_multiclass_duplicate_and_constant():
    a = np.repeat([0, 1], 4)
    b = np.tile([0, 0, 1, 1], 2)
    y = 2*a + b
    X = np.column_stack((a, a, b, a*0))
    order, quality = mrmr_mid_order(X, y, 4)
    np.testing.assert_array_equal(order, [0, 2, 1, 3])
    np.testing.assert_allclose(quality, [np.log(2), np.log(2), np.log(2)/2, 0], atol=1e-14)
    np.testing.assert_array_equal(check_oracle(X, y, 4), order)


def test_unequal_priors_and_legacy_mid_without_quantiles_or_row_sampling():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 7, size=(203, 11))
    y = rng.choice(3, size=len(X), p=[0.65, 0.3, 0.05])
    order = check_oracle(X, y, 8)
    relevance = np.array([_association(x, y, 'mi') for x in X.T])
    legacy, _ = _mrmr(X.T, relevance, 8)
    np.testing.assert_array_equal(order, legacy)
    np.testing.assert_array_equal(order, mrmr_mid_order(8-7*X, 19*y-8, 8)[0])


def test_constant_ties_and_category_ids_with_gaps():
    y = np.tile([5, 900], 5)
    X = np.column_stack((y, -3*y, y*0))
    np.testing.assert_array_equal(mrmr_mid_order(X, y, 3)[0], [0, 1, 2])


def test_large_alphabet_fallback():
    n = 1050
    X = np.stack((np.arange(n), np.arange(n) % 3))
    y = np.arange(n)[::-1]
    levels = X.max(axis=1) + 1
    values = np.arange(n+1) * np.log(np.maximum(np.arange(n+1), 1))
    marginal = np.array([values[np.bincount(x)].sum() for x in X])
    # An injective response discloses the entire feature, so MI equals entropy.
    observed = _information(X, levels, marginal, y, values)
    np.testing.assert_allclose(observed, [np.log(n), np.log(3)], atol=2e-12)


@pytest.mark.parametrize('x,y,k', [
    ([[0.1], [2]], [0, 1], 1), ([[1], [2]], [0, 0], 1),
    ([[1], [2]], [0, 1], 2), ([[1], [2]], [0, 1], True),
    ([[1], [2]], [0, 1], 0), ([[1], [2]], [0], 1),
    ([[np.nan], [2]], [0, 1], 1),
])
def test_invalid_input(x, y, k):
    with pytest.raises(ValueError):
        mrmr_mid_order(x, y, k)


def test_misspelled_comparison_method_fails_before_model_fit():
    from research.comparators.digits_comparison import DigitsFold, pipeline
    X = np.array([[0, 1], [1, 0]])
    panel = DigitsFold(X, np.array([0, 1]), X)
    with pytest.raises(ValueError, match="Unknown Digits comparison method"):
        pipeline(panel, "radnom50")


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_three_real_rankings_against_cell_mi_and_legacy_mid_core(fold):
    from experiments.stable_minpi_stress import inputs
    X, y, *_ = inputs("digits")
    path = Path(__file__).parent / "evidence" / "digits-20261118b" / "pass-1" / f"fold-{fold}.npz"
    with np.load(path) as stored:
        rows = stored["train"]
        order = check_oracle(X[rows], y[rows], 32)
        np.testing.assert_array_equal(order, stored["mrmr_mid_order"])
    columns = np.array([np.unique(col, return_inverse=True)[1] for col in X[rows].T])
    relevance = np.array([_association(col, y[rows], "mi") for col in columns])
    legacy, _ = _mrmr(columns, relevance, 32)
    np.testing.assert_array_equal(legacy, order)
