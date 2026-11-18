import numpy as np
import pytest

from minppi_mofs.exact_minpi import _witness_margins_from_folds, witness_margins
from minppi_mofs.piv2 import _folds
from minppi_mofs.grouped_minpi import (
    _representatives, grouped_control_from_training, grouped_top_from_training,
)
from minppi_mofs.word_minpi import exact_top_from_votes


def test_duplicate_columns_cast_one_vote_and_keep_original_positions():
    rng = np.random.default_rng(133)
    y = np.tile([0, 1], 60)
    original = rng.normal(size=(len(y), 24))
    original[:, :3] += y[:, None]
    copied = np.column_stack((original, original[:, 1], -original[:, 1]))
    selected, old = grouped_top_from_training(original, y, 6)
    actual, info = grouped_top_from_training(copied, y, 6)
    np.testing.assert_array_equal(actual, selected)
    assert old["representatives"] == info["representatives"] == 17
    assert info["budget_fill"] == 0


def test_exact_membership_on_reconstructed_representative_game():
    rng = np.random.default_rng(59)
    X = rng.normal(size=(390, 31))
    X[::3, :3] = np.nan
    X[:, -1] = np.nan
    y = .8 * X[:, 5] - X[:, 12] + rng.normal(size=len(X))
    args = dict(task="regression", random_state=19, tiers=(0, 2, 9))
    reps, _, _, _ = _representatives(X, y, "regression", 19, 4, (0, 2, 9))
    expected_control = np.sort(reps[:4])
    reps.sort()
    votes = np.sign(witness_margins(X[:, reps], y, "regression", 19)).astype(np.int8)
    expected = reps[exact_top_from_votes(votes, 4, tiers=(0, 2, 9))[0]]
    np.testing.assert_array_equal(grouped_top_from_training(X, y, 4, **args)[0], expected)
    np.testing.assert_array_equal(grouped_control_from_training(X, y, 4, **args)[0], expected_control)


def test_constant_columns_and_budget_fill():
    X = np.zeros((30, 9))
    y = np.repeat(["yes", "no"], 15)
    for method in [grouped_top_from_training, grouped_control_from_training]:
        selected, info = method(X, y, 7)
        np.testing.assert_array_equal(selected, np.arange(7))
        assert info["budget_fill"] == 7
        full, _ = method(X, y, 9)
        np.testing.assert_array_equal(full, np.arange(9))


def test_invalid_budget_and_target():
    X = np.ones((20, 5))
    y = np.arange(20, dtype=float)
    for value in (0, 6, True, 2.3):
        with pytest.raises(ValueError, match="count"):
            grouped_top_from_training(X, y, value)
    with pytest.raises(ValueError, match="finite"):
        grouped_top_from_training(X, np.where(y > 10, np.inf, y), 2, task="regression")


@pytest.mark.parametrize("task, classes", [("classification", 2), ("classification", 5), ("regression", 0)])
def test_bounded_projection_preserves_full_projection_margins(task, classes):
    rng = np.random.default_rng(73)
    X = rng.normal(size=(520, 78))
    X[::4, 2::5] = np.nan
    y = np.arange(len(X)) % classes if classes else X[:, 0] + rng.normal(size=len(X))
    reps = np.arange(2, X.shape[1], 3)
    folds = _folds(y, task, np.random.default_rng(1129), row_limit=256)
    expected = witness_margins(X[:, reps], y, task, 1129)
    actual = _witness_margins_from_folds(X, y, task, *folds, columns=reps)
    assert expected.tobytes() == actual.tobytes()
