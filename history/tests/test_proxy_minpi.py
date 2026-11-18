import numpy as np
import pytest

from minppi_mofs.pilot_minpi import pilot_top_from_training
from minppi_mofs.proxy_minpi import _target_direction, _unit_columns, proxy_top_from_training


def test_constant_decimal_column_with_missingness_has_no_unit_direction():
    rng = np.random.default_rng(8)
    X = np.full((96, 6), .1)
    X[0, 2] = np.nan
    X[5, 3] = np.inf
    X[:, 4] = np.nan
    X[:, 5] = rng.normal(size=len(X))
    z = _unit_columns(X)
    np.testing.assert_array_equal(z[:, :5], np.zeros_like(z[:, :5]))
    np.testing.assert_allclose(z[:, 5], (X[:, 5] - X[:, 5].mean()) /
                               np.linalg.norm(X[:, 5] - X[:, 5].mean()), atol=1e-15)
    np.testing.assert_array_equal(_target_direction(np.full(96, .1), None), np.zeros((96, 1)))


def test_missing_and_nonconstant_groups_survive_observed_value_anchoring():
    rng = np.random.default_rng(17)
    signal = rng.normal(size=240)
    X = np.column_stack((signal, -2 * signal, np.full(240, .1), np.full(240, -.1),
                         rng.normal(size=240), rng.normal(size=240)))
    X[::7, 4] = np.nan
    groups = np.repeat(np.arange(4), 60)
    selected, info = proxy_top_from_training(X, np.full(240, .1), task="regression",
                                              groups=groups, ordered_groups=True)
    np.testing.assert_array_equal(info["scores"], np.zeros(X.shape[1]))
    assert info["parent"][0] == info["parent"][1]
    np.testing.assert_array_equal(info["parent"][2:4], [2, 3])
    fit, held = (info["folds"][0][s] for s in ("fit_rows", "held_rows"))
    assert groups[fit].max() < groups[held].min() and len(selected) < X.shape[1]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_complete_nonconstant_inputs_preserve_frozen_pilot_output(task):
    rng = np.random.default_rng(811)
    X = rng.normal(size=(217, 33))
    X[:, 1] = X[:, 0]
    y = X[:, 0] + .4 * rng.normal(size=len(X))
    if task == "classification":
        y = np.digitize(y, [0., 1.])
    for count in (None, 25):
        original, old = pilot_top_from_training(X, y, count, task=task)
        revised, new = proxy_top_from_training(X, y, count, task=task)
        np.testing.assert_array_equal(revised, original)
        np.testing.assert_array_equal(new["parent"], old["parent"])
        np.testing.assert_array_equal(new["scores"], old["scores"])
