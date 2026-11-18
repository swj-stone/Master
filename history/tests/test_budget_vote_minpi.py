import numpy as np
import pytest

from minppi_mofs.budget_vote_minpi import _choose, budget_vote_top_from_training
from minppi_mofs.native_pivotal_minpi import library_path


def test_gain_floor_responds_to_disagreement_and_tiny_training_groups():
    reliable = np.tile([2., 1., 0., -1., -2., -3., -4., -5.], (2, 2, 1))
    cols, info = _choose(reliable, None, 96)
    np.testing.assert_array_equal(cols, [0, 1])
    assert info["agreement"] == 1 and info["effective_dimension"] == pytest.approx(9 / 5)
    uncertain = reliable.copy()
    uncertain[0] = -1.
    assert len(_choose(uncertain, None, 96)[0]) == 8
    assert len(_choose(reliable, None, 2)[0]) > len(cols)
    assert len(_choose(np.zeros_like(reliable), None, 96)[0]) == 8
    np.testing.assert_array_equal(_choose(np.zeros_like(reliable), 3, 96)[0], np.arange(3))


def test_uniform_positive_mass_keeps_every_column():
    _, info = _choose(np.ones((2, 2, 100)), None, 96)
    assert info["budget"] == 100


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_backend_agreement_and_group_isolation(ordered, task):
    rng = np.random.default_rng(372)
    X = rng.normal(size=(380, 47)).astype(np.float32)
    y = X[:, 2] - X[:, 4] + rng.normal(size=len(X))
    if task == "classification":
        y = np.digitize(y, [-.4, .4])
    X[::11, ::4] = np.nan
    X[:, -1] = .1
    groups = np.tile(np.arange(10), 38)
    options = dict(task=task, groups=groups, ordered_groups=ordered, random_state=132)
    cols, info = budget_vote_top_from_training(X, y, backend="numpy", **options)
    assert 1 <= len(cols) <= X.shape[1]
    assert len(info["folds"]) == (1 if ordered else 2)
    for fold in info["folds"]:
        fit, held = groups[fold["fit_rows"]], groups[fold["held_rows"]]
        assert not np.intersect1d(fit, held).size
        if ordered:
            assert fit.max() < held.min()
    fixed, _ = budget_vote_top_from_training(X, y, len(cols), backend="numpy", **options)
    np.testing.assert_array_equal(cols, fixed)
    if library_path().is_file():
        actual, native = budget_vote_top_from_training(X, y, backend="native", **options)
        np.testing.assert_array_equal(cols, actual)
        for name in ("scores", "zero_peer_scores", "one_peer_scores"):
            np.testing.assert_allclose(native[name], info[name], rtol=1e-10, atol=1e-12)
