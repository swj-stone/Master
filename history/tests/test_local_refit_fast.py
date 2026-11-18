import numpy as np
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from minppi_mofs.local_refit_fast import _FoldState, local_refit_fast_top_from_training
from minppi_mofs.local_refit_minpi import local_refit_top_from_training


@pytest.mark.parametrize("n,p,max_size,outputs", [(40, 15, 11, 3), (8, 37, 29, 1)])
def test_incremental_scores_equal_direct_refits_at_each_prefix(n, p, max_size, outputs):
    rng = np.random.default_rng(106)
    fit, check = rng.normal(size=(n, p)), rng.normal(size=(17, p))
    fit[:, 5] = fit[:, 3]
    check[:, 5] = check[:, 3]
    target, held = rng.normal(size=(n, outputs)), rng.normal(size=(len(check), outputs))
    state = _FoldState(fit, check, target, held, max_size)
    order = rng.permutation(p)[:max_size]
    sizes = [0, 1, max_size - 1, max_size]
    previous = 0

    def error(columns):
        if len(columns):
            model = Ridge(alpha=1, solver="svd", fit_intercept=False).fit(fit[:, columns], target)
            pred = model.predict(check[:, columns]).reshape(held.shape)
        else:
            pred = 0.
        return np.square(held - pred).sum()

    for size in sizes:
        state.append(order[previous:size])
        selected = order[:size]
        score = state.scores()
        base = error(selected)
        for j in np.setdiff1d(np.arange(p), selected):
            direct = base - error(np.append(selected, j))
            np.testing.assert_allclose(score[j], direct, atol=1e-10, rtol=5e-12)
        previous = size


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("n,p,h", [(90, 19, 11), (34, 65, 42)])
def test_fast_support_across_seeds_schedules_and_rank_deficiency(task, n, p, h):
    with threadpool_limits(limits=1):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(n, p))
            X[:, 2] = X[:, 0]
            X[::4, 4] = np.nan
            X[:, -3] = 0.1
            X[:, -1] = np.nan
            y = np.arange(n) % 3 if task == "classification" else rng.normal(size=n)
            rng.shuffle(y)
            for tiers in (None, (0, 3, h - 2), (0,)):
                reference = local_refit_top_from_training(X, y, h, task=task, random_state=seed, tiers=tiers)
                fast = local_refit_fast_top_from_training(X, y, h, task=task, random_state=seed, tiers=tiers)
                np.testing.assert_array_equal(fast[0], reference[0])
                assert fast[1] == reference[1]


def test_fast_all_ties_and_all_features():
    X, y = np.ones((20, 8)), np.arange(20.)
    assert local_refit_fast_top_from_training(X, y, 5, task="regression")[0].tolist() == list(range(5))
    support, info = local_refit_fast_top_from_training(X, y, 8, task="regression")
    assert support.tolist() == list(range(8)) and info["ballot_rows"] == 0
    with pytest.raises(ValueError):
        local_refit_fast_top_from_training(X, y, 5, task="regression", tiers=(1, 2))
