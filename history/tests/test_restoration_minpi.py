from itertools import combinations, permutations

import numpy as np
import pytest
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.restoration_minpi import _layer_scores, _screen_rows, restoration_top_from_training


@pytest.mark.parametrize("task,classes", [("regression", 0), ("classification", 2), ("classification", 4)])
def test_zero_tier_equals_direct_model_permutation(task, classes):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(122, 9))
    X[::3, 0] = np.nan
    X[:, -1] = 3.
    y = rng.normal(size=len(X)) if not classes else (np.arange(len(X)) % classes).astype(str)
    cols, info = restoration_top_from_training(X, y, 4, task=task, random_state=41, tiers=(0,))
    fit, held = info["fit_rows"], info["assessment_rows"]
    assert not np.intersect1d(fit, held).size
    train, check = _standardize(X[fit], X[held])
    cls = RidgeClassifier if classes else Ridge
    model = cls(alpha=1., solver="lsqr", tol=.001, max_iter=250).fit(train, y[fit])
    score = (lambda pred: 1 - balanced_accuracy_score(y[held], pred)) if classes else (
        lambda pred: mean_squared_error(y[held], pred))
    clean = score(model.predict(check))
    oracle = np.zeros(check.shape[1])
    donors = np.random.default_rng(41)
    for _ in range(3):
        order = donors.permutation(len(held))
        for j in range(check.shape[1]):
            perturbed = check.copy()
            perturbed[:, j] = check[order, j]
            oracle[j] += score(model.predict(perturbed)) - clean
    np.testing.assert_allclose(info["scores"], oracle / 3, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(cols, np.sort(np.argsort(-info["scores"], kind="stable")[:4]))


@pytest.mark.parametrize("outputs", [1, 3])
def test_all_feature_permutations_equal_exact_uniform_k_peer_oracle(outputs):
    rng = np.random.default_rng(55)
    n, p = 6, 4
    check = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, outputs))
    intercept = rng.normal(size=outputs)
    prediction = check @ coef + intercept
    delta = check[rng.permutation(n)] - check
    target = rng.normal(size=n) if outputs == 1 else np.arange(n) % outputs
    weights = None if outputs == 1 else np.full(n, 1 / n)

    def loss(matrix):
        pred = matrix @ coef + intercept
        if outputs == 1:
            return np.mean((target - pred[:, 0]) ** 2)
        return np.mean(pred.argmax(axis=1) != target)

    for k in range(p):
        score = np.mean([_layer_scores(delta, prediction, coef, np.tile(order, (n, 1)),
                                       k, target, weights) for order in permutations(range(p))], axis=0)
        for feature in range(p):
            gains = []
            for peers in combinations([j for j in range(p) if j != feature], k):
                masked = check.copy()
                masked[:, peers] += delta[:, peers]
                restored_loss = loss(masked)
                masked[:, feature] += delta[:, feature]
                gains.append(loss(masked) - restored_loss)
            assert score[feature] == pytest.approx(np.mean(gains), abs=1e-12)


def test_groups_caps_minimum_and_determinism():
    y = np.arange(3000) % 3
    groups = np.arange(len(y)) // 500
    for ordered in (False, True):
        fit, assess = _screen_rows(y, "classification", np.random.default_rng(99), groups, ordered)
        assert len(fit) == 1024 and len(assess) == 256
        assert not np.intersect1d(groups[fit], groups[assess]).size
        if ordered:
            assert np.all(groups[assess] == 5)
    rng = np.random.default_rng(55)
    X = rng.normal(size=(len(y), 7))
    cols, info = restoration_top_from_training(X, y, 3, groups=groups, ordered_groups=True)
    assert info["tiers"] == [0, 4]
    for i, k in enumerate(info["tiers"]):
        _, single = restoration_top_from_training(X, y, 3, groups=groups, ordered_groups=True, tiers=k)
        np.testing.assert_array_equal(info["layer_scores"][i], single["scores"])
    np.testing.assert_array_equal(info["scores"], info["layer_scores"].min(axis=0))
    assert len(cols) == 3
    with pytest.raises(ValueError, match="requires training groups"):
        restoration_top_from_training(X, y, 3, ordered_groups=True)
