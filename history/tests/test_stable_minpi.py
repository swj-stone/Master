import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.stable_minpi import StableMinPISelector, _calibrate_supports, _partition
from minppi_mofs.environment_minpi import environment_path_from_training


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_held_comparison_matches_independent_ridge_and_returned_support(task):
    rng = np.random.default_rng(92)
    X = rng.normal(size=(680, 9))
    X[::9, 4] = np.nan
    z = X[:, 2] - X[:, 5] + rng.normal(scale=.25, size=len(X))
    y = np.digitize(z, [-.3, .8]).astype(str) if task == "classification" else z
    groups = np.repeat(np.arange(10), 68)
    discovery, val = _partition(y, groups, True, task, 11)
    assert np.array_equal(np.unique(groups[discovery]), np.arange(7))
    assert [np.unique(groups[part]).tolist() for part in val] == [[7], [8], [9]]
    supports = {1: np.array([0]), 2: np.array([2, 5]), 7: np.arange(7)}
    chosen, info = _calibrate_supports(X, y, discovery, val, supports, task, 11, .025, bounded_calibration=True)
    scores = {}
    n = np.asarray(info["calibration_fit_rows"])
    for width, cols in {**supports, 9: np.arange(9)}.items():
        scores[width] = []
        for held in info["calibration_validation_rows"]:
            fit, test = _standardize(X[n][:, cols], X[held][:, cols])
            model = RidgeClassifier(solver="svd") if task == "classification" else Ridge(solver="svd")
            pred = model.fit(fit, y[n]).predict(test)
            score = balanced_accuracy_score if task == "classification" else r2_score
            scores[width].append(score(y[held], pred))
    for row in info["validation_candidates"]:
        gaps = np.subtract(scores[row["width"]], scores[9])
        np.testing.assert_allclose(row["gaps"], gaps, atol=1e-11)
        assert row["eligible"] == np.all(gaps >= -.025)
    qualified = [w for w in supports if np.all(np.subtract(scores[w], scores[9]) >= -.025)]
    expected = supports[min(qualified)] if qualified else np.arange(9)
    np.testing.assert_array_equal(chosen, expected)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_default_calibration_uses_all_rows_and_deployment_solver(task):
    rng = np.random.default_rng(71)
    X = rng.normal(size=(1270, 31)).astype(np.float32)
    noise = rng.normal(scale=.1, size=len(X)).astype(np.float32)
    z = X[:, 5] + X[:, 7] - X[:, 1] + noise
    X[::13, 17] = np.nan
    X[:, 19] = np.nan
    y = np.where(z < -.7, "a", np.where(z < .8, "b", "c")) if task == "classification" else z
    discovery = np.arange(612)
    helds = [np.arange(612, 913), np.arange(913, len(X))]
    supports = {3: np.array([1, 5, 7]), 23: np.arange(23)}
    _, info = _calibrate_supports(X, y, discovery, helds, supports, task, 71, .02)
    assert info["calibration_mode"] == "deployment"
    assert info["calibration_fit_rows"] == discovery.tolist()
    assert info["calibration_validation_rows"] == [a.tolist() for a in helds]
    data = X[discovery]
    mask = np.isfinite(data)
    mean = (np.where(mask, data, 0).sum(axis=0, dtype=np.float64) / np.maximum(mask.sum(axis=0), 1)).astype(np.float32)
    clean = np.where(np.isfinite(X), X, mean)
    for width, columns in {**supports, 31: slice(None)}.items():
        estimator = RidgeClassifier if task == "classification" else Ridge
        score = balanced_accuracy_score if task == "classification" else r2_score
        model = make_pipeline(StandardScaler(), estimator(alpha=1., solver="lsqr", tol=.001, max_iter=250))
        model.fit(clean[discovery][:, columns], y[discovery])
        truth = [score(y[held], model.predict(clean[held][:, columns])) for held in helds]
        actual = info["full_scores"] if width == 31 else next(
            row["scores"] for row in info["validation_candidates"] if row["width"] == width)
        np.testing.assert_allclose(actual, truth, atol=2e-7, rtol=0)


def test_end_to_end_path_never_refits_on_calibration_rows():
    rng = np.random.default_rng(43)
    X = pd.DataFrame(rng.normal(size=(192, 8)), columns=list("abcdefgh"))
    y = (2 * X["c"] - X["e"] + rng.normal(scale=.1, size=len(X)) > 0).to_numpy()
    groups = np.repeat(np.arange(8), 24)
    selector = StableMinPISelector(random_state=12, fractions=(.25, .5), tolerance=1)
    sel = clone(selector).fit(X, y, groups=groups, ordered_groups=True)
    info = sel.diagnostics_
    assert info["rule"] == "environment" and info["width"] == 2
    disc = np.asarray(info["discovery_rows"])
    selected, _ = environment_path_from_training(X.to_numpy()[disc], y[disc], (2, 4),
        random_state=12, groups=groups[disc], ordered_groups=True, task="classification")
    np.testing.assert_array_equal(sel.get_support(indices=True), selected[2])
    assert set(sel.get_feature_names_out()) == set(X.columns[selected[2]])
    assert sel.transform(X.iloc[:4]).shape == (4, 2)
    twin = clone(selector).fit(X, y, groups=groups, ordered_groups=True)
    np.testing.assert_array_equal(twin.get_support(), sel.get_support())
    assert twin.diagnostics_["validation_candidates"] == info["validation_candidates"]


def test_calibration_requires_every_environment_and_can_keep_all_features():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 2))
    X[:100] = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (25, 1))
    X[100:125, 1] = 0
    y = X[:, 0] + 3 * X[:, 1]
    support, info = _calibrate_supports(
        X, y, np.arange(100), [np.arange(100, 125), np.arange(125, 150)],
        {1: np.array([0])}, "regression", 42, .02)
    assert info["validation_candidates"][0]["gaps"][0] >= -.02
    assert info["validation_candidates"][0]["gaps"][1] < -.02
    assert info["rule"] == "full" and list(support) == [0, 1]


def test_fallback_and_invalid_metadata():
    X = np.ones((24, 4))
    y = np.arange(24) % 2
    selector = StableMinPISelector().fit(X, y, groups=np.ones(24))
    assert selector.get_support().all() and selector.diagnostics_["reason"] == "insufficient_environments"
    with pytest.raises(ValueError, match="nonnegative integer"):
        StableMinPISelector(random_state=None).fit(X, y)
    with pytest.raises(ValueError, match="requires training groups"):
        StableMinPISelector().fit(X, y, ordered_groups=True)
    with pytest.raises(ValueError, match="groups aligned"):
        StableMinPISelector().fit(X, y, groups=[1, 2])
