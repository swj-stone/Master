import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from minppi_mofs import prefix_factor_minpi as factor
from minppi_mofs.bestcal_grouped_minpi import best_calibrated_support
from minppi_mofs.scaled_pilot_grouped_minpi import scaled_pilot_grouped_top_from_training


@pytest.mark.parametrize("classes,shape", [(2, (43, 13)), (4, (21, 47)), (3, (13, 13))])
def test_shared_factor_prefixes_match_explicit_ridge_with_constants_and_dependence(classes, shape):
    rng = np.random.default_rng(901)
    X, Xt = rng.normal(size=shape), rng.normal(size=(17, shape[1]))
    X[:, -1], Xt[:, -1] = 7., 7.
    X[:, -2], Xt[:, -2] = X[:, 0], Xt[:, 0]
    y = np.array([f"label-{k}" for k in np.arange(shape[0]) % classes])
    order = rng.permutation(shape[1])
    widths = sorted({1, shape[1] // 3, shape[1]})
    decisions, predictions = factor._factorized_prefix_ridge(X, y, Xt, order, widths)
    for width, decision, prediction in zip(widths, decisions, predictions):
        columns = order[:width]
        scaler = StandardScaler().fit(X[:, columns])
        model = RidgeClassifier(alpha=1., solver="cholesky").fit(scaler.transform(X[:, columns]), y)
        np.testing.assert_allclose(decision, model.decision_function(scaler.transform(Xt[:, columns])),
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(prediction, model.predict(scaler.transform(Xt[:, columns])))


def test_narrow_branch_uses_one_full_factor_for_all_prefixes(monkeypatch):
    rng = np.random.default_rng(902)
    X, Xt = rng.normal(size=(71, 23)), rng.normal(size=(19, 23))
    calls = []
    original = factor.cho_factor

    def observe(matrix, **kwargs):
        calls.append(matrix.shape)
        return original(matrix, **kwargs)

    monkeypatch.setattr(factor, "cho_factor", observe)
    factor._factorized_prefix_ridge(X, np.arange(71) % 3, Xt, np.arange(23), [1, 3, 11, 23])
    assert calls == [(23, 23)]


def test_all_rankings_and_both_budget_rules_match_unchanged_scaled_pilot():
    rng = np.random.default_rng(903)
    X = rng.normal(size=(801, 27))
    y = np.arange(len(X)) % 3
    for ranking in ("pivotal", "representatives", "F"):
        old, before = scaled_pilot_grouped_top_from_training(X, y, ranking=ranking)
        best, _ = best_calibrated_support(X.shape[1], before)
        for budget_rule, expected in (("first_acceptable", old), ("bestcal", best)):
            actual, after = factor.factorized_grouped_top_from_training(X, y, ranking=ranking,
                                                                       budget_rule=budget_rule)
            np.testing.assert_array_equal(actual, expected)
            for key in ("ranking", "ranking_fit_rows", "pilot_fit_rows", "calibration_rows",
                        "calibration_predictions", "calibration_scores"):
                np.testing.assert_array_equal(after[key], before[key])
            np.testing.assert_allclose(after["calibration_decisions"], before["calibration_decisions"], atol=1e-9, rtol=1e-9)


def test_expanded_and_calibration_rows_do_not_enter_ranking(monkeypatch):
    rng = np.random.default_rng(904)
    X = rng.normal(size=(800, 260))
    y = np.arange(len(X)) % 3
    rank, held = np.arange(512), np.arange(512, 768)
    monkeypatch.setattr(factor, "_pilot_rows", lambda *args, **kwargs: (rank, held))
    _, before = factor.factorized_grouped_top_from_training(X, y)
    changed = X.copy()
    changed[512:] = rng.normal(size=changed[512:].shape) * 5
    labels = y.copy()
    labels[held] = rng.permutation(labels[held])
    _, after = factor.factorized_grouped_top_from_training(changed, labels)
    assert len(before["pilot_fit_rows"]) == 520
    assert not np.intersect1d(before["pilot_fit_rows"], held).size
    for field in ("ranking", "representatives", "votes", "pivotal_scores", "ranking_fit_rows"):
        np.testing.assert_array_equal(before[field], after[field])
