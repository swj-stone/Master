import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from minppi_mofs import scaled_pilot_grouped_minpi as scaled
from minppi_mofs.shared_grouped_minpi import shared_grouped_top_from_training


def test_larger_pilot_preserves_ranking_rows_calibration_and_votes():
    rng = np.random.default_rng(93)
    X = rng.normal(size=(800, 260))
    y = np.arange(800) % 4
    _, before = shared_grouped_top_from_training(X, y)
    _, after = scaled.scaled_pilot_grouped_top_from_training(X, y)
    assert len(after["pilot_fit_rows"]) == 520
    np.testing.assert_array_equal(before["pilot_fit_rows"], after["ranking_fit_rows"])
    for key in ("calibration_rows", "representatives", "ranking", "votes", "pivotal_scores"):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.intersect1d(after["pilot_fit_rows"], after["calibration_rows"]).size
    assert np.isin(after["ranking_fit_rows"], after["pilot_fit_rows"]).all()


def test_expanded_pilot_all_prefix_predictions_match_direct_ridge():
    rng = np.random.default_rng(97)
    X = rng.normal(size=(800, 260))
    y = np.arange(800) % 3
    _, info = scaled.scaled_pilot_grouped_top_from_training(X, y, ranking="F")
    fit, held = info["pilot_fit_rows"], info["calibration_rows"]
    for i, width in enumerate(info["evaluated_widths"]):
        columns = info["ranking"][:width]
        scaler = StandardScaler().fit(X[np.ix_(fit, columns)])
        model = RidgeClassifier(alpha=1., solver="cholesky").fit(
            scaler.transform(X[np.ix_(fit, columns)]), y[fit])
        reference = model.decision_function(scaler.transform(X[np.ix_(held, columns)]))
        np.testing.assert_allclose(reference, info["calibration_decisions"][i], atol=1e-9, rtol=1e-9)
        np.testing.assert_array_equal(model.predict(scaler.transform(X[np.ix_(held, columns)])),
                                      info["calibration_predictions"][i])


def test_pilot_never_exceeds_available_noncalibration_rows():
    y = np.arange(90) % 3
    rank, held = np.arange(60), np.arange(60, 90)
    np.testing.assert_array_equal(scaled._extended_fit_rows(y, rank, held, 200, 17), rank)
