import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from minppi_mofs.rescaled_calibrated_document_minpi import (
    _prefix_moments, rescaled_calibrated_document_top_from_training,
)


@pytest.mark.parametrize("classes", [2, 3])
def test_every_prefix_scale_equals_explicit_one_dimensional_ridge(classes):
    rng = np.random.default_rng(195)
    X = rng.binomial(1, .3, (41, 9)).astype(float)
    X[:, 0], X[:, 1] = 0, 1
    y = np.arange(len(X)) % classes
    mean = X.mean(axis=0)
    centered = X - mean
    target = np.eye(classes)[y] - np.eye(classes)[y].mean(axis=0)
    beta = (centered.T @ target) / (np.square(centered).sum(axis=0) + 1.)[:, None]
    order = rng.permutation(X.shape[1])
    base, cross, square, saved = _prefix_moments(
        sparse.csr_matrix(X), target, order, mean, beta, np.ones(len(X)), set(range(10)))
    penalty = np.r_[0., np.square(beta[order]).sum(axis=1).cumsum()]
    eta = np.divide(cross, square + penalty, out=np.zeros_like(cross), where=square + penalty > 1e-14)
    for k in range(1, len(order) + 1):
        cols = order[:k]
        g = centered[:, cols] @ beta[cols]
        np.testing.assert_allclose(saved[k], g, atol=2e-13)
        np.testing.assert_allclose(cross[k], np.sum(target * g), atol=2e-13)
        np.testing.assert_allclose(square[k], np.square(g).sum(), atol=2e-13)
        if penalty[k] > 0:
            fitted = Ridge(alpha=penalty[k], fit_intercept=False, solver="cholesky").fit(
                g.reshape(-1, 1), target.reshape(-1)).coef_[0]
            np.testing.assert_allclose(eta[k], fitted, atol=2e-13)
        expected_loss = np.square(target - eta[k] * g).sum()
        np.testing.assert_allclose(base - 2 * eta[k] * cross[k] + eta[k]**2 * square[k],
                                   expected_loss, atol=2e-13)


def test_calibration_labels_cannot_change_rank_or_any_fitted_prefix_scale():
    rng = np.random.default_rng(431)
    X = sparse.csr_matrix(rng.binomial(1, .15, (621, 37)))
    y = np.arange(X.shape[0]) % 2
    _, a = rescaled_calibrated_document_top_from_training(X, y, random_state=781, backend="numpy")
    changed = y.copy()
    changed[a["calibration_rows"]] = 1 - changed[a["calibration_rows"]]
    _, b = rescaled_calibrated_document_top_from_training(X, changed, random_state=781, backend="numpy")
    for field in ("scores", "rank", "fit_rows", "calibration_rows", "eta_path", "direction"):
        np.testing.assert_array_equal(a[field], b[field])
    assert len(a["ba_grid"]) <= 16
    assert a["ba_grid"].min() >= a["floor"] and a["ba_grid"].max() <= a["cap"]
    for rule in ("brier", "balanced_accuracy"):
        cols, info = rescaled_calibrated_document_top_from_training(
            X, y, random_state=781, backend="numpy", budget_rule=rule)
        k = len(cols)
        if info["feasible"]:
            if rule == "brier":
                assert info["prefix_losses"][k] <= info["prefix_losses"][-1] + 1e-12
            else:
                where = np.flatnonzero(info["ba_grid"] == k)[0]
                assert info["ba_grid_scores"][where] >= info["full_teacher_ba"] - .02 - 1e-12
