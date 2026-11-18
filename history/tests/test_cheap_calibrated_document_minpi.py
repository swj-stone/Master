import numpy as np
import pytest
from scipy import sparse

from minppi_mofs.cheap_calibrated_document_minpi import (
    _fit_direction, _prefix_losses, cheap_calibrated_document_top_from_training,
)


@pytest.mark.parametrize("classes", [2, 3])
def test_incremental_prefix_losses_equal_explicit_dense_predictions(classes):
    rng = np.random.default_rng(913)
    fit = rng.binomial(1, .3, (37, 11)).astype(float)
    held = rng.binomial(1, .3, (19, 11)).astype(float)
    fit[:, 0], held[:, 0] = 0, 0
    fit[:, 1], held[:, 1] = 1, 1
    yf, yh = np.arange(len(fit)) % classes, np.arange(len(held)) % classes
    mean, prior, beta, eta = _fit_direction(sparse.csr_matrix(fit), yf, classes)
    centered = fit - fit.mean(axis=0)
    target = np.eye(classes)[yf] - prior
    diagonal = (centered.T @ target) / (np.square(centered).sum(axis=0) + 1)[:, None]
    combined = centered @ diagonal
    expected_eta = np.sum(target * combined) / (np.square(combined).sum() + np.square(diagonal).sum())
    np.testing.assert_allclose(eta, expected_eta, atol=1e-14)
    np.testing.assert_allclose(beta, expected_eta * diagonal, atol=1e-14)
    order = rng.permutation(fit.shape[1])
    losses = _prefix_losses(sparse.csr_matrix(held), yh, order, mean, prior, beta)
    weights = 1 / (classes * np.bincount(yh, minlength=classes)[yh])
    for k in range(len(order) + 1):
        cols = order[:k]
        prediction = prior + (held[:, cols] - mean[cols]) @ beta[cols]
        expected = np.sum(weights[:, None] * np.square(np.eye(classes)[yh] - prediction))
        np.testing.assert_allclose(losses[k], expected, atol=2e-13, rtol=1e-12)


def test_calibration_labels_do_not_enter_votes_ties_or_fixed_teacher():
    rng = np.random.default_rng(611)
    X = sparse.csr_matrix(rng.binomial(1, .17, (641, 31)))
    y = np.arange(641) % 2
    _, first = cheap_calibrated_document_top_from_training(X, y, random_state=173, backend="numpy")
    changed = y.copy()
    changed[first["calibration_rows"]] = 1 - changed[first["calibration_rows"]]
    _, second = cheap_calibrated_document_top_from_training(X, changed, random_state=173, backend="numpy")
    for name in ("scores", "rank", "fit_rows", "calibration_rows", "coefficient", "feature_mean", "prior"):
        np.testing.assert_array_equal(first[name], second[name])
    assert not np.intersect1d(first["fit_rows"], first["calibration_rows"]).size


def test_small_samples_and_groups_are_explicitly_unsupported():
    X = sparse.csr_matrix(np.ones((513, 3)))
    y = np.arange(513) % 2
    with pytest.raises(ValueError, match="512"):
        cheap_calibrated_document_top_from_training(X[:512], y[:512])
    with pytest.raises(ValueError, match="groups"):
        cheap_calibrated_document_top_from_training(X, y, groups=np.arange(len(y)))
