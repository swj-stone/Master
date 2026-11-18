from unittest.mock import Mock

import numpy as np
from scipy.special import softmax
from sklearn.metrics import r2_score, roc_auc_score

from minppi_mofs.piv0 import _PrefixUtility, negative_vote


def test_multiclass_prefix_uses_class_biased_probabilities_for_macro_ovr():
    labels = np.repeat([0, 1, 2], [4, 5, 7])
    contributions = np.random.default_rng(31).integers(-3, 4, size=(3, len(labels), 4))
    bias = np.array([2.0, -1.0, 0.5])
    order = [2, 1, 3, 0]
    margins = []
    expected = []
    for k in range(len(order) + 1):
        logits = contributions[:, :, order[:k]].sum(axis=2).T + bias
        probabilities = softmax(logits, axis=1)
        expected.append(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
        margins.append(logits.T)

    actual = _PrefixUtility("ovr_auc", labels, 3)(np.stack(margins))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    # Check that the voting path itself carries the per-class TreeSHAP bias.
    model = Mock()
    parts = np.moveaxis(contributions, 0, 1)
    bias_column = np.broadcast_to(bias[None, :, None], (len(labels), 3, 1))
    model.get_booster.return_value.predict.side_effect = [
        np.concatenate([parts, bias_column], axis=2), margins[-1].T,
    ]
    unbiased = softmax(parts.sum(axis=2), axis=1)
    assert abs(roc_auc_score(labels, unbiased, multi_class="ovr") - expected[-1]) > 1e-3
    vote = negative_vote(model, np.zeros((len(labels), 4)), labels, metric="ovr_auc")
    np.testing.assert_allclose(vote["full_proxy_utility"], expected[-1], rtol=0, atol=1e-14)


def test_regression_prefix_matches_clipped_r2_for_each_subset():
    labels = np.array([3., 5., 6., 7., 9.])
    bias = 6.0
    signal = labels - bias
    contributions = np.column_stack([np.full_like(labels, 4.), signal, np.full_like(labels, -4.)])
    margins = [bias + contributions[:, :k].sum(axis=1) for k in range(4)]
    raw = [r2_score(labels, prediction) for prediction in margins]
    assert raw[1] < 0 and raw[-1] == 1

    actual = _PrefixUtility("r2", labels, 1)(np.stack(margins)[:, None, :])
    np.testing.assert_allclose(actual, np.maximum(raw, 0.), rtol=0, atol=1e-14)
