import numpy as np
import pytest

from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training
from minppi_mofs.support_selector import PivotalSupportSelector


def _prepare(train, test):
    prepared = [np.zeros_like(train), np.zeros_like(test)]
    for j in range(train.shape[1]):
        valid = np.isfinite(train[:, j])
        if not valid.any():
            continue
        values = train[valid, j]
        mean = values[0] + (values - values[0]).mean()
        sd = np.sqrt(np.sum((values - mean)**2) / len(train))
        for output, raw in zip(prepared, (train, test)):
            observed = np.isfinite(raw[:, j])
            output[observed, j] = (raw[observed, j] - mean) / (sd or 1.)
    return prepared


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("ordered,missing", [(False, False), (True, False), (False, True), (True, True)])
def test_signal_vote_explicit_zero_and_strong_peer_ridge_refits(task, ordered, missing):
    rng = np.random.default_rng(587)
    X = rng.normal(size=(161, 8))
    y = X[:, 2] + .8 * X[:, 3] + rng.normal(size=len(X)) * .2
    X[:, 6] = .1
    X[:, 1] = X[:, 2] * 1e-15
    groups = np.repeat([0, 1, 2], [45, 55, 61])
    if missing:
        X[rng.random(X.shape) < .22] = np.nan
        X[groups < 2, 0] = np.inf
        X[groups == 2, 7] = -np.inf
    if task == "classification":
        y = np.digitize(y, [-1., .2])
        target = np.eye(3)[y]
    else:
        target = y[:, None]
    selected, info = signal_vote_top_from_training(X, y, task=task, groups=groups, ordered_groups=ordered)
    expected = np.zeros((2, X.shape[1]))
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(groups[fit], groups[held]).size
        if ordered:
            assert groups[fit].max() < groups[held].min()
        train, test = _prepare(X[fit], X[held])
        mean = target[fit].mean(axis=0)
        def loss(cols):
            x = train[:, cols]
            beta = np.linalg.solve(x.T @ x + np.eye(len(cols)), x.T @ (target[fit] - mean))
            return np.square(target[held] - mean - test[:, cols] @ beta).sum() / len(held)
        first, second = fold["peers"]
        strength = np.square(train.T @ (target[fit] - mean)).sum(axis=1) / (1 + np.square(train).sum(axis=0))
        np.testing.assert_allclose(strength[first], np.max(strength), rtol=1e-13)
        np.testing.assert_allclose(strength[second], np.delete(strength, first).max(), rtol=1e-13)
        for j in range(X.shape[1]):
            peer = second if j == first else first
            expected[:, j] += (loss([]) - loss([j]), loss([peer]) - loss([peer, j]))
    expected /= len(info["folds"])
    np.testing.assert_allclose(info["zero_peer_scores"], expected[0], atol=2e-12)
    np.testing.assert_allclose(info["one_peer_scores"], expected[1], atol=2e-12)
    np.testing.assert_allclose(info["scores"], expected.min(axis=0), atol=2e-12)
    scores = np.maximum(info["scores"], 0.)
    assert scores[selected].sum() >= .99 * scores.sum() - 1e-15
    assert scores[selected].sum() - scores[selected].min() < .99 * scores.sum() + 1e-15


def test_conditional_vote_changes_order_when_a_peer_supplies_the_same_signal():
    rng = np.random.default_rng(193)
    X = rng.normal(size=(1500, 8))
    X[:, 1] = X[:, 0]
    y = X[:, 0] + .7 * X[:, 2] + .1 * X[:, 3]
    _, info = signal_vote_top_from_training(X, y, task="regression")
    assert np.argmax(info["zero_peer_scores"]) in (0, 1)
    assert np.argmax(info["scores"]) == 2
    for fold in info["folds"]:
        assert len(fold["fit_rows"]) == len(fold["held_rows"]) == 256
    fixed, replay = signal_vote_top_from_training(X, y, 1, task="regression")
    np.testing.assert_array_equal(fixed, [2])
    np.testing.assert_array_equal(replay["scores"], info["scores"])


def test_empty_positive_mass_keeps_all_and_exact_ties_keep_input_order():
    X = np.full((27, 7), .1)
    X[::2, 4] = np.nan
    y = np.arange(len(X)) % 2
    all_cols, info = signal_vote_top_from_training(X, y)
    np.testing.assert_array_equal(all_cols, np.arange(7))
    np.testing.assert_array_equal(info["scores"], np.zeros(7))
    fixed, _ = signal_vote_top_from_training(X, y, 3)
    np.testing.assert_array_equal(fixed, [0, 1, 2])
    singleton, _ = signal_vote_top_from_training(X[:1, :1], y[:1])
    assert singleton.tolist() == [0]
    with pytest.raises(ValueError, match="two training rows"):
        signal_vote_top_from_training(X[:1], y[:1])
    with pytest.raises(ValueError, match="two distinct"):
        signal_vote_top_from_training(X, y, groups=np.ones(len(X)))


@pytest.mark.parametrize("fraction", ["auto", .5])
def test_signal_vote_transformer_uses_training_support(fraction):
    rng = np.random.default_rng(47)
    X = rng.normal(size=(160, 24))
    y = (X[:, 5] > X[:, 8]).astype(int)
    groups = np.repeat(np.arange(4), 40)
    params = dict(groups=groups[:120], ordered_groups=True)
    sel = PivotalSupportSelector(rule="signal_vote", fraction=fraction, random_state=301)
    sel.fit(X[:120], y[:120], **params)
    support, _ = signal_vote_top_from_training(X[:120], y[:120], None if fraction == "auto" else 12,
                                               random_state=301, **params)
    np.testing.assert_array_equal(sel.get_support(indices=True), support)
    np.testing.assert_array_equal(sel.transform(X[120:]), X[120:, support])
