from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.moment_minpi import moment_top_from_training


def _explicit_values(raw_train, raw_held, target, query, mate):
    p = raw_train.shape[1]
    train, held = np.zeros_like(raw_train), np.zeros_like(raw_held)
    for j in range(p):
        observed = np.isfinite(raw_train[:, j])
        if not observed.any():
            continue
        x = raw_train[observed, j]
        if np.ptp(x) == 0:
            continue
        mean = x.mean()
        std = np.sqrt(np.sum((x - mean)**2) / len(train))
        train[observed, j] = (x - mean) / std
        seen = np.isfinite(raw_held[:, j])
        held[seen, j] = (raw_held[seen, j] - mean) / std
    center = target.mean(axis=0)
    target, query = target - center, query - center
    null_loss = np.square(query).sum(axis=1).mean()
    blocks = [{j, other} for j, other in enumerate(mate) if j <= other]
    block_gain = {}
    for block in blocks:
        for k in range(1, len(block) + 1):
            for subset in combinations(sorted(block), k):
                A = np.vstack((train[:, subset], np.eye(k)))
                Y = np.vstack((target, np.zeros((k, target.shape[1]))))
                beta = np.linalg.lstsq(A, Y, rcond=None)[0]
                err = query - held[:, subset] @ beta
                block_gain[frozenset(subset)] = null_loss - np.square(err).sum(axis=1).mean()
    values = {}
    for k in range(p + 1):
        for subset in combinations(range(p), k):
            S = frozenset(subset)
            values[S] = sum(block_gain.get(S & block, 0.) for block in blocks)
    return values


@pytest.mark.parametrize("ordered,missing", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_moment_votes_equal_explicit_imputation_refits_and_all_peer_subsets(ordered, missing, task):
    rng = np.random.default_rng(52)
    X = rng.normal(size=(74, 6))
    y = X[:, 2] - .25 * X[:, 4] + .2 * rng.normal(size=len(X))
    X[:, 5] = -X[:, 3]
    X[:, 2] *= 1e69
    X[:, 4] *= 1e-69
    groups = np.repeat(["early", "middle", "recent"], [19, 21, 34])
    if missing:
        X[rng.random(size=X.shape) < .28] = np.nan
        X[groups != "recent", 1] = np.inf
        X[groups == "recent", 3] = -np.inf
        X[::7, 5] = np.nan
    target = y[:, None]
    if task == "classification":
        y = np.digitize(y, [-.7, .2, .9])
        target = np.eye(4)[y]
    cols, info = moment_top_from_training(X, y, 4, task=task, groups=groups, ordered_groups=ordered)
    expected = np.zeros((2, X.shape[1]))
    assert len(info["folds"]) == (1 if ordered else 2)
    for fold in info["folds"]:
        fit, held, mate = fold["fit_rows"], fold["held_rows"], fold["mate"]
        np.testing.assert_array_equal(mate[mate], np.arange(len(mate)))
        assert not np.intersect1d(groups[fit], groups[held]).size
        if ordered:
            assert set(groups[held]) == {"recent"}
        values = _explicit_values(X[fit], X[held], target[fit], target[held], mate)
        for j in range(X.shape[1]):
            for tier, k in enumerate([0, 3]):
                peers = set(range(X.shape[1])) - {j}
                gains = []
                for subset in combinations(peers, k):
                    S = frozenset(subset)
                    gains.append(values[S | {j}] - values[S])
                expected[tier, j] += np.mean(gains) / len(info["folds"])
    score = expected.min(axis=0)
    np.testing.assert_allclose(info["scores"], score, atol=2e-12, rtol=2e-12)
    # Fully unobserved or constant candidates can share an exact zero.
    assert min(score[cols]) >= max(np.delete(score, cols)) - 2e-12


def test_one_group_in_sketch_is_repaired_within_budget_and_without_group_overlap():
    n = 2000
    seed = 19
    split_seed = np.random.SeedSequence(seed).spawn(2)[0]
    initial = np.random.default_rng(split_seed).choice(n, 192, replace=False)
    groups = np.zeros(n, dtype=int)
    groups[np.setdiff1d(np.arange(n), initial)[:10]] = 1
    X = np.arange(n * 3).reshape(n, 3)
    y = np.arange(n) % 3
    _, info = moment_top_from_training(X, y, 2, groups=groups, random_state=seed)
    fit, held = (info["folds"][0][key] for key in ("fit_rows", "held_rows"))
    assert len(fit) + len(held) == 192
    assert set(groups[fit]) == {0} and set(groups[held]) == {1}
    with pytest.raises(ValueError, match="distinct"):
        moment_top_from_training(X, y, 2, groups=np.zeros(n))


def test_singleton_ties_and_sketch_replay():
    rng = np.random.default_rng(61)
    X = np.full((450, 5), .1)
    X[:, 3] = np.inf
    y = rng.choice(["one", "two", "three"], size=len(X))
    cols, info = moment_top_from_training(X, y, 3)
    np.testing.assert_array_equal(cols, [0, 1, 2])
    np.testing.assert_array_equal(info["scores"], np.zeros(5))
    replay, again = moment_top_from_training(X, y, 3)
    np.testing.assert_array_equal(cols, replay)
    for a, b in zip(info["folds"], again["folds"], strict=True):
        for key in ("mate", "fit_rows", "held_rows"):
            np.testing.assert_array_equal(a[key], b[key])
        assert len(a["fit_rows"]) + len(a["held_rows"]) == 192
        assert sum(a["mate"] == np.arange(5)) == 1
    only, full = moment_top_from_training(X[:, :1], y, 1)
    np.testing.assert_array_equal(only, [0])
    assert full["folds"] == []
