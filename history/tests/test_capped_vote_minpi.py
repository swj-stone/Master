import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from minppi_mofs.capped_vote_minpi import capped_vote_top_from_training


@pytest.mark.parametrize("task,ordered,p", [("classification", False, 13), ("regression", True, 3)])
def test_all_votes_are_exact_held_refit_gains(task, ordered, p):
    rng = np.random.default_rng(109)
    X = rng.normal(size=(218, p))
    y = X[:, 0] + X[:, 1] * .3
    groups = np.repeat(np.arange(2), 109)
    if task == "classification":
        y = np.digitize(y, [-.3, .3])
        targets = np.eye(3)[y]
    else:
        targets = y[:, None]
    X[::7, ::3] = np.nan
    support, info = capped_vote_top_from_training(X, y, task=task, groups=groups, ordered_groups=ordered)
    expected = np.zeros((3, p))
    k = info["peer_window"][1]
    for fold in info["folds"]:
        tr, te = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(groups[tr], groups[te]).size
        if ordered:
            assert groups[tr].max() < groups[te].min()
        imputer = SimpleImputer().fit(X[tr])
        a = imputer.transform(X[tr])
        scaler = StandardScaler().fit(a)
        a, b = scaler.transform(a), scaler.transform(imputer.transform(X[te]))

        def loss(cols):
            if not len(cols):
                pred = targets[tr].mean(axis=0)
            else:
                reg = Ridge(alpha=1., solver="svd").fit(a[:, cols], targets[tr])
                pred = reg.predict(b[:, cols]).reshape(len(te), -1)
            return np.square(targets[te] - pred).sum() / len(te)

        for side, candidates in enumerate(info["column_halves"]):
            peers = fold["peers"][side]
            assert not np.intersect1d(candidates, peers).size
            for endpoint, subset in enumerate((peers[:0], peers[:int(k > 1)], peers)):
                base = loss(subset)
                for j in candidates:
                    expected[endpoint, j] += base - loss(np.r_[subset, j])
    expected /= len(info["folds"])
    for key, values in zip(("zero_peer_scores", "low_peer_scores", "high_peer_scores"), expected):
        np.testing.assert_allclose(info[key], values, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(info["scores"], np.minimum(expected[1], expected[2]), atol=1e-12)
    assert len(support) <= (p + 1) // 2


def test_concentrated_gain_can_choose_less_than_half_and_fixed_overrides_cap():
    rng = np.random.default_rng(415)
    X = rng.normal(size=(800, 20))
    y = X[:, 5]
    support, info = capped_vote_top_from_training(X, y, task="regression")
    assert 5 in support and len(support) < info["cap"] == 10
    fixed, details = capped_vote_top_from_training(X, y, 15, task="regression")
    np.testing.assert_array_equal(details["scores"], info["scores"])
    assert len(fixed) == 15
    null, _ = capped_vote_top_from_training(X, y*0, task="regression")
    np.testing.assert_array_equal(null, np.arange(10))
