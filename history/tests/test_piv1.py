import numpy as np
from xgboost import XGBClassifier

from minppi_mofs.piv1 import _order_blocks, _sample_rows, _vote_summary, relative_votes


def test_capped_binary_sampling_keeps_the_rare_class():
    y = np.repeat([0, 1], [2000, 11])
    sample = _sample_rows(y, "roc_auc", np.random.default_rng(9))
    assert len(sample) == len(np.unique(sample)) == 512
    assert y[sample].sum() == 11
    np.testing.assert_array_equal(sample, _sample_rows(y, "roc_auc", np.random.default_rng(9)))


def test_antithetic_blocks_and_worst_tier_votes():
    d = 29
    orders = _order_blocks(d, np.random.default_rng(40))
    assert orders.shape == (32, d)
    np.testing.assert_array_equal(np.sort(orders), np.tile(np.arange(d), (32, 1)))
    np.testing.assert_array_equal(orders[::2], orders[1::2, ::-1])
    tiers = np.minimum(np.arange(d) * 3 // d, 2)
    changes = np.zeros_like(orders, dtype=float)
    changes[orders == 2] = .06
    changes[orders == 5] = .12
    changes[(orders == 5) & (tiers == 1)] = -.15
    summary = _vote_summary(orders, changes, tiers)
    tier_scores = np.array(summary["tier_votes"])
    assert np.min(summary["tier_counts"]) > 0
    np.testing.assert_allclose(summary["min_vote"], tier_scores.min(axis=0))
    assert not summary["negative"][2]
    assert summary["negative"][5]
    assert tier_scores[0, 5] > 0 and tier_scores[2, 5] > 0
    assert tier_scores[1, 5] < 0
    folds = np.array(summary["fold_tier_votes"])
    expected = ((folds[0] < 0) & (folds[1] < 0)).any(axis=0)
    np.testing.assert_array_equal(summary["negative_consistent"], expected)
    assert summary["negative_consistent"][5]


def test_voting_reconstructs_fitted_full_model_and_is_deterministic():
    rng = np.random.default_rng(789)
    x = rng.normal(size=(120, 6))
    y = (2 * x[:, 0] - x[:, 1] > .25).astype(int)
    model = XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1, device="cpu").fit(x, y)
    a = relative_votes(model, x, y, metric="roc_auc", seed=42)
    b = relative_votes(model, x, y, metric="roc_auc", seed=42)
    assert a == b
    assert a["samples"] == len(x) and a["paths"] == 32
    assert a["full_margin_max_abs_error"] < 1e-4
    assert len(a["min_vote"]) == x.shape[1]
    assert all(-1 < value < 1 for value in a["min_vote"])
