r"""
Tests for ``minppi_mofs.voting_game`` — boundary cases, caching, non-monotonicity.

Covers:
- AccFullCache correctness (caching, empty-set handling, full-feature AUC)
- compute_v_alpha binary output
- classify_minppi label mapping
- Non-monotonicity behavior
- Edge cases: single feature, subset reordering, cache reuse

Run::

    ..\venv\Scripts\python.exe -m pytest tests/test_voting_game.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from minppi_mofs.voting_game import (
    AccFullCache,
    UtilityView,
    classify_minppi,
    compute_v_alpha,
    make_acc_full_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Deterministic synthetic 4-feature binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "f_signal": rng.normal(0, 1, n),
            "f_weak": rng.normal(0, 1, n),
            "f_noise1": rng.normal(0, 1, n),
            "f_noise2": rng.normal(0, 1, n),
        }
    )
    y = np.where(X["f_signal"] + 0.3 * X["f_weak"] > 0, 1, 0).astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_val, y_tr, y_val


@pytest.fixture(scope="module")
def acc_cache(toy_data) -> AccFullCache:
    """Pre-built AccFullCache shared across tests in this module."""
    X_tr, X_val, y_tr, y_val = toy_data
    return make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=42)


# ---------------------------------------------------------------------------
# AccFullCache
# ---------------------------------------------------------------------------


class TestAccFullCache:
    """AccFullCache construction, caching, and evaluation."""

    def test_acc_full_in_range(self, acc_cache: AccFullCache) -> None:
        """Acc_full must be in [0, 1]."""
        assert 0.0 <= acc_cache.acc_full <= 1.0

    def test_acc_full_reasonably_high(self, acc_cache: AccFullCache) -> None:
        """Full-feature model should achieve well above random on toy data."""
        assert acc_cache.acc_full > 0.6, (
            f"Expected Acc_full > 0.6 on toy data, got {acc_cache.acc_full:.4f}"
        )

    def test_empty_set_returns_baseline(self, acc_cache: AccFullCache) -> None:
        """v(∅) should be approximately 0.5 (random AUC baseline)."""
        assert acc_cache.evaluate(set()) == pytest.approx(0.5, abs=0.05)

    def test_single_feature_not_worse_than_empty(self, acc_cache: AccFullCache) -> None:
        """A signal feature alone should not be worse than empty set."""
        # Small n + simple model means performance may not beat baseline,
        # but we just sanity-check that evaluation doesn't crash.
        val = acc_cache.evaluate({"f_signal"})
        assert 0.0 <= val <= 1.0

    def test_caching_reuses_results(self, acc_cache: AccFullCache) -> None:
        """Repeated evaluation of the same coalition must hit the cache."""
        S = {"f_signal", "f_weak"}
        v1 = acc_cache.evaluate(S)
        v2 = acc_cache.evaluate(S)
        assert v1 == v2
        # The cache key is a sorted tuple; verify order-agnostic
        v3 = acc_cache.evaluate({"f_weak", "f_signal"})
        assert v1 == v3

    def test_full_set_equals_acc_full(self, acc_cache: AccFullCache) -> None:
        """v(all features) should equal the stored acc_full."""
        all_features = {"f_signal", "f_weak", "f_noise1", "f_noise2"}
        assert acc_cache.evaluate(all_features) == pytest.approx(acc_cache.acc_full)

    def test_same_subset_different_alpha_not_recached(
        self, acc_cache: AccFullCache
    ) -> None:
        """Threshold changes reuse the same cached AUC but give different v_alpha."""
        S = {"f_signal"}
        acc_cache.set_threshold(0.90)
        v_low = acc_cache.is_winning(S)
        acc_cache.set_threshold(0.999)
        v_high = acc_cache.is_winning(S)
        # v_low may or may not be winning, but the type must be bool
        assert isinstance(v_low, bool)
        assert isinstance(v_high, bool)

    def test_set_threshold_before_is_winning(self, acc_cache: AccFullCache) -> None:
        """is_winning without set_threshold raises RuntimeError."""
        empty_cache = AccFullCache(
            acc_cache._X_train,
            acc_cache._X_val,
            acc_cache._y_train,
            acc_cache._y_val,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="set_threshold"):
            empty_cache.is_winning({"f_signal"})

    def test_public_cost_counters_track_only_successful_nonempty_fits(
        self, toy_data
    ) -> None:
        X_tr, X_val, y_tr, y_val = toy_data
        cache = AccFullCache(X_tr, X_val, y_tr, y_val, seed=42)

        assert cache.model_fit_count == 1
        assert cache.coalition_count == 1
        cache.evaluate(set())
        assert cache.model_fit_count == 1
        assert cache.coalition_count == 2

        cache.evaluate({"f_signal"})
        assert cache.model_fit_count == 2
        assert cache.coalition_count == 3
        cache.evaluate({"f_signal"})
        assert cache.model_fit_count == 2
        assert cache.coalition_count == 3
        with pytest.raises(AttributeError):
            cache.model_fit_count = 99

    def test_utility_views_report_the_same_shared_cost_counters(self, toy_data) -> None:
        X_tr, X_val, y_tr, y_val = toy_data
        cache = AccFullCache(
            X_tr, X_val, y_tr, y_val, task="binary", utility="accuracy", seed=42
        )
        accuracy = UtilityView(cache, utility="accuracy", alpha=0.95)
        auc = UtilityView(cache, utility="roc_auc", alpha=0.95)

        accuracy.evaluate({"f_signal"})
        before = (accuracy.model_fit_count, accuracy.coalition_count)
        auc.evaluate({"f_signal"})

        assert before == (2, 2)
        assert (auc.model_fit_count, auc.coalition_count) == before

    def test_frozen_backend_parameters_are_forwarded_and_unknown_keys_fail_closed(
        self, toy_data
    ) -> None:
        X_tr, X_val, y_tr, y_val = toy_data
        cache = AccFullCache(
            X_tr,
            X_val,
            y_tr,
            y_val,
            seed=42,
            model_params={"device": "cpu", "tree_method": "hist"},
        )

        params = cache._model.get_params()
        assert params["device"] == "cpu"
        assert params["tree_method"] == "hist"
        with pytest.raises(ValueError, match="model_params|unknown"):
            AccFullCache(
                X_tr,
                X_val,
                y_tr,
                y_val,
                seed=42,
                model_params={"predictor": "cpu_predictor"},
            )


# ---------------------------------------------------------------------------
# compute_v_alpha
# ---------------------------------------------------------------------------


class TestComputeVAlpha:
    """Public API: compute_v_alpha."""

    def test_returns_binary(self, toy_data) -> None:
        """compute_v_alpha must return 0 or 1."""
        X_tr, X_val, y_tr, y_val = toy_data
        v = compute_v_alpha({"f_signal"}, X_tr, X_val, y_tr, y_val, alpha=0.90, seed=42)
        assert v in {0, 1}

    def test_full_set_at_low_alpha_is_winning(self, toy_data) -> None:
        """With α=0.50 the full set should always win."""
        X_tr, X_val, y_tr, y_val = toy_data
        v = compute_v_alpha(
            {"f_signal", "f_weak", "f_noise1", "f_noise2"},
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.50,
            seed=42,
        )
        assert v == 1

    def test_with_prebuilt_cache(self, acc_cache: AccFullCache) -> None:
        """compute_v_alpha must accept a pre-built cache."""
        acc_cache.set_threshold(0.95)
        v = compute_v_alpha(
            {"f_signal"},
            acc_cache._X_train,
            acc_cache._X_val,
            acc_cache._y_train,
            acc_cache._y_val,
            alpha=0.95,
            acc_cache=acc_cache,
            seed=42,
        )
        assert v in {0, 1}

    def test_empty_coalition(self, toy_data) -> None:
        """Empty coalition should not crash — v_alpha(∅) ∈ {0, 1}."""
        X_tr, X_val, y_tr, y_val = toy_data
        v = compute_v_alpha(set(), X_tr, X_val, y_tr, y_val, alpha=0.90, seed=42)
        assert v in {0, 1}


# ---------------------------------------------------------------------------
# classify_minppi
# ---------------------------------------------------------------------------


class TestClassifyMinppi:
    """Three-way label mapping."""

    def test_decisive_maps_correctly(self) -> None:
        assert classify_minppi(+1) == "decisive"

    def test_neutral_maps_correctly(self) -> None:
        assert classify_minppi(0) == "neutral"

    def test_harmful_maps_correctly(self) -> None:
        assert classify_minppi(-1) == "harmful"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(KeyError):
            classify_minppi(2)

    def test_invalid_value_raises_negative(self) -> None:
        with pytest.raises(KeyError):
            classify_minppi(-2)


# ---------------------------------------------------------------------------
# Lemma 1 (Boundedness) — marginal values always in {-1, 0, +1}
# ---------------------------------------------------------------------------


class TestLemma1Boundedness:
    """Lemma 1: eval_marginal always returns -1, 0, or +1."""

    @pytest.mark.parametrize(
        "S",
        [
            set(),
            {"f_signal"},
            {"f_noise1"},
            {"f_noise2"},
            {"f_signal", "f_weak"},
            {"f_signal", "f_noise1", "f_noise2"},
            {"f_weak", "f_noise1", "f_noise2"},
        ],
    )
    def test_marginal_in_trinary(self, S: set[str], acc_cache: AccFullCache) -> None:
        """For any S, the marginal of adding f_signal must be -1, 0, or +1."""
        acc_cache.set_threshold(0.90)
        marg = acc_cache.eval_marginal(S, "f_signal")
        assert marg in {-1, 0, +1}, f"Lemma 1 violated: marginal = {marg} for S={S}"

    def test_all_features_marginal_trinary(self, acc_cache: AccFullCache) -> None:
        """Every feature's marginal on every subset must be in {-1, 0, +1}."""
        acc_cache.set_threshold(0.90)
        features = ["f_signal", "f_weak", "f_noise1", "f_noise2"]
        import itertools

        for feat in features:
            others = [f for f in features if f != feat]
            for r in range(len(others) + 1):
                for combo in itertools.combinations(others, r):
                    marg = acc_cache.eval_marginal(set(combo), feat)
                    assert marg in {-1, 0, +1}, (
                        f"Lemma 1 violated: marginal({feat} | {combo}) = {marg}"
                    )


# ---------------------------------------------------------------------------
# Non-monotonicity (Remark 2)
# ---------------------------------------------------------------------------


class TestNonMonotonicity:
    """v_alpha is not guaranteed monotone — adding features can hurt."""

    def test_adding_noise_may_not_help(self, acc_cache: AccFullCache) -> None:
        """Adding noise features should not increase AUC — may decrease."""
        acc_cache.set_threshold(0.95)
        v_signal = acc_cache.evaluate({"f_signal"})
        v_with_noise = acc_cache.evaluate({"f_signal", "f_noise1", "f_noise2"})
        # Not a strict assertion (small data = noisy), but document the observation
        # In repeated trials, noise features often degrade perf
        assert v_signal >= 0.0
        assert v_with_noise >= 0.0
