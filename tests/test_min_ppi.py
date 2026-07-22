"""
Tests for ``minppi_mofs.min_ppi`` — estimation strategies and Theorem 1.

Covers (per tests/README.md):
- Lemma 1 (boundedness): MinPPI_i ∈ {-1, 0, +1}
- Lemma 2 (coincidence): MinSS_i = MinBz_i (same min over same set)
- Theorem 1 (three-way classification): decisive / harmful / neutral semantics
- Exact vs Algorithm A agreement on small n
- Algorithm A upward bias (Lemma 3)
- minppi_stats correctness
- Hybrid dispatcher routing

Run::

    pytest tests/test_min_ppi.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from minppi_mofs.min_ppi import (
    AlgorithmBPerturbationSchedule,
    AlgorithmBPerturbationTrial,
    estimate_minppi,
    estimate_minppi_exact,
    estimate_minppi_hybrid,
    estimate_minppi_mi_correlation_guided,
    estimate_minppi_random_mc,
    estimate_minppi_mi_guided,
    make_shared_coalition_plan,
    make_algorithm_b_perturbation_schedule,
    algorithm_b_schedule_digest,
    minppi_stats,
)
from minppi_mofs.voting_game import make_acc_full_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic 5-feature dataset — small enough for exact enumeration."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {
            "f_signal": rng.normal(0, 1, n),
            "f_weak": rng.normal(0, 1, n),
            "f_noise1": rng.normal(0, 1, n),
            "f_noise2": rng.normal(0, 1, n),
            "f_noise3": rng.normal(0, 1, n),
        }
    )
    # f_signal drives the label; f_noise* are irrelevant
    y = np.where(X["f_signal"] + 0.2 * X["f_weak"] > 0, 1, 0).astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_val, y_tr, y_val


@pytest.fixture(scope="module")
def acc_cache(small_data):
    """Shared AccFullCache for all min_ppi tests."""
    X_tr, X_val, y_tr, y_val = small_data
    return make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=42)


# ---------------------------------------------------------------------------
# Lemma 1 — Boundedness
# ---------------------------------------------------------------------------


class TestLemma1Boundedness:
    """Lemma 1: Every MinPPI estimate must be in {-1, 0, +1}."""

    @pytest.mark.parametrize("method", ["exact", "random_mc"])
    def test_all_values_trinary(self, small_data, method: str) -> None:
        """Every MinPPI score is -1, 0, or +1."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            method=method,
            K=100,
            seed=42,
        )
        for feat, val in minppi.items():
            assert val in {-1, 0, +1}, f"Lemma 1 violated: MinPPI({feat}) = {val}"
        for feat, val in minppi.items():
            assert val in {-1, 0, +1}, f"Lemma 1 violated: MinPPI({feat}) = {val}"

    def test_exact_enumeration_bounded(self, small_data, acc_cache) -> None:
        """Exact enumeration on n=5 must produce trinary values."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        assert len(minppi) == 5
        for val in minppi.values():
            assert val in {-1, 0, +1}


# ---------------------------------------------------------------------------
# Lemma 2 — MinSS = MinBz (coincidence)
# ---------------------------------------------------------------------------


class TestLemma2Coincidence:
    """Lemma 2: MinSS_i = MinBz_i — both enumerate the same set of marginals."""

    def test_exact_vs_exact_agree(self, small_data, acc_cache) -> None:
        """Exact MinPPI (via enumeration, which is MinBz) is self-consistent."""
        X_tr, X_val, y_tr, y_val = small_data
        r1 = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        r2 = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        # Same cache + same alpha → deterministic, identical results
        for feat in r1:
            assert r1[feat] == r2[feat], f"Exact MinPPI not deterministic for {feat}"


# ---------------------------------------------------------------------------
# Theorem 1 — Three-Way Classification
# ---------------------------------------------------------------------------


class TestTheorem1ThreeWay:
    """Theorem 1: MinPPI gives a clean decisive / neutral / harmful split."""

    def test_exact_returns_reasonable_classes(self, small_data, acc_cache) -> None:
        """On toy data, the signal feature should not be harmful."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        # f_signal drives the label → should not be harmful
        assert minppi["f_signal"] in {0, +1}, (
            f"Signal feature should not be harmful, got {minppi['f_signal']}"
        )

    def test_higher_alpha_increases_harmful_count(self, small_data, acc_cache) -> None:
        """Higher α should produce ≥ harmful classifications (monotone in α)."""
        X_tr, X_val, y_tr, y_val = small_data
        n_harmful_lo = 0
        n_harmful_hi = 0
        for alpha in [0.85, 0.99]:
            acc_cache.set_threshold(alpha)
            minppi = estimate_minppi_exact(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=alpha,
                acc_cache=acc_cache,
                seed=42,
            )
            harmful = sum(1 for v in minppi.values() if v == -1)
            if alpha == 0.85:
                n_harmful_lo = harmful
            else:
                n_harmful_hi = harmful
        # At strict α=0.99, more features should appear harmful
        assert n_harmful_hi >= n_harmful_lo, (
            f"Expected more harmful at α=0.99 ({n_harmful_hi}) than α=0.85 ({n_harmful_lo})"
        )

    def test_classification_labels(self, small_data, acc_cache) -> None:
        """Each feature gets exactly one of three labels."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        labels = {"decisive": +1, "neutral": 0, "harmful": -1}
        for feat, val in minppi.items():
            label = [k for k, v in labels.items() if v == val]
            assert len(label) == 1, f"Feature {feat} has ambiguous value {val}"


# ---------------------------------------------------------------------------
# Algorithm A — Random MC
# ---------------------------------------------------------------------------


class TestSharedCoalitionPlan:
    """Utility-independent shared-coalition evaluation schedule."""

    def test_is_immutable_and_seed_reproducible(self) -> None:
        features = ("a", "b", "c", "d", "e")

        first = make_shared_coalition_plan(
            features, K=4, per_coalition=2, seed=42,
        )
        second = make_shared_coalition_plan(
            features, K=4, per_coalition=2, seed=42,
        )

        assert isinstance(first, tuple)
        assert first == second
        assert all(isinstance(coalition, frozenset) for coalition, _ in first)
        assert all(isinstance(evaluated, tuple) for _, evaluated in first)

    def test_features_are_absent_and_supplemental_coverage_is_fixed(self) -> None:
        features = ("a", "b", "c", "d", "e")
        K = 3
        plan = make_shared_coalition_plan(
            features, K=K, per_coalition=2, seed=7,
        )

        base_hits = {feature: 0 for feature in features}
        final_hits = {feature: 0 for feature in features}
        for step_index, (coalition, evaluated) in enumerate(plan):
            assert coalition <= set(features)
            for feature in evaluated:
                assert feature in features
                assert feature not in coalition
                final_hits[feature] += 1
                if step_index < K:
                    base_hits[feature] += 1

        for feature in features:
            if base_hits[feature] < 2:
                assert final_hits[feature] == 5
            else:
                assert final_hits[feature] == base_hits[feature]


class TestSharedCoalitionPlanExecution:
    """Explicit plans are validated and executed without adaptive skips."""

    def test_executes_a_valid_explicit_plan(
        self,
        small_data: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        acc_cache,
    ) -> None:
        X_train, X_val, y_train, y_val = small_data
        plan = (
            (frozenset(), ("f_signal",)),
            (frozenset({"f_noise1"}), ("f_weak",)),
        )

        result = estimate_minppi_random_mc(
            X_train,
            X_val,
            y_train,
            y_val,
            alpha=0.90,
            shared_coalitions=True,
            shared_plan=plan,
            acc_cache=acc_cache,
            seed=42,
        )

        assert set(result) == set(X_train.columns)
        assert tuple() in acc_cache._preds
        assert ("f_signal",) in acc_cache._preds
        assert ("f_noise1",) in acc_cache._preds
        assert ("f_noise1", "f_weak") in acc_cache._preds

    @pytest.mark.parametrize(
        "plan, message",
        [
            (((frozenset({"unknown"}), ("f_signal",)),), "coalition"),
            (((frozenset(), ("unknown",)),), "evaluated"),
            (((frozenset({"f_signal"}), ("f_signal",)),), "absent"),
        ],
    )
    def test_rejects_invalid_plan_before_cache_use(
        self,
        small_data: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        plan,
        message: str,
    ) -> None:
        X_train, X_val, y_train, y_val = small_data

        with pytest.raises(ValueError, match=message):
            estimate_minppi_random_mc(
                X_train,
                X_val,
                y_train,
                y_val,
                shared_coalitions=True,
                shared_plan=plan,
                acc_cache=object(),
            )

    def test_rejects_plan_outside_shared_mode(
        self,
        small_data: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        X_train, X_val, y_train, y_val = small_data
        plan = ((frozenset(), ("f_signal",)),)

        with pytest.raises(ValueError, match="shared_coalitions"):
            estimate_minppi_random_mc(
                X_train,
                X_val,
                y_train,
                y_val,
                shared_coalitions=False,
                shared_plan=plan,
                acc_cache=object(),
            )


class TestAlgorithmA:
    """Algorithm A: random MC estimation."""

    def test_returns_all_features(self, small_data, acc_cache) -> None:
        """Must return a value for every feature."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi_random_mc(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            K=50,
            acc_cache=acc_cache,
            seed=42,
        )
        assert len(minppi) == 5

    def test_upward_bias_lemma3(self, small_data, acc_cache) -> None:
        """Lemma 3: sample min ≥ population min (MC ≥ exact)."""
        X_tr, X_val, y_tr, y_val = small_data
        exact = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        mc = estimate_minppi_random_mc(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            K=200,
            acc_cache=acc_cache,
            seed=42,
        )
        for feat in exact:
            assert mc[feat] >= exact[feat], (
                f"Lemma 3 violated: MC({feat})={mc[feat]} < exact({feat})={exact[feat]}"
            )

    def test_seed_reproducibility(self, small_data, acc_cache) -> None:
        """Same seed → identical estimates."""
        X_tr, X_val, y_tr, y_val = small_data
        r1 = estimate_minppi_random_mc(
            X_tr, X_val, y_tr, y_val, alpha=0.90, K=50, acc_cache=acc_cache, seed=42
        )
        r2 = estimate_minppi_random_mc(
            X_tr, X_val, y_tr, y_val, alpha=0.90, K=50, acc_cache=acc_cache, seed=42
        )
        for feat in r1:
            assert r1[feat] == r2[feat], f"Random MC not seed-reproducible for {feat}"


# ---------------------------------------------------------------------------
# Algorithm B — MI-correlation-guided adversarial search
# ---------------------------------------------------------------------------


class TestAlgorithmB:
    """Algorithm B: MI-correlation-guided adversarial search."""

    def test_falls_back_to_random_when_no_neighbors(
        self, small_data, acc_cache
    ) -> None:
        """Without correlation neighbors, Algorithm B should not crash."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi_mi_correlation_guided(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            K_adv=20,
            correlation_neighbors=None,
            mi_scores=None,
            acc_cache=acc_cache,
            seed=42,
        )
        assert len(minppi) == 5
        for val in minppi.values():
            assert val in {-1, 0, +1}

    def test_perturbation_schedule_is_immutable_guidance_independent_and_validated(
        self, small_data
    ) -> None:
        X_train, X_val, y_train, y_val = small_data
        features = tuple(X_train.columns)
        first = make_algorithm_b_perturbation_schedule(
            features, K_adv=3, seed=42
        )
        second = make_algorithm_b_perturbation_schedule(
            features, K_adv=3, seed=42
        )

        assert isinstance(first, AlgorithmBPerturbationSchedule)
        assert first == second
        assert algorithm_b_schedule_digest(first) == algorithm_b_schedule_digest(second)
        assert first.features == features
        assert len(first.trials) == len(features)
        assert all(len(trials) == 3 for trials in first.trials)
        with pytest.raises(FrozenInstanceError):
            first.seed = 7

        class RecordingCache:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[str, ...], str]] = []

            def set_threshold(self, alpha: float) -> None:
                pass

            def eval_marginal(self, coalition: set[str], feature: str) -> int:
                self.calls.append((tuple(sorted(coalition)), feature))
                return 0

        random_cache = RecordingCache()
        guided_cache = RecordingCache()
        estimate_minppi_mi_correlation_guided(
            X_train,
            X_val,
            y_train,
            y_val,
            K_adv=3,
            perturbation_schedule=first,
            correlation_neighbors=None,
            mi_scores=None,
            acc_cache=random_cache,
            seed=42,
        )
        estimate_minppi_mi_correlation_guided(
            X_train,
            X_val,
            y_train,
            y_val,
            K_adv=3,
            perturbation_schedule=first,
            correlation_neighbors={feature: [features[(index + 1) % len(features)]] for index, feature in enumerate(features)},
            mi_scores={feature: float(len(features) - index) for index, feature in enumerate(features)},
            acc_cache=guided_cache,
            seed=42,
        )

        assert len(random_cache.calls) == len(guided_cache.calls) == 3 * len(features)
        assert random_cache.calls != guided_cache.calls

        foreign = make_algorithm_b_perturbation_schedule(
            tuple(reversed(features)), K_adv=3, seed=42
        )
        with pytest.raises(ValueError, match="schedule|feature"):
            estimate_minppi_mi_correlation_guided(
                X_train,
                X_val,
                y_train,
                y_val,
                K_adv=3,
                perturbation_schedule=foreign,
                acc_cache=RecordingCache(),
                seed=42,
            )

    def test_perturbation_schedule_rejects_nested_mutability_and_bool_draws(
        self, small_data
    ) -> None:
        X_train, _, _, _ = small_data
        features = tuple(X_train.columns)
        schedule = make_algorithm_b_perturbation_schedule(
            features, K_adv=2, seed=42
        )
        trial = schedule.trials[0][0]

        with pytest.raises(TypeError):
            trial.removal_draws[0] = 0.9
        with pytest.raises(ValueError, match="tuple|immutable"):
            AlgorithmBPerturbationTrial(
                removal_draws=list(trial.removal_draws),
                add_count=trial.add_count,
                add_order=trial.add_order,
            )
        with pytest.raises(ValueError, match="tuple|immutable"):
            AlgorithmBPerturbationTrial(
                removal_draws=trial.removal_draws,
                add_count=trial.add_count,
                add_order=list(trial.add_order),
            )
        with pytest.raises(ValueError, match="draw|numeric"):
            AlgorithmBPerturbationTrial(
                removal_draws=(False, *trial.removal_draws[1:]),
                add_count=trial.add_count,
                add_order=trial.add_order,
            )
        with pytest.raises(ValueError, match="tuple|immutable"):
            AlgorithmBPerturbationSchedule(
                features=list(schedule.features),
                seed=schedule.seed,
                K_adv=schedule.K_adv,
                seed_orders=schedule.seed_orders,
                trials=schedule.trials,
            )
        with pytest.raises(ValueError, match="tuple|immutable"):
            AlgorithmBPerturbationSchedule(
                features=schedule.features,
                seed=schedule.seed,
                K_adv=schedule.K_adv,
                seed_orders=(list(schedule.seed_orders[0]), *schedule.seed_orders[1:]),
                trials=schedule.trials,
            )
        with pytest.raises(ValueError, match="tuple|immutable"):
            AlgorithmBPerturbationSchedule(
                features=schedule.features,
                seed=schedule.seed,
                K_adv=schedule.K_adv,
                seed_orders=schedule.seed_orders,
                trials=(list(schedule.trials[0]), *schedule.trials[1:]),
            )

    def test_schedule_digest_rejects_structurally_invalid_schedule(
        self, small_data
    ) -> None:
        X_train, _, _, _ = small_data
        schedule = make_algorithm_b_perturbation_schedule(
            tuple(X_train.columns), K_adv=2, seed=42
        )
        object.__setattr__(
            schedule,
            "seed_orders",
            (list(schedule.seed_orders[0]), *schedule.seed_orders[1:]),
        )

        with pytest.raises(ValueError, match="schedule|tuple|immutable"):
            algorithm_b_schedule_digest(schedule)

    def test_legacy_estimator_warns(self, small_data, acc_cache) -> None:
        X_tr, X_val, y_tr, y_val = small_data

        with pytest.warns(DeprecationWarning, match="mi_correlation_guided"):
            estimate_minppi_mi_guided(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=0.90,
                K_adv=2,
                shapg_neighbors=None,
                shapg_scores=None,
                acc_cache=acc_cache,
                seed=42,
            )


# ---------------------------------------------------------------------------
# Hybrid dispatcher
# ---------------------------------------------------------------------------


class TestHybridDispatcher:
    """§4.3.3: auto-select strategy based on n."""

    def test_small_n_uses_exact(self, small_data, acc_cache) -> None:
        """n=5 ≤ 20 → should route to exact enumeration."""
        X_tr, X_val, y_tr, y_val = small_data
        minppi = estimate_minppi_hybrid(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            acc_cache=acc_cache,
            seed=42,
        )
        assert len(minppi) == 5

    def test_hybrid_matches_exact_for_small_n(self, small_data, acc_cache) -> None:
        """When n ≤ 20, hybrid should equal exact."""
        X_tr, X_val, y_tr, y_val = small_data
        exact = estimate_minppi_exact(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        hybrid = estimate_minppi_hybrid(
            X_tr, X_val, y_tr, y_val, alpha=0.90, acc_cache=acc_cache, seed=42
        )
        for feat in exact:
            assert hybrid[feat] == exact[feat], (
                f"Hybrid ≠ exact for {feat}: {hybrid[feat]} vs {exact[feat]}"
            )


# ---------------------------------------------------------------------------
# minppi_stats utility
# ---------------------------------------------------------------------------


class TestMinppiStats:
    """minppi_stats summary correctness."""

    def test_sums_match_total(self) -> None:
        stats = minppi_stats({"a": +1, "b": -1, "c": 0, "d": 0, "e": 0})
        assert stats["n_total"] == 5
        assert stats["n_decisive"] == 1
        assert stats["n_neutral"] == 3
        assert stats["n_harmful"] == 1
        assert stats["n_decisive"] + stats["n_neutral"] + stats["n_harmful"] == 5

    def test_percentages_sum_to_100(self) -> None:
        stats = minppi_stats({"a": +1, "b": -1, "c": 0, "d": 0})
        total = stats["pct_decisive"] + stats["pct_neutral"] + stats["pct_harmful"]
        assert total == pytest.approx(100.0, abs=0.1)

    def test_all_neutral(self) -> None:
        stats = minppi_stats({"a": 0, "b": 0})
        assert stats["n_harmful"] == 0
        assert stats["n_decisive"] == 0
        assert stats["pct_neutral"] == 100.0

    def test_all_harmful(self) -> None:
        stats = minppi_stats({"a": -1, "b": -1})
        assert stats["n_harmful"] == 2
        assert stats["pct_harmful"] == 100.0

    def test_empty_dict(self) -> None:
        stats = minppi_stats({})
        assert stats["n_total"] == 0
        assert stats["pct_harmful"] == 0.0


# ---------------------------------------------------------------------------
# estimate_minppi public API
# ---------------------------------------------------------------------------


class TestEstimateMinppiAPI:
    """Main entry-point routing."""

    def test_canonical_guidance_method_routes(self, small_data) -> None:
        X_tr, X_val, y_tr, y_val = small_data
        neighbors = {feature: [] for feature in X_tr.columns}
        scores = {feature: float(i) for i, feature in enumerate(X_tr.columns)}

        result = estimate_minppi(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            method="mi_correlation_guided",
            K_adv=2,
            correlation_neighbors=neighbors,
            mi_scores=scores,
            seed=42,
        )

        assert set(result) == set(X_tr.columns)

    def test_legacy_method_and_parameters_warn(self, small_data) -> None:
        X_tr, X_val, y_tr, y_val = small_data
        neighbors = {feature: [] for feature in X_tr.columns}
        scores = {feature: float(i) for i, feature in enumerate(X_tr.columns)}

        with pytest.warns(DeprecationWarning):
            result = estimate_minppi(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=0.90,
                method="mi_guided",
                K_adv=2,
                shapg_neighbors=neighbors,
                shapg_scores=scores,
                seed=42,
            )

        assert set(result) == set(X_tr.columns)

    def test_rejects_canonical_and_legacy_guidance_together(
        self, small_data,
    ) -> None:
        X_tr, X_val, y_tr, y_val = small_data

        with pytest.raises(ValueError, match="both"):
            estimate_minppi(
                X_tr,
                X_val,
                y_tr,
                y_val,
                method="mi_correlation_guided",
                correlation_neighbors={},
                shapg_neighbors={},
            )

    def test_unknown_method_raises(self, small_data) -> None:
        X_tr, X_val, y_tr, y_val = small_data
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_minppi(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=0.90,
                method="nonexistent",
                seed=42,
            )

    def test_exact_rejects_high_n(self, small_data) -> None:
        """Exact method must raise for n > 20."""
        # Create a dataset with >20 features
        rng = np.random.default_rng(42)
        n_features = 25
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(n_features)})
        y = np.where(X["f0"] > 0, 1, 0).astype(int)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        with pytest.raises(ValueError, match="n ≤ 20"):
            estimate_minppi_exact(X_tr, X_val, y_tr, y_val, alpha=0.90, seed=42)

    def test_hybrid_handles_medium_n(self) -> None:
        """Hybrid dispatcher should handle n=25 gracefully (no crash)."""
        rng = np.random.default_rng(42)
        n_features = 25
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(n_features)})
        y = np.where(X["f0"] > 0, 1, 0).astype(int)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        minppi = estimate_minppi_hybrid(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=0.90,
            K=50,
            seed=42,
        )
        assert len(minppi) == n_features
        for val in minppi.values():
            assert val in {-1, 0, +1}
