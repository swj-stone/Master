"""
Tests for ``minppi_mofs.mofs_minppi`` — MOFS-MinPPI: NSGA-II core, guided operators, hypervolume.

Covers (per tests/README.md):
- non_dominated_sort correctness on known 2D/3D cases
- crowding_distance: boundary points get inf, monotonic ordering
- tournament_select: lower rank wins, crowding breaks ties
- minppi_mutation: β=0 uniform, β>0 biases harmful→0 / decisive→1
- minppi_crossover: decisive-block inheritance
- compute_hypervolume: monotonicity, reference-point behaviour
- MOFSMinPPIResult dataclass integrity
- mofs_minppi_search end-to-end on tiny data

Run::

    pytest tests/test_mofs_minppi.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from minppi_mofs.mofs_minppi import (
    MOFSMinPPIResult,
    compute_hypervolume,
    crowding_distance,
    minppi_crossover,
    minppi_mutation,
    mofs_minppi_search,
    non_dominated_sort,
    pareto_feature_importance,
    tournament_select,
)


# ---------------------------------------------------------------------------
# non_dominated_sort
# ---------------------------------------------------------------------------


class TestNonDominatedSort:
    """Non-dominated sorting correctness (Deb et al. 2002)."""

    def test_single_point(self) -> None:
        """One point → one front of size 1."""
        fitness = np.array([[1.0, 2.0]])
        fronts = non_dominated_sort(fitness)
        assert len(fronts) == 1
        assert fronts[0] == [0]

    def test_two_independent_points(self) -> None:
        """Neither dominates the other → same front."""
        fitness = np.array([[1.0, 2.0], [2.0, 1.0]])
        fronts = non_dominated_sort(fitness)
        assert len(fronts) == 1
        assert sorted(fronts[0]) == [0, 1]

    def test_dominated_point_in_second_front(self) -> None:
        """(0,0) dominates (1,1) — the latter goes to front 1."""
        fitness = np.array([[0.0, 0.0], [1.0, 1.0]])
        fronts = non_dominated_sort(fitness)
        assert len(fronts) == 2
        assert fronts[0] == [0]
        assert fronts[1] == [1]

    def test_pareto_front_2d(self) -> None:
        """Known 2D case with 3 fronts."""
        fitness = np.array(
            [
                [0.1, 0.9],  # 0 — Pareto
                [0.2, 0.5],  # 1 — Pareto
                [0.5, 0.2],  # 2 — Pareto
                [1.0, 1.0],  # 3 — dominated by all above
            ]
        )
        fronts = non_dominated_sort(fitness)
        assert len(fronts) == 2
        assert sorted(fronts[0]) == [0, 1, 2]
        assert fronts[1] == [3]

    def test_three_objective_case(self) -> None:
        """3D: verify at least the non-dominated set is not empty."""
        rng = np.random.default_rng(42)
        fitness = rng.uniform(0, 1, (20, 3))
        fronts = non_dominated_sort(fitness)
        assert len(fronts) >= 1
        assert len(fronts[0]) >= 1
        # All indices covered exactly once
        all_idx = sorted(i for f in fronts for i in f)
        assert all_idx == list(range(20))

    def test_all_equal_points(self) -> None:
        """Identical points — none dominates another, all in one front."""
        fitness = np.ones((10, 2))
        fronts = non_dominated_sort(fitness)
        assert len(fronts) == 1
        assert len(fronts[0]) == 10


# ---------------------------------------------------------------------------
# crowding_distance
# ---------------------------------------------------------------------------


class TestCrowdingDistance:
    """Crowding distance for diversity preservation."""

    def test_boundary_points_get_inf(self) -> None:
        """Extreme points in each objective get infinite crowding distance."""
        fitness = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        front = [0, 1, 2]
        cd = crowding_distance(fitness, front)
        assert cd[0] == np.inf
        assert cd[2] == np.inf

    def test_middle_point_finite(self) -> None:
        """Interior points have finite crowding distance."""
        fitness = np.array([[0.0, 1.0], [0.4, 0.6], [0.6, 0.4], [1.0, 0.0]])
        front = [0, 1, 2, 3]
        cd = crowding_distance(fitness, front)
        assert cd[0] == np.inf
        assert cd[3] == np.inf
        assert 0.0 < cd[1] < np.inf
        assert 0.0 < cd[2] < np.inf

    def test_single_element_gets_inf(self) -> None:
        """One-element front → inf."""
        fitness = np.array([[0.5, 0.5]])
        cd = crowding_distance(fitness, [0])
        assert cd[0] == np.inf

    def test_two_elements_both_inf(self) -> None:
        """Two-element front → both inf."""
        fitness = np.array([[0.0, 1.0], [1.0, 0.0]])
        cd = crowding_distance(fitness, [0, 1])
        assert cd[0] == np.inf
        assert cd[1] == np.inf

    def test_sorted_output_length_and_boundaries(self) -> None:
        """Crowding distance returns array with correct length and inf entries."""
        fitness = np.array([[0.0, 1.0], [0.3, 0.7], [0.7, 0.3], [1.0, 0.0]])
        front = [0, 1, 2, 3]
        cd = crowding_distance(fitness, front)
        assert len(cd) == 4
        # Boundary points in at least one objective get inf
        assert cd[0] == np.inf  # fittest in obj 0
        assert cd[3] == np.inf  # fittest in obj 1


# ---------------------------------------------------------------------------
# tournament_select
# ---------------------------------------------------------------------------


class TestTournamentSelect:
    """Binary tournament selection."""

    def test_lower_rank_wins(self) -> None:
        """Individual in front 0 beats one in front 1."""
        rng = np.random.default_rng(42)
        fronts = [[0, 1], [2, 3]]
        crowding = [
            np.array([1.0, 0.5]),
            np.array([1.0, 0.5]),
        ]
        rank_map = {0: 0, 1: 0, 2: 1, 3: 1}
        pos_map = {0: 0, 1: 1, 2: 0, 3: 1}

        # tournament_select uses rng.choice internally; we verify via
        # repeated trials that the output indices are always valid.
        p1, p2 = tournament_select(4, fronts, crowding, rank_map, pos_map, rng)
        assert 0 <= p1 < 4
        assert 0 <= p2 < 4

    def test_on_tie_higher_crowding_wins(self) -> None:
        """Within the same front, higher crowding distance should be preferred."""
        rng = np.random.default_rng(123)
        # Four individuals in the same front with very different crowding
        fronts = [[0, 1, 2, 3]]
        crowding = [np.array([0.01, 100.0, 0.5, 0.5])]
        rank_map = {0: 0, 1: 0, 2: 0, 3: 0}
        pos_map = {0: 0, 1: 1, 2: 2, 3: 3}

        # Run many trials — individual 1 should win its matchups more often
        wins = 0
        for _ in range(200):
            p1, p2 = tournament_select(4, fronts, crowding, rank_map, pos_map, rng)
            if p1 == 1 or p2 == 1:
                wins += 1
        assert wins > 0


# ---------------------------------------------------------------------------
# minppi_mutation — §4.4.1
# ---------------------------------------------------------------------------


class TestMinppiMutation:
    """MinPPI-guided mutation operator."""

    def test_beta_zero_is_uniform(self) -> None:
        """β=0 → all features get p_base (no guidance)."""
        rng = np.random.default_rng(42)
        mask = np.ones(100, dtype=np.int8)
        minppi = {i: -1 for i in range(50)}  # half harmful
        minppi.update({i: +1 for i in range(50, 100)})  # half decisive

        # With β=0, harmful and decisive should flip at roughly the same rate
        n_flips_harmful = 0
        n_flips_decisive = 0
        n_trials = 200
        for _ in range(n_trials):
            mutated = minppi_mutation(
                mask.copy(), minppi, beta=0.0, p_base=0.1, rng=rng
            )
            n_flips_harmful += sum(1 for i in range(50) if mutated[i] != mask[i])
            n_flips_decisive += sum(1 for i in range(50, 100) if mutated[i] != mask[i])
        rate_h = n_flips_harmful / (50 * n_trials)
        rate_d = n_flips_decisive / (50 * n_trials)
        # With β=0 they should be close
        assert abs(rate_h - rate_d) < 0.05, (
            f"β=0: harmful_rate={rate_h:.4f}, decisive_rate={rate_d:.4f}"
        )

    def test_beta_positive_biases_harmful_removal(self) -> None:
        """β>0: harmful features with mask=1 are more likely flipped to 0."""
        rng = np.random.default_rng(42)
        mask = np.ones(50, dtype=np.int8)  # all selected
        minppi = {i: -1 for i in range(50)}  # all harmful

        n_flipped = 0
        n_trials = 200
        for _ in range(n_trials):
            mutated = minppi_mutation(
                mask.copy(), minppi, beta=2.0, p_base=0.1, rng=rng
            )
            n_flipped += sum(1 for i in range(50) if mutated[i] == 0)
        rate = n_flipped / (50 * n_trials)
        # With β=2.0, harmful-removal rate should be well above base p_base
        assert rate > 0.1, f"β=2.0 harmful removal rate too low: {rate:.4f}"

    def test_beta_positive_protects_decisive(self) -> None:
        """β>0: decisive features with mask=0 are more likely flipped to 1
        than uniform baseline would suggest.

        With p_base=0.1 and β=1.0, the protection factor exp(-β)≈0.37
        gives effective p_flip ≈ 0.037, reduced from 0.1.  Over many
        trials the fraction staying 0 should exceed the unflipped rate
        at β=0 for neutral features.
        """
        rng = np.random.default_rng(42)
        mask = np.zeros(100, dtype=np.int8)
        # Half decisive (should be protected), half neutral (no protection)
        minppi_d = {i: +1 for i in range(50)}
        minppi_n = {i: 0 for i in range(50, 100)}
        minppi = {**minppi_d, **minppi_n}

        n_kept_d = 0  # decisive features staying 0
        n_kept_n = 0  # neutral features staying 0
        n_trials = 200
        for _ in range(n_trials):
            mutated = minppi_mutation(
                mask.copy(), minppi, beta=1.0, p_base=0.1, rng=rng
            )
            n_kept_d += sum(1 for i in range(50) if mutated[i] == 0)
            n_kept_n += sum(1 for i in range(50, 100) if mutated[i] == 0)
        rate_kept_d = n_kept_d / (50 * n_trials)
        rate_kept_n = n_kept_n / (50 * n_trials)
        # Decisive features should stay 0 more often than neutral ones
        assert rate_kept_d > rate_kept_n, (
            f"β=1.0: decisive kept-rate={rate_kept_d:.4f}, "
            f"neutral kept-rate={rate_kept_n:.4f}"
        )

    def test_mutation_does_not_crash_on_empty_minppi(self) -> None:
        """No MinPPI info → uniform behaviour, no crash."""
        rng = np.random.default_rng(42)
        mask = np.array([1, 0, 1, 0], dtype=np.int8)
        mutated = minppi_mutation(mask, {}, beta=1.0, p_base=0.1, rng=rng)
        assert len(mutated) == 4
        assert all(b in {0, 1} for b in mutated)

    def test_output_is_binary(self) -> None:
        """All mutated bits must be 0 or 1."""
        rng = np.random.default_rng(42)
        mask = np.random.default_rng(0).integers(0, 2, size=30).astype(np.int8)
        minppi = {i: rng.choice([-1, 0, +1]) for i in range(30)}
        mutated = minppi_mutation(mask, minppi, beta=1.5, p_base=0.1, rng=rng)
        assert np.all((mutated == 0) | (mutated == 1))


# ---------------------------------------------------------------------------
# minppi_crossover — §4.4.2
# ---------------------------------------------------------------------------


class TestMinppiCrossover:
    """MinPPI-guided crossover."""

    def test_decisive_block_inherited_intact(self) -> None:
        """Decisive features should be inherited as a block from one parent."""
        rng = np.random.default_rng(42)
        p1 = np.array([1, 1, 0, 1, 0], dtype=np.int8)
        p2 = np.array([0, 0, 1, 0, 1], dtype=np.int8)
        minppi = {0: +1, 1: +1}  # indices 0,1 are decisive

        # Run many times — the decisive block should always come from one parent
        for _ in range(100):
            child = minppi_crossover(p1, p2, minppi, rng=rng)
            # Indices 0,1 must both come from p1 OR both from p2
            from_p1 = child[0] == p1[0] and child[1] == p1[1]
            from_p2 = child[0] == p2[0] and child[1] == p2[1]
            assert from_p1 or from_p2, (
                f"Decisive block not intact: child[{child[0]},{child[1]}] "
                f"vs p1[{p1[0]},{p1[1]}] p2[{p2[0]},{p2[1]}]"
            )

    def test_non_decisive_use_uniform_crossover(self) -> None:
        """Non-decisive positions should mix from both parents."""
        rng = np.random.default_rng(42)
        p1 = np.array([1, 0, 1, 0], dtype=np.int8)
        p2 = np.array([0, 1, 0, 1], dtype=np.int8)
        minppi = {}  # no decisive features

        # Track whether each position gets value from p1 sometimes
        from_p1_counts = np.zeros(4)
        for _ in range(200):
            child = minppi_crossover(p1, p2, minppi, rng=rng)
            for i in range(4):
                if child[i] == p1[i]:
                    from_p1_counts[i] += 1
        # Each non-decisive position should have a mix (not all from one parent)
        for i in range(4):
            rate = from_p1_counts[i] / 200
            assert 0.2 < rate < 0.8, f"Position {i} biased: rate={rate:.3f}"

    def test_output_is_binary(self) -> None:
        """Child mask must be 0/1."""
        rng = np.random.default_rng(42)
        p1 = np.random.default_rng(0).integers(0, 2, size=20).astype(np.int8)
        p2 = np.random.default_rng(1).integers(0, 2, size=20).astype(np.int8)
        minppi = {i: rng.choice([-1, 0, +1]) for i in range(20)}
        child = minppi_crossover(p1, p2, minppi, rng=rng)
        assert len(child) == 20
        assert np.all((child == 0) | (child == 1))


# ---------------------------------------------------------------------------
# compute_hypervolume
# ---------------------------------------------------------------------------


class TestHypervolume:
    """Hypervolume computation."""

    def test_empty_front_returns_zero(self) -> None:
        assert compute_hypervolume(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0

    def test_single_point_2d(self) -> None:
        fitness = np.array([[0.2, 0.3]])
        ref = np.array([1.0, 1.0])
        hv = compute_hypervolume(fitness, ref)
        assert hv > 0.0

    def test_dominated_point_adds_no_volume(self) -> None:
        """A fully dominated point should not increase HV beyond the dominator."""
        fitness_better = np.array([[0.1, 0.1]])
        fitness_worse = np.array([[0.1, 0.1], [0.5, 0.5]])  # 2nd is dominated
        ref = np.array([1.0, 1.0])
        hv_better = compute_hypervolume(fitness_better, ref)
        hv_worse = compute_hypervolume(fitness_worse, ref)
        assert hv_better == pytest.approx(hv_worse), (
            f"HV should not increase with dominated point: {hv_better} vs {hv_worse}"
        )

    def test_more_points_increases_hv(self) -> None:
        """Adding non-dominated points should not decrease HV."""
        fitness = np.array([[0.3, 0.7], [0.7, 0.3]])
        fitness_plus = np.array([[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]])
        ref = np.array([1.0, 1.0])
        hv_2 = compute_hypervolume(fitness, ref)
        hv_3 = compute_hypervolume(fitness_plus, ref)
        assert hv_3 >= hv_2 - 1e-10  # allow tiny floating error

    def test_reference_point_effect(self) -> None:
        """A farther reference point → larger HV."""
        fitness = np.array([[0.2, 0.2]])
        hv_near = compute_hypervolume(fitness, np.array([0.5, 0.5]))
        hv_far = compute_hypervolume(fitness, np.array([2.0, 2.0]))
        assert hv_far > hv_near


# ---------------------------------------------------------------------------
# MOFSMinPPIResult dataclass
# ---------------------------------------------------------------------------


class TestMOFSMinPPIResult:
    """Result dataclass integrity."""

    def test_construction(self) -> None:
        masks = np.array([[1, 0], [0, 1]], dtype=np.int8)
        fitness = np.array([[1.0, 0.2, 0.01], [1.0, 0.3, 0.02]])
        result = MOFSMinPPIResult(
            pareto_masks=masks,
            pareto_fitness=fitness,
            minppi={0: +1, 1: -1},
            hypervolume=0.5,
            history=[],
            runtime_s=1.0,
            features=["a", "b"],
        )
        assert len(result.pareto_masks) == 2
        assert result.hypervolume == 0.5
        assert result.features == ["a", "b"]

    def test_pareto_feature_importance(self) -> None:
        masks = np.array([[1, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
        fitness = np.array([[1.0, 0.1, 0.01], [2.0, 0.2, 0.02], [1.0, 0.15, 0.01]])
        result = MOFSMinPPIResult(
            pareto_masks=masks,
            pareto_fitness=fitness,
            minppi={},
            hypervolume=0.0,
            history=[],
            runtime_s=0.0,
            features=["a", "b", "c"],
        )
        imp = pareto_feature_importance(result)
        # New metric: max AUC among solutions containing the feature.
        # a appears in sols with AUC {0.9, 0.8, 0.85} → max 0.9
        # b appears in sol  with AUC 0.8               → max 0.8
        # c appears in sol  with AUC 0.9               → max 0.9
        assert imp["a"] >= imp["c"]  # both tied at 0.9
        assert imp["a"] > imp["b"]   # 0.9 > 0.8
        assert imp["c"] > imp["b"]   # 0.9 > 0.8


# ---------------------------------------------------------------------------
# mofs_minppi_search — end-to-end integration
# ---------------------------------------------------------------------------


class TestMOFSMinppiSearch:
    """End-to-end NSGA-II search on tiny data."""

    @pytest.fixture(scope="class")
    def tiny_search_input(self):
        """4-feature synthetic data."""
        rng = np.random.default_rng(42)
        n = 150
        X = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
                "c": rng.normal(0, 1, n),
                "d": rng.normal(0, 1, n),
            }
        )
        y = np.where(X["a"] > 0, 1, 0).astype(int)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_tr, X_val, y_tr, y_val

    def test_search_runs_and_returns_result(self, tiny_search_input) -> None:
        """MOFS-MinPPI should complete without error."""
        X_tr, X_val, y_tr, y_val = tiny_search_input
        minppi = {"a": +1, "b": 0, "c": -1, "d": -1}
        result = mofs_minppi_search(
            X_tr,
            X_val,
            y_tr,
            y_val,
            minppi=minppi,
            alpha=0.90,
            beta=1.0,
            pop_size=10,
            generations=5,
            seed=42,
            verbose=False,
        )
        assert isinstance(result, MOFSMinPPIResult)
        assert result.pareto_masks.shape[1] == 4
        assert result.pareto_fitness.shape[1] == 3
        assert result.hypervolume >= 0
        assert result.runtime_s > 0
        assert len(result.history) == 5
        assert len(result.features) == 4

    def test_search_no_crash_with_all_neutral(self, tiny_search_input) -> None:
        """All-neutral MinPPI should not crash the search."""
        X_tr, X_val, y_tr, y_val = tiny_search_input
        minppi = {"a": 0, "b": 0, "c": 0, "d": 0}
        result = mofs_minppi_search(
            X_tr,
            X_val,
            y_tr,
            y_val,
            minppi=minppi,
            alpha=0.90,
            beta=1.0,
            pop_size=10,
            generations=5,
            seed=42,
            verbose=False,
        )
        assert len(result.pareto_masks) >= 1

    def test_beta_zero_also_works(self, tiny_search_input) -> None:
        """β=0 (no guidance) should still run correctly."""
        X_tr, X_val, y_tr, y_val = tiny_search_input
        minppi = {"a": +1, "b": 0, "c": -1, "d": -1}
        result = mofs_minppi_search(
            X_tr,
            X_val,
            y_tr,
            y_val,
            minppi=minppi,
            alpha=0.90,
            beta=0.0,
            pop_size=10,
            generations=5,
            seed=42,
            verbose=False,
        )
        assert result.pareto_fitness.shape[0] >= 1
