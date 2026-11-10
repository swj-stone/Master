"""A skipped endpoint may preserve a sampled minimum without revealing a marginal."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.minimum_sampling import advance_minimum, sample_random_round, update_sample_minimum
from minppi_mofs.coalition_cache import CoalitionUtilityView
from minppi_mofs.min_ppi import (
    estimate_minppi_hybrid,
    estimate_minppi_mi_correlation_guided,
    estimate_minppi_random_mc,
    make_shared_coalition_plan,
)
from minppi_mofs.voting_game import AccFullCache, UtilityView


def test_balanced_round_preserves_frozen_pilot_rng_sequence():
    rng = np.random.default_rng(42)
    assert list(sample_random_round(4, rng)) == [(3, 2, 10), (2, 11, 15), (1, 12, 14), (0, 12, 13)]
    assert list(sample_random_round(4, rng)) == [(0, 2, 3), (3, 7, 15), (1, 5, 7), (2, 0, 4)]
    assert list(sample_random_round(1, rng)) == [(0, 0, 1)]


@pytest.mark.parametrize("current,before,after", product((-1, 0, 1), (False, True), (False, True)))
def test_update_compares_exact_minimum_and_only_queries_necessary_endpoint(current, before, after):
    calls = []

    def read_after():
        calls.append(after)
        return after

    assert update_sample_minimum(current, before, read_after) == min(current, int(after) - int(before))
    assert calls == ([after] if (current, before) in {(0, True), (1, False), (1, True)} else [])


@pytest.mark.parametrize("view_class", [AccFullCache, UtilityView, CoalitionUtilityView])
def test_fixed_schedule_keeps_each_prefix_minimum_with_only_required_endpoint_reads(view_class):
    # Bypass model training and substitute a deterministic threshold game.
    view = view_class.__new__(view_class)
    view._cache = SimpleNamespace(feature_order=("i", "a", "b", "c", "d"))
    queries = []
    winners = {frozenset(("i", "a")), frozenset(("i", "c")), frozenset(("d",))}

    def is_winning(coalition):
        queries.append(frozenset(coalition))
        return frozenset(coalition) in winners

    view.is_winning = is_winning
    best = 1
    actual = []
    for context in ("a", "b", "c", "d", "a"):
        best = view.update_minimum(best, {context}, "i")
        actual.append(best)

    assert actual == [1, 0, 0, -1, -1]
    assert queries == list(map(frozenset, (("a",), ("a", "i"), ("b",), ("b", "i"),
                                           ("c",), ("d",), ("d", "i"))))


def test_advance_honors_view_tracing_and_supports_legacy_marginal_views():
    calls = []

    class LegacyGame:
        def eval_marginal(self, coalition, feature):
            calls.append((frozenset(coalition), feature))
            return 1

    class TracedGame(LegacyGame):
        def update_minimum(self, current, coalition, feature):
            calls.append((current, frozenset(coalition), feature))
            return current

    for view in (LegacyGame(), TracedGame()):
        assert advance_minimum(view, 0, set(), "i") == 0
        assert advance_minimum(view, -1, set(), "i") == -1
    assert calls == [(frozenset(), "i"), (0, frozenset(), "i")]


class LosingGame(AccFullCache):
    """Every coalition loses; each feature needs just one augmented endpoint."""

    def __init__(self):
        self.queries = []
        self.acc_full = 0

    def is_winning(self, coalition):
        self.queries.append(frozenset(coalition))
        return False


@pytest.mark.parametrize("estimator,options", [
    (estimate_minppi_random_mc, {"K": 4}),
    (estimate_minppi_mi_correlation_guided, {"K_adv": 4}),
    (estimate_minppi_hybrid, {"K": 4}),
])
def test_sampled_public_estimators_use_lazy_endpoints(estimator, options):
    frame = pd.DataFrame(columns=[f"f{n}" for n in range(21)])
    game = LosingGame()
    scores = estimator(frame, frame, [], [], acc_cache=game, **options)
    assert scores == dict.fromkeys(frame.columns, 0)
    assert len(game.queries) == len(frame.columns) * 5


def test_shared_mode_keeps_all_endpoints_for_explicit_metric_pair_plans():
    frame = pd.DataFrame(columns=["a", "b", "c"])
    plan = make_shared_coalition_plan(frame.columns, K=4, per_coalition=2, seed=7)
    frozen = LosingGame()
    adaptive = LosingGame()
    arguments = dict(K=4, per_coalition=2, seed=7, shared_coalitions=True)
    frozen_scores = estimate_minppi_random_mc(
        frame, frame, [], [], acc_cache=frozen, shared_plan=plan, **arguments,
    )
    adaptive_scores = estimate_minppi_random_mc(frame, frame, [], [], acc_cache=adaptive, **arguments)

    assert frozen_scores == adaptive_scores == dict.fromkeys(frame.columns, 0)
    assert len(frozen.queries) == sum(1 + len(features) for _, features in plan)
    assert len(adaptive.queries) == sum(bool(features) for _, features in plan) + len(frame.columns)


def test_public_default_is_finite_sampling_even_on_tiny_games(monkeypatch):
    import minppi_mofs
    import minppi_mofs.min_ppi as implementation

    class RareWitness(LosingGame):
        def is_winning(self, coalition):
            self.queries.append(frozenset(coalition))
            return coalition in ({"a"}, {"a", "b", "c", "d"})

    game = RareWitness()
    monkeypatch.setattr(implementation, "make_acc_full_cache", lambda *args, **kwargs: game)
    frame = pd.DataFrame(columns=list("abcd"))
    inputs = (frame, frame, [], [])
    # Seed 42 checks {c,d} before adding b. The witness at {a} is missed.
    assert minppi_mofs.estimate_minppi(*inputs, K=1, seed=42)["b"] == 0
    assert len(game.queries) == 8
    assert minppi_mofs.estimate_minppi_sampled(*inputs, K=1, seed=42)["b"] == 0
    assert minppi_mofs.estimate_minppi(*inputs, method="exact")["b"] == -1
    assert minppi_mofs.estimate_minppi(*inputs, method="hybrid")["b"] == -1


def test_public_round_budget_and_explicit_random_mc_budget_keep_their_defaults(monkeypatch):
    import minppi_mofs.min_ppi as implementation

    game = LosingGame()
    monkeypatch.setattr(implementation, "make_acc_full_cache", lambda *args, **kwargs: game)
    frame = pd.DataFrame(columns=[f"f{i}" for i in range(11)])
    inputs = (frame, frame, [], [])
    implementation.estimate_minppi(*inputs)
    # Every draw reads a base; the first draw for each feature also reads its augmentation.
    assert len(game.queries) == 11 * (5 + 1)
    game.queries.clear()
    implementation.estimate_minppi(*inputs, method="random_mc")
    assert len(game.queries) == 11 * (200 + 1)


@pytest.mark.parametrize("options", [{"K": 0}, {"K": True}, {"shared_coalitions": True},
                                     {"K_adv": 2}, {"correlation_neighbors": {}}, {"mi_scores": {}}])
def test_sampled_dispatch_rejects_invalid_budget_or_unused_options_before_training(monkeypatch, options):
    import minppi_mofs.min_ppi as implementation

    frame = pd.DataFrame(columns=["a"])

    def reject_fit(*args, **kwargs):
        pytest.fail("model initialized before option validation")

    monkeypatch.setattr(implementation, "make_acc_full_cache", reject_fit)
    with pytest.raises(ValueError):
        implementation.estimate_minppi(frame, frame, [], [], method="sampled", **options)


@pytest.mark.parametrize("options", [
    {"correlation_neighbors": {"f0": ["f1"]}, "K_adv": 2},
    {"K_adv": 2}, {"mi_scores": {}}, {"shared_coalitions": True}, {"shared_coalitions": False},
])
def test_omitted_method_with_legacy_options_reproduces_hybrid_proposals(monkeypatch, options):
    import minppi_mofs.min_ppi as implementation

    game = LosingGame()
    monkeypatch.setattr(implementation, "make_acc_full_cache", lambda *args, **kwargs: game)
    frame = pd.DataFrame(columns=[f"f{i}" for i in range(21)])
    inputs = (frame, frame, [], [])
    expected = implementation.estimate_minppi(*inputs, method="hybrid", K=2, **options)
    historical_queries = game.queries.copy()
    game.queries.clear()
    assert implementation.estimate_minppi(*inputs, K=2, **options) == expected
    assert game.queries == historical_queries
