import numpy as np

from experiments.test.piv1_racing import replay_vote_prefix, search_vote_prefix
from minppi_mofs.piv1_search import search_subsets


def test_actual_early_stop_matches_original_proposals_and_validation_replay():
    rng = np.random.default_rng(8)
    width = 100
    shap, gain = rng.uniform(size=(2, width))
    votes = np.r_[-np.ones(60), np.ones(40)]
    consistent = np.arange(width) < 50
    full = search_subsets(votes, shap, gain, lambda _: 0.8, recipe="vote")
    replay = replay_vote_prefix(full["trials"], votes, consistent_negative=consistent)
    calls = []

    def score(indices):
        calls.append(indices)
        assert len(calls) <= 18
        return .8, .6, .7

    online = search_vote_prefix(votes, shap, gain, score, consistent_negative=consistent)
    assert online["earliest_stop"] == 18 and online["patience"] == 5
    assert online["evaluated"] == 18 and online["early_stopped"]
    assert calls == [trial["indices"] for trial in full["trials"][:18]]
    assert online["selected"] == replay["selected"] and online["utility"] == replay["utility"]


def test_recent_improvement_resets_patience_and_no_negatives_disable_early_stop():
    width = 100
    votes = -np.ones(width)
    consistent = np.ones(width, dtype=bool)
    # Earliest check is 12. Improvement at observation 11 protects four more misses.
    trials = [dict(indices=tuple(range(i)), utility=float(i >= 11), source="fixture")
              for i in range(1, 25)]
    raced = replay_vote_prefix(trials, votes, consistent_negative=consistent)
    assert raced["earliest_stop"] == 12 and raced["evaluated"] == 16
    assert raced["retained"] == 11
    unaffected = replay_vote_prefix(trials, votes, consistent_negative=np.zeros(width, dtype=bool))
    assert unaffected["evaluated"] == 24 and not unaffected["early_stopped"]


def test_refinement_protection_can_retain_improvements_after_first_eligible_stop():
    votes = -np.ones(100)
    consistent = np.ones(100, dtype=bool)
    trials = [dict(indices=tuple(range(i)), utility=float(i >= 17), source="fixture")
              for i in range(1, 25)]
    first = replay_vote_prefix(trials, votes, consistent_negative=consistent)
    guarded = replay_vote_prefix(trials, votes, consistent_negative=consistent, protect_refinement=True)
    assert first["evaluated"] == 12 and first["utility"] == 0.
    assert guarded["earliest_stop"] == 18 and guarded["evaluated"] == 22
    assert guarded["utility"] == 1. and guarded["retained"] == 17
