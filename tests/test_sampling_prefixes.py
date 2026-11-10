"""A single K=3 experiment must reproduce the three cold sampling prefixes."""

import pytest

from experiments import thesis_sampling_worker as worker


def run_sampler(k, callback_factory=None):
    features = ("a", "b", "c", "d", "e", "f", "g")
    queries, fits, cache = [], [], {}
    clock = [0.]

    def evaluate(mask):
        queries.append(mask)
        if mask not in cache:
            fits.append(mask)
            cache[mask] = float((mask * 3 + mask.bit_count()) % 7 >= 2)
            clock[0] += 2.
        return cache[mask]

    kwargs = {}
    if callback_factory is not None:
        kwargs["on_prefix"] = callback_factory(queries, fits, clock)
    result = worker.sample(features, k, None, .5, evaluate, lambda **_: None,
                           seed=123, clock=lambda: clock[0], **kwargs)
    return result, queries, fits


def test_each_prefix_matches_separate_cold_run_including_oracle_fit_history_and_elapsed():
    captured = {}

    def factory(queries, fits, clock):
        def save(k, evidence):
            captured[k] = (evidence, list(queries), list(fits))
            clock[0] += 400.  # The next prefix must exclude the selection/serialization time.
        return save

    final = run_sampler(3, factory)
    assert list(captured) == [1, 2, 3]
    for k, prefix in captured.items():
        assert prefix == run_sampler(k)
        assert prefix[0]["planned_trials"] == prefix[0]["resolved_trials"] == 7 * k
        assert prefix[0]["sampling_eta_s"] == 0.
    assert final == captured[3]
    assert captured[1][0]["sample_minima"] != captured[3][0]["sample_minima"]


def test_all_certified_prefixes_do_not_refit_or_expose_mutable_future_evidence(monkeypatch):
    monkeypatch.setattr(worker, "sample_random_round", lambda *_: iter([(0, 0, 1), (1, 0, 2)]))
    evidence = {}
    queries = []
    clock = [0.]

    def evaluate(mask):
        queries.append(mask)
        clock[0] += 1.
        return float(mask == 0)

    def save(k, record):
        evidence[k] = record
        assert record["witnesses"]["a"]["round"] == 1
        if k == 1:
            # Even callers that mutate their private data cannot alter subsequent snapshots.
            record["witnesses"]["a"]["round"] = 77
            record["draws_per_feature"]["a"] = 77
            record["sample_minima"]["a"] = 0
        clock[0] += 100.

    result = worker.sample(("a", "b"), 3, None, .5, evaluate, lambda **_: None,
                           on_prefix=save, clock=lambda: clock[0])
    assert queries == [0, 1, 0, 2]
    for k in (1, 2, 3):
        record = evidence[k]
        assert record["stop"] == "all_witnessed" and record["completed_rounds"] == record["round"] == 1
        assert record["resolved_trials"] == record["planned_trials"] == 2 * k
        assert record["sampled_draws"] == 2 and record["sample_elapsed_s"] == 4.
    assert result == evidence[3]
    assert evidence[2]["sample_minima"] == {"a": -1, "b": -1}
    assert evidence[2]["draws_per_feature"] == {"a": 1, "b": 1}


def test_prefix_checkpoints_require_a_full_sampling_plan():
    with pytest.raises(ValueError, match="unbounded"):
        worker.sample(("a",), 3, 15, .5, lambda _: 0., lambda **_: None, on_prefix=lambda *_: None)
