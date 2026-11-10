import pytest

from experiments.run_thesis_predictor import preparation_budget


def test_budget_charges_verified_sampling_and_latest_possible_freeze_time():
    watch = {"status": "completed", "elapsed_s": 79.4, "inherited_sampling_s": 780.93}
    log = [
        "time | isolet phase=validation_gate | elapsed=60s | misc",
        "time | gisette phase=test_full | elapsed=2s | misc",
        "time | isolet phase=training_confirmation | elapsed=70s | misc",
        "time | isolet phase=test_retained | elapsed=79s | misc",
    ]
    charge, source = preparation_budget("isolet", watch, log)
    assert charge == pytest.approx(851.93)
    assert source == "first_post_selection_heartbeat"


def test_budget_uses_full_wall_if_no_post_selection_observation():
    watch = {"status": "completed", "elapsed_s": 10., "inherited_sampling_s": 40.}
    assert preparation_budget("madelon", watch, []) == (50., "completed_source_run")
    with pytest.raises(ValueError):
        preparation_budget("madelon", {**watch, "status": "timed_out"}, [])
