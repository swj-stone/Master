"""The wall-clock limit must interrupt a worker even inside a stalled fit."""

import json
import multiprocessing
import sys
import time

import pytest

from experiments import run_thesis_sampling as runner
from experiments.run_thesis_sampling import reuse_budget, summary_row, supervise


def _stalled_fit(output):
    (output / "progress.json").write_text('{"phase":"sampling","coverage":7}')
    time.sleep(60)


def _late_write(path, when):
    time.sleep(max(0, when - time.monotonic()))
    path.write_text("worker survived the limit")
    time.sleep(60)


def test_deadline_kills_worker_and_keeps_partial_progress(tmp_path):
    process = multiprocessing.get_context("spawn").Process(target=_stalled_fit, args=(tmp_path,))
    outcome = supervise(process, tmp_path, 2.0, lambda *_: None)
    assert outcome["status"] == "timed_out"
    assert outcome["elapsed_s"] < 6
    assert not process.is_alive()
    assert outcome["progress"]["coverage"] == 7
    assert json.loads((tmp_path / "watchdog.json").read_text())["status"] == "timed_out"


def test_deadline_still_kills_worker_when_live_log_blocks(tmp_path):
    late = tmp_path / "late"
    process = multiprocessing.get_context("spawn").Process(target=_late_write, args=(late, time.monotonic() + 2))
    result = supervise(process, tmp_path, 1.3, lambda *_: time.sleep(2.3), heartbeat_s=0.05)
    assert result["status"] == "timed_out" and not late.exists()


def test_matched_replay_charges_sampling_to_the_same_dataset_limit(tmp_path):
    directory = tmp_path / "isolet"
    directory.mkdir()
    record = {"phase": "complete", "dataset": "isolet", "seed": 42, "requested_rounds": 5,
              "sample_seconds": 780, "sample_elapsed_s": 780.9}
    path = directory / "result.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    supplied, remaining = reuse_budget(tmp_path, "isolet", 900, 5, 780)
    assert supplied == path and remaining == pytest.approx(119.1)
    with pytest.raises(ValueError, match="budget"):
        reuse_budget(tmp_path, "isolet", 700, 5, 780)
    with pytest.raises(ValueError, match="Mismatched"):
        reuse_budget(tmp_path, "isolet", 900, 3, 780)


def test_interrupted_replay_reports_time_charged_before_worker_verification():
    outcome = {"status": "timed_out", "elapsed_s": 119.068, "inherited_sampling_s": 780.932,
               "progress": {"phase": "verify_sampling", "inherited_sampling_s": 0.0}, "result": {}}
    row = summary_row("isolet", outcome)
    assert row["inherited_sampling_s"] == pytest.approx(780.932)
    assert row["wall_plus_reused_sampling_s"] == pytest.approx(900)


def test_sign_aware_run_records_its_source_and_primary_method_before_start(tmp_path, monkeypatch):
    files = ("experiments/thesis_sampling_worker.py", "src/minppi_mofs/removal_selection.py",
             "src/minppi_mofs/minimum_sampling.py", "src/minppi_mofs/baselines/shap_top_k.py",
             "src/minppi_mofs/sign_aware_removal.py")
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run", "--datasets", "abalone", "--selection-policy", "sign_aware"])

    def check_protocol(process, directory, seconds, log, sample_seconds):
        protocol = json.loads((directory.parent / "protocol.json").read_bytes())
        assert protocol["primary_method"] == "sign_worst"
        assert protocol["sensitivity_method"] == "sign_pooled"
        assert protocol["matched_controls"] == ["matched_gain", "matched_treeshap"]
        assert set(protocol["selection_source_sha256"]) == set(files)
        assert seconds == 900
        return {"status": "timed_out", "elapsed_s": seconds, "progress": {}, "result": {}}

    monkeypatch.setattr(runner, "supervise", check_protocol)
    assert runner.main() == 1
