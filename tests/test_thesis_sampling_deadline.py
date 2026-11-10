"""The wall-clock limit must interrupt a worker even inside a stalled fit."""

import json
import multiprocessing
import time

from experiments.run_thesis_sampling import supervise


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
