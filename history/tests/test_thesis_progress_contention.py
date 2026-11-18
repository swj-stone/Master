"""A Windows snapshot read collision must not stop the running experiment."""

import errno
import json
from pathlib import Path

import pytest

from experiments import run_thesis_sampling as runner


class PublishingWorker:
    """Publish one new snapshot per monitor tick, then write a final result and exit."""

    def __init__(self, directory):
        self.directory = directory
        self.tick = 0
        self.now = 0.
        self.exitcode = None
        self.killed = False

    def start(self):
        pass

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout=None):
        if timeout is None:
            assert not self.is_alive()
            return
        self.tick += 1
        self.now += timeout
        if self.tick == 4:
            (self.directory / "result.json").write_text('{"phase":"complete","score":0.9}', encoding="utf-8")
            self.exitcode = 0
        snapshot = {"phase": "sampling", "resolved_trials": self.tick}
        (self.directory / "progress.json").write_text(json.dumps(snapshot), encoding="utf-8")

    def kill(self):
        self.killed = True
        self.exitcode = -1

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def worker(tmp_path, monkeypatch):
    proc = PublishingWorker(tmp_path)
    monkeypatch.setattr(runner.time, "monotonic", lambda: proc.now)
    monkeypatch.setattr(runner.time, "sleep", proc.sleep)
    return proc


def supervise(worker, callback):
    return runner.supervise(
        worker, worker.directory, None, lambda _: None, heartbeat_s=10,
        heartbeat_formatter=callback,
    )


def deny_while(monkeypatch, path, predicate):
    read_text = Path.read_text

    def read(file, *args, **kwargs):
        if file == path and predicate():
            raise PermissionError(errno.EACCES, "Permission denied", str(file))
        return read_text(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)


def test_transient_progress_denial_recovers_on_same_heartbeat(worker, monkeypatch):
    denials = []

    def lock_once():
        if worker.tick == 1 and not denials:
            denials.append(worker.tick)
            return True
        return False

    deny_while(monkeypatch, worker.directory / "progress.json", lock_once)
    observations = []
    result = supervise(worker, lambda p, elapsed, left: observations.append(p.get("resolved_trials")))
    assert denials == [1] and observations == [1, 2, 3]
    assert result["status"] == "completed" and not worker.killed


def test_unreadable_heartbeat_and_final_progress_retain_last_readable_snapshot(worker, monkeypatch):
    deny_while(monkeypatch, worker.directory / "progress.json", lambda: worker.tick in (2, 4))
    observations = []
    def observe(progress, elapsed, left):
        observations.append((progress.get("resolved_trials"), elapsed, left))

    result = supervise(worker, observe)
    assert [n for n, _, _ in observations] == [1, 1, 3]
    assert all(left is None for _, _, left in observations)
    assert observations[1][1] >= observations[0][1] + 10
    assert result["status"] == "completed" and not worker.killed
    assert result["result"] == {"phase": "complete", "score": 0.9}
    assert result["progress"]["resolved_trials"] == 3
    assert json.loads((worker.directory / "watchdog.json").read_bytes())["status"] == "completed"


def test_result_read_denial_cannot_succeed_on_progress_alone(worker, monkeypatch):
    deny_while(monkeypatch, worker.directory / "result.json", lambda: True)
    result = supervise(worker, lambda *_: None)
    assert result["status"] == "error" and not result["result"] and not worker.killed
    assert "result.json" in result["error"]
    assert json.loads((worker.directory / "watchdog.json").read_bytes())["error"] == result["error"]
