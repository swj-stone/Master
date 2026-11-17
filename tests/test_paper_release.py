"""Keep the short paper reproducible without replacing historical archives."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


MODULE = Path(__file__).resolve().parents[1] / "thesis-paper/package_revision.py"


@pytest.fixture
def release():
    spec = importlib.util.spec_from_file_location("paper_release", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_uses_file_contents_independently_of_checkout_time(release, tmp_path):
    member = tmp_path / "method.tex"
    member.write_bytes(b"Method.\n")
    archive = tmp_path / "source.zip"
    release.write_archive(archive, {"method.tex": member})
    first = archive.read_bytes()
    os.utime(member, (3600, 3600))
    release.write_archive(archive, {"method.tex": member})
    assert archive.read_bytes() == first


def test_historical_flag_replays_pinned_archive_without_writes(release, tmp_path, monkeypatch):
    frozen = tmp_path / "r13.zip"
    original = b"frozen fixture"
    frozen.write_bytes(original)
    ledger = tmp_path / "artifacts.json"
    ledger.write_text(json.dumps([{"path": "r13.zip", "sha256": hashlib.sha256(original).hexdigest()}]))
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "DECISION", frozen)
    monkeypatch.setattr(release, "ARTIFACTS", ledger)
    checked = []
    monkeypatch.setattr(release, "verify_transport", checked.append)
    monkeypatch.setattr(sys, "argv", ["package_revision.py", "--decision-evidence"])
    release.main()
    assert checked == [frozen] and frozen.read_bytes() == original
    assert set(tmp_path.iterdir()) == {frozen, ledger}
    frozen.write_bytes(b"damaged archive")
    with pytest.raises(ValueError, match="ledger"):
        release.main()
    assert checked == [frozen]


def test_ledger_keeps_releases_absent_from_clean_checkout(release, tmp_path, monkeypatch):
    ledger = tmp_path / "artifacts.json"
    historic = {"path": "earlier.zip", "sha256": "pinned"}
    ledger.write_text(json.dumps([historic]))
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "ARTIFACTS", ledger)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"paper")
    release.record_artifacts([paper])
    rows = json.loads(ledger.read_text())
    assert rows[0] == historic
    assert rows[1]["sha256"] == hashlib.sha256(paper.read_bytes()).hexdigest()


def test_screening_flag_verifies_old_release_without_packaging_new_paper(release, tmp_path, monkeypatch):
    frozen = tmp_path / "r14.zip"
    frozen.write_bytes(b"frozen screening release")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"path": frozen.name, "sha256": release.digest(frozen)}]))
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "ARTIFACTS", ledger)
    monkeypatch.setattr(release, "SCREENING", frozen)
    calls = []
    monkeypatch.setattr(release, "verify_screening", lambda paper, prior: calls.append(paper))
    monkeypatch.setattr(sys, "argv", ["package_revision.py", "--screening-release"])
    release.main()
    assert calls == [frozen] and frozen.read_bytes() == b"frozen screening release"
    frozen.write_bytes(b"modified")
    with pytest.raises(ValueError, match="ledger"):
        release.main()
