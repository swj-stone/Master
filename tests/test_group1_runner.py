from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def test_checked_command_sequence_stops_after_first_failure(tmp_path: Path) -> None:
    """Catches later formal stages running after an upstream command failed."""
    module = importlib.import_module("experiments.run_icomp2026_group1")
    observed: list[tuple[str, ...]] = []

    def fake_runner(command, *, cwd, check):
        assert cwd == tmp_path
        assert check is True
        arguments = tuple(command[1:])
        observed.append(arguments)
        if arguments == ("second",):
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        module.run_checked_command_sequence(
            (("first",), ("second",), ("must-not-run",)),
            python_executable=Path("project-python"),
            project_root=tmp_path,
            runner=fake_runner,
        )

    assert observed == [("first",), ("second",)]


@pytest.mark.parametrize("status", (" M experiments/runner.py\n", "?? new_source.py\n"))
def test_group1_rejects_dirty_source_tree_before_formal_work(
    tmp_path: Path, status: str
) -> None:
    """Catches a formal run freezing uncommitted or untracked source bytes."""
    module = importlib.import_module("experiments.run_icomp2026_group1")
    observed: list[tuple[object, ...]] = []

    def fake_runner(command, **kwargs):
        observed.append(tuple(command))
        assert kwargs == {
            "cwd": tmp_path,
            "check": True,
            "capture_output": True,
            "text": True,
        }
        return SimpleNamespace(stdout=status)

    with pytest.raises(RuntimeError, match="clean Git source tree"):
        module._require_clean_source_tree(tmp_path, runner=fake_runner)

    assert observed == [("git", "status", "--porcelain", "--untracked-files=all")]


def test_group1_accepts_ignored_artifacts_in_clean_source_tree(tmp_path: Path) -> None:
    module = importlib.import_module("experiments.run_icomp2026_group1")
    observed: list[tuple[object, ...]] = []

    def fake_runner(command, **kwargs):
        observed.append(tuple(command))
        output = "" if command[1] == "status" else "a" * 40 + "\n"
        return SimpleNamespace(stdout=output)

    revision = module._require_clean_source_tree(tmp_path, runner=fake_runner)

    assert revision == "a" * 40
    assert observed == [
        ("git", "status", "--porcelain", "--untracked-files=all"),
        ("git", "rev-parse", "--verify", "HEAD"),
    ]


def test_group1_rejects_git_status_failure(tmp_path: Path) -> None:
    module = importlib.import_module("experiments.run_icomp2026_group1")

    def fake_runner(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)

    with pytest.raises(subprocess.CalledProcessError):
        module._require_clean_source_tree(tmp_path, runner=fake_runner)


def test_group1_rejects_clean_but_changed_head(tmp_path: Path) -> None:
    module = importlib.import_module("experiments.run_icomp2026_group1")

    def fake_runner(command, **kwargs):
        output = "" if command[1] == "status" else "b" * 40 + "\n"
        return SimpleNamespace(stdout=output)

    with pytest.raises(RuntimeError, match="Git HEAD changed"):
        module._require_clean_source_tree(
            tmp_path,
            expected_head="a" * 40,
            runner=fake_runner,
        )


def test_group1_stops_if_tree_changes_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches validate starting after tests or acquisition alter provenance."""
    module = importlib.import_module("experiments.run_icomp2026_group1")
    revision = "a" * 40
    clean_checks: list[str | None] = []
    executed: list[tuple[tuple[str, ...], ...]] = []

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module, "_require_adjacent_project_python", lambda _: Path("python")
    )
    monkeypatch.setattr(
        module, "_normalize_output_root", lambda *args: Path("results/formal")
    )

    def fake_clean(project_root, *, expected_head=None, runner=None):
        clean_checks.append(expected_head)
        if expected_head is None:
            return revision
        raise RuntimeError("source changed after preflight")

    def fake_run(commands, **kwargs):
        executed.append(tuple(tuple(command) for command in commands))

    monkeypatch.setattr(module, "_require_clean_source_tree", fake_clean)
    monkeypatch.setattr(module, "_preflight_commands", lambda: (("preflight",),))
    monkeypatch.setattr(module, "_acquisition_commands", lambda: (("acquire",),))
    monkeypatch.setattr(module, "run_checked_command_sequence", fake_run)
    monkeypatch.setattr(module, "_verify_registered_raw_snapshots", lambda _: None)
    monkeypatch.setattr(
        module,
        "_experiment_commands",
        lambda **kwargs: pytest.fail("formal experiment commands must not be built"),
    )

    with pytest.raises(RuntimeError, match="source changed after preflight"):
        module.main([])

    assert clean_checks == [None, revision]
    assert executed == [(("preflight",),), (("acquire",),)]


def test_existing_malformed_model_policy_fails_closed(tmp_path: Path) -> None:
    """Catches file existence being mistaken for completed calibration."""
    module = importlib.import_module("experiments.run_icomp2026_group1")
    policy = (
        tmp_path
        / "results"
        / "icomp2026_formal"
        / "bootstrap-s42-v1"
        / "calibration"
        / "model_policy.json"
    )
    policy.parent.mkdir(parents=True)
    policy.write_text("not canonical json", encoding="utf-8")

    with pytest.raises(ValueError, match="model policy"):
        module._experiment_commands(
            project_root=tmp_path,
            output_root=Path("results/icomp2026_formal"),
            bootstrap_id="bootstrap-s42-v1",
            smoke_id="smoke-s42-v1",
            campaign_id="icomp2026-formal-v1",
            max_parallel=3,
        )


def test_existing_model_policy_must_match_frozen_group1_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a structurally valid but scientifically different policy reuse."""
    module = importlib.import_module("experiments.run_icomp2026_group1")
    policy_path = (
        tmp_path
        / "results"
        / "icomp2026_formal"
        / "bootstrap-s42-v1"
        / "calibration"
        / "model_policy.json"
    )
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("placeholder", encoding="utf-8")
    valid_rules = module.CalibrationRules()
    invalid_policies = (
        SimpleNamespace(
            calibration_seed=7,
            candidate_rounds=(10, 20, 40, 100),
            rules=valid_rules,
        ),
        SimpleNamespace(
            calibration_seed=42,
            candidate_rounds=(20, 40, 100),
            rules=valid_rules,
        ),
        SimpleNamespace(
            calibration_seed=42,
            candidate_rounds=(10, 20, 40, 100),
            rules=module.CalibrationRules(max_full_utility_loss=0.006),
        ),
    )

    for invalid_policy in invalid_policies:
        monkeypatch.setattr(
            module,
            "load_model_policy",
            lambda *args, _policy=invalid_policy, **kwargs: _policy,
        )
        with pytest.raises(ValueError, match="frozen Group 1 contract"):
            module._experiment_commands(
                project_root=tmp_path,
                output_root=Path("results/icomp2026_formal"),
                bootstrap_id="bootstrap-s42-v1",
                smoke_id="smoke-s42-v1",
                campaign_id="icomp2026-formal-v1",
                max_parallel=3,
            )


@pytest.mark.parametrize(
    "configured",
    (Path("../outside"), Path("results/../configs"), Path("results")),
)
def test_group1_output_root_must_be_a_project_results_child(
    tmp_path: Path, configured: Path
) -> None:
    """Catches relative traversal and collision with the shared results root."""
    module = importlib.import_module("experiments.run_icomp2026_group1")

    with pytest.raises(ValueError, match="project results"):
        module._normalize_output_root(tmp_path, configured)


def test_group1_output_root_is_normalized_relative_to_project(tmp_path: Path) -> None:
    module = importlib.import_module("experiments.run_icomp2026_group1")
    expected = Path("results/icomp2026_formal")

    assert module._normalize_output_root(tmp_path, expected) == expected
    assert module._normalize_output_root(tmp_path, tmp_path / expected) == expected


def test_group1_pytest_basetemp_preserves_windows_path_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the runner reintroducing MAX_PATH failures in artifact tests."""
    module = importlib.import_module("experiments.run_icomp2026_group1")
    monkeypatch.setattr(module.os, "getpid", lambda: 52276)
    pytest_command = module._preflight_commands()[2]
    basetemp = Path(pytest_command[pytest_command.index("--basetemp") + 1])
    representative_artifact = (
        Path("test_status_store_resume_decis3")
        / "results"
        / "symlink-complete"
        / "raw_metrics"
        / "preprocess"
        / f".{('3' * 64)}.json.{('0' * 32)}.tmp"
    )

    assert len(str(module.PROJECT_ROOT / basetemp / representative_artifact)) <= 240
