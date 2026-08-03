"""Run the complete fail-fast ICOMP 2026 Group 1 workflow."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
import os
from pathlib import Path
import re
import subprocess
import sys

from minppi_mofs.multifidelity import CalibrationRules, load_model_policy

try:
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        verify_registered_snapshot,
    )
except ModuleNotFoundError:  # Direct script execution from experiments/.
    from preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        verify_registered_snapshot,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("abalone", "abalone.yaml", "preprocess_abalone.py"),
    (
        "credit_card_fraud",
        "credit_card_fraud.yaml",
        "preprocess_Credit-Card-Fraud.py",
    ),
    ("gisette", "gisette.yaml", "preprocess_Gisette.py"),
    ("isolet", "isolet.yaml", "preprocess_ISOLET.py"),
    ("madelon", "madelon.yaml", "preprocess_Madelon.py"),
    ("spambase", "spambase.yaml", "preprocess_Spambase.py"),
)
Runner = Callable[..., object]
GROUP1_CALIBRATION_SEED = 42
GROUP1_CANDIDATE_ROUNDS = (10, 20, 40, 100)
GROUP1_CALIBRATION_RULES = CalibrationRules()


def run_checked_command_sequence(
    commands: Iterable[Sequence[str]],
    *,
    python_executable: Path,
    project_root: Path,
    runner: Runner = subprocess.run,
) -> None:
    """Run commands in order and propagate the first non-zero exit."""
    for arguments in commands:
        command = [str(python_executable), *map(str, arguments)]
        print(f"\n> {subprocess.list2cmdline(command)}", flush=True)
        runner(command, cwd=project_root, check=True)


def _require_adjacent_project_python(project_root: Path) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    expected = (project_root.parent / "venv" / scripts_dir / executable_name).resolve()
    actual = Path(sys.executable).resolve()
    if actual != expected:
        raise RuntimeError(
            "Group 1 must run with the adjacent project venv: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _require_clean_source_tree(
    project_root: Path,
    *,
    expected_head: str | None = None,
    runner: Runner = subprocess.run,
) -> str:
    """Fail before formal work if Git cannot reproduce the current source."""
    completed = runner(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = getattr(completed, "stdout", None)
    if not isinstance(status, str):
        raise RuntimeError("cannot verify the Group 1 Git source tree")
    if status.strip():
        raise RuntimeError(
            "Group 1 requires a clean Git source tree; commit the intended "
            "source and documentation changes before the formal run:\n"
            f"{status.rstrip()}"
        )
    revision_result = runner(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = getattr(revision_result, "stdout", None)
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision.strip()) is None
    ):
        raise RuntimeError("cannot verify the Group 1 Git HEAD")
    revision = revision.strip().lower()
    if expected_head is not None and revision != expected_head.lower():
        raise RuntimeError(
            "Git HEAD changed during Group 1 preflight; rerun the complete "
            "entry from the new clean revision"
        )
    return revision


def _normalize_output_root(project_root: Path, configured: Path) -> Path:
    """Return a project-relative output path confined below ``results/``."""
    root = project_root.resolve()
    results_root = (root / "results").resolve()
    candidate = Path(configured)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(
            "output root must be a child of the project results directory"
        ) from exc
    if resolved == results_root:
        raise ValueError("output root must be a child of the project results directory")
    return resolved.relative_to(root)


def _load_group1_model_policy(policy_path: Path) -> None:
    """Reject a cached policy that differs from the frozen Group 1 contract."""
    policy = load_model_policy(
        policy_path,
        expected_datasets=tuple(dataset_id for dataset_id, _, _ in DATASETS),
    )
    if (
        policy.calibration_seed != GROUP1_CALIBRATION_SEED
        or policy.candidate_rounds != GROUP1_CANDIDATE_ROUNDS
        or policy.rules != GROUP1_CALIBRATION_RULES
    ):
        raise ValueError(
            "existing model policy does not match the frozen Group 1 contract"
        )


def _preflight_commands() -> tuple[tuple[str, ...], ...]:
    basetemp = f".t{os.getpid()}"
    return (
        ("-m", "pip", "check"),
        ("-c", "import minppi_mofs; print('import OK')"),
        (
            "-m",
            "pytest",
            "tests",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            basetemp,
        ),
        (
            "-m",
            "ruff",
            "check",
            "src",
            "experiments",
            "tests",
            "conference-paper/sync_figures.py",
        ),
    )


def _acquisition_commands() -> tuple[tuple[str, ...], ...]:
    return tuple(
        (f"experiments/preprocess_data/{script_name}",)
        for _, _, script_name in DATASETS
    )


def _verify_registered_raw_snapshots(project_root: Path) -> None:
    print("\nRegistered raw snapshot verification:", flush=True)
    for dataset_id, config_name, _ in DATASETS:
        snapshot = load_registered_snapshot(
            project_root / "configs" / config_name,
            expected_dataset_id=dataset_id,
        )
        verify_registered_snapshot(snapshot)
        print(
            f"  {dataset_id}: bytes={snapshot.byte_count} sha256={snapshot.sha256}",
            flush=True,
        )


def _experiment_commands(
    *,
    project_root: Path,
    output_root: Path,
    bootstrap_id: str,
    smoke_id: str,
    campaign_id: str,
    max_parallel: int,
) -> tuple[tuple[str, ...], ...]:
    output_text = output_root.as_posix()
    bootstrap_root = output_root / bootstrap_id
    smoke_root = output_root / smoke_id
    policy = bootstrap_root / "calibration" / "model_policy.json"

    validate = [
        "experiments/run_icomp2026_pipeline.py",
        "--stage",
        "validate",
        "--config-dir",
        "configs",
        "--output-root",
        output_text,
        "--seeds",
        "42",
        "--backend",
        "cuda",
        "--run-id",
        bootstrap_id,
    ]
    if (project_root / bootstrap_root).exists():
        validate.append("--resume")

    commands: list[tuple[str, ...]] = [
        tuple(validate),
        (
            "experiments/run_icomp2026_pipeline.py",
            "--stage",
            "preprocess",
            "--config-dir",
            "configs",
            "--output-root",
            output_text,
            "--seeds",
            "42",
            "--backend",
            "cuda",
            "--run-id",
            bootstrap_id,
            "--resume",
        ),
    ]
    policy_path = project_root / policy
    if policy_path.is_file():
        _load_group1_model_policy(policy_path)
    else:
        commands.append(
            (
                "experiments/calibrate_icomp2026_models.py",
                "--run-root",
                bootstrap_root.as_posix(),
                "--config-dir",
                "configs",
                "--backend",
                "cuda",
                "--seed",
                "42",
                "--candidate-rounds",
                "10",
                "20",
                "40",
                "100",
                "--output",
                policy.as_posix(),
            )
        )

    smoke = [
        "experiments/run_icomp2026_pipeline.py",
        "--stage",
        "all",
        "--config-dir",
        "configs",
        "--output-root",
        output_text,
        "--seeds",
        "42",
        "--backend",
        "cuda",
        "--run-id",
        smoke_id,
        "--smoke",
        "--model-policy",
        policy.as_posix(),
    ]
    if (project_root / smoke_root).exists():
        smoke.append("--resume")
    commands.extend(
        (
            tuple(smoke),
            (
                "experiments/run_icomp2026_campaign.py",
                "--config-dir",
                "configs",
                "--output-root",
                output_text,
                "--campaign-id",
                campaign_id,
                "--model-policy",
                policy.as_posix(),
                "--through",
                "estimator",
                "--max-parallel",
                str(max_parallel),
            ),
            (
                "experiments/screen_icomp2026_results.py",
                "--run-root",
                f"{output_text}/{campaign_id}-s42-123",
                "--config-dir",
                "configs",
                "--model-policy",
                policy.as_posix(),
                "--backend",
                "cuda",
                "--seed",
                "42",
                "--output-dir",
                f"{output_text}/{campaign_id}/screening",
            ),
        )
    )
    return tuple(commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run environment checks, raw acquisition, calibration, smoke, "
            "sealed estimator campaign, and validation-only screening."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/icomp2026_formal")
    )
    parser.add_argument("--bootstrap-id", default="bootstrap-s42-v1")
    parser.add_argument("--smoke-id", default="smoke-s42-v1")
    parser.add_argument("--campaign-id", default="icomp2026-formal-v1")
    parser.add_argument("--max-parallel", type=int, choices=(1, 2, 3), default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python_executable = _require_adjacent_project_python(PROJECT_ROOT)
    output_root = _normalize_output_root(PROJECT_ROOT, args.output_root)
    source_revision = _require_clean_source_tree(PROJECT_ROOT)
    run_checked_command_sequence(
        _preflight_commands(),
        python_executable=python_executable,
        project_root=PROJECT_ROOT,
    )
    run_checked_command_sequence(
        _acquisition_commands(),
        python_executable=python_executable,
        project_root=PROJECT_ROOT,
    )
    _verify_registered_raw_snapshots(PROJECT_ROOT)
    _require_clean_source_tree(PROJECT_ROOT, expected_head=source_revision)
    run_checked_command_sequence(
        _experiment_commands(
            project_root=PROJECT_ROOT,
            output_root=output_root,
            bootstrap_id=args.bootstrap_id,
            smoke_id=args.smoke_id,
            campaign_id=args.campaign_id,
            max_parallel=args.max_parallel,
        ),
        python_executable=python_executable,
        project_root=PROJECT_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
