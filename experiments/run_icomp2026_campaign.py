"""Run the frozen ICOMP 2026 protocol as three safe concurrent seed shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import yaml

from minppi_mofs.experiment_protocol import atomic_write_json

if __package__:
    from .summarize_multifidelity_campaign import (
        build_campaign_index,
        build_campaign_summary,
    )
else:
    from summarize_multifidelity_campaign import (  # type: ignore[no-redef]
        build_campaign_index,
        build_campaign_summary,
    )


_SHARD_STAGES = (
    "validate",
    "preprocess",
    "estimator",
    "removal",
    "alpha",
    "nsga2",
    "ablation",
)
_STAGES = (*_SHARD_STAGES, "summarize")
_TEST_ACCESS_STAGES = frozenset({"removal", "alpha", "nsga2"})


def default_seed_shards() -> tuple[tuple[int, ...], ...]:
    """Return the frozen three-worker split of the canonical five seeds."""
    return ((42, 123), (456, 789), (1024,))


def campaign_execution_stages(
    through: str,
) -> tuple[tuple[str, ...], bool]:
    """Resolve shard stages plus the single campaign-level summary action."""
    if through not in _STAGES:
        raise ValueError(f"unsupported campaign stage: {through}")
    if through == "summarize":
        return _SHARD_STAGES, True
    index = _SHARD_STAGES.index(through)
    return _SHARD_STAGES[: index + 1], False


def build_pipeline_command(
    *,
    python: Path,
    pipeline_script: Path,
    stage: str,
    config_dir: Path,
    output_root: Path,
    model_policy: Path,
    run_id: str,
    seeds: tuple[int, ...],
    resume: bool,
) -> tuple[str, ...]:
    """Build one explicit shard command without shell interpolation."""
    if stage not in _SHARD_STAGES:
        raise ValueError(f"unsupported campaign stage: {stage}")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
    ):
        raise ValueError("campaign shard seeds must be unique non-negative integers")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("campaign shard run_id must be non-empty")
    command = (
        Path(python).as_posix(),
        Path(pipeline_script).as_posix(),
        "--stage",
        stage,
        "--config-dir",
        Path(config_dir).as_posix(),
        "--output-root",
        Path(output_root).as_posix(),
        "--backend",
        "cuda",
        "--model-policy",
        Path(model_policy).as_posix(),
        "--run-id",
        run_id,
        "--seeds",
        *(str(seed) for seed in seeds),
    )
    return (*command, "--resume") if resume else command


def _shard_name(campaign_id: str, seeds: tuple[int, ...]) -> str:
    return f"{campaign_id}-s{'-'.join(str(seed) for seed in seeds)}"


def _run_command(command: tuple[str, ...], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} command="
            f"{json.dumps(command, ensure_ascii=False)}\n"
        )
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} "
            f"exit_code={completed.returncode}\n"
        )
        handle.flush()
    return int(completed.returncode)


def _eligible_shards(
    shards: tuple[tuple[tuple[int, ...], str], ...],
    *,
    stage: str,
    config_dir: Path,
) -> tuple[tuple[tuple[int, ...], str], ...]:
    if stage not in {"alpha", "ablation"}:
        return shards
    try:
        document = yaml.safe_load(
            (config_dir / "icomp2026_pipeline.yaml").read_text(
                encoding="utf-8"
            )
        )
        scopes = document.get("stage_scopes", {})
        seeds = set(scopes.get(stage, {}).get("seeds", ()))
    except (OSError, UnicodeDecodeError, AttributeError, yaml.YAMLError) as exc:
        raise ValueError("cannot read campaign stage scope") from exc
    if not seeds:
        return shards
    return tuple(
        item for item in shards if seeds.intersection(item[0])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run three source-identical seed shards through a frozen stage."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/icomp2026_48h"),
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--model-policy", type=Path, required=True)
    parser.add_argument(
        "--through",
        choices=_STAGES,
        default="estimator",
        help="last stage to execute; default keeps the held-out test sealed",
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument(
        "--allow-test-access",
        action="store_true",
        help="required before running removal/alpha/nsga2 stages",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_parallel < 1 or args.max_parallel > 3:
        raise ValueError("max_parallel must be between one and three")
    selected_stages, summarize_campaign = campaign_execution_stages(
        args.through
    )
    if (
        _TEST_ACCESS_STAGES.intersection(selected_stages)
        and not args.allow_test_access
    ):
        raise ValueError(
            "test-access stages require the explicit --allow-test-access gate"
        )
    project_root = Path(__file__).resolve().parents[1]
    pipeline_script = project_root / "experiments" / "run_icomp2026_pipeline.py"
    python = Path(sys.executable).resolve()
    config_dir = (
        args.config_dir
        if args.config_dir.is_absolute()
        else (project_root / args.config_dir).resolve()
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (project_root / args.output_root).resolve()
    )
    model_policy = (
        args.model_policy
        if args.model_policy.is_absolute()
        else (project_root / args.model_policy).resolve()
    )
    shards = tuple(
        (
            seeds,
            _shard_name(args.campaign_id, seeds),
        )
        for seeds in default_seed_shards()
    )
    campaign_root = output_root / args.campaign_id
    logs_root = campaign_root / "logs"
    campaign_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        campaign_root / "campaign_plan.json",
        {
            "schema_version": 1,
            "campaign_id": args.campaign_id,
            "backend": "cuda",
            "max_parallel": args.max_parallel,
            "through": args.through,
            "test_access_authorized": args.allow_test_access,
            "seed_shards": [list(seeds) for seeds, _ in shards],
            "run_ids": [run_id for _, run_id in shards],
        },
    )
    for stage in selected_stages:
        active_shards = _eligible_shards(
            shards, stage=stage, config_dir=config_dir
        )
        if not active_shards:
            print(f"[campaign] stage={stage} has empty frozen scope", flush=True)
            continue
        print(
            f"[campaign] stage={stage} shards={len(active_shards)} "
            f"parallel={args.max_parallel}",
            flush=True,
        )
        failures: list[tuple[str, int]] = []
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {}
            for seeds, run_id in active_shards:
                run_root = output_root / run_id
                resume = run_root.exists()
                command = build_pipeline_command(
                    python=python,
                    pipeline_script=pipeline_script,
                    stage=stage,
                    config_dir=config_dir,
                    output_root=output_root,
                    model_policy=model_policy,
                    run_id=run_id,
                    seeds=seeds,
                    resume=resume,
                )
                future = pool.submit(
                    _run_command,
                    command,
                    cwd=project_root,
                    log_path=logs_root / f"{run_id}.log",
                )
                futures[future] = run_id
            for future in as_completed(futures):
                run_id = futures[future]
                code = future.result()
                print(
                    f"[campaign] stage={stage} run={run_id} exit={code}",
                    flush=True,
                )
                if code != 0:
                    failures.append((run_id, code))
        if failures:
            print(
                f"[campaign] stopped after failures={failures}",
                flush=True,
            )
            return 1
    if "estimator" in selected_stages:
        index = build_campaign_index(
            tuple(output_root / run_id for _, run_id in shards)
        )
        atomic_write_json(campaign_root / "campaign_index.json", index)
        print(
            f"[campaign] index={campaign_root / 'campaign_index.json'}",
            flush=True,
        )
    if summarize_campaign:
        report = build_campaign_summary(
            tuple(output_root / run_id for _, run_id in shards),
            campaign_root=campaign_root,
            config_dir=config_dir,
            model_policy_path=model_policy,
            project_root=project_root,
        )
        print(
            f"[campaign] summary={campaign_root / 'campaign_summary.json'} "
            f"publication_allowed={report['publication_allowed']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
