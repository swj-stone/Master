"""Benchmark two locked XGBoost recipes on the same previously selected columns."""

import argparse
from datetime import datetime
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import re
import runpy
import sys

from experiments.run_thesis_sampling import DATASETS, MSK, ROOT, _read, _save, supervise


def preparation_budget(dataset, watchdog, lines):
    """Upper-bound elapsed preparation by an observation *after* both candidates froze."""
    wall = watchdog["elapsed_s"]
    sample = watchdog["inherited_sampling_s"]
    if (watchdog["status"] != "completed" or not math.isfinite(wall + sample)
            or min(wall, sample) < 0):
        raise ValueError(f"Missing completed source timing for {dataset}")
    pattern = re.compile(
        rf"\| {re.escape(dataset)} phase=(?:training_confirmation|test_full|test_retained|complete) \| elapsed=(\d+)s \|")
    for line in lines:
        match = pattern.search(line)
        if match:
            # The source printed elapsed seconds rounded to the nearest integer.
            upper = min(wall, int(match[1]) + 1.)
            return sample + upper, "first_post_selection_heartbeat"
    return sample + wall, "completed_source_run"


def run_worker(dataset, directory, manifest, recipe, training_check):
    with (directory / "worker.log").open("w", encoding="utf-8", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.argv = ["predictor", "--dataset", dataset, "--output-dir", str(directory), "--mask-manifest", str(manifest)]
        if training_check:
            runpy.run_module("experiments.test.replicate_fixed_rounds", run_name="__main__")
        else:
            sys.argv.extend(["--recipe", recipe])
            runpy.run_module("experiments.thesis_fixed_predictor", run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmed_run", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--dataset-seconds", type=int, default=900)
    parser.add_argument("--recipe", choices=("column_blend", "rounds"), default="column_blend")
    parser.add_argument("--training-check", action="store_true", help="Re-evaluate the rounds recipe inside the frozen train split.")
    args = parser.parse_args()
    if not 0 < args.dataset_seconds <= 900 or len(args.datasets) != len(set(args.datasets)):
        parser.error("Use distinct datasets and a limit from 1 to 900 seconds.")
    if args.training_check and args.recipe != "rounds":
        parser.error("The training-only replication is defined for --recipe rounds.")
    source = args.confirmed_run.resolve()
    manifest = source / "comparison/guarded_validation_manifest.json"
    manifest_data = _read(manifest)
    if manifest_data.get("source_run") != source.name or manifest_data.get("seed") != 42:
        parser.error("Frozen guarded selection and timing source must match.")
    log_lines = (source / "run.log").read_text(encoding="utf-8").splitlines()
    costs = {}
    for dataset in args.datasets:
        if dataset not in manifest_data["datasets"]:
            parser.error(f"Missing frozen feature set for {dataset}")
        watch = _read(source / dataset / "watchdog.json")
        charge, evidence = preparation_budget(dataset, watch, log_lines)
        if charge >= args.dataset_seconds:
            parser.error(f"{dataset} has no remaining budget after feature selection.")
        costs[dataset] = dict(charged_s=charge, basis=evidence)

    role = "training-check" if args.training_check else "predictor"
    run = ROOT / "results/test/thesis_sampling" / datetime.now(MSK).strftime(f"seed42-{role}-{args.recipe}-%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    worker_file = "test/replicate_fixed_rounds.py" if args.training_check else "thesis_fixed_predictor.py"
    files = (manifest, ROOT / "experiments" / worker_file)

    def digests():
        return [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]

    hashes = digests()
    settings = ({"colsample_bynode": 0.5} if args.recipe == "column_blend" else
                {"checkpoint_multipliers": [1.5, 2.0], "rounding": "ceil"})
    _save(run / "protocol.json", dict(seed=42, datasets=list(args.datasets), dataset_seconds=args.dataset_seconds,
          primary_method="guarded_chosen", fixed_sensitivity_method="guarded_blend", predictor_recipe=args.recipe,
          recipe="equal_average", evaluation_partition="training_check" if args.training_check else "official_test",
          alternative_parameters=settings, mask_source=source.name, mask_manifest_sha256=hashes[0],
          predictor_source_sha256=hashes[1], preparation=costs, minimum_improved_datasets=2,
          practical_change_fraction_of_margin=0.1, paired_bootstrap_replicates=500, paired_bootstrap_seed=42))

    def log(message):
        line = f"{datetime.now(MSK):%H:%M:%S} MSK | {message}"
        with (run / "run.log").open("a", encoding="utf-8") as handle:
            print(line, file=handle)
        print(line, flush=True)

    log(f"run={run.name} | frozen subsets={source.name} | per-dataset limit={args.dataset_seconds}s")
    rows = []
    for ds in args.datasets:
        if hashes != digests():
            raise RuntimeError("Predictor source or frozen feature sets changed during the experiment.")
        folder = run / ds
        folder.mkdir()
        charged = costs[ds]["charged_s"]
        remaining = args.dataset_seconds - charged
        process = multiprocessing.get_context("spawn").Process(
            target=run_worker, args=(ds, folder, manifest, args.recipe, args.training_check))
        log(f"start {ds} | prior preparation={charged:.3f}s | new work budget={remaining:.3f}s")
        outcome = supervise(process, folder, remaining, log)
        process.close()
        total = charged + outcome["elapsed_s"]
        outcome.update(inherited_preparation_s=charged, total_charged_s=total, dataset_limit_s=args.dataset_seconds)
        _save(folder / "watchdog.json", outcome)
        rows.append(dict(dataset=ds, status=outcome["status"], model_wall_s=outcome["elapsed_s"], total_charged_s=total))
        _save(run / "summary.json", rows)
        log(f"finish {ds} | status={outcome['status']} | model wall={outcome['elapsed_s']:.3f}s | charged={total:.3f}s")
    return int(any(r["status"] != "completed" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
