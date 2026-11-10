"""Run the six frozen seed-42 benchmark partitions with one deadline per dataset."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("abalone", "spambase", "madelon", "gisette", "credit_card_fraud", "isolet")
MSK = timezone(timedelta(hours=3))


def _save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _worker_entry(dataset, directory, rounds, sample_seconds, selection_policy, reuse_sampling, local_deadline=None):
    # XGBoost fits in this spawned process using native threads. Killing this
    # process interrupts a native fit without waiting for a Python callback.
    with (directory / "worker.log").open("w", encoding="utf-8", buffering=1) as worker_log:
        os.dup2(worker_log.fileno(), 1)
        os.dup2(worker_log.fileno(), 2)
        if local_deadline is not None:
            os.environ["MINPPI_DATASET_DEADLINE"] = str(local_deadline)
        sys.argv = ["thesis_sampling_worker", "--dataset", dataset, "--output-dir", str(directory),
                    "--rounds", str(rounds), "--sample-seconds", str(sample_seconds),
                    "--selection-policy", selection_policy]
        if reuse_sampling is not None:
            sys.argv.extend(("--reuse-sampling", str(reuse_sampling)))
        runpy.run_module("experiments.thesis_sampling_worker", run_name="__main__")


def supervise(process, directory, seconds, log, sample_seconds=None, *, heartbeat_s=10.0):
    started = time.monotonic()
    deadline = started + seconds
    process.start()
    kill_requested = []

    def expire():
        # Termination is independent of disk and console I/O in the monitor.
        if process.is_alive():
            kill_requested.append(time.monotonic() - started)
            process.kill()

    timer = threading.Timer(max(0.0, deadline - time.monotonic()), expire)
    timer.start()
    try:
        while process.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.join()
                break
            process.join(timeout=min(heartbeat_s, remaining))
            if not process.is_alive() or time.monotonic() >= deadline:
                continue
            p = _read(directory / "progress.json")
            elapsed = time.monotonic() - started
            eta = p.get("sampling_eta_s") if p.get("phase") == "sampling" else None
            if p.get("phase") == "sampling" and eta is None and p.get("resolved_trials") and p.get("planned_trials"):
                done = p["resolved_trials"]
                eta = p.get("sample_elapsed_s", 0) * (p["planned_trials"] - done) / done
            if eta is not None and sample_seconds is not None:
                eta = max(0.0, min(eta, sample_seconds - p.get("sample_elapsed_s", 0), seconds - elapsed))
            eta_text = "pending" if eta is None else f"~{max(0, eta):.0f}s"
            pool = (p.get("proposed_count", 0) if p.get("selection_policy") in ("sign_aware", "confirmed", "guarded", "local_refine")
                    else p.get("structured_proposed_count", p.get("proposed_count", 0)))
            log(f"{directory.name} phase={p.get('phase', 'starting')} | elapsed={elapsed:.0f}s"
                f" | coverage={p.get('coverage', 0)}/{p.get('n_features', '?')}"
                f" | sample_fits={p.get('sample_fits', 0)} | skipped={p.get('skipped_after', 0)}"
                f" | witnesses={p.get('witness_count', len(p.get('witnesses', {})))}"
                f" | nominees={pool} | masks checked={len(p.get('selection_trials', []))}"
                f" | sampling ETA={eta_text}"
                f" | deadline in {max(0, seconds-elapsed):.0f}s")
    finally:
        timer.cancel()
        timer.join()
        if process.is_alive():
            process.kill()
        process.join()
    result = _read(directory / "result.json")
    progress = _read(directory / "progress.json")
    status = "timed_out" if kill_requested else "completed" if result and process.exitcode == 0 else "error"
    outcome = {"status": status, "elapsed_s": time.monotonic() - started, "limit_s": seconds,
               "kill_requested_s": kill_requested[0] if kill_requested else None,
               "exit_code": process.exitcode, "progress": progress, "result": result}
    _save(directory / "watchdog.json", outcome)
    return outcome


def summary_row(dataset, outcome):
    data = {**outcome["progress"], **outcome["result"]}
    row = {"dataset": dataset, "seed": 42, "status": outcome["status"],
           "wall_s": round(outcome["elapsed_s"], 3), "phase": data.get("phase"),
           "metric": data.get("policy", {}).get("utility")}
    for key in ("sampling_stop", "sample_fits", "skipped_after", "coverage", "n_features",
                "proposed_count", "removed_count", "gate_passed", "full_validation_utility",
                "proposed_validation_utility", "retained_validation_utility", "full_test_utility",
                "retained_test_utility", "test_delta", "validation_fits", "test_fits", "completed_rounds",
                "selection_policy", "selected_method", "ranking_fits", "sampling_reused", "inherited_sample_fits"):
        row[key] = data.get(key)
    row["witness_count"] = data.get("witness_count", len(data.get("witnesses", {})))
    row["selection_masks_evaluated"] = len(data.get("selection_trials", []))
    row["inherited_sampling_s"] = outcome.get("inherited_sampling_s", data.get("inherited_sampling_s", 0.0))
    row["wall_plus_reused_sampling_s"] = round(outcome["elapsed_s"] + row["inherited_sampling_s"], 3)
    row["compression_fraction"] = (row["removed_count"] / row["n_features"]
                                   if row["removed_count"] is not None and row["n_features"] else None)
    draws = data.get("sampled_draws", data.get("marginals"))
    row["sampled_draws"] = draws
    row["observed_marginals"] = None if draws is None else draws - data.get("skipped_after", 0)
    return row


def write_summary(output, rows):
    _save(output / "summary.json", rows)
    path = output / "summary.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_method_comparison(output, datasets):
    """Tabulate comparisons only after the workers have frozen their choices and exited."""
    rows = []
    for dataset in datasets:
        result = _read(output / dataset / "result.json")
        if not result:
            continue
        selection = result.get("selection", {})
        for method, record in result.get("method_results", {}).items():
            scores = record.get("scores", ())
            reference = selection.get("full_scores", [result["full_validation_utility"]])
            deltas = [s - r for s, r in zip(scores, reference, strict=True)]
            rows.append(dict(dataset=dataset, metric=result["policy"]["utility"], method=method,
                             delivered=(method == result["selected_method"]), n=result["n_features"],
                             removed=record["removed_count"], removed_fraction=record["removed_count"]/result["n_features"],
                             pooled_validation_delta=deltas[0], worst_validation_half_delta=min(deltas[1:]) if len(deltas)>1 else None,
                             test_delta=record["test_delta"], test_utility=record["test_utility"]))
    if rows:
        with (output / "methods.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def reuse_budget(source, dataset, seconds, rounds, sample_seconds):
    """Charge the measured sampling time once when replaying a frozen sampling stage."""
    if source is None:
        return None, float(seconds)
    path = source / dataset / "result.json"
    result = _read(path)
    if (result.get("phase") != "complete" or result.get("dataset") != dataset or result.get("seed") != 42
            or result.get("requested_rounds") != rounds or result.get("sample_seconds") != sample_seconds):
        raise ValueError(f"Mismatched sampling result for {dataset}: {path}")
    inherited = result["sample_elapsed_s"]
    if not math.isfinite(inherited) or not 0 <= inherited < seconds:
        raise ValueError(f"No valid remaining budget for {dataset}: {inherited!r}s previously sampled")
    return path, seconds - inherited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dataset-seconds", type=int, default=900)
    parser.add_argument("--sample-seconds", type=int, default=780)
    parser.add_argument("--selection-policy", choices=("structured", "wide", "witness_only", "sign_aware", "confirmed", "guarded", "local_refine"),
                        default="structured")
    parser.add_argument("--reuse-sampling-run", type=Path, help="Reuse verified seed-42 sampling evidence from an earlier run.")
    args = parser.parse_args()
    if (not 1 <= args.dataset_seconds <= 900 or not 0 < args.sample_seconds < args.dataset_seconds
            or args.rounds < 1 or len(set(args.datasets)) != len(args.datasets)):
        parser.error("Use distinct datasets, positive rounds, and 0 < sample-seconds < dataset-seconds <= 900.")
    budgets = {dataset: reuse_budget(args.reuse_sampling_run, dataset, args.dataset_seconds,
                                     args.rounds, args.sample_seconds) for dataset in args.datasets}
    output = ROOT / "results/thesis_sampling" / datetime.now(MSK).strftime(f"seed42-{args.selection_policy}-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    def selection_sources():
        files = ("experiments/thesis_sampling_worker.py", "src/minppi_mofs/removal_selection.py",
                 "src/minppi_mofs/minimum_sampling.py", "src/minppi_mofs/baselines/shap_top_k.py")
        if args.selection_policy in ("sign_aware", "confirmed", "guarded", "local_refine"):
            files += ("src/minppi_mofs/sign_aware_removal.py",)
        if args.selection_policy == "confirmed":
            files += ("src/minppi_mofs/removal_confirmation.py",)
        if args.selection_policy in ("guarded", "local_refine"):
            files += ("src/minppi_mofs/guarded_update.py",)
        if args.selection_policy == "local_refine":
            files += ("src/minppi_mofs/local_refinement.py", "src/minppi_mofs/residual_ranking.py")
        return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}

    sources = selection_sources()
    protocol = {"datasets": list(args.datasets), "seed": 42, "rounds": args.rounds,
                "selection_policy": args.selection_policy, "sample_seconds": args.sample_seconds,
                "dataset_seconds": args.dataset_seconds,
                "reuse_sampling_run": str(args.reuse_sampling_run) if args.reuse_sampling_run else None,
                "selection_source_sha256": sources}
    if args.selection_policy == "sign_aware":
        protocol.update(primary_method="sign_worst", sensitivity_method="sign_pooled",
                        matched_controls=["matched_gain", "matched_treeshap"])
    elif args.selection_policy == "confirmed":
        protocol.update(primary_method="confirmed", confirmation_candidates=["structured", "sign_worst", "full_set"],
                        confirmation_training_holdout_fraction=0.25, confirmation_split_seed=42)
    elif args.selection_policy == "guarded":
        protocol.update(primary_method="guarded", incumbent="structured", challenger="sign_worst")
    elif args.selection_policy == "local_refine":
        protocol.update(primary_method="local_robust", incumbent="guarded_incumbent",
                        comparators=["local_pooled", "local_drop", "validation_frontier"],
                        max_local_masks=8, minimum_improved_datasets=2,
                        practical_change_fraction_of_margin=0.1, paired_bootstrap_replicates=500,
                        paired_bootstrap_seed=42)
    _save(output / "protocol.json", protocol)

    def log(message):
        line = f"{datetime.now(MSK):%H:%M:%S} MSK | {message}"
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            print(line, file=handle)
        print(line, flush=True)

    log(f"run={output.name} | seed=42 | rounds={args.rounds} | datasets={','.join(args.datasets)}"
        f" | selection={args.selection_policy} | sampling allowance={args.sample_seconds}s"
        f" | per-dataset hard limit={args.dataset_seconds}s | reuse={args.reuse_sampling_run}")
    summary = []
    for dataset in args.datasets:
        if sources != selection_sources():
            raise RuntimeError("Selection code changed during this run; start a new run for the revised policy.")
        directory = output / dataset
        directory.mkdir()
        reuse, remaining = budgets[dataset]
        local_deadline = time.monotonic() + remaining if args.selection_policy == "local_refine" else None
        process = multiprocessing.get_context("spawn").Process(
            target=_worker_entry, args=(dataset, directory, args.rounds, args.sample_seconds, args.selection_policy, reuse, local_deadline))
        cutoff = datetime.now(MSK) + timedelta(seconds=remaining)
        log(f"start {dataset} | inherited sampling={args.dataset_seconds-remaining:.2f}s"
            f" | new work budget={remaining:.2f}s | hard stop by {cutoff:%H:%M:%S} MSK")
        outcome = supervise(process, directory, remaining, log, args.sample_seconds)
        process.close()
        outcome["inherited_sampling_s"] = args.dataset_seconds - remaining
        outcome["dataset_limit_s"] = args.dataset_seconds
        _save(directory / "watchdog.json", outcome)
        row = summary_row(dataset, outcome)
        summary.append(row)
        write_summary(output, summary)
        log(f"finish {dataset} | status={row['status']} | wall={row['wall_s']}s"
            f" | charged={row['wall_plus_reused_sampling_s']}s | removed={row['removed_count']}"
            f" | nominees={row['proposed_count']} | policy={row['selected_method']}"
            f" | full test={row['full_test_utility']} | retained test={row['retained_test_utility']}"
            f" | delta={row['test_delta']}")
    write_method_comparison(output, args.datasets)
    return int(any(row["status"] != "completed" for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
