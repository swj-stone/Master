"""Run the six frozen seed-42 benchmark partitions with one deadline per dataset."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
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


def _worker_entry(dataset, directory, rounds, sample_seconds):
    # XGBoost fits in this spawned process using native threads. Killing this
    # process interrupts a native fit without waiting for a Python callback.
    with (directory / "worker.log").open("w", encoding="utf-8", buffering=1) as worker_log:
        os.dup2(worker_log.fileno(), 1)
        os.dup2(worker_log.fileno(), 2)
        sys.argv = ["thesis_sampling_worker", "--dataset", dataset, "--output-dir", str(directory),
                    "--rounds", str(rounds), "--sample-seconds", str(sample_seconds)]
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
            log(f"{directory.name} phase={p.get('phase', 'starting')} | elapsed={elapsed:.0f}s"
                f" | coverage={p.get('coverage', 0)}/{p.get('n_features', '?')}"
                f" | sample_fits={p.get('sample_fits', 0)} | skipped={p.get('skipped_after', 0)}"
                f" | witnesses={p.get('proposed_count', 0)} | sampling ETA={eta_text}"
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
                "retained_test_utility", "test_delta", "validation_fits", "test_fits", "completed_rounds"):
        row[key] = data.get(key)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dataset-seconds", type=int, default=900)
    parser.add_argument("--sample-seconds", type=int, default=780)
    args = parser.parse_args()
    if (not 1 <= args.dataset_seconds <= 900 or not 0 < args.sample_seconds < args.dataset_seconds
            or args.rounds < 1 or len(set(args.datasets)) != len(args.datasets)):
        parser.error("Use distinct datasets, positive rounds, and 0 < sample-seconds < dataset-seconds <= 900.")
    output = ROOT / "results/thesis_sampling" / datetime.now(MSK).strftime("seed42-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)

    def log(message):
        line = f"{datetime.now(MSK):%H:%M:%S} MSK | {message}"
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            print(line, file=handle)
        print(line, flush=True)

    log(f"run={output.name} | seed=42 | rounds={args.rounds} | datasets={','.join(args.datasets)}"
        f" | sampling allowance={args.sample_seconds}s | per-dataset hard limit={args.dataset_seconds}s")
    summary = []
    for dataset in args.datasets:
        directory = output / dataset
        directory.mkdir()
        process = multiprocessing.get_context("spawn").Process(
            target=_worker_entry, args=(dataset, directory, args.rounds, args.sample_seconds))
        cutoff = datetime.now(MSK) + timedelta(seconds=args.dataset_seconds)
        log(f"start {dataset} | hard stop by {cutoff:%H:%M:%S} MSK")
        outcome = supervise(process, directory, args.dataset_seconds, log, args.sample_seconds)
        process.close()
        row = summary_row(dataset, outcome)
        summary.append(row)
        write_summary(output, summary)
        log(f"finish {dataset} | status={row['status']} | wall={row['wall_s']}s | removed={row['removed_count']}"
            f" | full test={row['full_test_utility']} | retained test={row['retained_test_utility']}"
            f" | delta={row['test_delta']}")
    return int(any(row["status"] != "completed" for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
