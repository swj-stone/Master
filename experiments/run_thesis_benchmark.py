"""Run the guarded MinPI and validation-selected boosting-round benchmark."""

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from experiments.run_thesis_sampling import MSK, ROOT, _save, read_progress, supervise, write_summary
from experiments.thesis_benchmark_summary import write_dataset_summary
from experiments.thesis_prepared_catalog import CAMPAIGN, FORMAL_ROOT, audit_prepared_catalog
from minppi_mofs.experiment_protocol import canonical_hash


POLICY = "guarded_rounds"
FORMAL_OUTPUT = ROOT / "results/thesis_benchmark"
TEST_OUTPUT = ROOT / "results/test/thesis_benchmark"
SCIENCE_FILES = (
    "experiments/run_thesis_benchmark.py", "experiments/run_thesis_sampling.py",
    "experiments/thesis_prepared_catalog.py", "experiments/thesis_sampling_worker.py", "experiments/thesis_fixed_predictor.py",
    "experiments/calibrate_icomp2026_models.py", "src/minppi_mofs/modeling.py",
    "src/minppi_mofs/minimum_sampling.py", "src/minppi_mofs/removal_selection.py",
    "src/minppi_mofs/sign_aware_removal.py", "src/minppi_mofs/guarded_update.py",
    "src/minppi_mofs/baselines/shap_top_k.py", "src/minppi_mofs/utility.py", "src/minppi_mofs/experiment_protocol.py",
)


def _code_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SCIENCE_FILES}


def _pinned_metadata(root, audit):
    pins = {root / CAMPAIGN / "campaign_plan.json": audit["campaign_plan_sha256"]}
    for shard, metadata in audit["shards"].items():
        for filename, field in (("data_manifest.csv", "manifest_sha256"), ("resolved_protocol.yaml", "protocol_file_sha256"),
                                ("metadata_complete.json", "metadata_complete_sha256")):
            pins[root / shard / filename] = metadata[field]
    return pins


def _changed_file(pins):
    for path, expected in pins.items():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            return f"{path}: {error}"
        if digest != expected:
            return str(path)
    return None


def _worker(dataset, seed, source, directory, rounds, sample_seconds):
    with (directory / "worker.log").open("w", encoding="utf-8", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                from experiments.thesis_sampling_worker import run

                run(dataset, directory, rounds, sample_seconds, selection_policy=POLICY, seed=seed, source=source)
            except Exception:
                traceback.print_exc(file=stream)
                raise SystemExit(1)


def _clock(seconds):
    minutes, sec = divmod(round(max(seconds, 0)), 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02}:{minute:02}:{sec:02}"


def _batch_eta(pending, rows, current_elapsed, limit):
    """A dataset-specific estimate becomes available once each pending dataset has been observed."""
    if not pending:
        return "batch remaining=00:00:00"
    observations = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            observations[row["dataset"]].append(row["wall_s"])
    bound = ""
    if limit is not None:
        upper = max(len(pending) * limit - current_elapsed, 0.)
        bound = f" | remaining job-budget upper bound={_clock(upper)}"
    if any(ds not in observations for ds, _ in pending):
        return f"batch ETA=pending{bound}"
    predicted = [sum(observations[ds]) / len(observations[ds]) for ds, _ in pending]
    remaining = max(predicted[0] - current_elapsed, 0.) + sum(predicted[1:])
    return f"batch ETA=~{_clock(remaining)}{bound}"


def _heartbeat(progress, elapsed, remaining, sample_seconds, rounds):
    phase = progress.get("phase", "starting")
    status = [f"phase={phase}", f"elapsed={_clock(elapsed)}"]
    if remaining is not None:
        status.append(f"deadline in {_clock(remaining)}")
    if phase == "sampling":
        status.extend([f"round={progress.get('round', 0)}/{rounds}",
                       f"coverage={progress.get('coverage', 0)}/{progress.get('n_features', '?')}",
                       f"resolved={progress.get('resolved_trials', 0)}/{progress.get('planned_trials', '?')}",
                       f"fits={progress.get('sample_fits', 0)}", f"witnesses={progress.get('witness_count', 0)}"])
        eta = progress.get("sampling_eta_s")
        if eta is not None:
            if sample_seconds is not None:
                eta = min(eta, sample_seconds - progress.get("sample_elapsed_s", 0.), remaining)
            status.append(f"sampling ETA=~{_clock(eta)}")
    elif phase.startswith("predictor_"):
        status.append(f"fits={progress.get('predictor_fits_done', 0)}/{progress.get('predictor_fits_planned', '?')}")
        status.append(f"role={progress.get('predictor_role', 'pending')}")
    elif phase == "validation_gate":
        status.append(f"masks checked={len(progress.get('selection_trials', []))}")
    return " | ".join(status)


def _summary_row(dataset, seed, outcome, mode):
    progress = outcome["progress"]
    result = outcome["result"] if outcome["status"] == "completed" else {}
    data = {**progress, **result}
    utilities = result.get("method_test_utilities", {})
    decisions = result.get("predictor_decisions", {})

    def delta(new, old):
        return utilities[new] - utilities[old] if new in utilities and old in utilities else None

    n = data.get("n_features")
    removed = data.get("removed_count")
    row = dict(dataset=dataset, seed=seed, scope=mode, status=outcome["status"],
               wall_s=round(outcome["elapsed_s"], 3), phase=data.get("failed_phase") or data.get("phase", "starting"),
               metric=data.get("policy", {}).get("utility"), n_features=n, removed_count=removed,
               retained_count=result.get("retained_count"), compression_fraction=removed/n if n and removed is not None else None,
               sampling_s=data.get("sample_elapsed_s"), sampling_stop=data.get("sampling_stop"),
               planned_trials=data.get("planned_trials"), resolved_trials=data.get("resolved_trials"),
               sampled_draws=data.get("sampled_draws"),
               coverage=data.get("coverage"), completed_rounds=data.get("completed_rounds"), sample_fits=data.get("sample_fits"),
               witnesses=data.get("witness_count"), nominees=data.get("proposed_count"),
               selection_delta=delta("guarded_original", "full_original"), predictor_delta=delta("guarded_chosen", "guarded_original"),
               net_delta=delta("guarded_chosen", "full_original"), matched_full_delta=delta("guarded_chosen", "full_chosen"),
               error=outcome.get("error") or data.get("error"))
    for role in ("full", "guarded"):
        row[f"{role}_recipe"] = None
        if role in decisions:
            row[f"{role}_recipe"] = "extended_average" if decisions[role]["adopt_blend"] else "original"
        for recipe in ("original", "chosen"):
            method = f"{role}_{recipe}"
            row[f"{method}_test"] = utilities.get(method)
    return row


def _configuration_line(config):
    catalog = config["prepared"]
    shared = catalog["shared"]
    settings = json.dumps(shared["model_params"], sort_keys=True, ensure_ascii=False)
    preparation = json.dumps(shared["preprocessing"], sort_keys=True, ensure_ascii=False)
    return (f"config audit=PASS | catalog jobs={len(catalog['jobs'])} | seed set={catalog['seeds']}"
            f" | backend={shared['backend']} | model={settings} | alpha={shared['alpha']}"
            f" | margins (classification/regression)={shared['classification_margin']}/{shared['regression_margin']}"
            f" | predictor checkpoints={shared['checkpoints']} | preprocessing={preparation}"
            f" | catalog SHA256={catalog['sha256']}")


def _formal_completion_error(result, dataset, seed, rounds):
    if (result.get("dataset"), result.get("seed"), result.get("run_mode"), result.get("phase")) != (
            dataset, seed, "formal", "complete"):
        return "Worker did not return a completed formal result for this dataset and seed"
    n = result.get("n_features", 0)
    expected = rounds * n
    stop = result.get("sampling_stop")
    finished = ((stop == "round_limit" and result.get("completed_rounds") == rounds)
                or (stop == "all_witnessed" and result.get("witness_count") == n))
    if (not finished or n <= 0 or result.get("sample_seconds") is not None
            or result.get("planned_trials") != expected or result.get("resolved_trials") != expected):
        return "Worker did not resolve the entire formal sampling plan"
    scores = result.get("method_test_utilities", {})
    decisions = result.get("predictor_decisions", {})
    for role in ("full", "guarded"):
        if not isinstance(decisions.get(role, {}).get("adopt_blend"), bool):
            return f"Worker did not provide the validation decision for {role}"
        for variant in ("original", "blend", "chosen"):
            method = f"{role}_{variant}"
            if not isinstance(scores.get(method), (int, float)) or not math.isfinite(scores[method]):
                return f"Worker did not provide a finite held-out score for {method}"
    if result.get("selected_method") != "guarded_chosen":
        return "Worker did not report the selected guarded predictor"
    return None


def _run(jobs, output_root, rounds, sample_seconds, limit, config, pins):
    code = _code_hashes()
    verified_inputs = {**pins, **{ROOT / name: digest for name, digest in code.items()}}
    mode = config["experiment"]["mode"]
    prefix = "test-guarded-rounds" if mode == "test" else "guarded-rounds-formal"
    run = output_root / datetime.now(MSK).strftime(f"{prefix}-%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True, exist_ok=False)
    _save(run / "config_audit.json", config)
    _save(run / "protocol.json", dict(mode=mode, selection_policy=POLICY, predictor_recipe="rounds", rounds=rounds,
          sample_seconds=sample_seconds, dataset_seconds=limit, heartbeat_seconds=10,
          jobs=[dict(dataset=ds, seed=seed, source=str(source)) for (ds, seed), source in jobs.items()],
          catalog_sha256=config["prepared"]["sha256"], config_sha256=config["sha256"], source_sha256=code))
    order = list(jobs)
    rows = []
    started = time.monotonic()

    def log(message, job_directory=None):
        line = f"{datetime.now(MSK):%Y-%m-%d %H:%M:%S} MSK | {message}"
        with (run / "run.log").open("a", encoding="utf-8") as stream:
            print(line, file=stream)
        if job_directory is not None:
            with (job_directory / "job.log").open("a", encoding="utf-8") as stream:
                print(line, file=stream)
        print(line, flush=True)

    log(_configuration_line(config))
    timing = "time limits=none" if limit is None else f"sample <= {sample_seconds}s | per-job hard limit={limit}s"
    log(f"{len(jobs)} jobs | mode={mode} | {POLICY} | rounds={rounds} | {timing} | log={run / 'run.log'}")
    for index, ((ds, seed), source) in enumerate(jobs.items()):
        changed = _changed_file(verified_inputs)
        if changed:
            log(f"Stopped before dataset={ds} seed={seed}: frozen configuration or source changed: {changed}")
            return 1
        folder = run / f"seed-{seed}" / ds
        folder.mkdir(parents=True)
        prefix = f"[{index + 1}/{len(jobs)}] dataset={ds} seed={seed}"
        job_start = time.monotonic()

        def job_log(message):
            eta = _batch_eta(order[index:], rows, time.monotonic() - job_start, limit)
            log(f"{prefix} | {message} | batch elapsed={_clock(time.monotonic()-started)} | {eta}", folder)

        job_log("start" if limit is None else f"start | deadline={limit}s")
        process = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(ds, seed, source, folder, rounds, sample_seconds))
        try:
            outcome = supervise(process, folder, limit, job_log, sample_seconds,
                                heartbeat_formatter=lambda p, t, left: _heartbeat(p, t, left, sample_seconds, rounds))
        except KeyboardInterrupt:
            if process.is_alive():
                process.kill()
                process.join()
            outcome = dict(status="interrupted", elapsed_s=time.monotonic()-job_start, limit_s=limit,
                           exit_code=process.exitcode, progress=read_progress(folder, log=job_log), result={})
            _save(folder / "watchdog.json", outcome)
        finally:
            process.close()
        if mode == "formal" and outcome["status"] == "completed":
            error = _formal_completion_error(outcome["result"], ds, seed, rounds)
            if error:
                outcome.update(status="error", error=error, result={})
                _save(folder / "watchdog.json", outcome)
        row = _summary_row(ds, seed, outcome, mode)
        rows.append(row)
        write_summary(run, rows)
        write_dataset_summary(run, rows, order)
        eta = _batch_eta(order[index + 1:], rows, 0., limit)
        log(f"{prefix} | {row['status']} | phase={row['phase']} | wall={row['wall_s']:.3f}s"
            f" | removed={row['removed_count']}/{row['n_features']} | guarded test={row['guarded_chosen_test']}"
            f" | {eta}" + (f" | error={row['error']}" if row["error"] else ""), folder)
        if row["status"] == "interrupted":
            return 130
    completed = sum(row["status"] == "completed" for row in rows)
    log(f"finished | completed={completed}/{len(jobs)} | total wall={_clock(time.monotonic()-started)} | summary={run/'summary.csv'}")
    return int(completed != len(jobs))


def main(argv=None, *, test_mode=False):
    mode = "test" if test_mode else "formal"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Validate the frozen input catalog and display jobs without training.")
    parser.add_argument("--prepared-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=TEST_OUTPUT if test_mode else FORMAL_OUTPUT)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--rounds", type=int, default=5)
    if test_mode:
        parser.add_argument("--sample-seconds", type=int, default=780)
        parser.add_argument("--dataset-seconds", type=int, default=900)
    else:
        parser.set_defaults(sample_seconds=None, dataset_seconds=None)
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive.")
    if test_mode and not 0 < args.sample_seconds < args.dataset_seconds <= 900:
        parser.error("Test runs require 0 < sample-seconds < dataset-seconds <= 900.")
    output_root = args.output_root.resolve()
    if output_root.is_relative_to((ROOT.parent / "Master").resolve()):
        parser.error("Use a thesis output directory outside the frozen conference checkout.")
    other = FORMAL_OUTPUT if test_mode else ROOT / "results/test"
    if output_root.is_relative_to(other.resolve()):
        parser.error(f"The {mode} entry cannot write into the other mode's result directory.")
    root = args.prepared_root.resolve()
    try:
        catalog, audit = audit_prepared_catalog(root)
        pins = _pinned_metadata(root, audit)
        changed = _changed_file(pins)
        if changed:
            raise ValueError(f"Archive metadata changed while verifying {changed}")
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"Prepared configuration audit failed: {error}")
    for name, values, axis in (("dataset", args.datasets, 0), ("seed", args.seeds, 1)):
        if values is not None and (len(set(values)) != len(values) or set(values) - {pair[axis] for pair in catalog}):
            parser.error(f"Use distinct registered {name}s from {sorted({pair[axis] for pair in catalog})}.")
    jobs = {(ds, seed): source for (ds, seed), source in catalog.items()
            if (not args.datasets or ds in args.datasets) and (not args.seeds or seed in args.seeds)}
    if not jobs:
        parser.error("No matching prepared jobs.")
    config = {"status": "PASS", "prepared": audit, "experiment": dict(mode=mode, selection_policy=POLICY, predictor_recipe="rounds",
        balanced_rounds=args.rounds, sampling_seconds=args.sample_seconds, job_limit_seconds=args.dataset_seconds,
        heartbeat_seconds=10, jobs_requested=len(jobs), seeds_requested=sorted({s for _,s in jobs}),
        datasets_requested=sorted({ds for ds,_ in jobs}))}
    config["sha256"] = canonical_hash(config)
    if args.plan:
        print(_configuration_line(config))
        timing = (f"sample <= {args.sample_seconds}s | hard limit={args.dataset_seconds}s/job"
                  f" | total job-budget upper bound={_clock(len(jobs)*args.dataset_seconds)}"
                  if test_mode else "time limits=none")
        print(f"{len(jobs)} jobs | mode={mode} | guarded MinPI + validation-selected rounds | sample rounds={args.rounds}"
              f" | {timing}")
        for i, ((ds, seed), source) in enumerate(jobs.items(), 1):
            print(f"[{i}/{len(jobs)}] dataset={ds} seed={seed} | {source}")
        return 0
    return _run(jobs, output_root, args.rounds, args.sample_seconds, args.dataset_seconds, config, pins)


if __name__ == "__main__":
    raise SystemExit(main())
