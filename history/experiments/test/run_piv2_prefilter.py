"""Seed-42 cold-start paired benchmark of downstream selectors with and without PIv2."""

import argparse
import csv
import multiprocessing
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_piv0_benchmark import exclusive_run, read, sha
from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test.run_sampled_permutation_seed42 import metrics
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict


SEED = 42
LIMIT = 900
DATASETS = ("madelon", "spambase", "abalone", "gisette", "credit_card_fraud", "isolet")
METHODS = ("pfi", "f_score", "mutual_info", "tree_shap", "l1", "banzhaf", "exact", "mc32", "gain")
ADDITIONAL = ("spearman", "mrmr", "chi2", "variance", "random_forest", "extra_trees", "rfe_linear", "elastic_net")
SMALL_ONLY = {"exact", "mc32"}
OUTPUT = ROOT / "results/test/thesis_sampling"
STAGES = ("prefilter_s", "backend_s", "final_s")
NUMBERS = ("prefilter_retained", "retained", "fits", *STAGES, "validation_utility",
           "validation_secondary", "test_utility", "test_secondary")
SOURCES = (
    "src/minppi_mofs/piv2.py",
    "experiments/test/piv2_selectors.py",
    "experiments/test/piv2_extra_selectors.py",
    "experiments/test/run_piv2_prefilter.py",
    "experiments/test/run_sampled_permutation_seed42.py",
    "experiments/calibrate_icomp2026_models.py",
    "experiments/thesis_prepared_catalog.py",
    "experiments/run_thesis_sampling.py",
    "src/minppi_mofs/modeling.py", "src/minppi_mofs/utility.py",
)


def check_columns(columns, universe):
    columns = list(columns)
    selected = set(columns)
    if not selected or columns != [name for name in universe if name in selected]:
        raise ValueError("Selector must return a nonempty ordered subset of its input columns")
    return columns


def run_arm(source, row, policy, backend, arm, out, fingerprint):
    dataset, metric = row["dataset_id"], row["utility"]
    state = dict(fits=0)
    last_publish = 0.

    def progress(phase, **counts):
        nonlocal last_publish
        now = time.monotonic()
        changed = state.get("phase") != phase
        if changed:
            for field in ("checked", "candidates", "total", "eta_s"):
                state.pop(field, None)
            state["phase_started"] = now
        state.update(phase=phase, **counts)
        if changed or now - last_publish >= .5:
            atomic_write_json(out / "progress.json", state)
            last_publish = now

    progress("loading")
    train, yt, val, yv = load_prepared_train_validation(source, row)
    names = list(train.columns)
    candidates, narrow_train, narrow_val = names, train, val
    prefilter_s, votes = 0., None
    if arm == "piv2":
        progress("prefilter")
        start = time.monotonic()
        from minppi_mofs.piv2 import prefilter

        proposed, votes = prefilter(train, yt, task=policy["task"], seed=SEED)
        candidates = check_columns(proposed, names)
        if candidates != names:
            narrow_train, narrow_val = train[candidates], val[candidates]
        prefilter_s = time.monotonic() - start
    progress("backend", prefilter_s=prefilter_s, prefilter_retained=len(candidates))
    if backend in SMALL_ONLY and dataset != "abalone":
        # The selector can hit the watchdog before a final column set exists.
        # Keep the completed prefilter decision for comparison across backends.
        atomic_write_json(out / "backend_input.json", dict(candidates=candidates, prefilter=votes,
                          protocol_fingerprint=fingerprint))
    start = time.monotonic()
    if row.get("k_selection") == "validation_auc_all_prefixes":
        from experiments.test.piv2_auc_prefix import select
    elif backend in ADDITIONAL:
        from experiments.test.piv2_conventional import select
    elif backend in SMALL_ONLY and dataset != "abalone":
        from experiments.test.piv2_streaming_coalitions import select_coalitions as select
    else:
        from experiments.test.piv2_selectors import select
    selected, vp, fits, diagnostics = select(
        backend, narrow_train, yt, narrow_val, yv, policy, metric, progress,
        original_width=len(names), original_columns=names,
    )
    selected = check_columns(selected, candidates)
    validation, validation_secondary = metrics(yv, np.asarray(vp), metric)
    backend_s = time.monotonic() - start
    frozen = dict(dataset=dataset, seed=SEED, backend=backend, arm=arm, metric=metric,
                  protocol_fingerprint=fingerprint, model_policy=policy, features=names,
                  candidates=candidates, selected=selected, prefilter=votes, selector=diagnostics,
                  partition_sha256={s: row[f"{s}_file_sha256"] for s in ("train", "val", "test")},
                  validation_utility=validation, validation_secondary=validation_secondary)
    identity = canonical_hash(frozen)
    atomic_write_json(out / "frozen.json", frozen)
    progress("final_model", retained=len(selected), fits=fits, backend_s=backend_s)
    start = time.monotonic()
    paths = PreparedSplitPaths(dataset, SEED, row["target"], *(source/row[f"{s}_relpath"] for s in ("train","val","test")))
    job = JobSpec(identity, row["protocol_hash"], "removal", dataset, SEED, backend+"_"+arm,
                  {"columns": selected}, (), "result.json")
    audit = AuditLog.from_manifest(out / "test_access.json", [job])
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=audit).load_test(
        reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
    if list(test.columns) != names or len(ye) != row["test_row_count"]:
        raise ValueError("Unexpected held-out partition")
    prediction = fit_predict(pd.concat([train[selected], val[selected]], ignore_index=True),
                             np.concatenate([yt, yv]), test[selected], **policy)
    test_primary, test_secondary = metrics(ye, prediction, metric)
    final_s = time.monotonic() - start
    np.savez_compressed(out/"predictions.npz", validation_y=yv, validation=vp, y=ye, selected=prediction)
    atomic_write_json(out/"result.json", dict(
        dataset=dataset, arm=arm, backend=backend, seed=SEED, metric=metric, width=len(names),
        prefilter_retained=len(candidates), retained=len(selected), fits=fits+1, prefilter_s=prefilter_s,
        backend_s=backend_s, final_s=final_s, validation_utility=validation, validation_secondary=validation_secondary,
        test_utility=test_primary, test_secondary=test_secondary, frozen_hash=identity,
    ))
    progress("complete", fits=fits+1, final_s=final_s)


def worker(source, row, policy, backend, arm, out, fingerprint):
    try:
        run_arm(source, row, policy, backend, arm, out, fingerprint)
    except Exception:
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def heartbeat(label, state, elapsed, remaining):
    checked = state.get("checked", 0)
    total = state.get("total", state.get("candidates", 0))
    eta = state.get("eta_s")
    if eta is None and 0 < checked <= total:
        stage_time = time.monotonic() - state["phase_started"]
        eta = stage_time * (total - checked) / checked
    forecast = f"{eta:.1f}s" if eta is not None else "pending"
    return (f"{label} {state.get('phase', 'starting')} {checked}/{total or '?'} "
            f"fits={state.get('fits', 0)} elapsed={elapsed:.1f}s stage_ETA={forecast} deadline_in={remaining:.0f}s")


def schedule(datasets, methods, extended=False):
    registry = (*ADDITIONAL, "mc32", "exact") if extended else METHODS
    for method_index, backend in enumerate(registry):
        if backend not in methods:
            continue
        for data_index, dataset in enumerate(DATASETS):
            if dataset not in datasets or (backend in SMALL_ONLY and (dataset == "abalone") == extended):
                continue
            arms = ("plain", "piv2") if (method_index+data_index) % 2 == 0 else ("piv2", "plain")
            for arm in arms:
                yield dataset, backend, arm


def campaign(root, datasets, methods, *, extended=False):
    jobs, catalog = audit_prepared_catalog()
    tasks = list(schedule(datasets, methods, extended))
    if not tasks:
        raise ValueError("The requested methods have no eligible datasets")
    registry = (*ADDITIONAL, "mc32", "exact") if extended else METHODS
    all_tasks = list(schedule(DATASETS, registry, extended))
    sources = SOURCES
    if extended:
        sources += ("experiments/test/piv2_conventional.py", "experiments/test/piv2_conventional_filters.py",
                    "experiments/test/piv2_conventional_models.py", "experiments/test/piv2_streaming_coalitions.py")
    protocol = dict(version="PIv2-test", seed=SEED, tasks=[list(t) for t in all_tasks], arm_limit_s=LIMIT,
                    catalog_sha256=catalog["sha256"], model_params=catalog["shared"]["model_params"],
                    hardware_backend=catalog["shared"]["backend"], cold_start=True,
                    source_sha256={name: sha(ROOT/name) for name in sources},
                    comparison="Same backend and frozen predictor; PIv2 sees training split only. "
                               "Where a fixed k is used, both arms derive it from original width. "
                               "All arms freeze selected columns before test access. No work shared between arms.",
                    timing="Outer wall includes worker startup, data checks, prefilter, selector, "
                           "final refit and test prediction, artifact writes and process exit.")
    protocol_path = root / "protocol.json"
    if extended:
        protocol["extension"] = ("Eight additional training-only selectors at up to 4096 rows; "
                                 "exact and 32-permutation retraining on full train and validation partitions "
                                 "for the five remaining datasets. Abalone comes from the original paired run. "
                                 "The 900-second wall limit applies separately to each cold arm.")
    if protocol_path.exists():
        if protocol != read(protocol_path):
            raise ValueError("Cannot resume with different source code, data, settings, or arm order")
    else:
        if any(p.name != "campaign.lock" for p in root.iterdir()):
            raise ValueError(f"Choose an empty test output directory: {root}")
        atomic_write_json(protocol_path, protocol)
    for name in sources:
        copy = root / "source" / name
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, copy)
        if sha(copy) != protocol["source_sha256"][name]:
            raise ValueError(f"Invalid benchmark source snapshot: {copy}")
    rows = {}
    for dataset in datasets:
        with (jobs[dataset, SEED]/"data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            row, = (r for r in csv.DictReader(stream) if r["dataset_id"]==dataset and int(r["seed"])==SEED)
        for split in ("train", "val", "test"):
            row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
        rows[dataset] = row
    fingerprint = canonical_hash(protocol)
    return run_campaign_tasks(root, tasks, all_tasks, jobs, rows, protocol, fingerprint)


def run_campaign_tasks(root, tasks, all_tasks, jobs, rows, protocol, fingerprint):
    """Run cold arms on the supplied prepared partitions, preserving completed arms on resume."""
    old = read(root/"summary.json") if (root/"summary.json").exists() else []
    records = {(record["dataset"], record["backend"], record["arm"]): record for record in old}
    with (root/"run.log").open("a", encoding="utf-8", buffering=1) as stream:
        def log(message):
            line = f"[{datetime.now(MSK):%H:%M:%S}] {message}"
            print(line, flush=True)
            print(line, file=stream, flush=True)

        log(f"PIv2 paired test {root} | {len(tasks)} cold arms | {LIMIT}s/arm")
        for task_index, (dataset, backend, arm) in enumerate(tasks, start=1):
            row = rows[dataset]
            label = f"{dataset}/{backend}/{arm}"
            folder = root / label
            policy = dict(task=row["task"], model_params=protocol["model_params"],
                          backend=protocol["hardware_backend"], seed=SEED)
            if (folder/"watchdog.json").exists():
                timer = read(folder/"watchdog.json")
            else:
                if folder.exists():
                    folder.rename(folder.with_name(arm + ".interrupted-" + datetime.now(MSK).strftime("%Y%m%d-%H%M%S-%f")))
                folder.mkdir(parents=True)
                log(f"{label} started ({task_index}/{len(tasks)})")
                proc = multiprocessing.get_context("spawn").Process(
                    target=worker, args=(jobs[dataset, SEED], row, policy, backend, arm, folder, fingerprint))
                try:
                    timer = supervise(proc, folder, LIMIT, log, heartbeat_s=10.,
                                      heartbeat_formatter=lambda p,t,left: heartbeat(label,p,t,left))
                finally:
                    proc.close()
            data = timer["progress"]
            if timer["status"] == "completed":
                result = read(folder/"result.json")
                frozen = read(folder/"frozen.json")
                if (timer["result"] != result or canonical_hash(frozen) != result["frozen_hash"]
                        or frozen["protocol_fingerprint"] != fingerprint):
                    raise ValueError(f"Invalid completed arm: {folder}")
                data = result
            records[dataset, backend, arm] = dict(
                dataset=dataset, seed=SEED, backend=backend, arm=arm, status=timer["status"],
                metric=row["utility"], secondary_metric="mae" if row["task"]=="regression" else "accuracy",
                width=int(row["selected_feature_count"]), wall_s=timer["elapsed_s"],
                **{key:data.get(key) for key in NUMBERS}, phase=timer["progress"].get("phase"))
            write_summary(root, [records[task] for task in all_tasks if task in records])
            log(f"{label} {timer['status']} | elapsed={timer['elapsed_s']:.2f}s PI={data.get('prefilter_s')}s "
                f"retained={data.get('retained')} test={data.get('test_utility')} | {len(records)}/{len(all_tasks)}")
            if timer["status"] == "error":
                log(f"Inspect {folder/'error.log'}")
    return int(any(records[t]["status"] != "completed" for t in tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Use or resume this test output directory")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--extended", action="store_true", help="Additional conventional and full-partition coalition comparisons")
    parser.add_argument("--methods", nargs="+", choices=(*METHODS,*ADDITIONAL))
    args = parser.parse_args()
    root = (args.run or OUTPUT / datetime.now(MSK).strftime("PIv2-seed42-%Y%m%d-%H%M%S-%f")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        registry = (*ADDITIONAL, "mc32", "exact") if args.extended else METHODS
        if args.methods and not set(args.methods).issubset(registry):
            parser.error("Methods must belong to the selected suite")
        return campaign(root, args.datasets, args.methods or registry, extended=args.extended)


if __name__ == "__main__":
    raise SystemExit(main())
