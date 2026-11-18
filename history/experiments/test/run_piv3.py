"""Madelon-500 paired cold starts for PIv3 and the previous adaptive deletion search."""

import argparse
from datetime import datetime
import multiprocessing
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_piv0_benchmark import exclusive_run, read, sha
from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test.piv2_madelon500_data import prepare_run
from experiments.test.run_sampled_permutation_seed42 import metrics
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict


OUTPUT = ROOT / "results/test/thesis_sampling"
HISTORY = OUTPUT / "adaptive-pruning-validation-20261113-175433-248"
SEEDS = (11, 23, 47)
TOLERANCES = (.01, .05)
BUDGET = 12
LIMIT = 900
SOURCES = (
    "src/minppi_mofs/piv3.py", "src/minppi_mofs/piv2.py", "src/minppi_mofs/modeling.py",
    "src/minppi_mofs/utility.py", "src/minppi_mofs/experiment_protocol.py",
    "experiments/test/adaptive_pruning_original.py", "experiments/test/run_piv3.py",
    "experiments/test/piv2_madelon500_data.py", "experiments/test/thesis_madelon_alpha_data.py",
    "experiments/test/run_sampled_permutation_seed42.py", "experiments/calibrate_icomp2026_models.py",
    "experiments/run_thesis_sampling.py",
)


def search_arm(source, row, policy, method, seed, tolerance, folder, fingerprint):
    if method == "piv3":
        from minppi_mofs.piv3 import search
    else:
        from experiments.test.adaptive_pruning_original import search
    fits = 0

    def progress(phase, **fields):
        atomic_write_json(folder/"progress.json", dict(phase=phase, fits=fits, **fields))

    progress("loading")
    train, yt, val, yv = load_prepared_train_validation(source, row)
    columns = train.columns.tolist()
    if len(columns) != 500 or len(yt) != 1664 or len(yv) != 416:
        raise ValueError("Unexpected Madelon split")
    progress("reference")
    reference_pred = fit_predict(train, yt, val, **policy)
    fits += 1
    reference = list(metrics(yv, reference_pred, "roc_auc"))
    predictions = {}

    def evaluate(candidate):
        nonlocal fits
        key = tuple(candidate)
        membership = set(key)
        if (not key or len(predictions) >= BUDGET or key in predictions or list(key) == columns or
                list(key) != [f for f in columns if f in membership]):
            raise ValueError("Invalid or repeated candidate, or candidate budget exhausted")
        prob = fit_predict(train[list(key)], yt, val[list(key)], **policy)
        fits += 1
        score = metrics(yv, prob, "roc_auc")
        predictions[key] = prob
        progress("candidate", checked=len(predictions), k=len(key), validation=list(score))
        return np.asarray(score)

    selected, diagnostics = search(train, yt, search_seed=seed, tolerance=tolerance,
                                   reference_scores=reference, evaluate=evaluate, budget=BUDGET)
    keys = [tuple(s["proposed"]) for s in diagnostics["steps"]]
    if keys != list(predictions) or not 0 <= len(keys) <= BUDGET:
        raise ValueError("Search trace and actual evaluated subsets disagree")
    validation_pred = reference_pred if selected == columns else predictions[tuple(selected)]
    validation = list(metrics(yv, validation_pred, "roc_auc"))
    if np.any(np.asarray(validation) + 1e-12 < np.asarray(reference)*(1-tolerance)):
        raise ValueError("Selected columns fail the validation constraint")
    frozen = dict(method=method, dataset="madelon", data_seed=42, predictor_seed=42, search_seed=seed,
                  tolerance=tolerance, model_policy=policy, columns=columns, selected=selected,
                  reference_scores=reference, validation_scores=validation, diagnostics=diagnostics,
                  protocol_fingerprint=fingerprint,
                  partition_sha256={s:row[f"{s}_file_sha256"] for s in ("train","val","test")})
    identity = canonical_hash(frozen)
    atomic_write_json(folder/"frozen.json", frozen)
    paths = PreparedSplitPaths("madelon", 42, row["target"],
                               *(source/row[s+"_relpath"] for s in ("train","val","test")))
    job = JobSpec(identity, row["protocol_hash"], "removal", "madelon", 42, method,
                  {"search_seed":seed, "tolerance":tolerance, "selected":selected}, (), "result.json")
    access = AuditLog.from_manifest(folder/"test_access.json", [job])
    progress("final_evaluation")
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=access).load_test(
        reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
    if list(test.columns) != columns or len(ye) != row["test_row_count"]:
        raise ValueError("Unexpected test split")
    combined = pd.concat([train,val], ignore_index=True)
    targets = np.concatenate([yt,yv])
    test_pred = fit_predict(combined[selected], targets, test[selected], **policy)
    test_full = fit_predict(combined, targets, test, **policy)
    fits += 2
    candidates = (np.stack(list(predictions.values())) if predictions else
                  np.empty((0, *reference_pred.shape), dtype=reference_pred.dtype))
    np.savez_compressed(folder/"predictions.npz", validation_y=yv, validation=validation_pred,
                        validation_full=reference_pred, candidate_validation=candidates,
                        y=ye, selected=test_pred, full=test_full)
    test_scores = list(metrics(ye, test_pred, "roc_auc"))
    test_reference = list(metrics(ye, test_full, "roc_auc"))
    result = dict(method=method, search_seed=seed, tolerance=tolerance, k=len(selected), fits=fits,
                  candidates=len(keys), frozen_hash=identity, reference_scores=reference,
                  validation_scores=validation, test_reference_scores=test_reference, test_scores=test_scores,
                  validation_pass=True, test_pass=bool(np.all(np.asarray(test_scores)+1e-12 >=
                                                               np.asarray(test_reference)*(1-tolerance))))
    atomic_write_json(folder/"result.json",result)
    progress("complete", checked=len(keys), k=len(selected))


def worker(*args):
    try:
        search_arm(*args)
    except Exception:
        (args[-2]/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run(root):
    prep_start = time.monotonic()
    source,row = prepare_run(root)
    setup_file = root/"input_preparation.json"
    if not setup_file.exists():
        atomic_write_json(setup_file, dict(elapsed_s=time.monotonic()-prep_start))
    data = read(source/"provenance.json")
    policy = {k:data["policy"][k] for k in ("seed","model_params","task","backend")}
    if policy["seed"] != 42 or row["selected_feature_count"] != 500:
        raise ValueError("Unexpected predictor or input")
    tasks = [(m,t,s) for j,t in enumerate(TOLERANCES) for i,s in enumerate(SEEDS)
             for m in (("old","piv3") if (i+j)%2==0 else ("piv3","old"))]
    hashes = {s:row[s+"_file_sha256"] for s in ("train","val","test")}
    protocol = dict(version="PIv3", dataset="madelon", width=500, data_seed=42, predictor_seed=42,
                    search_seeds=list(SEEDS), tolerances=list(TOLERANCES), candidate_budget=BUDGET, arm_limit_s=LIMIT,
                    tasks=[list(t) for t in tasks], methods=["old","piv3"], model_policy=policy,
                    input_sha256=sha(source/"provenance.json"), catalog_sha256=row["protocol_hash"],
                    input_partition_sha256=hashes, data_sha256=hashes,
                    source_sha256={p:sha(ROOT/p) for p in SOURCES}, old_probe_sha256=sha(HISTORY/"probe.py"),
                    admission="Both full-validation metrics >= (1-tolerance) * the respective full model metric.",
                    test="Only after the method's support is frozen. The selected and 500-column models are "
                         "independently trained on train+validation and evaluated on the same held-out rows.",
                    timing="Separate cold process per arm; includes reference, search, final selected predictor "
                           "and independent final full predictor, I/O and startup; dataset preparation is shared.")
    path = root/"protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Cannot resume a different protocol")
    if not path.exists():
        atomic_write_json(path, protocol)
    for name,digest in protocol["source_sha256"].items():
        dest = root/"source"/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        if sha(dest) != digest:
            raise ValueError(f"Invalid source snapshot: {dest}")
    fingerprint = canonical_hash(protocol)
    records=[]
    with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(message):
            line = f"[{datetime.now(MSK):%H:%M:%S}] {message}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        for method,tol,seed in tasks:
            folder = root/method/f"tol-{round(100*tol):02d}"/f"s{seed}"
            if not (folder/"watchdog.json").exists():
                if folder.exists():
                    folder.rename(folder.with_name(folder.name+".interrupted-"+datetime.now(MSK).strftime("%H%M%S%f")))
                folder.mkdir(parents=True)
                log(f"{method} tolerance={tol:.0%} search_seed={seed} started ({len(records)+1}/{len(tasks)})")
                proc = multiprocessing.get_context("spawn").Process(
                    target=worker,args=(source,row,policy,method,seed,tol,folder,fingerprint))
                try:
                    timer=supervise(proc,folder,LIMIT,log,heartbeat_s=5.,
                        heartbeat_formatter=lambda p,t,left:f"{p.get('phase','starting')} fits={p.get('fits',0)} "
                        f"candidates={p.get('checked',0)}/{BUDGET} elapsed={t:.1f}s deadline_in={left:.0f}s")
                finally:
                    proc.close()
            else:
                timer=read(folder/"watchdog.json")
            result=timer["result"]
            if timer["status"]=="completed" and (result!=read(folder/"result.json") or
                    canonical_hash(read(folder/"frozen.json"))!=result["frozen_hash"]):
                raise ValueError(f"Inconsistent result {folder}")
            records.append(dict(method=method, tolerance=tol, search_seed=seed, status=timer["status"],
                                wall_s=timer["elapsed_s"], k=result.get("k"), fits=result.get("fits"),
                                test_pass=result.get("test_pass")))
            write_summary(root,records)
            log(f"{method} {timer['status']} wall={timer['elapsed_s']:.2f}s k={result.get('k')} "
                f"fits={result.get('fits')} test={result.get('test_scores')}")
            if timer["status"]=="error":
                raise RuntimeError(f"Failed arm: {folder/'error.log'}")
    return int(any(r["status"]!="completed" for r in records))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    args=parser.parse_args()
    root=(args.run or OUTPUT/datetime.now(MSK).strftime("PIv3-madelon500-%Y%m%d-%H%M%S-%f")).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root):
        raise SystemExit(run(root))
