"""Paired Madelon-500 tests of PIv3 as a selector and as a front end to PFI and RF prefixes."""

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
from experiments.test.run_piv3 import BUDGET, LIMIT, OUTPUT, SEEDS
from experiments.test.run_sampled_permutation_seed42 import metrics
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.piv3 import search


METHODS = ("piv3", "pfi", "piv3_pfi", "rf_prefix", "piv3_rf_prefix")
TOLERANCES = (.03, .05, .08)
SOURCES = (
    "experiments/test/run_piv3_representatives.py", "experiments/test/piv3_representative_selectors.py",
    "experiments/test/run_piv3.py", "src/minppi_mofs/piv3.py", "src/minppi_mofs/piv2.py",
    "src/minppi_mofs/piv1.py", "src/minppi_mofs/modeling.py", "src/minppi_mofs/utility.py",
    "src/minppi_mofs/experiment_protocol.py", "experiments/test/piv2_madelon500_data.py",
    "experiments/test/thesis_madelon_alpha_data.py", "experiments/test/run_sampled_permutation_seed42.py",
    "experiments/test/piv2_conventional_models.py", "experiments/calibrate_icomp2026_models.py",
    "experiments/run_thesis_sampling.py",
)


def trial(source, row, policy, method, seed, tol, out, fingerprint):
    from experiments.test.piv3_representative_selectors import select

    def progress(phase, **fields):
        atomic_write_json(out/"progress.json", dict(phase=phase, **fields))

    progress("loading")
    train, yt, val, yv = load_prepared_train_validation(source, row)
    columns = train.columns.tolist()
    predictions, models = {}, {}
    need_model = method.endswith("pfi")

    def evaluate(names):
        key = tuple(names)
        if key not in predictions:
            members = set(key)
            if not key or list(key) != [name for name in columns if name in members]:
                raise ValueError("Expected a nonempty unique feature subset in input order")
            names_safe = [f"f{i}" for i in range(len(names))]
            model = make_model(**policy)
            model.fit(train[names].set_axis(names_safe, axis=1), yt)
            pred = model.predict_proba(val[names].set_axis(names_safe, axis=1))
            predictions[key] = np.asarray(pred, dtype=float)
            if need_model:
                models[key] = model
        return np.asarray(metrics(yv, predictions[key], "roc_auc"))

    full_key = tuple(columns)
    progress("reference")
    reference = evaluate(columns)
    support, front, front_s = columns, None, 0.
    if method.startswith("piv3"):
        start = time.monotonic()
        def pi_evaluate(names):
            score = evaluate(names)
            progress("voting_search", checked=len(predictions)-1, candidates=BUDGET)
            return score
        support, front = search(train, yt, search_seed=seed, tolerance=tol,
                                reference_scores=reference, evaluate=pi_evaluate, budget=BUDGET)
        front_s = time.monotonic()-start
        if len(front["steps"]) != len(predictions)-1 or len(front["steps"]) > BUDGET:
            raise ValueError("PIv3 candidate count mismatch")
    support_key = tuple(support)
    if support_key not in predictions:
        raise ValueError("PIv3 returned an unverified feature set")
    front_scores = evaluate(support)
    front_candidates = len(predictions)-1
    selected, backend, backend_s = support, None, 0.
    if method != "piv3":
        start = time.monotonic()
        selected, backend = select(method.removeprefix("piv3_"), train, yt, val, yv,
                                   support=support, search_seed=seed, policy=policy, reference_scores=reference,
                                   tolerance=tol, evaluate=evaluate,
                                   get_input_model=lambda: (models[support_key], predictions[support_key]),
                                   progress=progress)
        backend_s = time.monotonic()-start
    selected_key = tuple(selected)
    if selected_key not in predictions or not set(selected) <= set(support):
        raise ValueError("Unverified downstream feature set")
    scores = evaluate(selected)
    if np.any(scores+1e-12 < reference*(1-tol)):
        raise ValueError("Selected feature set failed relative validation floors")
    keys = [key for key in predictions if key != full_key]
    aux_fits = 0 if backend is None else backend["selector_aux_fits"]
    frozen = dict(dataset="madelon", data_seed=42, predictor_seed=42, model_policy=policy,
                  method=method, search_seed=seed, tolerance=tol, columns=columns, support=support,
                  selected=selected, reference_scores=reference.tolist(), front_scores=front_scores.tolist(),
                  validation_scores=scores.tolist(), protocol_fingerprint=fingerprint,
                  partition_sha256={s: row[f"{s}_file_sha256"] for s in ("train","val","test")},
                  diagnostics=dict(front=front, backend=backend, evaluated_masks=[list(k) for k in keys]))
    identity = canonical_hash(frozen)
    atomic_write_json(out/"frozen.json", frozen)
    paths = PreparedSplitPaths("madelon",42,row["target"],*(source/row[s+"_relpath"] for s in ("train","val","test")))
    job = JobSpec(identity,row["protocol_hash"],"removal","madelon",42,method,
                  dict(selected=selected,tolerance=tol,search_seed=seed),(),"result.json")
    audit = AuditLog.from_manifest(out/"test_access.json", [job])
    progress("final_model", fits=len(predictions)+aux_fits)
    test, ye = AuditedSplitAccess(paths,stage="removal",job_id=identity,audit=audit).load_test(
        reason="final_removal_evaluation",expected_sha256=row["test_file_sha256"])
    combined = pd.concat([train,val],ignore_index=True)
    targets = np.concatenate([yt,yv])
    test_pred = fit_predict(combined[selected], targets, test[selected], **policy)
    test_full = fit_predict(combined, targets, test, **policy)
    test_scores, test_ref = (np.asarray(metrics(ye, p, "roc_auc")) for p in (test_pred, test_full))
    np.savez_compressed(out/"predictions.npz", validation_y=yv, validation=predictions[selected_key],
                        validation_full=predictions[full_key], y=ye, selected=test_pred, full=test_full,
                        candidate_validation=np.stack([predictions[k] for k in keys]) if keys else
                        np.empty((0,*predictions[full_key].shape)))
    result = dict(method=method, search_seed=seed, tolerance=tol, fits=len(predictions)+aux_fits+2, aux_fits=aux_fits,
                  candidates=len(keys), k=len(selected), front_k=len(support), front_candidates=front_candidates,
                  front_s=front_s, backend_s=backend_s, reference_scores=reference.tolist(),
                  validation_scores=scores.tolist(), test_scores=test_scores.tolist(),
                  test_reference_scores=test_ref.tolist(), validation_pass=True,
                  test_pass=bool(np.all(test_scores+1e-12 >= test_ref*(1-tol))), frozen_hash=identity)
    atomic_write_json(out/"result.json", result)
    progress("complete", fits=result["fits"])


def worker(*args):
    try:
        trial(*args)
    except Exception:
        (args[-2]/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run(root):
    source, row = prepare_run(root)
    data = read(source/"provenance.json")
    policy = {k:data["policy"][k] for k in ("seed","model_params","backend","task")}
    tasks = []
    for i,tol in enumerate(TOLERANCES):
        for j,seed in enumerate(SEEDS):
            shift=(3*i+j)%len(METHODS)
            for method in METHODS[shift:]+METHODS[:shift]:
                tasks.append((method,tol,seed))
    hashes = {s:row[f"{s}_file_sha256"] for s in ("train","val","test")}
    p = dict(version="PIv3-standalone-and-prefilter", dataset="madelon", width=500, data_seed=42, predictor_seed=42,
             search_seeds=list(SEEDS), tolerances=list(TOLERANCES), methods=list(METHODS), candidate_budget=BUDGET,
             tasks=[list(t) for t in tasks], arm_limit_s=LIMIT, model_policy=policy, input_partition_sha256=hashes,
             catalog_sha256=row["protocol_hash"], input_sha256=sha(source/"provenance.json"),
             source_sha256={name:sha(ROOT/name) for name in SOURCES},
             representatives="Selected once by peak archived plain validation AUC per historical family; "
                 "sampling: native sampled PFI vs masked Banzhaf (fixed-k arm); prefix ranker: RF vs 11 others.",
             sampling="PFI >0 mean drop after five shuffles, validate native support once, fallback to input if it fails.",
             traversal="Random forest importance (100 trees, seed 42), ascending ranked prefixes, stop at first two-metric "
                  "pass. No all-subsets enumeration. Refit ranking on front output. Three cold timing repeats for plain RF.",
             validation_rule="Both AUC and accuracy at least (1-tol)*their full 500-col validation score.",
             timing="Separate serial watchdog cold starts. Reference and per-candidate models can be shared within "
                 "a single arm only. Two train+validation refits (selected and full) for held-out comparison.",
             reuse="For native PFI use the actual fitted model on its input (full or accepted PI support); "
                  "do not charge a second fit for the identical model in that arm.")
    if (root/"protocol.json").exists():
        if read(root/"protocol.json") != p:
            raise ValueError("Cannot resume a changed protocol")
    else:
        atomic_write_json(root/"protocol.json",p)
    for f,h in p["source_sha256"].items():
        dest=root/"source"/f
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/f,dest)
        if sha(dest) != h:
            raise ValueError(f"Invalid source snapshot {f}")
    with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(msg):
            line=f"[{datetime.now(MSK):%H:%M:%S}] {msg}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        rows=[]
        for m,t,s in tasks:
            folder=root/m/f"tol-{round(100*t):02d}"/f"s{s}"
            if (folder/"watchdog.json").exists():
                timer=read(folder/"watchdog.json")
            else:
                if folder.exists():
                    folder.rename(folder.with_name(folder.name+".interrupted-"+datetime.now(MSK).strftime("%H%M%S-%f")))
                folder.mkdir(parents=True)
                log(f"Start {m} {t:.0%} s{s}: {len(rows)+1}/{len(tasks)}")
                proc=multiprocessing.get_context("spawn").Process(target=worker,
                    args=(source,row,policy,m,s,t,folder,canonical_hash(p)))
                try:
                    timer=supervise(proc,folder,LIMIT,log,heartbeat_s=10.,
                        heartbeat_formatter=lambda state,elapsed,left:f"{m} {state.get('phase','startup')} "
                        f"{state.get('checked',0)}/{state.get('candidates','?')} elapsed={elapsed:.1f}s remaining_limit={left:.0f}s")
                finally:
                    proc.close()
            r=timer["result"]
            if timer["status"]=="completed" and (r!=read(folder/"result.json") or
                    canonical_hash(read(folder/"frozen.json"))!=r["frozen_hash"]):
                raise ValueError(f"Changed run {folder}")
            rows.append(dict(method=m,tolerance=t,search_seed=s,status=timer["status"],wall_s=timer["elapsed_s"],
                             k=r.get("k"),fits=r.get("fits"),test_pass=r.get("test_pass")))
            write_summary(root,rows)
            log(f"{timer['status']} {m} {t:.0%} s{s}: {timer['elapsed_s']:.2f}s k={r.get('k')} fits={r.get('fits')}")
            if timer["status"]=="error":
                raise RuntimeError(f"Inspect {folder/'error.log'}")
    return int(any(r["status"]!="completed" for r in rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    args=parser.parse_args()
    root=(args.run or OUTPUT/datetime.now(MSK).strftime("PIv3-frontend-%Y%m%d-%H%M%S-%f")).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root):
        raise SystemExit(run(root))
