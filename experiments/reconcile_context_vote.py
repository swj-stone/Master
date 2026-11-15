"""Audit whole-group deadlines and repeat predeclared arms without replacing raw records."""
import argparse
import csv
from datetime import datetime
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import time

from experiments.native_process_guard import WorkerContract
from experiments.run_context_vote import DEFAULT, REFERENCE, digest, read, worker
from experiments.run_thesis_sampling import ROOT, supervise
from minppi_mofs.experiment_protocol import atomic_write_json


def invalid_group(group, arms, limit=3600):
    reasons, spent = [], 0.
    for arm in arms:
        path = group/arm/"watchdog.json"
        if not path.is_file():
            reasons.append(f"{arm}: missing completion")
            continue
        timer = read(path)
        elapsed = timer["elapsed_s"]
        spent += elapsed
        if timer["status"] != "completed":
            reasons.append(f"{arm}: {timer['status']}")
        elif elapsed >= timer["limit_s"]:
            reasons.append(f"{arm}: returned after watchdog deadline")
    if spent >= limit:
        reasons.append(f"group wall time {spent:.3f}s reached {limit}s")
    return reasons


def retry(group, dataset, seed, arms, log, controls):
    from experiments.run_context_vote_controls import matched_worker
    from experiments.run_formal_literature import worker as control_worker
    group.mkdir(parents=True, exist_ok=False)
    start=time.monotonic()
    inputs = REFERENCE/"inputs"/f"seed-{seed}"
    for arm in arms:
        folder = group/arm
        folder.mkdir()
        seconds = 3600-(time.monotonic()-start)
        if seconds<=0:
            atomic_write_json(folder/"watchdog.json",dict(status="budget_exhausted",elapsed_s=0))
            continue
        label=f"{dataset}/s{seed}/{arm}"
        log(f"retry {group.name} {label} start")
        receipt=WorkerContract(folder/"worker-completion.json",secrets.token_hex(32))
        if controls:
            method="gain" if arm=="matched_gain" else arm
            job=dict(dataset=dataset,seed=seed,method=method,mode="shared")
            target=matched_worker if arm=="matched_gain" else control_worker
            args=(job,folder,inputs,receipt)
        else:
            target=worker
            args=(dataset,seed,arm,folder,receipt)
        proc=mp.get_context("spawn").Process(target=target,args=args)
        try:
            outcome=supervise(proc,folder,seconds,log,completion_contract=receipt,heartbeat_s=20)
        finally:
            proc.close()
        status=outcome["status"]
        if outcome["elapsed_s"]>=seconds:
            status="invalid_wall_deadline"
        log(f"{label} {status} wall={outcome['elapsed_s']:.2f}s")
        if outcome["status"]=="error":
            raise RuntimeError(f"Invalid worker completion in {folder}")


def run(root,controls=False):
    protocol=read(root/"protocol.json")
    if controls:
        methods=protocol["methods"]
        jobs=[]
        for i,seed in enumerate(protocol["seeds"]):
            for j,dataset in enumerate(protocol["datasets"]):
                off=(i+j)%len(methods)
                jobs.append(dict(dataset=dataset,seed=seed,arms=methods[off:]+methods[:off]))
        sources=protocol["sources"]
    else:
        jobs=protocol["jobs"]
        sources=protocol["source"]
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f"Source changed: {name}")
    total=sum(len(job["arms"]) for job in jobs)
    if len(list((root/"runs").glob("seed-*/*/*/watchdog.json")))!=total:
        raise ValueError("Finish the original run before timing retries")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    rows=[]
    batches=[]
    with (root/"recovery.log").open("a",encoding="utf-8",buffering=1) as f:
        def log(message):
            line=f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
            print(line,flush=True)
            print(line,file=f,flush=True)
        for job in jobs:
            seed,dataset,arms=job["seed"],job["dataset"],job["arms"]
            base=root/"runs"/f"seed-{seed}"/dataset
            group=base
            invalid=[]
            while reasons:=invalid_group(group,arms):
                invalid.append(dict(folder=group.relative_to(root).as_posix(),reasons=reasons))
                group=root/"retries"/f"attempt-{len(invalid)}"/f"seed-{seed}"/dataset
                if not group.is_dir():
                    log(f"Repeat entire {dataset}/s{seed}: {reasons}")
                    retry(group,dataset,seed,arms,log,controls)
            batches.append(dict(dataset=dataset,seed=seed,folder=group.relative_to(root).as_posix(),
                                invalid_attempts=invalid))
            for arm in arms:
                folder=group/arm
                result=read(folder/"result.json")
                timer=read(folder/"watchdog.json")
                if result!=timer["result"]:
                    raise ValueError(f"Receipt mismatch in {folder}")
                row=dict(dataset=dataset,seed=seed,arm=arm,status="completed",
                         run_folder=folder.relative_to(root).as_posix(),wall_s=timer["elapsed_s"])
                for name in ("width","k","fits","reference_s","selector_s","guard_s","final_s",
                             "selection_prediction_calls","sample_prediction_calls"):
                    row[name]=result.get(name)
                if not controls:
                    e=read(folder/"estimator.json")
                    row.update(anchor_k=e["anchor_k"],contexts=len(e["context_extensions"]),
                               proposals=len(e["trace"]),stop=e["stop"])
                row.update(result["test_scores"])
                row.update({"validation_"+k:v for k,v in result["validation_selected"].items()})
                rows.append(row)
    atomic_write_json(root/"effective.json",dict(protocol_sha256=digest(root/"protocol.json"),
        rule="replace entire dataset-seed if any incomplete arm, cold elapsed >= remaining limit, or sum >= 3600s",
        groups=batches))
    tmp=root/"per-seed-effective.tmp"
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp,root/"per-seed-effective.csv")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--controls",action="store_true")
    args=parser.parse_args()
    run(args.run.resolve(),args.controls)
