"""Single-seed stress test on every formal benchmark truncated from >100 to 100 columns."""
import argparse
import csv
from datetime import datetime
import gc
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import time
import traceback

from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_context_vote import DATASETS, DEFAULT as PRIMARY, REFERENCE, digest, read
from experiments.run_context_vote_controls import DEFAULT as CONTROLS
from experiments.run_formal_literature import worker as standard_worker
from experiments.run_thesis_sampling import ROOT, supervise


INPUT=ROOT/"results/test/thesis_sampling/PIv3-six-raw-seed42-input"
DEFAULT=ROOT/"results/thesis_benchmark/context-vote-width-20261115"


def vote_worker(job,folder,inputs,receipt):
    from functools import partial
    from unittest.mock import patch
    from threadpoolctl import threadpool_limits
    from experiments import literature_worker
    from experiments.context_vote_selector import select
    try:
        with threadpool_limits(limits=1),patch.object(literature_worker,"get_selector",
                return_value=partial(select,arm="min")):
            literature_worker.trial(**job,out=folder,input_root=inputs,enforce_guard=False)
        gc.collect()
        write_worker_receipt(receipt,0)
    except Exception:
        (folder/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


def protocol():
    import numpy as np
    from experiments.test.piv3_raw_splits import prepare
    inputs={}
    for dataset in DATASETS:
        prepared=read(REFERENCE/"inputs/seed-42"/dataset/"metadata.json")
        if prepared["width"]!=100 or prepared["raw_width"]<=100:
            continue
        raw_meta=INPUT/dataset/"metadata.json"
        if not raw_meta.is_file():
            raise FileNotFoundError(raw_meta)
        raw=prepare(INPUT,dataset)
        for key in ("dataset","seed","task","utility","policy","raw_sha256","index_sha256"):
            if raw[key]!=prepared[key]:
                raise ValueError(f"{dataset}: raw and preprocessed protocols differ for {key}")
        if raw["width"]!=prepared["raw_width"]:
            raise ValueError(f"{dataset}: raw feature count disagrees")
        for role in ("train","val","test"):
            for suffix in ("_indices.npy","_y.npy"):
                # Independent exports can have different NPY storage layouts.
                name=role+suffix
                a=np.load(INPUT/dataset/name,allow_pickle=False)
                b=np.load(REFERENCE/"inputs/seed-42"/dataset/name,allow_pickle=False)
                if not np.array_equal(a,b):
                    raise ValueError(f"{dataset}: {name} differs from its prepared counterpart")
        inputs[dataset]=dict(metadata_sha256=digest(raw_meta),width=raw["width"],
                             index_sha256=raw["index_sha256"],arrays=raw["sha256"],policy=raw["policy"])
    if set(inputs)!={"madelon","gisette","isolet"}:
        raise ValueError("The predeclared width-stress cohort differs")
    return dict(seed=42,inputs=inputs,input_root=INPUT.relative_to(ROOT).as_posix(),
                group_seconds=3600,guard_fallback=False,arms=["min","gain","full"],
                cohort="all six formal tasks with raw_width>100 and prepared_width=100",
                order="rotate min, gain, full by dataset index; Madelon, Gisette, ISOLET",
                primary_protocol_sha256=digest(PRIMARY/"protocol.json"),
                source={k:digest(ROOT/k) for k in (
                    "experiments/run_context_vote_width.py","experiments/context_vote_selector.py",
                    "src/minppi_mofs/context_vote.py","experiments/literature_worker.py",
                    "src/minppi_mofs/piv3_compact_tree.py","src/minppi_mofs/piv3_compact_witness.py",
                    "experiments/test/minpi_convergence_metrics.py","experiments/test/piv3_raw_splits.py")})


def run(root,prepare_only):
    from minppi_mofs.experiment_protocol import atomic_write_json
    root.mkdir(parents=True,exist_ok=True)
    spec=protocol()
    config=root/"protocol.json"
    if config.exists() and read(config)!=spec:
        raise ValueError("Width-study protocol differs from the frozen copy")
    if not config.exists():
        atomic_write_json(config,spec)
    if prepare_only:
        return
    if not (CONTROLS/"effective.json").exists():
        raise ValueError("Finish contemporaneous controls before using the same device")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    rows=[]
    with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(message):
            line=f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        for idx,dataset in enumerate(("madelon","gisette","isolet")):
            start=time.monotonic()
            budget=spec["group_seconds"]
            for arm in spec["arms"][idx:]+spec["arms"][:idx]:
                folder=root/"runs/seed-42"/dataset/arm
                wd=folder/"watchdog.json"
                if wd.exists():
                    timer=read(wd)
                    budget-=timer["elapsed_s"]
                else:
                    folder.mkdir(parents=True,exist_ok=False)
                    left=budget-(time.monotonic()-start)
                    if left<=0:
                        timer=dict(status="budget_exhausted",elapsed_s=0)
                        atomic_write_json(wd,timer)
                    else:
                        method="minpi" if arm=="min" else arm
                        job=dict(dataset=dataset,mode="shared",method=method,seed=42)
                        receipt=WorkerContract(folder/"worker-completion.json",secrets.token_hex(32))
                        target=vote_worker if arm=="min" else standard_worker
                        proc=mp.get_context("spawn").Process(target=target,args=(job,folder,INPUT,receipt))
                        log(f"{dataset} {arm} start")
                        try:
                            timer=supervise(proc,folder,left,log,heartbeat_s=20,completion_contract=receipt)
                        finally:
                            proc.close()
                status=timer["status"]
                if status=="completed" and timer["elapsed_s"]>=timer["limit_s"]:
                    status="late_completion"
                row=dict(dataset=dataset,seed=42,arm=arm,run_folder=folder.relative_to(root).as_posix(),
                         status=status,width=spec["inputs"][dataset]["width"],wall_s=timer["elapsed_s"])
                if status=="completed":
                    res=read(folder/"result.json")
                    if res!=timer["result"]:
                        raise ValueError(f"{folder}: result differs from receipt")
                    row.update(k=res["k"],fits=res["fits"],selector_s=res["selector_s"],**res["test_scores"])
                rows.append(row)
                log(f"{dataset} {arm} {status} elapsed={timer['elapsed_s']:.2f}s")
                tmp=root/"per-seed.tmp"
                with tmp.open("w",encoding="utf-8-sig",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                    writer.writeheader()
                    writer.writerows(rows)
                os.replace(tmp,root/"per-seed.csv")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    run(args.run.resolve(),args.prepare_only)
