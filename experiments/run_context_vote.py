"""Collect matched context-vote arms on the frozen formal benchmark."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import time
import traceback

from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_thesis_sampling import ROOT, supervise


REFERENCE = ROOT / "results/thesis_benchmark/baselines-eight-preprocessed-five-seeds"
DEFAULT = ROOT / "results/thesis_benchmark/context-vote-20261115"
SEEDS = (42, 123, 456, 789, 1024)
DATASETS = ("abalone", "spambase", "madelon", "credit_card_fraud", "gisette", "isolet")
SOURCE = ("src/minppi_mofs/context_vote.py", "experiments/context_vote_selector.py",
          "experiments/run_context_vote.py", "experiments/literature_worker.py",
          "experiments/test/minpi_convergence_metrics.py", "src/minppi_mofs/piv3_compact_tree.py",
          "src/minppi_mofs/piv3_compact_witness.py", "src/minppi_mofs/prediction_meter.py")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def worker(dataset, seed, arm, folder, receipt):
    try:
        from functools import partial
        from unittest.mock import patch
        from threadpoolctl import threadpool_limits
        from experiments import literature_worker
        from experiments.context_vote_selector import select
        from minppi_mofs.experiment_protocol import atomic_write_json
        atomic_write_json(folder/"arm.json", dict(arm=arm, seed=seed, dataset=dataset))
        # Reuse the unchanged formal evaluation pipeline in the private worker process.
        with threadpool_limits(limits=1), patch.object(literature_worker, "get_selector",
                return_value=partial(select, arm=arm)):
            literature_worker.trial(dataset,"shared","minpi",folder,seed=seed,enforce_guard=False,
                                    input_root=REFERENCE/"inputs"/f"seed-{seed}")
        import gc
        gc.collect()
        write_worker_receipt(receipt,0)
    except Exception:
        (folder/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


def summarize(root):
    rows = []
    for path in sorted((root/"runs").glob("seed-*/*/*/watchdog.json")):
        arm, dataset, seed_dir = path.parent.name, path.parents[1].name, path.parents[2].name
        w = read(path)
        r = w.get("result", {})
        row = dict(dataset=dataset, seed=int(seed_dir.split("-")[1]), arm=arm, status=w["status"],
                   wall_s=w["elapsed_s"])
        if w["status"] == "completed":
            e = read(path.parent/"estimator.json")
            row.update({k:r[k] for k in ("width","k","fits","reference_s","selector_s","guard_s",
                                         "final_s","selection_prediction_calls","sample_prediction_calls")})
            row.update(anchor_k=e["anchor_k"], contexts=len(e["context_extensions"]),
                       proposals=len(e["trace"]), stop=e["stop"], **r["test_scores"])
            row.update({"validation_"+k:v for k,v in r["validation_selected"].items()})
        rows.append(row)
    if rows:
        temp=root/"per-seed.tmp"
        with temp.open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp,root/"per-seed.csv")


def run(root):
    from minppi_mofs.experiment_protocol import atomic_write_json
    root.mkdir(parents=True,exist_ok=True)
    spec = read(REFERENCE/"protocol.json")
    jobs=[]
    for i,seed in enumerate(SEEDS):
        for j,dataset in enumerate(DATASETS):
            arms=["min","mean","point"]+(["eager"] if seed==42 else [])
            # Rotate the cold-process order across datasets and across seeds.
            offset=(i+j)%len(arms)
            jobs.append(dict(seed=seed,dataset=dataset,arms=arms[offset:]+arms[:offset]))
    protocol=dict(reference=str(REFERENCE),reference_protocol_sha256=digest(REFERENCE/"protocol.json"),
        source={s:digest(ROOT/s) for s in SOURCE},inputs=spec["inputs"],jobs=jobs,
        dataset_seed_seconds=3600,guard_fallback=False,epsilon=1e-12,
        tolerance=.03,average_precision_tolerance=.05,anchor="archived compact Gain, 8% mask tolerance",
        panels="point plus three paired class-stratified bootstraps on at most 256 ranking rows",
        contexts="distinct empty, next anchor-width, all remaining model-active columns",
        endpoints="all six, all five seeds, compression, held-out metrics and independent cold timing",
        acquisition="archived partitions reused; selection uses only fit and ranking data")
    path=root/"protocol.json"
    if path.exists() and read(path)!=protocol:
        raise ValueError("Frozen protocol changed; use a separate output directory")
    if not path.exists():
        atomic_write_json(path,protocol)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(msg):
            line=f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        for job in jobs:
            seed, dataset=job["seed"],job["dataset"]
            input_folder=REFERENCE/"inputs"/f"seed-{seed}"/dataset
            start=time.monotonic()
            if digest(input_folder/"metadata.json")!=spec["inputs"][str(seed)][dataset]["metadata_sha256"]:
                raise ValueError("Input metadata changed")
            for name,expected in read(input_folder/"metadata.json")["sha256"].items():
                if digest(input_folder/name)!=expected:
                    raise ValueError(f"Corrupt input {input_folder/name}")
            budget=3600
            for arm in job["arms"]:
                folder=root/"runs"/f"seed-{seed}"/dataset/arm
                if (folder/"watchdog.json").is_file():
                    past=read(folder/"watchdog.json")
                    if past["status"]=="completed" and read(folder/"result.json")!=past["result"]:
                        raise ValueError(f"Completion mismatch in {folder}")
                    budget-=past["elapsed_s"]
                    continue
                folder.mkdir(parents=True,exist_ok=False)
                left=budget-(time.monotonic()-start)
                label=f"{dataset} seed={seed} arm={arm}"
                if left<=0:
                    atomic_write_json(folder/"watchdog.json",dict(status="budget_exhausted",elapsed_s=0))
                    log(label+" group exceeded time budget")
                    summarize(root)
                    continue
                log(label+" start")
                receipt=WorkerContract(folder/"worker-completion.json",secrets.token_hex(32))
                proc=mp.get_context("spawn").Process(target=worker,args=(dataset,seed,arm,folder,receipt))
                try:
                    out=supervise(proc,folder,left,log,heartbeat_s=20,completion_contract=receipt,
                         heartbeat_formatter=lambda p,t,left: f"{label} phase={p.get('phase','start')} elapsed={t:.1f}s")
                finally:
                    proc.close()
                log(f"{label} {out['status']} elapsed={out['elapsed_s']:.2f}s")
                summarize(root)
                if out["status"]=="error":
                    raise RuntimeError(f"Failed {label}, see {folder}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    args=parser.parse_args()
    run(args.run.resolve())
