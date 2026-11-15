"""Contemporaneous cold replications of the compact Gain anchor and five-block minPI."""
import argparse
import csv
from datetime import datetime
import gc
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import time

from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_context_vote import DEFAULT as PRIMARY, DATASETS, REFERENCE, SEEDS, digest, read
from experiments.run_formal_literature import worker
from experiments.run_thesis_sampling import ROOT, supervise


DEFAULT = ROOT / "results/thesis_benchmark/context-vote-controls-20261115-final"


def matched_worker(job, out, inputs, receipt):
    """The realized MinPI k is supplied as side information, never optimized on test."""
    from functools import partial
    import traceback
    from unittest.mock import patch
    from threadpoolctl import threadpool_limits
    from experiments import literature_worker
    from minppi_mofs.baselines._selection_context import Selection
    from minppi_mofs.experiment_protocol import canonical_hash
    from minppi_mofs.piv3_compact_tree import _model_columns
    group, = (g for g in read(PRIMARY/"effective.json")["groups"]
              if g["dataset"]==job["dataset"] and g["seed"]==job["seed"])
    donor=PRIMARY/group["folder"]/"min"
    frozen=read(donor/"frozen.json")
    k=len(frozen["selected"])
    if canonical_hash(frozen)!=read(donor/"test_access.json")["frozen_hash"]:
        raise ValueError("The already-frozen donor support does not match its test-access record")

    def matched(context, donor_order):
        booster, mapping=_model_columns(context.full_model,context.columns)
        gains={mapping[f]:g for f,g in booster.get_score(importance_type="gain").items()}
        order=sorted(gains,key=lambda name: (-gains[name],context.columns.index(name))) or context.columns[:1]
        if order!=donor_order or not 1<=k<=len(order):
            raise ValueError("The freshly fitted Gain order and MinPI donor disagree")
        keep=set(order[:k])
        return Selection(selected=[c for c in context.columns if c in keep],
                         diagnostics=dict(k=k,order=order,donor_hash=canonical_hash(frozen),
                                          size_source="frozen min support",arm="matched_gain"))
    try:
        source_order=read(donor/"estimator.json")["order"]
        with threadpool_limits(limits=1),patch.object(literature_worker,"get_selector",
                return_value=partial(matched,donor_order=source_order)):
            literature_worker.trial(job["dataset"],"shared","gain",out,seed=job["seed"],
                                    input_root=inputs,enforce_guard=False)
        gc.collect()
        write_worker_receipt(receipt,0)
    except Exception:
        (out/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


def run(root, prepare_only=False):
    from minppi_mofs.experiment_protocol import atomic_write_json
    original=read(REFERENCE/"protocol.json")
    root.mkdir(parents=True,exist_ok=True)
    methods=("gain","minpi","matched_gain")
    protocol=dict(primary_protocol_sha256=digest(PRIMARY/"protocol.json"),
                  baseline_protocol_sha256=digest(REFERENCE/"protocol.json"),
                  seeds=SEEDS,datasets=DATASETS,methods=methods,inputs=original["inputs"],
                  route="unmodified original shared worker; enforce_guard=False",
                  method_order="rotate by dataset+seed index",dataset_seed_seconds=3600,
                  matched_gain="independently refit Gain, same k as signed MinPI support; conditional timing excludes donor selection",
                  sources={name:digest(ROOT/name) for name in (
                      "experiments/run_context_vote_controls.py","experiments/run_formal_literature.py",
                      "experiments/literature_worker.py","src/minppi_mofs/minpi_selector.py",
                      "src/minppi_mofs/piv3_compact_tree.py","src/minppi_mofs/piv3_compact_witness.py",
                      "src/minppi_mofs/baselines/gain.py")})
    protocol_path=root/"protocol.json"
    if protocol_path.is_file():
        import json
        if read(protocol_path)!=json.loads(json.dumps(protocol)):
            raise ValueError("Frozen control protocol changed")
    else:
        atomic_write_json(protocol_path,protocol)
    if prepare_only:
        return
    if not (PRIMARY/"effective.json").is_file():
        raise ValueError("Finish the primary run before timing controls on this machine")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    rows=[]
    with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(message):
            line=f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        for i,seed in enumerate(SEEDS):
            for j,dataset in enumerate(DATASETS):
                inputs=REFERENCE/"inputs"/f"seed-{seed}"
                start=time.monotonic()
                budget=3600
                meta=inputs/dataset/"metadata.json"
                if digest(meta)!=original["inputs"][str(seed)][dataset]["metadata_sha256"]:
                    raise ValueError(f"Input metadata changed for {dataset}/s{seed}")
                for name,expected in read(meta)["sha256"].items():
                    if digest(meta.parent/name)!=expected:
                        raise ValueError(f"Control input changed: {meta.parent/name}")
                shift=(i+j)%len(methods)
                for method in methods[shift:]+methods[:shift]:
                    job=dict(dataset=dataset,seed=seed,mode="shared",method="gain" if method=="matched_gain" else method)
                    folder=root/"runs"/f"seed-{seed}"/dataset/method
                    wd=folder/"watchdog.json"
                    label=f"{dataset} seed={seed} arm={method}"
                    if wd.is_file():
                        out=read(wd)
                        budget-=out["elapsed_s"]
                    else:
                        folder.mkdir(parents=True,exist_ok=False)
                        remaining=budget-(time.monotonic()-start)
                        if remaining<=0:
                            atomic_write_json(wd,dict(status="budget_exhausted",elapsed_s=0))
                            out=read(wd)
                        else:
                            receipt=WorkerContract(folder/"worker-completion.json",secrets.token_hex(32))
                            target=matched_worker if method=="matched_gain" else worker
                            proc=mp.get_context("spawn").Process(target=target,args=(job,folder,inputs,receipt))
                            log(label+" start")
                            try:
                                out=supervise(proc,folder,remaining,log,heartbeat_s=20,completion_contract=receipt)
                            finally:
                                proc.close()
                    row=dict(dataset=dataset,seed=seed,arm=method,status=out["status"],wall_s=out["elapsed_s"])
                    if out["status"]=="completed":
                        res=read(folder/"result.json")
                        if res!=out["result"] or any(res[k]!=v for k,v in job.items()):
                            raise ValueError(f"Mismatch with completion receipt: {folder}")
                        row.update({k:res[k] for k in ("width","k","fits","selector_s","reference_s","guard_s","final_s")})
                        row.update(res["test_scores"])
                    log(f"{label} {out['status']} elapsed={out['elapsed_s']:.2f}s")
                    rows.append(row)
                    tmp=root/"per-seed.tmp"
                    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
                        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                        writer.writeheader()
                        writer.writerows(rows)
                    os.replace(tmp,root/"per-seed.csv")
                    if out["status"]=="error":
                        raise RuntimeError(f"Failed {label}, see {folder}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--prepare-only",action="store_true")
    options=parser.parse_args()
    run(options.run.resolve(),options.prepare_only)
