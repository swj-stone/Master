"""Evaluate the published ISOLET speaker split (seed 42, all 617 features).

Download https://archive.ics.uci.edu/static/public/54/isolet.zip to the default
tmp/isolet-uci-official-20261115.zip or pass --zip PATH. GNU gzip must be on PATH
to decode the two Unix compress (.Z) streams in the original UCI archive.
"""
import argparse
import csv
from datetime import datetime
from functools import partial
import gc
import hashlib
from io import BytesIO
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import subprocess
import time
import traceback
from unittest.mock import patch
from zipfile import ZipFile

from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_context_vote import DEFAULT as PRIMARY, REFERENCE, digest, read
from experiments.run_context_vote_width import vote_worker
from experiments.run_formal_literature import worker as standard_worker
from experiments.run_thesis_sampling import ROOT, supervise


DEFAULT=ROOT/"results/thesis_benchmark/context-vote-isolet-official-20261115"
ZIP=ROOT/"tmp/isolet-uci-official-20261115.zip"
ZIP_SHA="fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053"
TRAIN_FILE="isolet1+2+3+4.data.Z"
TEST_FILE="isolet5.data.Z"
ARMS=("min","full","gain","matched_gain")


def prepare(root, archive):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from minppi_mofs.experiment_protocol import atomic_write_json
    if digest(archive)!=ZIP_SHA:
        raise ValueError("Unexpected UCI ISOLET archive SHA-256")
    local_path=ROOT/"data/raw/isolet.csv"
    local=pd.read_csv(local_path)
    names=[f"f{i}" for i in range(1,618)]
    if list(local.columns)!=names+["letter"]:
        raise ValueError("The local ISOLET feature schema differs")
    mapping={int(s):j for j,s in enumerate(sorted(map(str,range(1,27))))}
    pos=0
    receipts={}
    dev_labels=None
    with ZipFile(archive) as bundle:
        for name in (TRAIN_FILE,TEST_FILE):
            compressed=bundle.read(name)
            payload=subprocess.run(["gzip","-dc"],input=compressed,capture_output=True,check=True).stdout
            official=pd.read_csv(BytesIO(payload),header=None)
            count,d=official.shape
            y=official.iloc[:,-1].to_numpy()
            local_block=local.iloc[pos:pos+count]
            if (d!=618 or not np.array_equal(y,y.astype(int)) or set(y)!=set(mapping)
                    or not np.array_equal(official.iloc[:,:-1].to_numpy(),local_block[names].to_numpy())
                    or not np.array_equal([mapping[int(t)] for t in y],local_block["letter"])):
                raise ValueError(f"The features/targets in {name} do not exactly match local row order")
            if name==TRAIN_FILE:
                dev_labels=y.astype(int)
            receipts[name]=dict(rows=count,local_row_start=pos,
                                compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                                decompressed_sha256=hashlib.sha256(payload).hexdigest())
            pos+=count
    if pos!=len(local) or (len(dev_labels),pos)!=(6238,7797):
        raise ValueError("Official/local ISOLET row count mismatch")
    # A group has 30 speakers, two examples per letter and no additional examples.
    # In the four-group file two examples are missing. Enumerate *all* boundaries
    # consistent with the sizes and the per-letter quotas on each side.
    missing=4*30*2*26-len(dev_labels)
    nominal=3*30*2*26
    bounds={}
    for idx in range(nominal-missing,nominal+1):
        left=np.bincount(dev_labels[:idx],minlength=27)[1:]
        right=np.bincount(dev_labels[idx:],minlength=27)[1:]
        bounds[str(idx)]=dict(train_max_per_letter=int(left.max()),val_max_per_letter=int(right.max()),
                              feasible=bool(np.all(left<=180) and np.all(right<=60)))
    boundary,=(int(i) for i,item in bounds.items() if item["feasible"])
    roles=dict(train=np.arange(boundary),val=np.arange(boundary,len(dev_labels)),
               test=np.arange(len(dev_labels),len(local)))
    x=local[names].to_numpy(dtype=np.float64)
    labels=local.letter.to_numpy(dtype=np.int64)
    scaler=StandardScaler().fit(x[roles["train"]])
    cached=root/"inputs/isolet"
    cached.mkdir(parents=True,exist_ok=True)
    hashes={}
    for role,ids in roles.items():
        for file,arr in ((role,scaler.transform(x[ids])),(role+"_y",labels[ids]),(role+"_indices",ids)):
            path=cached/(file+".npy")
            if path.is_file():
                if not np.array_equal(np.load(path,allow_pickle=False,mmap_mode="r"),arr):
                    raise ValueError(f"Existing prepared file differs: {path}")
            else:
                np.save(path,arr,allow_pickle=False)
            hashes[path.name]=digest(path)
    model_policy=read(REFERENCE/"inputs/seed-42/isolet/metadata.json")["policy"]
    metadata=dict(dataset="isolet",seed=42,width=617,task="multiclass",utility="ovr_auc",features=names,
                  policy=model_policy,sha256=hashes,partitions={r:dict(rows=len(ids)) for r,ids in roles.items()},
                  scale_fit_role="train",scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist())
    meta=cached/"metadata.json"
    if meta.is_file() and read(meta)!=metadata:
        raise ValueError("Saved speaker-split metadata differs")
    if not meta.is_file():
        atomic_write_json(meta,metadata)
    sources=["experiments/run_context_vote_isolet_official.py","experiments/run_context_vote_width.py",
             "experiments/context_vote_selector.py","experiments/literature_worker.py",
             "src/minppi_mofs/context_vote.py","src/minppi_mofs/piv3_compact_tree.py",
             "src/minppi_mofs/piv3_compact_witness.py","experiments/test/minpi_convergence_metrics.py"]
    return dict(dataset="isolet",seed=42,source_url="https://archive.ics.uci.edu/static/public/54/isolet.zip",
                citation="R. Cole and M. Fanty, ISOLET, UCI, DOI 10.24432/C51G69, CC BY 4.0",
                source_description="https://archive.ics.uci.edu/dataset/54/isolet",
                zip_sha256=ZIP_SHA,official_files=receipts,local_csv_sha256=digest(local_path),
                compared_feature_cells=len(local)*617,max_abs_feature_difference=0,
                targets_compared=len(local),target_mapping={str(k):v for k,v in mapping.items()},
                boundary_candidates=bounds,development_missing=missing,
                rows={r:len(ids) for r,ids in roles.items()},training="isolet1+2+3",
                validation="isolet4; stratified 50/50 row split for ranking/guard",
                final_fit="isolet1+2+3+4",test="isolet5",fallback=False,
                arms=list(ARMS),matched_gain="same k as signed min, independently trained Gain ranking",
                dataset_seed_seconds=3600,metadata_sha256=digest(meta),policy=model_policy,
                source={s:digest(ROOT/s) for s in sources},primary_protocol_sha256=digest(PRIMARY/"protocol.json"))


def equal_width_worker(job,folder,inputs,receipt):
    from threadpoolctl import threadpool_limits
    from experiments import literature_worker
    from minppi_mofs.baselines._selection_context import Selection
    from minppi_mofs.experiment_protocol import canonical_hash
    from minppi_mofs.piv3_compact_tree import _model_columns
    donor=folder.parent/"min"
    frozen=read(donor/"frozen.json")
    fingerprint=canonical_hash(frozen)
    if fingerprint!=read(donor/"test_access.json")["frozen_hash"]:
        raise ValueError("Matched-k donor disagrees with its frozen decision")
    k=len(frozen["selected"])
    def choose(context,expected_order):
        booster,names=_model_columns(context.full_model,context.columns)
        gain={names[f]:v for f,v in booster.get_score(importance_type="gain").items()}
        order=sorted(gain,key=lambda f:(-gain[f],context.columns.index(f))) or context.columns[:1]
        if order!=expected_order or not 1<=k<=len(order):
            raise ValueError("Fresh Gain order differs from MinPI anchor")
        selected=set(order[:k])
        return Selection(selected=[c for c in context.columns if c in selected],
                         diagnostics=dict(arm="matched_gain",k=k,order=order,donor_hash=fingerprint))
    try:
        selector=partial(choose,expected_order=read(donor/"estimator.json")["order"])
        with threadpool_limits(limits=1), patch.object(literature_worker,"get_selector",return_value=selector):
            literature_worker.trial(**job,out=folder,input_root=inputs,enforce_guard=False)
        gc.collect()
        write_worker_receipt(receipt,0)
    except Exception:
        (folder/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


def run(root,archive):
    from minppi_mofs.experiment_protocol import atomic_write_json
    root.mkdir(parents=True,exist_ok=True)
    config=prepare(root,archive)
    path=root/"protocol.json"
    if path.is_file() and read(path)!=config:
        raise ValueError("Official speaker-split protocol changed")
    if not path.is_file():
        atomic_write_json(path,config)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    rows=[]
    budget=3600
    start=time.monotonic()
    with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(msg):
            line=f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        for arm in ARMS:
            folder=root/"runs/seed-42/isolet"/arm
            wd=folder/"watchdog.json"
            if wd.is_file():
                timer=read(wd)
                budget-=timer["elapsed_s"]
            else:
                folder.mkdir(parents=True,exist_ok=False)
                left=budget-(time.monotonic()-start)
                if left<=0:
                    timer=dict(status="budget_exhausted",elapsed_s=0)
                    atomic_write_json(wd,timer)
                else:
                    target={"min":vote_worker,"matched_gain":equal_width_worker}.get(arm,standard_worker)
                    method={"min":"minpi","matched_gain":"gain"}.get(arm,arm)
                    job=dict(dataset="isolet",seed=42,mode="shared",method=method)
                    receipt=WorkerContract(folder/"worker-completion.json",secrets.token_hex(32))
                    proc=mp.get_context("spawn").Process(target=target,args=(job,folder,root/"inputs",receipt))
                    log(f"isolet5 {arm} start")
                    try:
                        timer=supervise(proc,folder,left,log,heartbeat_s=20,completion_contract=receipt)
                    finally:
                        proc.close()
            status=timer["status"]
            if status=="completed" and timer["elapsed_s"]>=timer["limit_s"]:
                status="late_completion"
            row=dict(dataset="isolet",seed=42,arm=arm,status=status,run_folder=folder.relative_to(root).as_posix(),
                     width=617,wall_s=timer["elapsed_s"])
            if status=="completed":
                res=read(folder/"result.json")
                if res!=timer["result"]:
                    raise ValueError("Result and worker receipt differ")
                row.update(k=res["k"],fits=res["fits"],selector_s=res["selector_s"],
                           reference_s=res["reference_s"],guard_s=res["guard_s"],final_s=res["final_s"],
                           **res["test_scores"])
                row.update({"validation_"+k:v for k,v in res["validation_selected"].items()})
            rows.append(row)
            tmp=root/"per-seed.tmp"
            with tmp.open("w",encoding="utf-8-sig",newline="") as f:
                wr=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                wr.writeheader()
                wr.writerows(rows)
            os.replace(tmp,root/"per-seed.csv")
            log(f"isolet5 {arm} {status} elapsed={timer['elapsed_s']:.2f}s")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip",type=Path,default=ZIP)
    p.add_argument("--run",type=Path,default=DEFAULT)
    args=p.parse_args()
    run(args.run.resolve(),args.zip.resolve())
