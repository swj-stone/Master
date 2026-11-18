"""Run the seed-42 raw-width PIv3-Lite benchmark serially with a hard per-route deadline."""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import ROOT, MSK, supervise
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, prepare, _sha
from minppi_mofs.experiment_protocol import atomic_write_json

DATASETS = ("abalone", "spambase", "madelon", "credit_card_fraud", "gisette", "isolet")
METHODS = ("full", "pi", "pi_mean", "gain", "pi_gain", "extra_trees", "pi_extra_trees",
           "mi", "pi_mi", "banzhaf", "pi_banzhaf", "pfi", "pi_pfi", "f_score", "pi_f_score")
LIMIT = 300.
DEFAULT = ROOT/"results/test/thesis_sampling/PIv3-lite-raw-six-seed42"
SOURCES = ("src/minppi_mofs/piv3_lite.py", "src/minppi_mofs/piv2.py", "src/minppi_mofs/piv1.py",
           "src/minppi_mofs/modeling.py", "src/minppi_mofs/utility.py", "experiments/test/piv3_lite_arm.py",
           "experiments/test/run_piv3_lite_six.py", "experiments/test/piv3_raw_splits.py",
           "experiments/test/piv2_extra_selectors.py", "experiments/test/run_sampled_permutation_seed42.py",
           "experiments/test/piv2_conventional_models.py", "experiments/run_thesis_sampling.py")

def worker(d,m,path):
    try:
        from threadpoolctl import threadpool_limits
        from experiments.test.piv3_lite_arm import run_arm
        with threadpool_limits(limits=1):
            run_arm(d,m,path)
    except Exception:
        (path/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise

def campaign(root):
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root), (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(msg):
            s=f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {msg}"
            print(s,flush=True)
            print(s,file=stream,flush=True)

        pins={s:_sha(ROOT/s) for s in SOURCES}
        metadata={}
        for dataset in DATASETS:
            obj=prepare(DEFAULT_ROOT,dataset)
            metadata[dataset]=dict(width=obj["width"],hash=_sha(DEFAULT_ROOT/dataset/"metadata.json"))
        protocol=dict(seed=42,datasets=list(DATASETS),methods=list(METHODS),data=metadata,seconds_limit=LIMIT,
                      backend="frozen XGB, each arm independent", source_sha256=pins,
                      timing="wall includes spawn, imports, input loads, frontend, validation and test output; "
                             "active elapsed includes inputs, selection, guard, train+val fit and test evaluation. "
                             "A shared immutable dataset cache is verified before training.",
                      analysis="Noninferiority descriptive: all test scores relative loss <=5% vs full AND "
                               "baseline. Time parity with fast selection: active wall ratio <=1.10. Report absolute "
                               "differences, gate rejections and low-width bypass separately. No test tuning.")
        if (root/"protocol.json").exists() and json.loads((root/"protocol.json").read_bytes())!=protocol:
            raise ValueError("Refusing to mix different source or input versions in an existing run")
        atomic_write_json(root/"protocol.json",protocol)
        jobs=[]
        for idx,d in enumerate(DATASETS):
            # Rotate and reverse the backend pairing order across datasets.
            pairs=[METHODS[j:j+2] for j in range(3,len(METHODS),2)]
            pairs=pairs[idx%len(pairs):]+pairs[:idx%len(pairs)]
            order=[*METHODS[:3]]
            for j,pair in enumerate(pairs):
                order.extend(reversed(pair) if (idx+j)%2 else pair)
            jobs.extend((d,m) for m in order)
        start=time.monotonic()
        for i,(d,m) in enumerate(jobs,1):
            folder=root/d/m
            if (folder/"watchdog.json").is_file():
                old=json.loads((folder/"watchdog.json").read_bytes())
                if old.get("status") in ("completed","timed_out"):
                    continue
            if folder.exists():
                folder.rename(folder.with_name(folder.name+"-interrupted-"+datetime.now().strftime("%H%M%S%f")))
            folder.mkdir(parents=True)
            log(f"{i}/{len(jobs)} {d} {m} started")
            proc=mp.get_context("spawn").Process(target=worker,args=(d,m,folder))
            try:
                timer=supervise(proc,folder,LIMIT,log,heartbeat_s=20.,
                               heartbeat_formatter=lambda p,e,left:f"{d}/{m} {p.get('phase','startup')} "
                               f"elapsed={e:.1f}s deadline_in={left:.1f}s fits={p.get('fits','?')}")
            finally:
                proc.close()
            if timer["status"]=="error":
                raise RuntimeError(f"Inspect {folder/'error.log'}")
            r=timer.get("result",{})
            log(f"{i}/{len(jobs)} {d}/{m} {timer['status']} wall={timer['elapsed_s']:.2f}s "
                f"active={r.get('elapsed_s','?')} k={r.get('k','?')} fits={r.get('fits','?')} "
                f"campaign={time.monotonic()-start:.0f}s")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    args=parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    campaign(args.run.resolve())

