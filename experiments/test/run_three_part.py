"""Three matched seed-42 pilot experiments on the original feature dimensions."""
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
import traceback

from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import ROOT, MSK, supervise
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, _sha, prepare
from minppi_mofs.experiment_protocol import atomic_write_json

DATASETS = ("abalone", "spambase", "madelon", "credit_card_fraud", "gisette", "isolet")
FAMILIES = ("gain", "pfi", "nsga")
METHODS = ("full","gain","min_gain","mean_gain","pfi","min_pfi","mean_pfi",
           "nsga","min_nsga","mean_nsga","mi","extra_trees")
LIMIT = 300.
DEFAULT = ROOT / "results/test/thesis_sampling/three-part-witness-raw-six-seed42"
SOURCES = ("experiments/test/run_three_part.py","experiments/test/three_part_worker.py",
           "experiments/test/three_part_nsga.py","src/minppi_mofs/piv3_compact_witness.py",
           "src/minppi_mofs/piv3_compact_tree.py","src/minppi_mofs/baselines/nsga2_fs.py",
           "experiments/test/piv3_lite_arm.py","experiments/test/piv3_raw_splits.py",
           "experiments/test/piv2_conventional_models.py","experiments/test/run_sampled_permutation_seed42.py",
           "experiments/run_thesis_sampling.py","src/minppi_mofs/modeling.py","src/minppi_mofs/utility.py")

def worker(dataset,method,folder):
    try:
        from threadpoolctl import threadpool_limits
        from experiments.test.three_part_worker import trial
        with threadpool_limits(limits=1):
            trial(dataset,method,folder)
    except Exception:
        (folder/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise

def order_for(i):
    families=FAMILIES[i%3:]+FAMILIES[:i%3]
    order=["full"]
    for j,family in enumerate(families):
        variants=(family,"min_"+family,"mean_"+family)
        off=(i+j)%3
        order.extend(variants[off:]+variants[:off])
    return order+["mi","extra_trees"]

def run(root):
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root), (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(message):
            line=f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {message}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        data={}
        for dataset in DATASETS:
            meta=prepare(DEFAULT_ROOT,dataset)
            data[dataset]=dict(width=meta["width"],hash=_sha(DEFAULT_ROOT/dataset/"metadata.json"),
                               task=meta["task"],policy=meta["policy"])
        protocol=dict(seed=42,datasets=list(DATASETS),methods=list(METHODS),limit_s=LIMIT,
                      source_sha256={f:_sha(ROOT/f) for f in SOURCES},data=data,
                      front="Unchanged src/minppi_mofs/piv3_compact_witness.select; per-family inputs and seed identical. "
                            "Mean only changes the aggregation of context votes. Full XGBoost training cost included.",
                      validation="Historical disjoint ranking and guard halves. One independent check on the proposed "
                                 "candidate; all arms use original width as fallback. Relative max loss 3%, AP 5%. "
                                 "Test evaluated once after feature list freezes and train+val refit.",
                      search=dict(algorithm="Gain-seeded binary NSGA-II",population=8,evaluations=24,
                                  ref_point=[1.1,1.1],objectives=["k/original_p","1-primary_rank_utility"],
                                  choose="smallest front subset passing all rank metrics, otherwise minimum worst "
                                         "relative deterioration; one independent guard"),
                      timing="Independent serial cold processes with a 300s external kill, one per dataset/method. "
                             "Cold wall and active time both include frontend when used, no cross-arm fit reuse. "
                             "Input SHA verification is done once before training.",
                      assessment="Test point tolerance 5% per metric versus full and versus unprefiltered matched "
                                 "backend. AP included on fraud, MAE lower is better. Single timing observation per arm.")
        if (root/"protocol.json").exists():
            recorded=json.loads((root/"protocol.json").read_bytes())
            if recorded!=protocol:
                raise ValueError("Run data, configuration or implementation changed; use a new run directory")
        else:
            atomic_write_json(root/"protocol.json",protocol)
        for i,d in enumerate(DATASETS):
            for j,m in enumerate(order_for(i)):
                idx=i*len(METHODS)+j+1
                folder=root/d/m
                marker=folder/"watchdog.json"
                if marker.exists():
                    old=json.loads(marker.read_bytes())
                    if old["status"] in ("completed","timed_out"): continue
                    raise RuntimeError(f"Inspect {folder/'error.log'}")
                if folder.exists():
                    raise RuntimeError(f"Interrupted route {folder} needs review before resuming")
                folder.mkdir(parents=True)
                log(f"{idx}/72 {d}/{m} start; cold limit {LIMIT:.0f}s")
                proc=mp.get_context("spawn").Process(target=worker,args=(d,m,folder))
                try:
                    timer=supervise(proc,folder,LIMIT,log,heartbeat_s=20.,
                        heartbeat_formatter=lambda p,t,left:f"{d}/{m} {p.get('phase','startup')} "
                        f"fits={p.get('fits','?')} candidates={p.get('checked','?')} "
                        f"elapsed={t:.1f}s budget_left={left:.1f}s")
                finally:
                    proc.close()
                if timer["status"]=="error":
                    raise RuntimeError(f"Check {folder/'error.log'} and {marker}")
                result=timer["result"]
                log(f"{idx}/72 {d}/{m} {timer['status']}, k={result.get('k')}, cold={timer['elapsed_s']:.2f}s, "
                    f"active={result.get('elapsed_s')}, full_fallback={result.get('fallback')}")
    return root

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    args=parser.parse_args()
    for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[var]="1"
    run(args.run.resolve())

