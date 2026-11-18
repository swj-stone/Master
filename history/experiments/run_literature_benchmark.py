"""Run the literature comparison from an independent, cold process for each route."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import MSK, ROOT, supervise
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, _sha, prepare
from minppi_mofs.experiment_protocol import atomic_write_json

DATASETS = ("abalone","spambase","madelon","credit_card_fraud","gisette","isolet")
CONTROLS = ("full","minpi","gain","pfi")
ESTIMATORS = ("minshap","shapley","banzhaf","shapg")
MODES = ("shared","native","frozen")
DEFAULT = ROOT/"results/test/thesis_sampling/literature-games-raw-six-seed42"
UNGATED = DEFAULT.with_name(DEFAULT.name+"-no-gate")
LIMIT = 300

def read(path):
    return json.loads(path.read_bytes())

def worker(d,mode,m,out,enforce_guard,support_only=False,native_workers=1):
    if native_workers>1:
        # Keep the handler alive through interpreter and native runtime teardown.
        import faulthandler
        crash=(out/"crash.log").open("a",encoding="utf-8",buffering=1)
        print(f"Native diagnostic handler active, pid={os.getpid()}",file=crash)
        faulthandler.enable(file=crash,all_threads=True)
        # Some Windows fail-fast exits bypass Python's exception handler. Periodic
        # stacks still provide a last observed location for each live thread.
        faulthandler.dump_traceback_later(15,repeat=True,file=crash)
        worker.crash_stream=crash
    try:
        from threadpoolctl import threadpool_limits
        from experiments.literature_worker import trial
        with threadpool_limits(limits=1):
            trial(d,mode,m,out,enforce_guard=enforce_guard,minshap_support_only=support_only,native_workers=native_workers)
        if native_workers>1:
            print("Trial returned and native thread limits restored.",file=crash)
            faulthandler.cancel_dump_traceback_later()
    except Exception:
        (out/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise

def jobs(datasets, modes, methods):
    schedule=[]
    for mode in ("shared","frozen","native"):
        if mode not in modes: continue
        for d in datasets:
            group=CONTROLS if mode=="shared" else ESTIMATORS
            i=DATASETS.index(d)%len(group)
            for m in group[i:]+group[:i]:
                if m in methods:
                    schedule.append((d,mode,m))
    return schedule

def run(root, datasets, modes, methods, *, enforce_guard=True, minshap_support_only=False,native_workers=1):
    todo=jobs(datasets,modes,methods)
    if not todo:
        raise ValueError("No selected route matches the requested mode and method")
    if native_workers not in (1,2):
        raise ValueError("Use one or two native coalition workers")
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root),(root/"run.log").open("a",buffering=1,encoding="utf-8") as stream:
        def log(msg):
            line=f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {msg}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        files=["experiments/run_literature_benchmark.py","experiments/literature_worker.py",
               "experiments/native_fit_pool.py",
               "experiments/test/piv3_raw_splits.py","experiments/test/piv3_lite_arm.py",
               "experiments/test/run_sampled_permutation_seed42.py","experiments/run_thesis_sampling.py",
               "src/minppi_mofs/minpi_selector.py","src/minppi_mofs/piv3_compact_tree.py",
               "src/minppi_mofs/piv3_compact_witness.py","src/minppi_mofs/modeling.py","src/minppi_mofs/utility.py"]
        files.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"src/minppi_mofs/baselines").glob("*.py")))
        if len(files)!=len(set(files)):
            raise ValueError("Duplicate source provenance paths")
        source_sha256={f:_sha(ROOT/f) for f in files}
        inputs={}
        for d in DATASETS:
            meta=prepare(DEFAULT_ROOT,d)
            inputs[d]=dict(width=meta["width"],task=meta["task"],metadata_sha256=_sha(DEFAULT_ROOT/d/"metadata.json"),
                           policy=meta["policy"])
        from importlib.metadata import version
        proto=dict(seed=42,limit_s=LIMIT,enforce_guard=enforce_guard,minshap_support_only=minshap_support_only,native_workers=native_workers,
                   datasets=DATASETS,controls=CONTROLS,methods=ESTIMATORS,
                   source=source_sha256,inputs=inputs,dependencies={n:version(n) for n in
                        ("numpy","xgboost","scikit-learn","pandas","scipy")},python=sys.version.split()[0],
                   scope="Native=original-style coalition refit (same archived train, independent ranking assessment). "
                         "Frozen=common full-trained predictor, masked inputs, same game utility. "
                         "The four shared controls are measured once per dataset and referenced in both views.",
                   guard=("Holdout validation half: if any metric falls by more than 5%, return all original columns."
                          if enforce_guard else
                          "Report validation loss for diagnostics only. Always deliver the method's own candidate.")+
                         " Score test only after decision and train+validation refit.",
                   deadline="One independent cold process per route with 300s watchdog including startup, full model, "
                            "estimator, gate, final fit, and score export. Timed-out arms do not receive test scores.",
                   spec={"minshap":"4 random permutations; native variance threshold alpha .05; frozen ranks by sampled minimum.",
                         "shapley":"4 random permutations. Average marginal on project global utility.",
                         "banzhaf":"32 complementary pairs of uniform masks using shared evaluations.",
                         "shapg":"Published graph-guided ShapG approximation: Pearson graph, depth 1, m=3; native F1/R2.",
                         "minpi":"Latest compact frozen witness voting, minimum of three context net votes, no downstream PFI.",
                         "gain":"Mean split gain + adaptive frozen prefix sizing.",
                         "pfi":"Positive sampled permutation importance, 3 repeats and <=128 ranking-side rows."})
        # Round-trip tuples into JSON arrays before comparing protocol identity on resume.
        proto=json.loads(json.dumps(proto))
        if (root/"protocol.json").exists():
            if read(root/"protocol.json")!=proto:
                raise ValueError("Sources, inputs or estimator protocol changed; use a new run directory.")
        else:
            atomic_write_json(root/"protocol.json",proto)
        for i,(d,mode,m) in enumerate(todo,1):
            out=root/d/mode/m
            if (out/"watchdog.json").is_file():
                timer=read(out/"watchdog.json")
                if timer["status"] in ("completed","timed_out"): continue
                raise RuntimeError(f"Earlier route failed: {out}")
            if out.exists():
                raise RuntimeError(f"Interrupted route requires audit of its spent time: {out}")
            out.mkdir(parents=True)
            log(f"{i}/{len(todo)} {d}/{mode}/{m} start")
            support_only = minshap_support_only and mode=="native" and m=="minshap"
            workers=native_workers if mode=="native" and m in ("minshap","shapley") else 1
            proc=mp.get_context("spawn").Process(target=worker,args=(d,mode,m,out,enforce_guard,support_only,workers))
            try:
                timer=supervise(proc,out,LIMIT,log,heartbeat_s=30,
                    heartbeat_formatter=lambda p,e,left:f"{d}/{mode}/{m} {p.get('phase','startup')} "
                        f"finished_fits={p.get('fits', p.get('fits_completed','?'))} elapsed={e:.1f}s "
                        f"remaining_budget={left:.1f}s")
            finally:
                proc.close()
            if timer["status"]=="error":
                raise RuntimeError(f"Inspect {out/'error.log'}")
            result=timer.get("result") or {}
            log(f"{i}/{len(todo)} {d}/{mode}/{m} {timer['status']} "
                f"cold={timer['elapsed_s']:.2f}s delivered_k={result.get('k')} "
                f"fits={result.get('fits',timer.get('progress',{}).get('fits','?'))}")
    return root

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--no-guard",action="store_true",help="Deliver each method's proposed columns regardless of validation loss")
    parser.add_argument("--minshap-support-only", action="store_true",
                        help="In native MinShap, certify negative features early and return the exact selected set.")
    parser.add_argument("--native-workers",type=int,choices=(1,2),default=1,
                        help="Concurrent independent fits on sampled native MinShap and Shapley prefixes.")
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    parser.add_argument("--modes",nargs="+",choices=MODES,default=MODES)
    parser.add_argument("--methods",nargs="+",choices=CONTROLS+ESTIMATORS,default=CONTROLS+ESTIMATORS)
    args=parser.parse_args()
    for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[var]="1"
    root=args.run or (UNGATED if args.no_guard else DEFAULT)
    if args.run is None and args.minshap_support_only:
        root = root.with_name(root.name+"-support-only")
    if args.run is None and args.native_workers!=1:
        root = root.with_name(root.name+f"-w{args.native_workers}")
    run(root.resolve(),args.datasets,args.modes,args.methods,enforce_guard=not args.no_guard,
        minshap_support_only=args.minshap_support_only,native_workers=args.native_workers)

