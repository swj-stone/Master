"""Matched raw-width benchmarks, with one real time budget per algorithm route."""
import argparse
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback

from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import MSK, read_progress
from experiments.test.report_piv3_raw_six import NAMES, WIDTHS, report
from minppi_mofs.experiment_protocol import atomic_write_json

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "results/test/thesis_sampling"
DATA = BASE / "PIv3-six-raw-seed42-input"
DEFAULT = BASE / "PIv3-raw-six-seed42-tol358"
LIMIT = 900.
DATASETS = ("abalone", "madelon", "spambase", "credit_card_fraud", "gisette", "isolet")
PCTS = (3, 5, 8)
SOURCES = ("experiments/test/run_piv3_raw_six.py", "experiments/test/piv3_raw_stages.py",
           "experiments/test/piv3_raw_splits.py", "experiments/test/piv3_raw_ranker.py",
           "experiments/test/piv2_conventional_filters.py", "experiments/test/piv2_conventional_models.py",
           "experiments/test/piv2_selectors.py", "experiments/test/piv2_extra_selectors.py",
           "experiments/test/piv2_prefix_rankings.py", "src/minppi_mofs/piv3.py", "src/minppi_mofs/piv2.py",
           "src/minppi_mofs/piv1.py", "src/minppi_mofs/utility.py", "src/minppi_mofs/modeling.py",
           "src/minppi_mofs/baselines/shap_top_k.py")

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def worker(pipe):
    from experiments.test.piv3_raw_stages import run_stage
    from experiments.test import piv3_raw_ranker
    from minppi_mofs import piv3
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        pipe.send("ready")
        while True:
            task = pipe.recv()
            if task is None:
                return
            try:
                start = time.monotonic()
                run_stage(*task)
                pipe.send(("completed", time.monotonic() - start, None))
            except Exception:
                pipe.send(("error", time.monotonic() - start, traceback.format_exc()))

class Engine:
    def __init__(self, log):
        self.proc = self.pipe = None
        self.log = log

    def close(self):
        if self.proc is not None:
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join()
            self.proc.close()
            self.pipe.close()
        self.proc = self.pipe = None

    def measure(self, args, budget):
        if self.proc is None:
            ctx = mp.get_context("spawn")
            self.pipe, remote = ctx.Pipe()
            self.proc = ctx.Process(target=worker, args=(remote,), daemon=True)
            self.proc.start()
            remote.close()
            if not self.pipe.poll(100) or self.pipe.recv() != "ready":
                raise RuntimeError("Stage worker failed to initialize")
        out = args[-1]
        started = time.monotonic()
        self.pipe.send(args)
        while True:
            remaining = max(0., started + budget - time.monotonic())
            if self.pipe.poll(min(30., remaining)):
                status, elapsed, err = self.pipe.recv()
                if status == "error":
                    (out/"error.log").write_text(err, encoding="utf-8")
                    raise RuntimeError(f"{out}\n{err}")
                return dict(status=status, elapsed_s=elapsed, budget_s=budget)
            progress = read_progress(out)
            if time.monotonic() >= started + budget:
                self.close()
                return dict(status="timeout", elapsed_s=budget, budget_s=budget, progress=progress)
            self.log(f"{out.relative_to(args[0])} {progress.get('phase','startup')} "
                     f"k={progress.get('checked','?')} fits={progress.get('fits','?')} "
                     f"elapsed={time.monotonic()-started:.0f}s left={started+budget-time.monotonic():.0f}s")

def stage_path(root, dataset, stage, method, arm, pct):
    base = root / "stages" / dataset / stage
    if stage in ("reference", "test_full"):
        return base
    if stage == "front":
        return base / f"tol-{pct:02d}"
    return base / arm / method / f"tol-{pct:02d}"

def stage_run(engine, root, d, step, method, arm, pct, budget):
    out = stage_path(root, d, step, method, arm, pct)
    timing = out / "timer.json"
    if timing.exists():
        timer = read(timing)
    else:
        if out.exists():
            out.rename(out.with_name(out.name+".interrupted-"+datetime.now(MSK).strftime("%H%M%S%f")))
        out.mkdir(parents=True)
        if step == "backend" and arm == "plain" and pct != 3:
            from experiments.test.piv3_raw_shared_prefix import materialize
            materialize(root, d, method, pct, out)
        if timing.exists():
            timer = read(timing)
        else:
            timer = engine.measure((root, DATA, d, step, method, arm, pct, out), budget)
            atomic_write_json(timing, timer)
    allowed = timer["status"] == "completed" and timer["elapsed_s"] <= budget
    result = read(out/"stage.json") if allowed else {}
    return out.relative_to(root).as_posix(), timer, result, allowed

def run(root, datasets):
    paths = {name:sha(PROJECT/name) for name in SOURCES}
    protocol = dict(version="PIv3-raw-six-1", data=str(DATA), data_seed=42, model_seed=42, search_seed=42,
                    datasets=list(DATASETS), methods=list(NAMES), tolerances=[.03,.05,.08],
                    meta_sha256={d:sha(DATA/d/"metadata.json") for d in DATASETS},
                    limit_s=LIMIT, PI_candidates=12, prefix_order="ascending to first validation pass",
                    timing="Sequential measured algorithm stages with inputs, ranking, validation, final full and "
                           "selected refits. All used stages charged per arm. Interpreter startup, common preparation "
                           "and orchestration excluded. Identical deterministic plain prefix shared across tolerances.",
                    source_sha256=paths)
    if (root/"protocol.json").exists() and read(root/"protocol.json") != protocol:
        raise ValueError("Protocol or input changed. Choose a fresh output directory.")
    atomic_write_json(root/"protocol.json",protocol)
    for name in SOURCES:
        dest = root/"source"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            shutil.copyfile(PROJECT/name,dest)
        if sha(dest) != paths[name]:
            raise ValueError(f"Source snapshot mismatch {name}")

    stream=(root/"run.log").open("a",encoding="utf-8",buffering=1)
    def log(message):
        line=f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {message}"
        print(line,flush=True)
        print(line,file=stream,flush=True)

    engine=Engine(log)
    try:
        for d in datasets:
            for method in NAMES:
                for pct in PCTS:
                    for arm in ("plain", "piv3"):
                        folder=root/"arms"/d/method/arm/f"tol-{pct:02}"
                        if (folder/"outcome.json").exists():
                            continue
                        start=time.monotonic()
                        stages, outputs={}, {}
                        charged=0.
                        fits=0
                        good=True
                        phase="completed"
                        steps=["reference"] + (["front"] if arm=="piv3" else []) + ["backend","test_full","test_selected"]
                        for step in steps:
                            rel,t,r,good=stage_run(engine,root,d,step,method,arm,pct,max(LIMIT-charged,0.))
                            stages[step]=rel
                            if not good:
                                phase=step+"/"+t.get("progress",{}).get("phase","deadline")
                                if t["status"]=="timeout":
                                    fits+=t.get("progress",{}).get("fits",0)
                                charged=LIMIT
                                break
                            charged+=t["elapsed_s"]
                            fits+=r["fits"]
                            outputs[step]=r
                        reference=outputs.get("reference",{}).get("scores")
                        test_reference=outputs.get("test_full",{}).get("scores")
                        test=outputs.get("test_selected",{}).get("scores")
                        passed=None
                        if good:
                            passed=all((score <= base*(1+pct/100)+1e-12 if d=="abalone" and i==1
                                        else score+1e-12 >= base*(1-pct/100))
                                       for i,(score,base) in enumerate(zip(test,test_reference,strict=True)))
                        front_s=0. if arm=="plain" else None
                        if "front" in outputs:
                            front_s=read(root/stages["front"]/"timer.json")["elapsed_s"]
                        result=dict(dataset=d,method=method,arm=arm,tolerance=pct/100,width=WIDTHS[d],
                                    status="completed" if good else "timeout", wall_s=charged,fits=fits,
                                    k=outputs.get("backend",{}).get("k"),
                                    front_k=WIDTHS[d] if arm=="plain" else outputs.get("front",{}).get("k"),
                                    front_s=front_s,reference_scores=reference,test_reference_scores=test_reference,
                                    validation_scores=outputs.get("backend",{}).get("scores"),test_scores=test,
                                    test_pass=passed,phase=phase,stages=stages)
                        folder.mkdir(parents=True,exist_ok=True)
                        atomic_write_json(folder/"outcome.json",result)
                        log(f"{d} {method} {arm} {pct}% {result['status']} k={result['k']} "
                            f"fits={fits} route={charged:.2f}s new work={time.monotonic()-start:.2f}s")
                report(root)
        return report(root,require_complete=set(datasets)==set(DATASETS))
    finally:
        engine.close()
        stream.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    args=parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    path=args.run.resolve()
    path.mkdir(parents=True,exist_ok=True)
    with exclusive_run(path):
        print(run(path,args.datasets))

