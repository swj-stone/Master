"""Freeze or resume the full, serial PMLB campaign; stream progress and enforce fold deadlines."""
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import sklearn
import numpy
import xgboost

from experiments.context_study.data import CONFIG, OUT, ROOT, config, digest


def lock():
    pinned = ["experiments/context_study/benchmark.py","experiments/context_study/selection.py",
              "experiments/context_study/run.py","experiments/context_study/data.py",
              "experiments/test/minpi_convergence_metrics.py","src/minppi_mofs/context_vote.py",
              "src/minppi_mofs/context_design.py","src/minppi_mofs/piv3_compact_tree.py",
              "src/minppi_mofs/piv3_compact_witness.py","src/minppi_mofs/utility.py"]
    spec = dict(design=config(), spec_sha256=digest(CONFIG), sources_sha256=digest(OUT/"sources.json"),
                code={p:digest(ROOT/p) for p in pinned},
                runtime=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                             numpy=numpy.__version__, xgboost=xgboost.__version__, system=platform.platform()))
    path = OUT/"protocol-lock.json"
    if path.exists():
        if json.loads(path.read_bytes()) != spec:
            raise RuntimeError("Protocol, input sources, runtime or implementation changed; use a new study ID.")
    else:
        path.write_text(json.dumps(spec, indent=2), encoding="utf8")


def main():
    lock()
    log = OUT/"run.log"
    tasks = [(d, f) for d in config()["datasets"] for f in range(3)]
    records = []
    observed = []
    errors = []
    def emit(line):
        entry = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {line}"
        print(entry,flush=True)
        with log.open("a",encoding="utf8") as stream:
            print(entry,file=stream)
    for n,(dataset,fold) in enumerate(tasks):
        folder = OUT/dataset/f"fold-{fold}"
        done = folder/"completed.json"
        if not done.exists():
            start = perf_counter()
            emit(f"Starting {dataset} fold {fold+1}/3 ({n+1}/{len(tasks)})")
            try:
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                subprocess.run([sys.executable,"-m","experiments.context_study.benchmark",dataset,str(fold)],
                               cwd=ROOT, env=env, timeout=3600, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                emit(f"FAILED {dataset} fold {fold}: {exc}")
                errors.append((dataset,fold,str(exc)))
                continue
            elapsed = perf_counter()-start
            observed.append(elapsed)
            emit(f"Completed {dataset} fold {fold+1} in {elapsed:.1f}s; estimated remaining "
                 f"{(len(tasks)-n-1)*sum(observed)/len(observed):.0f}s")
        result = json.loads(done.read_bytes())
        if (result["protocol_sha256"] != digest(OUT/"protocol-lock.json") or
            result["predictions_sha256"] != digest(folder/"predictions.npz") or
            result["selection_sha256"] != digest(folder/"selection.json")):
            raise ValueError(f"Receipt verification failed in {folder}")
        records.extend(result["records"])
    if records:
        with (OUT/"per-fold.csv").open("w",newline="",encoding="utf8") as stream:
            writer = csv.DictWriter(stream,fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    (OUT/"failures.json").write_text(json.dumps(errors,indent=2),encoding="utf8")
    if errors:
        raise RuntimeError(f"{len(errors)} unsuccessful folds; all successes and failures written.")
    emit(f"Completed {len(records)} evaluations in {len(tasks)} disjoint outer folds")


if __name__ == "__main__":
    main()

