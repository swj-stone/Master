"""Time MinShap and permutation Shapley together on the same four sampled paths."""
import argparse
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

from experiments.run_literature_benchmark import DATASETS, LIMIT, worker
from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import MSK, ROOT, supervise
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, _sha, prepare
from minppi_mofs.experiment_protocol import atomic_write_json


def run(root,datasets,modes,*,enforce_guard=False,native_workers=1):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    ids=list(dict.fromkeys(datasets))
    games=list(dict.fromkeys(modes))
    if not ids or not games or set(ids)-set(DATASETS) or set(games)-{"native","frozen"}:
        raise ValueError("Select at least one supported dataset and game")
    if native_workers not in (1,2) or (native_workers>1 and "native" not in games):
        raise ValueError("Select the native game to run concurrent coalition fits")
    with exclusive_run(root):
        manifest=root/"protocol.json"
        if not manifest.exists() and any(p.name!="campaign.lock" for p in root.iterdir()):
            raise ValueError(f"Expected a fresh pair results directory: {root}")
        files=[Path(__file__),ROOT/"experiments/literature_worker.py",ROOT/"experiments/native_fit_pool.py",
               ROOT/"experiments/run_literature_benchmark.py",
               ROOT/"experiments/run_thesis_sampling.py",ROOT/"experiments/test/piv3_raw_splits.py",
               ROOT/"experiments/test/piv3_lite_arm.py",ROOT/"src/minppi_mofs/piv3_compact_tree.py",
               ROOT/"src/minppi_mofs/modeling.py",ROOT/"src/minppi_mofs/utility.py"]
        files.extend(sorted((ROOT/"src/minppi_mofs/baselines").glob("*.py")))
        inputs={}
        for dataset in ids:
            metadata=prepare(DEFAULT_ROOT,dataset)
            inputs[dataset]=dict(width=metadata["width"],policy=metadata["policy"],
                                 sha256=_sha(DEFAULT_ROOT/dataset/"metadata.json"))
        protocol=dict(seed=42,methods=["minshap","shapley"],datasets=ids,modes=games,
                      limit_s=LIMIT,enforce_guard=enforce_guard,native_workers=native_workers,
                      inputs=inputs,python=sys.version.split()[0],
                      source={f.resolve().relative_to(ROOT).as_posix():_sha(f) for f in files},
                      timing="One cold process per pair including loading, sampling and both final predictions.",
                      fits="One ledger entry per physical fit; per-method counts include shared work.")
        if manifest.exists() and json.loads(manifest.read_bytes())!=protocol:
            raise ValueError("Pair protocol or sources changed; use a new run directory")
        if not manifest.exists():
            atomic_write_json(manifest,protocol)
        with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
            def log(message):
                line=f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {message}"
                print(line,flush=True)
                print(line,file=stream,flush=True)
            for mode in games:
                for dataset in ids:
                    folder=root/dataset/mode
                    if (folder/"watchdog.json").is_file():
                        status=json.loads((folder/"watchdog.json").read_bytes())["status"]
                        if status in ("completed","timed_out"):
                            continue
                        raise RuntimeError(f"Previous run ended with {status}: {folder}")
                    if folder.exists():
                        raise RuntimeError(f"Interrupted or unfinished run (see fits.jsonl and progress.json): {folder}")
                    folder.mkdir(parents=True)
                    workers=native_workers if mode=="native" else 1
                    log(f"{dataset}/{mode} pair start native_workers={workers}")
                    proc=mp.get_context("spawn").Process(target=worker,
                        args=(dataset,mode,"permutation_pair",folder,enforce_guard,False,workers))
                    started=time.monotonic()
                    try:
                        outcome=supervise(proc,folder,LIMIT,log,heartbeat_s=10,
                            heartbeat_formatter=lambda p,t,left:
                                f"{dataset}/{mode} {p.get('phase','startup')} fits={p.get('fits',0)} "
                                f"inflight={p.get('inflight_fit_ids',[])} "
                                f"prefixes={p.get('checked',0)}/{p.get('total','?')} elapsed={t:.1f}s remaining={left:.1f}s")
                    except KeyboardInterrupt:
                        elapsed=time.monotonic()-started
                        atomic_write_json(folder/"interruption.json",dict(dataset=dataset,mode=mode,status="interrupted",
                                          reason="stopped_by_user",elapsed_with_cleanup_s=elapsed))
                        log(f"{dataset}/{mode} pair interrupted, cold including cleanup={elapsed:.2f}s")
                        raise
                    finally:
                        proc.close()
                    log(f"{dataset}/{mode} pair {outcome['status']} cold={outcome['elapsed_s']:.2f}s")
                    if outcome["status"]=="error":
                        raise RuntimeError(f"See {folder/'error.log'}")
    return root


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,required=True)
    parser.add_argument("--modes",nargs="+",choices=["native","frozen"],default=["frozen"])
    parser.add_argument("--run",type=Path)
    parser.add_argument("--guard",action="store_true",help="Revert to full width on a validation loss above 5%%")
    parser.add_argument("--native-workers",type=int,choices=(1,2),default=1,
                        help="Maximum concurrent coalition refits in native games")
    args=parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    folder=args.run or ROOT/"results/test/thesis_sampling"/datetime.now(MSK).strftime("permutation-pair-%Y%m%d-%H%M%S-%f")
    run(folder,args.datasets,args.modes,enforce_guard=args.guard,native_workers=args.native_workers)
