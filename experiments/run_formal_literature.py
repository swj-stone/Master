"""Run both literature games on the archived five-seed prepared inputs."""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
from statistics import median
import sys
import traceback

from experiments.run_literature_benchmark import DATASETS, read
from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import MSK, ROOT, supervise
from experiments.test.piv3_raw_splits import _sha
from experiments.thesis_prepared_catalog import FORMAL_SEEDS
from minppi_mofs.baselines.sampling_shap import BACKGROUND_LIMIT, EXPLANATION_LIMIT, PAIRS_PER_FEATURE
from minppi_mofs.experiment_protocol import atomic_write_json


DEFAULT = ROOT / "results/thesis_benchmark/literature-preprocessed-samplingshap-five-seeds"
CONTROLS = ("full", "minpi", "gain", "pfi", "sampling_shap")
ESTIMATORS = ("minshap", "banzhaf", "shapg")


def protocol(inputs, seeds=FORMAL_SEEDS, datasets=DATASETS, methods=CONTROLS+ESTIMATORS):
    schedule = []
    for mode in ("shared", "frozen", "native"):
        group = CONTROLS if mode == "shared" else ESTIMATORS
        for seed in seeds:
            for dataset in datasets:
                offset = DATASETS.index(dataset) % len(group)
                schedule.extend(dict(seed=seed, dataset=dataset, mode=mode, method=m)
                                for m in group[offset:]+group[:offset] if m in methods)
    sources = ["experiments/run_formal_literature.py", "experiments/formal_literature_inputs.py",
               "experiments/literature_worker.py", "experiments/run_literature_benchmark.py",
               "experiments/run_thesis_sampling.py", "experiments/native_process_guard.py",
               "experiments/thesis_prepared_catalog.py",
               "experiments/test/piv3_raw_splits.py", "experiments/test/piv3_lite_arm.py",
               "experiments/test/run_sampled_permutation_seed42.py",
               "src/minppi_mofs/minpi_selector.py", "src/minppi_mofs/piv3_compact_tree.py",
               "src/minppi_mofs/piv3_compact_witness.py", "src/minppi_mofs/modeling.py",
               "src/minppi_mofs/utility.py", "src/minppi_mofs/experiment_protocol.py",
               "src/minppi_mofs/prediction_meter.py"]
    sources.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"src/minppi_mofs/baselines").glob("*.py")))
    return dict(seeds=list(seeds), datasets=list(datasets), controls=[m for m in CONTROLS if m in methods],
                methods=[m for m in ESTIMATORS if m in methods], jobs=schedule, inputs=inputs, limit_s=None,
                enforce_guard=False, native_workers=1, minshap_support_only=True,
                timing="Independent, serial cold processes; includes startup, selection, refits and export.",
                selection="Deliver the proposed subset. The 5% validation diagnostic never triggers fallback.",
                sampling_shap=dict(implementation="shap.SamplingExplainer",
                                   max_background_train_rows=BACKGROUND_LIMIT,
                                   max_explained_ranking_rows=EXPLANATION_LIMIT,
                                   paired_samples_budget_per_feature_per_row=PAIRS_PER_FEATURE,
                                   ranking="mean absolute local contribution over explained rows and outputs"),
                games="SamplingSHAP estimates local model-output attributions once per seed and dataset. "
                      "The other game estimators have native refit and frozen masked variants.",
                source={f:_sha(ROOT/f) for f in sources}, python=sys.version.split()[0],
                dependencies={n:version(n) for n in ("numpy","xgboost","scikit-learn","pandas","scipy","shap")})


def worker(job, out, inputs, contract):
    try:
        from threadpoolctl import threadpool_limits
        from experiments.literature_worker import trial
        support_only = job["mode"] == "native" and job["method"] == "minshap"
        with threadpool_limits(limits=1):
            trial(**job, out=out, input_root=inputs, enforce_guard=False,
                  minshap_support_only=support_only)
        # Release cyclic Python references while XGBoost and its device runtime are still live.
        import gc
        gc.collect()
        write_worker_receipt(contract, 0)
    except Exception:
        (out/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def heartbeat(label, progress, elapsed, history):
    done, total = None, None
    for numerator, denominator in (("pairs_completed","pairs_planned"), ("checked","total"), ("covered","total")):
        if progress.get(denominator):
            done, total = progress.get(numerator), progress[denominator]
            break
    if progress.get("phase") == "minshap_support":
        width = progress["path_width"]
        done, total = (progress["path"]-1)*width + progress["position"], 4*width
    eta = "pending"
    if history:
        eta = f"~{max(0,median(history)-elapsed):.0f}s (prior seeds)"
    elif done and done < total:
        eta = f"~{elapsed*(total-done)/done:.0f}s (stage projection)"
    return (f"{label} {progress.get('phase','startup')} | fits={progress.get('fits',0)}"
            + (f" | progress={done}/{total}" if done is not None else "")
            + f" | elapsed={elapsed:.0f}s | remaining={eta}")


def run(root, *, seeds=FORMAL_SEEDS, datasets=DATASETS, methods=CONTROLS+ESTIMATORS):
    from experiments.formal_literature_inputs import prepare_inputs
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root), (root/"run.log").open("a",buffering=1,encoding="utf-8") as stream:
        def log(msg):
            line = f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {msg}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        log(f"Checking {len(seeds)*len(datasets)} archived dataset/seed partitions and formal protocol.")
        inputs = prepare_inputs(root, seeds=seeds, datasets=datasets)
        spec = json.loads(json.dumps(protocol(inputs, seeds, datasets, methods)))
        path = root/"protocol.json"
        if path.exists() and read(path) != spec:
            raise ValueError("Run protocol differs from its saved version; choose a new output directory.")
        if not path.exists():
            atomic_write_json(path,spec)
        history = {}
        for index,job in enumerate(spec["jobs"],1):
            d, mode, method = (job[k] for k in ("dataset","mode","method"))
            seed = job["seed"]
            key = (d,mode,method)
            times = history.setdefault(key,[])
            rel = Path(f"seed-{seed}",d,mode,method)
            out = root/"runs"/rel
            watchdog = out/"watchdog.json"
            outcome = read(watchdog) if watchdog.exists() else {}
            if outcome.get("status") == "completed":
                res = read(out/"result.json")
                if res != outcome["result"] or any(res[k]!=v for k,v in job.items()):
                    raise ValueError(f"Inconsistent saved result in {out}")
                times.append(outcome["elapsed_s"])
                continue
            if out.exists():
                archive = root/"attempts"/datetime.now().strftime("%Y%m%d-%H%M%S-%f")/rel
                archive.parent.mkdir(parents=True,exist_ok=True)
                out.rename(archive)
                log(f"Preserved incomplete attempt at {archive}")
            out.mkdir(parents=True)
            label = f"{index}/{len(spec['jobs'])} {rel.as_posix()}"
            log(f"{label} start")
            contract = WorkerContract(out/"worker-completion.json", secrets.token_hex(32))
            process = mp.get_context("spawn").Process(target=worker,
                        args=(job,out,root/"inputs"/f"seed-{seed}",contract))
            try:
                outcome = supervise(process,out,None,log,heartbeat_s=30,completion_contract=contract,
                    heartbeat_formatter=lambda p,t,_:heartbeat(label,p,t,times))
            finally:
                process.close()
            if outcome["status"] != "completed":
                raise RuntimeError(f"Failed {label}, exit={outcome['exit_code']}; see {out}")
            res = outcome["result"]
            times.append(outcome["elapsed_s"])
            log(f"{label} completed | cold={times[-1]:.2f}s | k={res['k']}/{res['width']} | "
                f"fits={res['fits']} | test={res['test_scores']}")
        log(f"All {len(spec['jobs'])} routes completed; verifying predictions and writing summaries.")
        from experiments.report_formal_literature import report
        report(root,partial=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,help="Separate output directory for this protocol or subset")
    parser.add_argument("--seeds",type=int,nargs="+",choices=FORMAL_SEEDS)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS)
    parser.add_argument("--methods",nargs="+",choices=CONTROLS+ESTIMATORS)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    if not args.run and any((args.seeds, args.datasets, args.methods)):
        parser.error("Pass --run for a subset so its results stay separate from the full experiment.")
    run((args.run or DEFAULT).resolve(),
        seeds=tuple(dict.fromkeys(args.seeds or FORMAL_SEEDS)),
        datasets=tuple(dict.fromkeys(args.datasets or DATASETS)),
        methods=tuple(dict.fromkeys(args.methods or CONTROLS+ESTIMATORS)))
