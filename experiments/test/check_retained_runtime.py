"""Time the two reserve selectors on archived seed-42 inputs, in isolated processes."""
import argparse
import csv
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import time
import traceback

from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import ROOT, supervise
from experiments.thesis_prepared_catalog import FORMAL_DATASETS
from minppi_mofs.experiment_protocol import atomic_write_json


METHODS = ("tree_shap_top_k", "nsga2")
EQUAL_WORK = ("tree_before", "tree_after", "nsga_before", "nsga_after")
INPUTS = ROOT / "results/thesis_benchmark/literature-preprocessed-formal-five-seeds"
TESTS = ROOT / "results/test/thesis_sampling"
LIMIT_S = 300


def measure(inputs, dataset, method, out):
    from importlib.metadata import version
    import numpy as np
    from sklearn.model_selection import train_test_split
    from experiments.test.piv3_raw_splits import _sha, load_split
    from minppi_mofs.baselines import nsga2_fs, shap_top_k

    source = inputs / "inputs/seed-42" / dataset
    meta = json.loads((source / "metadata.json").read_bytes())
    root_pins = json.loads((inputs / "protocol.json").read_bytes())["inputs"]["42"][dataset]
    if (meta["dataset"] != dataset or meta["seed"] != 42 or meta["input_type"] != "archived_preprocessed_csv"
            or _sha(source / "metadata.json") != root_pins["metadata_sha256"]):
        raise ValueError(f"Unexpected prepared inputs: {source}")
    parts = ("train",) if method == "tree_shap_top_k" else ("train","val")
    for part in parts:
        for filename in (f"{part}.npy", f"{part}_y.npy"):
            if _sha(source / filename) != meta["sha256"][filename]:
                raise ValueError(f"Prepared input changed: {source / filename}")
    train, yt = load_split(source.parent, dataset, "train")
    policy = meta["policy"]
    work = dict(dataset=dataset, method=method, seed=42, width=len(train.columns),
                train_rows=len(train), backend=policy["backend"], xgboost=version("xgboost"),
                metadata_sha256=root_pins["metadata_sha256"])

    def progress(phase, **fields):
        atomic_write_json(out / "progress.json", dict(phase=phase, **fields))

    if method == "tree_shap_top_k":
        diagnostics = {}
        progress("fit_and_explain")
        start = time.perf_counter()
        scores = shap_top_k.compute_shap_topk_importance(train, yt, diagnostics=diagnostics, **policy)
        work.update(selector_s=time.perf_counter()-start, fits=diagnostics["model_fit_count"],
                    explained_rows=min(len(train), shap_top_k.DEFAULT_EXPLANATION_ROWS), scores=scores)
    else:
        val, yv = load_split(source.parent, dataset, "val")
        rank, _ = train_test_split(np.arange(len(val)), test_size=.5, random_state=42,
                                   stratify=None if policy["task"] == "regression" else yv)
        rank = np.sort(rank)
        total = nsga2_fs.DEFAULT_POP_SIZE * (1 + nsga2_fs.DEFAULT_MAX_GEN)
        fits = 0
        def evaluated(*, logical_evaluation_count, unique_subset_count, memoization_hit_count, selected_count, cached):
            nonlocal fits
            fits += int(not cached and selected_count > 0)
            progress("search", done=logical_evaluation_count, total=total, fits=fits,
                     unique=unique_subset_count, cache_hits=memoization_hit_count)
        progress("search", done=0, total=total, fits=0)
        start = time.perf_counter()
        front = nsga2_fs.nsga2_feature_selection(train, yt, val.iloc[rank], yv[rank], progress=evaluated, **policy)
        work.update(selector_s=time.perf_counter()-start, fits=fits, rank_rows=len(rank),
                    population=nsga2_fs.DEFAULT_POP_SIZE, generations=nsga2_fs.DEFAULT_MAX_GEN,
                    unique=front.unique_subset_count, requests=front.logical_evaluation_count,
                    cache_hits=front.memoization_hit_count, pareto_size=len(front.pareto_front), hv=front.hv,
                    front=[(mask.tolist(), list(values)) for mask, values in front.pareto_front],
                    hv_history=front.hv_history)
    work["selector_sha256"] = _sha(Path((shap_top_k if method == "tree_shap_top_k" else nsga2_fs).__file__))
    atomic_write_json(out / "result.json", work)
    progress("completed", fits=work["fits"])


def worker(inputs, dataset, method, out, receipt):
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            if method in EQUAL_WORK:
                from experiments.test.retained_equal_work import measure as compare
                compare(inputs, dataset, method, out)
            else:
                measure(inputs, dataset, method, out)
        write_worker_receipt(receipt, 0)
    except Exception:
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=FORMAL_DATASETS, default=FORMAL_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS+EQUAL_WORK, default=METHODS)
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="Existing formal run with audited prepared inputs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = (args.out or TESTS / f"retained-runtime-{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    with exclusive_run(out), (out / "run.log").open("a", encoding="utf-8", buffering=1) as stream:
        def log(message):
            line = f"[{datetime.now():%H:%M:%S}] {message}"
            print(line, flush=True)
            print(line, file=stream, flush=True)
        for dataset in dict.fromkeys(args.datasets):
            for method in dict.fromkeys(args.methods):
                folder = out / dataset / method
                folder.mkdir(parents=True, exist_ok=False)
                label = f"{dataset}/{method}"
                log(f"{label} start (300 s deadline)")
                receipt = WorkerContract(folder / "completion.json", secrets.token_hex(32))
                proc = mp.get_context("spawn").Process(target=worker,
                          args=(args.inputs.resolve(), dataset, method, folder, receipt))
                def heartbeat(p, elapsed, left):
                    done, total = p.get("done",0), p.get("total")
                    eta = f" | ETA ~{elapsed*(total-done)/done:.0f}s" if done and total else ""
                    return (f"{label} {p.get('phase','startup')} | progress={done}/{total or '?'}"
                            f" | fits={p.get('fits','?')} | elapsed={elapsed:.1f}s | budget left={left:.1f}s{eta}")
                try:
                    outcome = supervise(proc, folder, LIMIT_S, log, heartbeat_s=15,
                                        heartbeat_formatter=heartbeat, completion_contract=receipt)
                finally:
                    proc.close()
                log(f"{label} {outcome['status']} | cold={outcome['elapsed_s']:.2f}s")
                if outcome["status"] == "error":
                    raise RuntimeError(f"Worker failed; inspect {folder}")
        fields = ["dataset","method","status","width","cold_s","selector_s","fits","explained_rows","requests",
                  "unique","cache_hits","pareto_size"]
        with (out / "timings.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for path in sorted(out.glob("*/*/watchdog.json")):
                watch = json.loads(path.read_bytes())
                row = watch["result"] if watch["status"] == "completed" else watch["progress"]
                writer.writerow({**row, "dataset":path.parent.parent.name, "method":path.parent.name,
                                 "status":watch["status"], "cold_s":watch["elapsed_s"]})
    print(out / "timings.csv")


if __name__ == "__main__":
    main()
