"""Isolated ISOLET pilot. Fix data/model/Gain seeds and vary only the voting sample."""
import argparse
import csv
from datetime import datetime
from itertools import combinations
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import traceback
from unittest.mock import patch

from experiments.native_process_guard import WorkerContract, write_worker_receipt
from experiments.run_thesis_sampling import ROOT, supervise
from minppi_mofs.experiment_protocol import atomic_write_json


REFERENCE = ROOT / "results/thesis_benchmark/baselines-eight-preprocessed-five-seeds"
MODES = ("five_blocks", "legacy", "full")
SEEDS = (42, 123, 456)


def worker(mode, seed, folder, receipt):
    from threadpoolctl import threadpool_limits
    from experiments import literature_worker
    from experiments.test.minpi_local_convergence import select
    from experiments.test.piv3_raw_splits import _sha
    from minppi_mofs import piv3_compact_witness as witness
    from minppi_mofs.baselines._selection_context import Selection
    import numpy as np

    def selector(ctx):
        weights_seed = np.random.SeedSequence([seed, 659])
        if mode == "five_blocks":
            sample = witness._sample
            with patch.object(witness, "_sample", lambda y,t,_: sample(y,t,weights_seed)):
                names, info = witness.select(ctx.full_model, ctx.train, ctx.yt, ctx.ranking, ctx.yr,
                                             task=ctx.task, utility=ctx.utility, seed=ctx.seed)
            info["stop_reason"] = ("proposal_cap" if len(info["witnesses"]) == witness.MAX_BLOCKS
                                   else "queue_empty_or_minimum_size")
        else:
            anchor, info = witness.gain_anchor(ctx.full_model, ctx.train, ctx.yt, ctx.ranking, ctx.yr,
                                     task=ctx.task, utility=ctx.utility, seed=ctx.seed, aggregate="gain")
            def progress(**fields):
                if fields["checked"] == 1 or fields["checked"] % 10 == 0:
                    ctx.progress("singleton_scan", **fields)
            names, detail = select(info, ctx.full_model, ctx.train, ctx.ranking, ctx.yr,
                                   task=ctx.task, utility=ctx.utility, search_seed=seed, vote_mode=mode,
                                   background_seed=ctx.seed, progress=progress)
            info.update(detail, anchor=anchor, k=len(names), witness_predictions=detail["voting_calls"],
                        witness_rows=detail["voting_row_ids"])
        info.update(search_seed=seed, search_mode=mode)
        return Selection(selected=names, diagnostics=info)

    try:
        inputs = REFERENCE / "inputs/seed-42"
        spec = json.loads((REFERENCE / "protocol.json").read_bytes())
        meta = inputs / "isolet/metadata.json"
        if _sha(meta) != spec["inputs"]["42"]["isolet"]["metadata_sha256"]:
            raise ValueError("ISOLET metadata differs from the reference run")
        for name,digest in json.loads(meta.read_bytes())["sha256"].items():
            if _sha(meta.parent / name) != digest:
                raise ValueError(f"ISOLET partition changed: {name}")
        with threadpool_limits(limits=1), patch.object(literature_worker, "get_selector", return_value=selector):
            literature_worker.trial("isolet", "shared", "minpi", folder, seed=42,
                                    input_root=inputs, enforce_guard=False)
        write_worker_receipt(receipt, 0)
    except Exception:
        (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def summarize(root):
    rows, sets = [], {}
    for mode in MODES:
        for seed in SEEDS:
            folder = root / mode / str(seed)
            watchdog = json.loads((folder / "watchdog.json").read_bytes())
            row = dict(mode=mode, search_seed=seed, status=watchdog["status"], cold_s=watchdog["elapsed_s"])
            if watchdog["status"] == "completed":
                result = watchdog["result"]
                info = json.loads((folder / "estimator.json").read_bytes())
                sets[mode,seed] = set(result["selected"])
                row.update(k=result["k"], stop=info["stop_reason"], fits=result["fits"],
                           proposals=info["proposals"] if "proposals" in info else len(info["witnesses"]),
                           terminal_checked=info.get("final_sweep_covered"), selector_s=result["selector_s"],
                           calls=result["sample_prediction_calls"], rows=result["sample_prediction_rows"],
                           validation_auc=result["validation_selected"]["auc"],
                           validation_accuracy=result["validation_selected"]["accuracy"], **result["test_scores"])
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    stability = {}
    for mode in MODES:
        data = [(seed,sets[mode,seed]) for seed in SEEDS if (mode,seed) in sets]
        stability[mode] = {f"{i}-{j}":len(a&b)/len(a|b) for (i,a),(j,b) in combinations(data,2)}
    atomic_write_json(root / "jaccard.json",stability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    root = parser.parse_args().out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    with (root / "run.log").open("w",encoding="utf-8",buffering=1) as stream:
        def log(message):
            line = f"[{datetime.now():%H:%M:%S}] {message}"
            print(line, flush=True)
            print(line, file=stream, flush=True)
        for seed in SEEDS:
            for mode in MODES:
                folder=root / mode / str(seed)
                folder.mkdir(parents=True)
                receipt=WorkerContract(folder/"completion.json",secrets.token_hex(32))
                label=f"{mode}/vote-{seed}"
                log(f"{label} started (300s cap)")
                proc=mp.get_context("spawn").Process(target=worker,args=(mode,seed,folder,receipt))
                try:
                    outcome=supervise(proc,folder,300,log,heartbeat_s=15,completion_contract=receipt,
                        heartbeat_formatter=lambda p,t,left: f"{label} {p.get('phase','startup')} "
                        f"sweep={p.get('sweep','-')} k={p.get('k','-')} elapsed={t:.0f}s left={left:.0f}s")
                finally:
                    proc.close()
                log(f"{label} {outcome['status']} in {outcome['elapsed_s']:.2f}s k={outcome['result'].get('k')}")
                if outcome["status"]=="error":
                    raise RuntimeError(f"See {folder}")
    summarize(root)


if __name__=="__main__":
    main()
