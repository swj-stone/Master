"""Run or resume the five-seed, six-dataset PIv0 versus direct Gain benchmark."""

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test import run_piv0_seed42 as paired
from experiments.thesis_prepared_catalog import FORMAL_SEEDS, audit_prepared_catalog
from minppi_mofs import piv0
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.utility import compute_utility


SEED42 = paired.RESULTS / "PIv0-seed42-20261111-173806-672616"
RESULTS = ROOT / "results/thesis_benchmark"
ORDER = ("madelon", "spambase", "abalone", "gisette", "credit_card_fraud", "isolet")
RESULT_FIELDS = (
    "negative", "retained", "full_validation_utility", "validation_utility", "test_utility",
    "ranking_fits", "scan_fits", "final_fits", "ranking_s", "voting_s", "scan_s",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


@contextmanager
def exclusive_run(directory):
    # The OS releases this lock when the controller exits, including on interruption.
    with (directory / "campaign.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_completed(out, row, policy, method, voter_hash):
    timer = read(out / "watchdog.json")
    ranking = read(out / "ranking.json")
    frozen = read(out / "frozen.json")
    result = read(out / "result.json")
    seed = int(row["seed"])
    width = int(row["selected_feature_count"])
    features = json.loads(row["selected_features_json"])
    expected = (row["dataset_id"], seed, method, row["utility"])
    if (timer["status"] != "completed" or timer["exit_code"] != 0 or timer["kill_requested_s"] is not None
            or timer["result"] != result or timer["elapsed_s"] <= 0):
        raise ValueError(f"Invalid timing or incomplete result in {out}")
    if (tuple(frozen[key] for key in ("dataset", "seed", "method", "metric")) != expected
            or tuple(ranking[key] for key in ("dataset", "seed", "method", "metric")) != expected
            or ranking["policy"] != policy or ranking["feature_order"] != features
            or ranking["partition_sha256"] != {s: row[f"{s}_file_sha256"] for s in ("train", "val", "test")}
            or (method == "PIv0" and ranking["voter_sha256"] != voter_hash)
            or frozen["ranking_sha256"] != sha(out / "ranking.json")
            or frozen["curve_sha256"] != sha(out / "validation_curve.csv")
            or result["frozen_hash"] != canonical_hash(frozen)):
        raise ValueError(f"Result identity differs from frozen protocol in {out}")
    order = ranking["descending_gain"]
    with (out / "validation_curve.csv").open(encoding="utf-8") as stream:
        trials = list(csv.DictReader(stream))
    negative = set(frozen["negative"])
    if (len(trials) != width - len(negative) or len(order) != len(set(order))
            or set(order) != set(features) - negative or len(trials) != len(order)
            or [int(t["retained"]) for t in trials] != list(range(1, len(order) + 1))):
        raise ValueError(f"Incomplete prefix scan in {out}")
    best = None
    for trial in trials:
        if best is None or float(trial["pooled_utility"]) > float(best["pooled_utility"]) + 1e-14:
            best = trial
    if (best["mask"] != frozen["mask"] or int(best["retained"]) != result["retained"]
            or set(frozen["selected"]) != set(order[:result["retained"]])
            or result["checked"] != len(trials) or result["fits"] != len(trials) + 1 + bool(negative)):
        raise ValueError(f"Frozen winner or fit count mismatch in {out}")
    with np.load(out / "predictions.npz", allow_pickle=False) as archive:
        score = compute_utility(archive["y"], archive["selected"], metric=row["utility"])
        if len(archive["y"]) != int(row["test_row_count"]) or abs(score - result["test_utility"]) > 1e-12:
            raise ValueError(f"Frozen prediction score mismatch in {out}")
    return timer


def arm_summary(row, method, timer, origin):
    result = timer["result"]
    return dict(dataset=row["dataset_id"], seed=int(row["seed"]), method=method, metric=row["utility"],
                width=int(row["selected_feature_count"]), status=timer["status"], checked=result["checked"],
                candidates=result["candidates"], fits=result["fits"], wall_s=timer["elapsed_s"], origin=origin,
                **{key: result[key] for key in RESULT_FIELDS})


def campaign(root, reuse, jobs, catalog):
    old = read(reuse / "protocol.json")
    config = catalog["shared"]
    voter_hash = sha(Path(piv0.__file__))
    if (old["version"] != "PIv0" or old["seed"] != 42 or old["catalog_hash"] != catalog["sha256"]
            or old["model_params"] != config["model_params"] or old["backend"] != config["backend"]
            or old["tie_tolerance"] != 1e-14 or set(old["paired_methods"]) != set(paired.METHODS)
            or sha(reuse/"source/piv0.py") != voter_hash or sha(reuse/"source/run_piv0_seed42.py") != old["run_sha256"]):
        raise ValueError("Archived seed 42 is incompatible with the five-seed protocol")
    sources = {"benchmark": Path(__file__), "worker": Path(paired.__file__), "voter": Path(piv0.__file__)}
    protocol = dict(version="PIv0", datasets=list(ORDER), seeds=list(FORMAL_SEEDS), methods=list(paired.METHODS),
                    limit_s_per_arm=None, catalog_sha256=catalog["sha256"], model_params=config["model_params"],
                    backend=config["backend"], seed42_source=str(reuse.resolve()), cold_start=True,
                    candidate_sizes=old["candidate_sizes"], criterion=old["criterion"], tie_tolerance=1e-14,
                    source_sha256={k:sha(v) for k,v in sources.items()})
    protocol_path = root/"protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != protocol:
            raise ValueError("Resume must use the original inputs and implementation")
    else:
        if any(root.glob("seed-*")):
            raise ValueError("Existing arm directories require their original protocol.json")
        atomic_write_json(protocol_path, protocol)
    (root/"source").mkdir(exist_ok=True)
    for file in sources.values():
        shutil.copy2(file, root/"source"/file.name)
    manifests = {}
    for src in set(jobs.values()):
        with (src/"data_manifest.csv").open(encoding="utf-8-sig") as stream:
            for record in csv.DictReader(stream):
                for split in ("train", "val", "test"):
                    record[f"{split}_row_count"] = int(record[f"{split}_row_count"])
                manifests[record["dataset_id"], int(record["seed"])] = record
    tasks = [(d,42,m) for d in ORDER for m in paired.METHODS]
    for i, dataset in enumerate(ORDER):
        for j, seed in enumerate(FORMAL_SEEDS[1:]):
            methods = paired.METHODS if (i+j)%2==0 else paired.METHODS[::-1]
            tasks.extend((dataset,seed,m) for m in methods)
    rows = []
    with (root/"run.log").open("a", encoding="utf-8", buffering=1) as log_file:
        def log(message):
            print(message, file=log_file, flush=True)
            print(message, flush=True)

        log(f"Results: {root}")
        for dataset, seed, method in tasks:
            row = manifests[dataset,seed]
            source = jobs[dataset,seed]
            if not 1 <= int(row["selected_feature_count"]) <= 100:
                raise ValueError(f"Unexpected prepared width: {dataset}/{seed}")
            policy = dict(task=row["task"], seed=seed, model_params=config["model_params"], backend=config["backend"])
            arm = f"seed-{seed}/{dataset}/{method}"
            out = root/arm
            if out.exists() and not (out/"watchdog.json").exists():
                stamp = datetime.now(MSK).strftime("%Y%m%d-%H%M%S-%f")
                out.rename(out.with_name(out.name + ".incomplete-" + stamp))
            if out.exists() and read(out/"watchdog.json")["status"] != "completed":
                stamp = datetime.now(MSK).strftime("%Y%m%d-%H%M%S-%f")
                out.rename(out.with_name(out.name + ".failed-" + stamp))
            if not out.exists() and seed==42:
                archived = reuse/dataset/method
                verify_completed(archived, row, policy, method, voter_hash)
                shutil.copytree(archived, out)
            if not out.exists():
                out.mkdir(parents=True)
                with (out/"run.log").open("x", encoding="utf-8", buffering=1) as arm_log:
                    def live(message):
                        print(message, file=arm_log, flush=True)
                        log(message)

                    live(f"{arm} starting | {len(rows)}/60 complete")
                    proc = multiprocessing.get_context("spawn").Process(
                        target=paired.worker, args=(source, row, policy, method, out))
                    try:
                        timer = supervise(proc, out, None, live, heartbeat_s=10.,
                            heartbeat_formatter=lambda state,t,left: paired.heartbeat(arm,state,t,left))
                    finally:
                        proc.close()
                    if timer["status"] != "completed":
                        raise RuntimeError(f"{arm} {timer['status']}; inspect {out} and resume with --output")
            timer = verify_completed(out, row, policy, method, voter_hash)
            origin = "archived_seed42" if seed==42 else "fresh"
            entry = arm_summary(row, method, timer, origin)
            rows.append(entry)
            write_summary(root, rows)
            log(f"{arm} {origin} | retained={entry['retained']} test={entry['test_utility']:.7f} "
                f"fits={entry['fits']} elapsed={entry['wall_s']:.2f}s | {len(rows)}/60 complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Resume an existing campaign or use a new output directory")
    parser.add_argument("--seed42-source", type=Path, default=SEED42, help="Previously completed paired seed 42 study")
    args = parser.parse_args()
    root = (args.output or RESULTS/datetime.now(MSK).strftime("PIv0-all-seeds-%Y%m%d-%H%M%S-%f")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        jobs, catalog = audit_prepared_catalog()
        campaign(root, args.seed42_source, jobs, catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
