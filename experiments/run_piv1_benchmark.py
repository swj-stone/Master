"""Run or resume the locked PIv1 versus direct Gain benchmark over the formal prepared splits."""

import argparse
import csv
import json
import multiprocessing
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.run_piv0_benchmark import exclusive_run, read, sha, verify_completed
from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test import run_piv1_pilot as pilot
from experiments.test.summarize_piv1_pilot import records, verify_heldout
from experiments.thesis_prepared_catalog import FORMAL_SEEDS, audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


PILOT = pilot.OUTPUT / "PIv1-pilot-20261111-234033-780781"
RESULTS = ROOT / "results/thesis_benchmark"


def validated_summary(folder, row, policy, method, origin):
    seed = int(row["seed"])
    dataset = row["dataset_id"]
    metric = row["utility"]
    features = json.loads(row["selected_features_json"])
    if method == "direct_gain":
        timer = verify_completed(folder, row, policy, method, None)
        result = timer["result"]
        retained = result["retained"]
        count = result["candidates"]
        validation = result["validation_utility"]
    else:
        ranking, frozen, result, timer = records(folder)
        choice = result["choices"]["vote"]
        count = choice["checked"]
        retained = choice["retained"]
        validation = choice["validation"][0]
        trials = read(folder / "trials-vote.json")
        scores = {tuple(t["indices"]): t["scores"] for t in read(folder / "validation.json")}
        maximum = max(t["utility"] for t in trials)
        winner = min((t for t in trials if t["utility"] >= maximum - pilot.TIE),
                     key=lambda t: (len(t["indices"]), t["indices"]))
        if (set(result["choices"]) != {"vote"} or result["choices"] != frozen["choices"]
                or ranking["policy"] != policy or ranking["features"] != features
                or (ranking["dataset"], ranking["seed"], ranking["metric"]) != (dataset, seed, metric)
                or ranking["partition_sha256"] != {s: row[f"{s}_file_sha256"] for s in ("train", "val", "test")}
                or frozen["ranking_hash"] != canonical_hash(ranking)
                or len(trials) != count or count != choice["budget"] or count + 1 != result["fits"]
                or set(scores) != {tuple(t["indices"]) for t in trials}
                or any(scores[tuple(t["indices"])][0] != t["utility"] for t in trials)
                or choice["indices"] != winner["indices"] or retained != len(choice["indices"])
                or choice["selected"] != [features[i] for i in choice["indices"]]
                or validation != scores[tuple(choice["indices"])][0]
                or (origin == "fresh" and timer["limit_s"] is not None)):
            raise ValueError(f"Invalid PIv1 selection: {folder}")
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        labels = archive["y"]
    if len(labels) != row["test_row_count"]:
        raise ValueError(f"Unexpected held-out sample count: {folder}")
    verify_heldout(folder, labels, metric, result)
    return dict(dataset=dataset, seed=seed, method=method, metric=metric, width=len(features),
                status=timer["status"], origin=origin, retained=retained, candidates=count, checked=count,
                fits=result["fits"], validation_utility=validation, test_utility=result["test_utility"],
                wall_s=timer["elapsed_s"], voting_s=result["voting_s"])


def campaign(root, pilot_run, gain_run):
    jobs, catalog = audit_prepared_catalog()
    model = catalog["shared"]
    study, lock = read(pilot_run / "protocol.json"), read(pilot_run / "lock.json")
    gain = read(gain_run / "protocol.json")
    if (study["source_hashes"] != pilot.source_hashes() or lock["primary_recipe"] != "vote"
            or lock["protocol_hash"] != canonical_hash(study) or catalog["sha256"] != study["catalog_hash"]
            or gain["catalog_sha256"] != catalog["sha256"] or gain["model_params"] != model["model_params"]
            or gain["backend"] != model["backend"] or lock["seeds"] != [42, 123]
            or study["replication_seed"] != 123 or study["development_seed"] != 42
            or set(study["datasets"]) != set(pilot.DATASETS) or gain["seeds"] != list(FORMAL_SEEDS)):
        raise ValueError("Locked recipe, frozen data or model configuration mismatch")
    if any(sha(pilot_run/"source"/name) != h for name, h in study["source_hashes"].items()):
        raise ValueError("Pilot source snapshot changed")
    protocol = dict(version="PIv1", primary_recipe="vote", frozen_lock=canonical_hash(lock),
                    limit_s_per_new_arm=None, seeds=list(FORMAL_SEEDS), datasets=list(pilot.DATASETS),
                    methods=["PIv1","direct_gain"], catalog_hash=catalog["sha256"],
                    model_params=model["model_params"], backend=model["backend"], cold_start=True,
                    pilot_run=str(pilot_run), direct_gain_run=str(gain_run),
                    pilot_source_hashes=study["source_hashes"], runner_sha256=sha(Path(__file__)))
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != protocol:
            raise ValueError("Resume requires the original inputs and code")
    elif any(p.name != "campaign.lock" for p in root.iterdir()):
        raise ValueError("A nonempty output directory requires its original protocol")
    else:
        atomic_write_json(root / "protocol.json", protocol)
    (root / "source").mkdir(exist_ok=True)
    for src in [Path(__file__), *pilot.source_files()]:
        shutil.copy2(src, root/"source"/src.name)
    rows = {}
    for source in set(jobs.values()):
        with (source / "data_manifest.csv").open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for split in ("train","val","test"):
                    row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
                rows[row["dataset_id"],int(row["seed"])] = row
    tasks = [(d,s,"direct_gain") for s in FORMAL_SEEDS for d in pilot.DATASETS]
    tasks.extend((d,s,"PIv1") for d in pilot.DATASETS for s in FORMAL_SEEDS)
    # Verify both archives first so the 42 existing records are available immediately.
    tasks.sort(key=lambda t: t[2] == "PIv1" and t[1] not in lock["seeds"])
    summary = []
    with (root / "run.log").open("a",encoding="utf-8",buffering=1) as log_file:
        def log(message):
            print(message, flush=True)
            print(message, file=log_file, flush=True)

        log(f"Results: {root}")
        for dataset, seed, method in tasks:
            row = rows[dataset,seed]
            if not 1 <= int(row["selected_feature_count"]) <= 100:
                raise ValueError(f"Expected a frozen split of at most 100 features: {dataset}/{seed}")
            policy = dict(task=row["task"], seed=seed, model_params=model["model_params"], backend=model["backend"])
            arm = f"seed-{seed}/{dataset}/{method}"
            out = root / arm
            archived = None
            origin = "fresh"
            if method == "direct_gain":
                archived = gain_run / arm
                origin = "archived_gain"
            elif seed in lock["seeds"]:
                archived = pilot_run / "final" / f"seed-{seed}" / dataset / "vote"
                origin = "pilot_piv1"
            if out.exists() and not ((out/"watchdog.json").exists()
                                      and read(out/"watchdog.json").get("status") == "completed"):
                out.rename(out.with_name(out.name + ".incomplete-" + datetime.now(MSK).strftime("%Y%m%d-%H%M%S-%f")))
            if not out.exists() and archived:
                validated_summary(archived, row, policy, method, origin)
                shutil.copytree(archived,out)
            if not out.exists():
                out.mkdir(parents=True)
                with (out/"run.log").open("x",encoding="utf-8",buffering=1) as arm_log:
                    def live(message):
                        print(message,file=arm_log,flush=True)
                        log(message)
                    live(f"{arm} starting | {len(summary)}/60 complete")
                    proc = multiprocessing.get_context("spawn").Process(
                        target=pilot.worker,args=(jobs[dataset,seed],row,policy,("vote",),"final",out,None))
                    try:
                        timer = supervise(proc,out,None,live,heartbeat_s=10.,
                            heartbeat_formatter=lambda p,t,left: f"{arm} {p.get('phase','starting')} "
                            f"{p.get('checked',0)}/{p.get('total','?')} fits={p.get('fits',0)} "
                            f"elapsed={t:.1f}s search_ETA={p.get('eta_s',0):.1f}s")
                    finally:
                        proc.close()
                    if timer["status"] != "completed":
                        raise RuntimeError(f"{arm} failed; inspect its log and resume with --output")
            entry = validated_summary(out, row, policy, method, origin)
            summary.append(entry)
            write_summary(root,summary)
            log(f"{arm} {origin} retained={entry['retained']} test={entry['test_utility']:.7f} "
                f"fits={entry['fits']} elapsed={entry['wall_s']:.2f}s | {len(summary)}/60 complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="Use or resume a specific formal output directory")
    parser.add_argument("--pilot",type=Path,default=PILOT)
    parser.add_argument("--gain",type=Path,default=pilot.BASELINE)
    args = parser.parse_args()
    root = (args.output or RESULTS / datetime.now(MSK).strftime("PIv1-all-seeds-%Y%m%d-%H%M%S-%f")).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root):
        campaign(root,args.pilot.resolve(),args.gain.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
