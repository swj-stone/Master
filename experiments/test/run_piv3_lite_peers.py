"""Seed-42 paired peer-screen experiment, using the source-pinned primary evaluation pipeline."""

import argparse
import csv
from datetime import datetime
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

from experiments.test.run_piv3_lite_six import (
    DATASETS, DEFAULT as MAIN, DEFAULT_ROOT, LIMIT, MSK, ROOT, _sha, exclusive_run, supervise,
)
from minppi_mofs.experiment_protocol import atomic_write_json


METHODS = ("pi", "pi_mean", "pi_pfi", "pi_mi", "pi_gain")
PEERS = "src/minppi_mofs/piv3_lite_peers.py"
ORIGINAL = "experiments/test/piv3_lite_arm.py"
SNAPSHOT = "source/piv3_lite_peer_arm.py"
OLD_IMPORT = b"from minppi_mofs.piv3_lite import screen"
NEW_IMPORT = b"from minppi_mofs.piv3_lite_peers import screen"
DEFAULT = ROOT/"results/test/thesis_sampling/PIv3-lite-peer-test-raw-six-seed42"


def read(path):
    return json.loads(path.read_bytes())


def source_copy(source, expected_sha, destination):
    if _sha(source) != expected_sha:
        raise ValueError("Primary experiment worker differs from its recorded digest")
    original = source.read_bytes()
    if original.count(OLD_IMPORT) != 1 or NEW_IMPORT in original:
        raise ValueError("Expected exactly one injectable frontend import in the pinned worker")
    payload = original.replace(OLD_IMPORT, NEW_IMPORT, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise ValueError("Peer worker snapshot was modified")
    else:
        destination.write_bytes(payload)


def worker(dataset, method, folder, snapshot, checksum):
    try:
        from threadpoolctl import threadpool_limits
        if _sha(snapshot) != checksum:
            raise ValueError("Worker snapshot invalid")
        spec = importlib.util.spec_from_file_location("piv3_lite_peer_worker", snapshot)
        if spec is None or spec.loader is None:
            raise ImportError(snapshot)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with threadpool_limits(limits=1):
            module.run_arm(dataset, method, folder)
    except Exception:
        (folder/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def protocol(root, main):
    from experiments.test.repeat_piv3_lite_times import verify_main
    parent = verify_main(main)
    repeat_table = main/"timing-repeats/timing-median.csv"
    with repeat_table.open(encoding="utf-8-sig", newline="") as f:
        repeats = list(csv.DictReader(f))
    expected = {(d,m) for d in DATASETS for m in ("pi","pfi","pi_pfi")}
    if len(repeats) != len(expected) or {(r["dataset"],r["method"]) for r in repeats} != expected:
        raise ValueError("Incomplete timing repeat matrix")
    if any(not r["r3_status"] for r in repeats):
        raise ValueError("Final timing round has not finished")
    source_copy(ROOT/ORIGINAL, parent["source_sha256"][ORIGINAL], root/SNAPSHOT)
    names = ("full","pfi","mi","gain",*METHODS)
    return dict(seed=42, datasets=list(DATASETS), methods=list(METHODS), limit_s=LIMIT, data=parent["data"],
                primary_protocol_sha256=_sha(main/"protocol.json"), source_sha256=parent["source_sha256"],
                timing_repeat_table_sha256=_sha(repeat_table), source_copy=SNAPSHOT,
                source_copy_sha256=_sha(root/SNAPSHOT), peer_sha256=_sha(ROOT/PEERS), runner_sha256=_sha(Path(__file__)),
                baseline_timer_sha256={f"{d}/{m}":_sha(main/d/m/"watchdog.json") for d in DATASETS for m in names},
                comparison="Unchanged primary pipeline, with only the screen import replaced by peer screen. "
                           "Each new pipeline is measured in its own process. All stages charged. "
                           "Direct PFI is also the reference for standalone min/mean PI.")


def verify_witnesses(method, result, shadow, diag):
    import numpy as np
    if diag["shadow_front_k"] != len(shadow["front"]):
        raise ValueError("Peer stage used a different starting shadow screen")
    removed = set(shadow["front"]) - set(result["front"])
    if not set(result["front"]).issubset(shadow["front"]):
        raise ValueError("Peer stage introduced unknown shadow features")
    witnesses = diag["witnesses"]
    if {w["removed"] for w in witnesses} != removed or len(witnesses) != len(removed):
        raise ValueError("Peer removals do not match the witness ledger")
    for w in witnesses:
        if w["representative"] not in result["front"] or w["representative"] == w["removed"]:
            raise ValueError("Peer replacement refers to a dropped representative")
        folds = np.asarray(w["fold_tier_rates"])
        corr = np.asarray(w["correlations"])
        if (folds.shape != (2,3) or not np.all(np.isfinite(folds)) or np.abs(folds).max() > 1 or
                corr.shape != (2,) or not np.all(np.isfinite(corr)) or
                min(abs(corr)) + 1e-12 < diag["peer_corr_threshold"] or corr.prod() <= 0):
            raise ValueError("Uncorroborated observed peer substitution")
        tier = folds.mean(axis=0)
        np.testing.assert_allclose(tier, w["tier_rates"], rtol=0, atol=1e-12)
        score = tier.mean() if method == "pi_mean" else tier.min()
        if abs(score-w["score"]) > 1e-12 or min(folds.mean(axis=1)) < 0 or score < 0:
            raise ValueError("Witness fails the fold or stratum vote rule")


def report(root, main):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from experiments.test.audit_piv3_lite import _route, _verify_scores
    from experiments.test.report_piv3_lite_six import losses, csv_table

    rows = []
    with (main/"timing-repeats/timing-median.csv").open(encoding="utf-8-sig", newline="") as f:
        medians = {(row["dataset"],row["method"]):row for row in csv.DictReader(f)}
    for d in DATASETS:
        meta = read(DEFAULT_ROOT/d/"metadata.json")
        vy = np.load(DEFAULT_ROOT/d/"val_y.npy")
        ty = np.load(DEFAULT_ROOT/d/"test_y.npy")
        ranking, guard = train_test_split(np.arange(len(vy)), test_size=.5, random_state=42,
                                         stratify=None if d == "abalone" else vy)
        ranking, guard = np.sort(ranking), np.sort(guard)
        measures, votes = {}, {}
        def measured(directory, method):
            key = directory, method
            if key not in measures:
                folder = directory/d/method
                timer = read(folder/"watchdog.json")
                result, diag = None, None
                if timer["status"] == "completed":
                    result, diag = _route(folder, meta, method, ty, vy, ranking, guard, timer)
                measures[key] = timer, result
                votes[key] = diag
            return measures[key]
        for m in METHODS:
            baseline = "pfi" if m in ("pi", "pi_mean") else m.removeprefix("pi_")
            timer, result = measured(root,m)
            base_timer, direct = measured(main,baseline)
            shadow_timer, shadow = measured(main,m)
            _, full = measured(main,"full")
            row = dict(dataset=d, method=m, baseline=baseline, status=timer["status"], cold_wall_s=timer["elapsed_s"])
            if result is not None and shadow is not None:
                diag = votes[root,m]
                verify_witnesses(m, result, shadow, diag)
                row.update(peer_removed_count=diag["peer_removed_count"], peer_verified_edges=diag.get("verified_edges",0),
                           peer_bypassed_small_input=bool(diag["shadow_bypass"] == "small input"))
            for prefix, ref, clock in (("direct",direct,base_timer), ("shadow",shadow,shadow_timer), ("peer",result,timer)):
                row[prefix+"_status"] = clock["status"]
                if ref is None:
                    continue
                row.update({prefix+"_"+k:ref[k] for k in
                    ("width","front_k","candidate_k","k","fits","front_fits","fallback","front_s",
                     "selector_s","rank_s","guard_s","elapsed_s","proposal_s","final_s")})
                for scores in ("test_scores","validation_candidate","validation_selected","validation_full"):
                    row.update({prefix+"_"+scores+"_"+metric:v for metric,v in ref[scores].items()})
                row[prefix+"_cold_wall_s"] = clock["elapsed_s"]
            for name, reference in (("full",full), ("direct",direct), ("shadow",shadow)):
                if reference is None or result is None:
                    continue
                _verify_scores(result["validation_full"], reference["validation_full"])
                loss = losses(result["test_scores"], reference["test_scores"])
                row.update({f"loss_vs_{name}_{metric}":number for metric,number in loss.items()})
                row["worst_loss_vs_"+name] = max(loss.values())
                row["pass_5pct_vs_"+name] = max(loss.values()) <= .05+1e-12
                row["same_selected_as_"+name] = result["selected"] == reference["selected"]
                if name != "full":
                    row["active_speedup_vs_"+name] = reference["elapsed_s"]/result["elapsed_s"]
            if result is not None and direct is not None:
                row["cold_speedup_vs_direct"] = base_timer["elapsed_s"]/timer["elapsed_s"]
                repeat = medians.get((d,baseline))
                if repeat and repeat["median_active_s"]:
                    row["direct_median_active_s"] = float(repeat["median_active_s"])
                    row["peer_single_vs_direct_median_speedup"] = float(repeat["median_active_s"])/result["elapsed_s"]
            rows.append(row)
        mn, av = votes[root,"pi"], votes[root,"pi_mean"]
        if mn and av and mn["fold_tier_rates"] != av["fold_tier_rates"]:
            raise ValueError(f"{d}: mean/min screen compared different voting observations")
    csv_table(root/"peer-comparison.csv", rows)
    print(f"Audited {sum(row['status']=='completed' for row in rows)}/{len(rows)} complete routes "
          f"and their available primary comparators; table: {root/'peer-comparison.csv'}",flush=True)
    return rows


def run(root, main, report_only=False):
    if root == main:
        raise ValueError("Peer results must be isolated from the primary campaign")
    root.mkdir(parents=True, exist_ok=True)
    # The primary/timing-repeat runner uses the same OS lock, also released on process termination.
    with exclusive_run(main), exclusive_run(root):
        spec = protocol(root, main)
        protocol_file = root/"protocol.json"
        if protocol_file.exists() and read(protocol_file) != spec:
            raise ValueError("Cannot mix changed input, code or baseline measurements in the peer run")
        if report_only:
            if not protocol_file.is_file():
                raise ValueError("No existing peer campaign protocol")
            return report(root, main)
        atomic_write_json(protocol_file,spec)
        with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
            def log(msg):
                line = f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {msg}"
                print(line,flush=True)
                print(line,file=stream,flush=True)
            jobs = [(d,m) for i,d in enumerate(DATASETS) for m in METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)]]
            for i, (d,m) in enumerate(jobs, 1):
                folder = root/d/m
                if (folder/"watchdog.json").exists():
                    state = read(folder/"watchdog.json")["status"]
                    if state in ("completed","timed_out","unmeasured_interruption"):
                        continue
                    raise ValueError(f"Inspect incomplete prior run {folder}")
                if folder.exists():
                    # The remaining time in a killed cold attempt is unknown; do not run it a second time.
                    atomic_write_json(folder/"watchdog.json",dict(status="unmeasured_interruption",elapsed_s=None))
                    continue
                folder.mkdir(parents=True)
                log(f"{i}/{len(jobs)} {d}/{m} start")
                proc = mp.get_context("spawn").Process(target=worker,
                    args=(d,m,folder,root/SNAPSHOT,spec["source_copy_sha256"]))
                try:
                    timer = supervise(proc,folder,LIMIT,log,heartbeat_s=20.,
                                      heartbeat_formatter=lambda p,t,left:f"{d}/{m} {p.get('phase','startup')} "
                                      f"elapsed={t:.1f}s deadline_in={left:.1f}s fits={p.get('fits','?')}")
                finally:
                    proc.close()
                log(f"{i}/{len(jobs)} {d}/{m} {timer['status']} {timer['elapsed_s']:.2f}s")
                if timer["status"] == "error":
                    raise RuntimeError(f"Inspect {folder/'error.log'}")
        return report(root,main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--main",type=Path,default=MAIN)
    parser.add_argument("--report-only",action="store_true")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    run(args.run.resolve(), args.main.resolve(), args.report_only)
