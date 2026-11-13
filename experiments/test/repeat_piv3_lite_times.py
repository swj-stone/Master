"""Counterbalanced cold timing repeats of the frozen PI/PFI routes under a cumulative wall budget."""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import median
from uuid import uuid4

from experiments.test.run_piv3_lite_six import (
    DATASETS, DEFAULT, DEFAULT_ROOT, LIMIT, METHODS, ROOT, SOURCES, _sha, exclusive_run, prepare, supervise, worker,
)
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


TIMED = ("pi", "pfi", "pi_pfi")
ROUNDS = ("r2", "r3")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def read(path):
    return json.loads(path.read_bytes())


def verify_main(root):
    protocol = read(root/"protocol.json")
    if (protocol["seed"] != 42 or tuple(protocol["datasets"]) != DATASETS or
            tuple(protocol["methods"]) != METHODS or protocol["seconds_limit"] != LIMIT or
            protocol["source_sha256"] != {name: _sha(ROOT/name) for name in SOURCES}):
        raise ValueError("Timing protocol differs from the original benchmark")
    for d in DATASETS:
        if any(not (root/d/m/"watchdog.json").is_file() or
               read(root/d/m/"watchdog.json")["status"] not in ("completed", "timed_out") for m in METHODS):
            raise ValueError("All 90 primary routes must finish before timing repeats")
        data = prepare(DEFAULT_ROOT, d)
        if protocol["data"][d] != {"width": data["width"], "hash": _sha(DEFAULT_ROOT/d/"metadata.json")}:
            raise ValueError(f"Changed source data for {d}")
    return protocol


def observation(folder, *, dataset, method):
    timer = read(folder/"watchdog.json")
    cold = float(timer["elapsed_s"])
    if not math.isfinite(cold) or cold < 0:
        raise ValueError(f"Invalid elapsed time in {folder}")
    if timer["status"] != "completed":
        return dict(status=timer["status"], cold=cold)
    result = read(folder/"result.json")
    if (timer["result"] != result or (result["dataset"], result["method"], result["seed"]) != (dataset, method, 42)
            or canonical_hash(read(folder/"frozen.json")) != result["frozen_hash"]
            or not 0 <= result["front_s"] <= result["elapsed_s"] <= cold + 1e-6):
        raise ValueError(f"Inconsistent or changed timing result in {folder}")
    return dict(status="completed", cold=cold, active=result["elapsed_s"], front=result["front_s"],
                result=result)


def comparisons(main, obs):
    fields = ("validation_full", "validation_candidate", "validation_selected", "test_scores")
    a, b = main["result"], obs["result"]
    if any(a[s].keys() != b[s].keys() for s in fields):
        raise ValueError("Metric schema changed between timing repeats")
    diffs = {k: max(abs(a[k][s] - b[k][s]) for s in a[k]) for k in fields}
    return dict(same_selected=a["selected"] == b["selected"], same_candidate=a["candidate"] == b["candidate"],
                same_front=a["front"] == b["front"], same_fallback=a["fallback"] == b["fallback"],
                same_failed_metrics=a["failed_metrics"] == b["failed_metrics"], same_fits=a["fits"] == b["fits"],
                max_validation_difference=max(diffs[s] for s in fields[:-1]), max_test_difference=diffs["test_scores"])


def schedule(round_id):
    # Every method occupies each within-dataset slot once across the three rounds
    # after restricting the original dataset-specific run order to these three methods.
    for idx in (reversed(range(len(DATASETS))) if round_id == "r2" else range(len(DATASETS))):
        pairs = [METHODS[j:j+2] for j in range(3, len(METHODS), 2)]
        pairs = pairs[idx % len(pairs):] + pairs[:idx % len(pairs)]
        ordered = ["pi"]
        for j, pair in enumerate(pairs):
            ordered.extend(m for m in (reversed(pair) if (idx+j) % 2 else pair) if m in TIMED)
        shift = ROUNDS.index(round_id)+1
        for m in ordered[shift:]+ordered[:shift]:
            yield DATASETS[idx], m


def run_one(root, base, round_id, dataset, method, observations, *, execute=supervise, target=worker):
    folder = base/round_id/dataset/method
    prior = sum(item["cold"] for item in observations)
    available = max(0., LIMIT - prior)
    identity = dict(main_protocol=_sha(root/"protocol.json"), round=round_id, dataset=dataset, method=method,
                    main_watchdog=_sha(root/dataset/method/"watchdog.json"), prior_cold_s=prior,
                    remaining_s=available)
    if folder.exists():
        record = read(folder/"allocation.json")
        if record != identity:
            raise ValueError(f"Cannot resume with changed run or timing budget: {folder}")
        if (folder/"watchdog.json").exists():
            return observation(folder, dataset=dataset, method=method)
        # An interruption with no terminal watchdog leaves actual expenditure unknowable.
        # Charge the entire remaining allowance instead of spending part of it twice.
        atomic_write_json(folder/"watchdog.json", dict(status="unmeasured_interruption", elapsed_s=available))
        return observation(folder, dataset=dataset, method=method)
    folder.mkdir(parents=True)
    atomic_write_json(folder/"allocation.json", identity)
    if observations[0]["status"] != "completed" or available <= 2.:
        atomic_write_json(folder/"watchdog.json", dict(status="insufficient_remaining_budget", elapsed_s=0.))
    else:
        process = mp.get_context("spawn").Process(target=target, args=(dataset, method, folder))
        try:
            # External watchdog includes interpreter start. Reserve two seconds for process
            # termination/collection so a timed-out worker is not granted extra runtime.
            execute(process, folder, available - 2., print, heartbeat_s=30.,
                    heartbeat_formatter=lambda p,e,left: f"{round_id}/{dataset}/{method} "
                    f"{p.get('phase','startup')} elapsed={e:.1f}s budget_remaining={left:.1f}s")
        finally:
            process.close()
    return observation(folder, dataset=dataset, method=method)


def publish(base, all_obs):
    rows = []
    for (dataset, method), series in all_obs.items():
        complete = len(series)==3 and all(item["status"]=="completed" for item in series)
        row = dict(dataset=dataset, method=method, complete=complete, total_cold_s=sum(x["cold"] for x in series),
                   completed_observations=sum(x["status"]=="completed" for x in series))
        checks = [comparisons(series[0], x) for x in series[1:]
                  if x["status"]=="completed" and series[0]["status"]=="completed"]
        for key in ("same_selected","same_front","same_candidate","same_fallback","same_fits","same_failed_metrics"):
            row[key] = all(x[key] for x in checks) if checks else None
        for key in ("max_validation_difference","max_test_difference"):
            row[key] = max((x[key] for x in checks), default=None)
        row["reproducible"] = complete and all(row[k] for k in ("same_selected","same_front","same_candidate",
                 "same_fallback","same_fits","same_failed_metrics")) and max(row["max_validation_difference"],
                 row["max_test_difference"]) <= 1e-12
        for i in range(3):
            single = series[i] if i < len(series) else {}
            for key in ("status","cold","active","front"):
                row[f"r{i+1}_{key}"] = single.get(key)
        for key in ("cold","active","front"):
            row[f"median_{key}_s"] = median(x[key] for x in series) if complete else None
        row["over_budget_s"] = max(0., row["total_cold_s"] - LIMIT)
        rows.append(row)
    # The same unprefixed PFI is the reference for both PI standalone and PI+PFI.
    pfi = {r["dataset"]:r for r in rows if r["method"]=="pfi"}
    for r in rows:
        for key in ("active","cold"):
            name = f"median_{key}_s"
            ref = pfi[r["dataset"]]
            r[f"speedup_vs_pfi_{key}"] = ref[name]/r[name] if r.get(name) and ref.get(name) else None
    out = base/"timing-median.csv"
    tmp = out.with_name(out.name+"."+uuid4().hex+".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out)
    return rows


def run(root):
    # Same process-level lock as the primary benchmark; parallel timing would distort results.
    with exclusive_run(root):
        protocol = verify_main(root)
        base = root/"timing-repeats"
        base.mkdir(exist_ok=True)
        spec = dict(methods=list(TIMED), rounds=list(ROUNDS), cumulative_per_route_s=LIMIT,
                    source_sha256=_sha(Path(__file__)), primary_protocol=canonical_hash(protocol), seed=42,
                    budget_unit="Cold wall of main plus repeats. Reserve 2s for watchdog collection per repeat.",
                    reproducibility="Selection, candidates, front, fallback and fits exact, metrics abs tol 1e-12.")
        if (base/"protocol.json").exists() and read(base/"protocol.json") != spec:
            raise ValueError("Timing protocol or source changed")
        atomic_write_json(base/"protocol.json", spec)
        series = {(d,m): [observation(root/d/m,dataset=d,method=m)] for d in DATASETS for m in TIMED}
        for round_id in ROUNDS:
            for d, m in schedule(round_id):
                observations = series[d,m]
                item = run_one(root, base, round_id, d, m, observations)
                observations.append(item)
                print(f"{round_id}/{d}/{m} {item['status']} {item['cold']:.2f}s; "
                      f"cumulative={sum(x['cold'] for x in observations):.2f}/{LIMIT:.0f}s", flush=True)
        rows = publish(base, series)
        print(f"{sum(r['complete'] for r in rows)}/{len(rows)} complete triplicates, "
              f"{sum(r['reproducible'] for r in rows)}/{len(rows)} reproducible, "
              f"{sum(r['over_budget_s'] > 0 for r in rows)} exceeded cold budget. "
              f"Table: {base/'timing-median.csv'}")
        return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT)
    args = parser.parse_args()
    for name in THREADS:
        os.environ[name] = "1"
    run(args.run.resolve())

