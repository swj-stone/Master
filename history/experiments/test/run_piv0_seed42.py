"""Compare PIv0 with direct Gain on all six frozen prepared datasets at seed 42."""

import argparse
import csv
import hashlib
import multiprocessing
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_thesis_sampling import DATASETS, MSK, ROOT, supervise, write_summary
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from experiments.thesis_sampling_worker import prediction_scores, subset_validation_scores, validation_halves
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.utility import compute_utility


SEED = 42
LIMIT_S = 900
METHODS = ("direct_gain", "PIv0")
ORDER = ("isolet", "madelon", "credit_card_fraud", "abalone", "gisette", "spambase")
RESULTS = ROOT / "results/test/thesis_sampling"


def rank_model(train, yt, val, yv, halves, metric, policy):
    names = [f"f{i}" for i in range(train.shape[1])]
    safe_val = val.set_axis(names, axis=1)
    model = make_model(**policy)
    model.fit(train.set_axis(names, axis=1), yt)
    predict = model.predict if policy["task"] == "regression" else model.predict_proba
    scores = prediction_scores(yv, np.asarray(predict(safe_val), dtype=float), halves, metric)
    return model, safe_val, scores


def run_arm(source, row, policy, method, out):
    started = time.monotonic()
    dataset, metric = row["dataset_id"], row["utility"]
    seed = int(row["seed"])
    if policy["seed"] != seed:
        raise ValueError("The model seed must match the frozen data split")
    fits = 0

    def progress(phase, **state):
        atomic_write_json(out / "progress.json", dict(phase=phase, fits=fits, **state))

    progress("load")
    train, yt, val, yv = load_prepared_train_validation(source, row)
    features = list(train.columns)
    halves, half_id = validation_halves(yv, policy["task"], seed=seed)
    progress("ranking")
    ranking_start = time.monotonic()
    model, safe_val, full_scores = rank_model(train, yt, val, yv, halves, metric, policy)
    fits += 1
    votes = None
    negative = []
    voting_s = 0.
    if method == "PIv0":
        from minppi_mofs import piv0

        voting_start = time.monotonic()
        progress("vote")

        def vote_progress(phase, step, total):
            progress(phase, vote_step=step, vote_total=total)

        votes = piv0.negative_vote(model, safe_val, yv, metric=metric, seed=seed, progress=vote_progress)
        voting_s = time.monotonic() - voting_start
        negative = [f for f, step in zip(features, votes["lock_round"], strict=True) if step]
    negative_set = set(negative)
    remaining = [f for f in features if f not in negative_set]
    if not remaining:
        raise ValueError("PIv0 returned no surviving features")
    endpoint_scores = full_scores
    if negative:
        progress("reranking")
        model, _, endpoint_scores = rank_model(train[remaining], yt, val[remaining], yv, halves, metric, policy)
        fits += 1
    gains = model.get_booster().get_score(importance_type="gain")
    ranking = sorted(range(len(remaining)), key=lambda i: (gains.get(f"f{i}", 0.), i), reverse=True)
    order = [remaining[i] for i in ranking]
    ranking_fits = fits
    ranking_s = time.monotonic() - ranking_start
    atomic_write_json(out / "ranking.json", dict(
        dataset=dataset, seed=seed, metric=metric, method=method, policy=policy, feature_order=features,
        descending_gain=order, descending_gain_values=[gains.get(f"f{i}", 0.) for i in ranking], negative=negative,
        vote=votes, full_validation=full_scores, halves=half_id, source=source.name,
        partition_sha256={s: row[f"{s}_file_sha256"] for s in ("train", "val", "test")},
        voter_sha256=hashlib.sha256(Path(piv0.__file__).read_bytes()).hexdigest() if votes is not None else None,
    ))
    positions = {f: i for i, f in enumerate(features)}
    mask = 0
    best = None
    scan_start = time.monotonic()
    total = len(order)
    progress("scan", checked=0, candidates=total, negative=len(negative))
    curve = out / "validation_curve.csv"
    with curve.open("x", encoding="utf-8", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(("retained", "mask", "pooled_utility", "half_a", "half_b", "seconds", "fit_source"))
        for k, feature in enumerate(order, 1):
            point_start = time.monotonic()
            mask |= 1 << positions[feature]
            columns = [f for i, f in enumerate(features) if mask & (1 << i)]
            scores = endpoint_scores
            if k < total:
                scores = subset_validation_scores(train, yt, val, yv, columns, halves, utility=metric, **policy)
                fits += 1
            if not np.isfinite(scores).all():
                raise ValueError(f"Nonfinite validation results for {dataset}/{method} k={k}")
            # Ascending k keeps the smaller model for utility ties within machine roundoff.
            if best is None or scores[0] > best["validation"][0] + 1e-14:
                best = dict(retained=k, selected=columns, mask=hex(mask), validation=list(scores))
            writer.writerow((k, hex(mask), *scores, time.monotonic()-point_start,
                             "current_ranking_model" if k == total else "fresh_subset_fit"))
            elapsed = time.monotonic() - scan_start
            progress("scan", checked=k, candidates=total, negative=len(negative), eta_s=elapsed*(total-k)/k)
    scan_s = time.monotonic() - scan_start
    frozen = dict(dataset=dataset, seed=seed, method=method, metric=metric, width=len(features), negative=negative,
                  checked=total, **best, curve_sha256=hashlib.sha256(curve.read_bytes()).hexdigest(),
                  ranking_sha256=hashlib.sha256((out/"ranking.json").read_bytes()).hexdigest())
    atomic_write_json(out/"frozen.json", frozen)
    identity = canonical_hash(frozen)
    progress("final_fit", checked=total, candidates=total, negative=len(negative))
    paths = PreparedSplitPaths(dataset, seed, row["target"], *(source/row[f"{s}_relpath"] for s in ("train","val","test")))
    job = JobSpec(identity, row["protocol_hash"], "removal", dataset, seed, method, {"mask": best["mask"]}, (), "result.json")
    audit = AuditLog.from_manifest(out/"test_access.json", [job])
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=audit).load_test(
        reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
    if list(test.columns) != features or len(ye) != row["test_row_count"]:
        raise ValueError("Test partition does not match the prepared split")
    columns = best["selected"]
    prediction = fit_predict(pd.concat([train[columns], val[columns]], ignore_index=True),
                             np.concatenate([yt, yv]), test[columns], **policy)
    fits += 1
    score = compute_utility(ye, prediction, metric=metric)
    np.savez_compressed(out/"predictions.npz", y=ye, selected=prediction)
    atomic_write_json(out/"result.json", dict(
        candidates=total, checked=total, retained=len(columns), negative=len(negative) if votes is not None else None,
        full_validation_utility=full_scores[0], validation_utility=best["validation"][0], test_utility=score,
        ranking_fits=ranking_fits, scan_fits=total-1, final_fits=1, fits=fits, ranking_s=ranking_s, voting_s=voting_s,
        scan_s=scan_s, worker_s=time.monotonic()-started, frozen_hash=identity,
    ))
    progress("complete", checked=total, candidates=total, negative=len(negative))


def worker(*args):
    try:
        run_arm(*args)
    except Exception:
        (args[-1]/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def heartbeat(arm, state, elapsed, left):
    counts = f"{state.get('checked',0)}/{state.get('candidates','?')}"
    if "vote_step" in state:
        counts = f"{state['vote_step']}/{state['vote_total']} voting paths"
    eta = f"{state['eta_s']:.1f}s" if "eta_s" in state else "pending"
    deadline = f" deadline_in={left:.1f}s" if left is not None else ""
    return (f"{arm} {state.get('phase','startup')} {counts} fits={state.get('fits',0)} "
            f"elapsed={elapsed:.1f}s scan_ETA={eta}{deadline}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=ORDER)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Select each dataset only once")
    jobs, catalog = audit_prepared_catalog()
    config = catalog["shared"]
    if config["backend"] != "cuda" or config["model_params"]["n_estimators"] != 100:
        raise ValueError("PIv0 requires the frozen 100-tree CUDA recipe")
    schedule = []
    for index, dataset in enumerate(args.datasets):
        source = jobs[dataset, SEED]
        with (source/"data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            row, = (r for r in csv.DictReader(stream) if r["dataset_id"] == dataset and int(r["seed"]) == SEED)
        for split in ("train","val","test"):
            row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
        if not 1 <= int(row["selected_feature_count"]) <= 100:
            raise ValueError(f"{dataset}: only prepared widths <=100 are supported")
        policy = dict(task=row["task"], seed=SEED, model_params=config["model_params"], backend=config["backend"])
        for method in (METHODS if index%2 == 0 else METHODS[::-1]):
            schedule.append((dataset, source, row, policy, method))
    root = RESULTS / datetime.now(MSK).strftime("PIv0-seed42-%Y%m%d-%H%M%S-%f")
    root.mkdir(parents=True)
    atomic_write_json(root/"protocol.json", dict(
        version="PIv0", seed=SEED, cold_start=True, paired_methods=METHODS, datasets=list(args.datasets),
        limit_s_per_arm=LIMIT_S, catalog_hash=catalog["sha256"], model_params=config["model_params"],
        backend=config["backend"], tie_tolerance=1e-14, criterion="max pooled validation utility; ties choose smaller k",
        candidate_sizes="all integers 1..surviving width; no validation-loss veto", cached_prior_fits=0,
        run_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs=[dict(dataset=d, method=m, source=s.name, metric=r["utility"], width=int(r["selected_feature_count"]),
                     protocol_hash=r["protocol_hash"]) for d,s,r,p,m in schedule],
    ))
    result_fields = ("negative", "retained", "full_validation_utility", "validation_utility", "test_utility",
                     "ranking_fits", "scan_fits", "final_fits", "ranking_s", "voting_s", "scan_s")
    rows = []
    with (root/"run.log").open("x", encoding="utf-8", buffering=1) as run_log:
        print(f"Results: {root}", flush=True)
        for dataset, source, row, policy, method in schedule:
            arm = f"{dataset}/{method}"
            out = root/arm
            out.mkdir(parents=True)
            with (out/"run.log").open("x", encoding="utf-8", buffering=1) as arm_log:
                def log(message):
                    print(message, file=arm_log, flush=True)
                    print(message, file=run_log, flush=True)
                    print(message, flush=True)

                log(f"{arm} starting, wall limit {LIMIT_S}s")
                proc = multiprocessing.get_context("spawn").Process(target=worker, args=(source,row,policy,method,out))
                try:
                    outcome = supervise(proc, out, LIMIT_S, log, heartbeat_s=10.,
                                        heartbeat_formatter=lambda s,t,left: heartbeat(arm,s,t,left))
                finally:
                    proc.close()
                result = outcome["result"]
                state = outcome["progress"]
                entry = dict(dataset=dataset, seed=SEED, method=method, metric=row["utility"],
                             width=int(row["selected_feature_count"]), status=outcome["status"],
                             checked=state.get("checked",0), candidates=state.get("candidates"),
                             fits=result.get("fits",state.get("fits")), wall_s=outcome["elapsed_s"])
                entry.update({key:result.get(key) for key in result_fields})
                rows.append(entry)
                write_summary(root, rows)
                log(str(entry))
    return int(any(r["status"] != "completed" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
