"""Cold-start sampled permutation importance with positive-mean-drop feature selection."""

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
from scipy.stats import rankdata

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_thesis_sampling import MSK, supervise, write_summary
from experiments.test.run_piv0_seed42 import heartbeat
from experiments.test.run_piv1_pilot import DATASETS, LIMIT, OUTPUT
from experiments.thesis_prepared_catalog import FORMAL_SEEDS, audit_prepared_catalog
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.piv1 import _sample_rows
from minppi_mofs.utility import compute_utility


REPEATS = 5
BATCH_COLS = 4


def batched_utility(y, predictions, metric):
    """Score batches of actual model predictions, shape (batch, rows[, classes])."""
    if metric == "r2":
        return np.asarray([compute_utility(y, p, metric=metric) for p in predictions])
    cols = [1] if metric == "roc_auc" else np.arange(predictions.shape[-1])
    ranks = rankdata(predictions[:, :, cols], axis=1, method="average")
    auc = []
    for output, label in enumerate(cols):
        positive = y == label
        n = positive.sum()
        wins = ranks[:, positive, output].sum(axis=1) - n * (n + 1) / 2
        auc.append(wins / (n * (len(y) - n)))
    return np.mean(auc, axis=0)


def predict_with(model, frame, policy):
    fn = model.predict if policy["task"] == "regression" else model.predict_proba
    return np.asarray(fn(frame), dtype=float)


def sampled_drops(model, valid, labels, full_prediction, metric, policy, progress):
    row_seed, shuffle_seed = np.random.SeedSequence(policy["seed"]).spawn(2)
    rows = _sample_rows(labels, metric, np.random.default_rng(row_seed))
    x = valid.iloc[rows].to_numpy()
    y = labels[rows]
    base = batched_utility(y, full_prediction[rows][None], metric)[0]
    if not np.isclose(base, compute_utility(y, full_prediction[rows], metric=metric), atol=1e-12, rtol=0):
        raise ValueError("Permutation scorer and canonical utility disagree on the original predictions")
    rng = np.random.default_rng(shuffle_seed)
    drops = np.empty((x.shape[1], REPEATS))
    for start in range(0, x.shape[1], BATCH_COLS):
        stop = min(start + BATCH_COLS, x.shape[1])
        batch = np.broadcast_to(x, ((stop - start) * REPEATS, *x.shape)).copy()
        for j in range(len(batch)):
            column = start + j // REPEATS
            batch[j, :, column] = x[rng.permutation(len(x)), column]
        frame = pd.DataFrame(batch.reshape(-1, x.shape[1]), columns=valid.columns)
        flat = predict_with(model, frame, policy)
        predictions = flat.reshape(len(batch), len(x), *flat.shape[1:])
        scores = batched_utility(y, predictions, metric)
        if not np.isclose(scores[0], compute_utility(y, predictions[0], metric=metric), rtol=0, atol=1e-12):
            raise ValueError("Permutation scorer disagrees on shuffled predictions")
        drops[start:stop] = (base - scores).reshape(stop - start, REPEATS)
        progress("permutation", checked=stop, candidates=x.shape[1])
    if not np.isfinite(drops).all():
        raise ValueError("Non-finite permutation importance")
    return rows, base, drops


def metrics(labels, predictions, metric):
    primary = compute_utility(labels, predictions, metric=metric)
    if metric == "r2":
        return primary, float(np.abs(labels - predictions).mean())
    return primary, float((predictions.argmax(axis=1) == labels).mean())


def run_arm(source, row, policy, out):
    dataset, metric = row["dataset_id"], row["utility"]
    seed = policy["seed"]
    fits = 0

    def progress(phase, **counts):
        atomic_write_json(out / "progress.json", dict(phase=phase, fits=fits, **counts))

    progress("loading")
    train, yt, val, yv = load_prepared_train_validation(source, row)
    names = list(train.columns)
    safe = [f"f{i}" for i in range(len(names))]
    safe_val = val.set_axis(safe, axis=1)
    progress("full_model")
    model = make_model(**policy)
    model.fit(train.set_axis(safe, axis=1), yt)
    fits += 1
    full = predict_with(model, safe_val, policy)
    progress("permutation", checked=0, candidates=len(names))
    started = time.monotonic()
    rows, base, drops = sampled_drops(model, safe_val, np.asarray(yv), full, metric, policy, progress)
    rank_s = time.monotonic() - started
    means = drops.mean(axis=1)
    selected = [name for name, drop in zip(names, means, strict=True) if drop > 0.]
    if not selected:
        selected = [names[int(np.argmax(means))]]
    progress("selected_validation")
    vp = full
    if len(selected) < len(names):
        vp = fit_predict(train[selected], yt, val[selected], **policy)
        fits += 1
    validation, validation_secondary = metrics(yv, vp, metric)
    frozen = dict(dataset=dataset, seed=seed, metric=metric, policy=policy, features=names, selected=selected,
                  partition_sha256={s:row[f"{s}_file_sha256"] for s in ("train", "val", "test")},
                  rows=rows.tolist(), sample_full_utility=float(base), drops=drops.tolist(),
                  mean_drops=means.tolist(), validation_utility=validation, validation_secondary=validation_secondary)
    atomic_write_json(out / "frozen.json", frozen)
    identity = canonical_hash(frozen)
    progress("final_test")
    paths = PreparedSplitPaths(dataset, seed, row["target"], *(source / row[f"{s}_relpath"] for s in ("train","val","test")))
    job = JobSpec(identity, row["protocol_hash"], "removal", dataset, seed, "sampled_permutation",
                  {"columns": selected}, (), "result.json")
    audit = AuditLog.from_manifest(out / "test_access.json", [job])
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=audit).load_test(
        reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
    if len(ye) != int(row["test_row_count"]) or list(test.columns) != names:
        raise ValueError("Unexpected held-out partition")
    pred = fit_predict(pd.concat([train[selected],val[selected]], ignore_index=True), np.concatenate([yt, yv]),
                       test[selected], **policy)
    fits += 1
    test_score, test_secondary = metrics(ye, pred, metric)
    np.savez_compressed(out / "predictions.npz", validation_y=yv, validation=vp, y=ye, selected=pred)
    result = dict(dataset=dataset, seed=seed, width=len(names), retained=len(selected), fits=fits, metric=metric,
                  full_validation_utility=compute_utility(yv, full, metric=metric), validation_utility=validation,
                  test_utility=test_score, secondary_metric="mae" if metric=="r2" else "accuracy",
                  validation_secondary=validation_secondary, test_secondary=test_secondary,
                  sample_rows=len(rows), permutation_s=rank_s, frozen_hash=identity)
    atomic_write_json(out / "result.json", result)
    progress("complete")


def worker(source, row, policy, out):
    try:
        run_arm(source, row, policy, out)
    except Exception:
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, default=42)
    seed = parser.parse_args().seed
    jobs, catalog = audit_prepared_catalog()
    root = OUTPUT / datetime.now(MSK).strftime(f"sampled-permutation-seed{seed}-%Y%m%d-%H%M%S-%f")
    root.mkdir(parents=True)
    atomic_write_json(root / "protocol.json", dict(seed=seed, repeats=REPEATS, datasets=DATASETS,
                      limit_s=LIMIT, catalog_sha256=catalog["sha256"], model=catalog["shared"],
                      sample="PIv1 validation row sample", selection="strictly positive mean drop; top 1 if none",
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(root, flush=True)
    records = []
    with (root / "run.log").open("x", encoding="utf-8", buffering=1) as stream:
        def live(message):
            print(message, flush=True)
            print(message, file=stream, flush=True)

        for dataset in DATASETS:
            source = jobs[dataset, seed]
            with (source / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
                row, = (r for r in csv.DictReader(handle) if r["dataset_id"]==dataset and int(r["seed"])==seed)
            for split in ("train", "val", "test"):
                row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
            policy = dict(task=row["task"], seed=seed, model_params=catalog["shared"]["model_params"],
                          backend=catalog["shared"]["backend"])
            out = root / dataset
            out.mkdir()
            proc = multiprocessing.get_context("spawn").Process(target=worker, args=(source, row, policy, out))
            live(f"{dataset} started, wall limit {LIMIT}s")
            try:
                outcome = supervise(proc, out, LIMIT, live, heartbeat_s=10.,
                                    heartbeat_formatter=lambda p,t,left: heartbeat(dataset,p,t,left))
            finally:
                proc.close()
            live(f"{dataset} {outcome['status']} in {outcome['elapsed_s']:.2f}s")
            records.append(dict(dataset=dataset, **{"status":outcome["status"], "wall_s":outcome["elapsed_s"]},
                                **{k:v for k,v in outcome["result"].items() if k!="dataset"}))
        write_summary(root, records)
    return int(any(r["status"] != "completed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
