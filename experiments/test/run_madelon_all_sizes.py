"""Madelon seed 42: compare every nested Gain prefix without a validation loss gate."""
import argparse
import csv
import hashlib
import json
import multiprocessing
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test.run_madelon_binary_widths import LIMIT_S, METHODS, SEED, fit_rank_model, negative_vote
from experiments.test.thesis_madelon_alpha_data import load_arm, load_heldout
from experiments.thesis_sampling_worker import subset_validation_scores, validation_halves
from minppi_mofs.experiment_protocol import AuditLog, JobSpec, atomic_write_json, canonical_hash
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


RESULTS = ROOT / "results/test/thesis_sampling"
REFERENCE = RESULTS / "madelon-s42-binary-widths-20261111-160617-502663"


def run_arm(width, method, reference, out):
    started = time.monotonic()
    train, yt, val, yv, top100, audit = load_arm(width, alpha=None)
    policy = {key: audit["policy"][key] for key in ("task", "seed", "backend", "model_params")}
    halves, half_id = validation_halves(yv, "binary", seed=SEED)
    features = list(train.columns)
    archived = json.loads((reference / "validation.json").read_bytes())
    archive_hash = canonical_hash(archived)
    expected_hash = json.loads((reference / "result.json").read_bytes())["frozen_hash"]
    if (archive_hash != expected_hash or audit != archived["data"] or features != archived["features"]
            or half_id != archived["half_identity"] or top100 != archived["top100"]):
        raise ValueError("Frozen source or data identity mismatch")

    atomic_write_json(out / "progress.json", dict(phase="ranking"))
    rank_start = time.monotonic()
    model, valid, full_scores = fit_rank_model(train, yt, val, yv, halves, policy)
    vote = negative_vote(model, valid, yv) if method == "hard_negative_then_gain" else None
    negative = [name for i, name in enumerate(features) if vote and vote["lock_round"][i]]
    negative_set = set(negative)
    remaining = [f for f in features if f not in negative_set]
    rank_scores = full_scores
    if negative:
        model, _, rank_scores = fit_rank_model(train[remaining], yt, val[remaining], yv, halves, policy)
    gains = model.get_booster().get_score(importance_type="gain")
    ascending = sorted(range(len(remaining)), key=lambda i: (gains.get(f"f{i}", 0.), i))
    order = [remaining[i] for i in reversed(ascending)]
    if (negative != archived["negative"] or order != archived["remaining_gain_order"][::-1]
            or not np.allclose(full_scores, archived["full_validation"], rtol=0, atol=1e-12)):
        raise ValueError("Independent ranking differs from the frozen four-arm comparison")
    ranking_s = time.monotonic() - rank_start
    ranking_fits = 1 + int(bool(negative))
    atomic_write_json(out / "ranking.json", dict(data=audit, halves=half_id, source_hash=archive_hash,
                      feature_order=features, negative=negative, descending_gain=order, vote=vote))
    prior_scores = {t["mask"]: t["scores"] for t in archived["trials"]}
    positions = {f: i for i, f in enumerate(features)}
    selected = set()
    mask = 0
    best_units = -1
    number_pairs = int((yv == 1).sum()) * int((yv == 0).sum())
    scan_start = time.monotonic()
    with (out / "validation_curve.csv").open("x", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.writer(handle)
        writer.writerow(["retained", "mask", "pooled_auc", "half_a_auc", "half_b_auc", "seconds", "fit_source"])
        for k, name in enumerate(order, 1):
            point_start = time.monotonic()
            selected.add(name)
            mask |= 1 << positions[name]
            columns = [f for f in features if f in selected]
            rank_endpoint = k == len(order)
            scores = rank_scores if rank_endpoint else subset_validation_scores(
                train, yt, val, yv, columns, halves, utility="roc_auc", **policy)
            previous = prior_scores.get(hex(mask))
            if previous and not np.allclose(previous, scores, rtol=0, atol=1e-12):
                raise ValueError(f"Fresh validation scores disagree at k={k}")
            # Pooled empirical AUC counts positive/negative wins plus half the ties.
            units = round(2 * number_pairs * scores[0])
            if units > best_units:
                best_units = units
                best = dict(retained=k, selected=columns, mask=hex(mask), validation=list(scores))
            elapsed = time.monotonic() - scan_start
            writer.writerow([k, hex(mask), *scores, time.monotonic()-point_start,
                             "current_ranking_model" if rank_endpoint else "fresh_subset_fit"])
            atomic_write_json(out / "progress.json", dict(phase="scan", checked=k, total=len(order),
                              scan_eta_s=elapsed * (len(order)-k)/k))
    scan_s = time.monotonic() - scan_start
    frozen = dict(width=width, method=method, source_hash=archive_hash, **best,
                  checked=len(order), negative=negative, pooled_auc_units=best_units,
                  curve_sha256=hashlib.sha256((out / "validation_curve.csv").read_bytes()).hexdigest())
    atomic_write_json(out / "frozen.json", frozen)
    identity = canonical_hash(frozen)
    job = JobSpec(identity, audit["protocol_hash"], "removal", "madelon", SEED, method,
                  dict(mask=best["mask"]), (), "result.json")
    log = AuditLog.from_manifest(out / "test_access.json", [job])
    event = log.append_test_access(job_id=identity, stage="removal", dataset_id="madelon", seed=SEED,
                                  reason="final_removal_evaluation", decision="allowed")
    if event["decision"] != "allowed":
        raise ValueError("Selected mask lacks held-out access")
    atomic_write_json(out / "progress.json", dict(phase="final_fit", checked=len(order), total=len(order)))
    test, ye = load_heldout(width, audit)
    columns = best["selected"]
    pred = fit_predict(pd.concat([train[columns], val[columns]], ignore_index=True),
                       np.concatenate([yt, yv]), test[columns], **policy)
    auc = float(compute_utility(ye, pred, metric="roc_auc"))
    np.savez_compressed(out / "predictions.npz", y=ye, selected=pred)
    result = dict(candidates=len(order), retained=len(columns), negative=len(negative) if vote else None,
                  outside_top100=len(set(columns)-set(top100)), validation_auc=best["validation"][0],
                  auc=auc, ranking_fits=ranking_fits, scan_fits=len(order)-1, final_fits=1,
                  fits=ranking_fits+len(order), ranking_s=ranking_s, scan_s=scan_s,
                  worker_s=time.monotonic()-started, frozen_hash=identity)
    atomic_write_json(out / "result.json", result)


def worker(*args):
    try:
        run_arm(*args)
    except Exception:
        (args[-1] / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="Original frozen four-arm run")
    args = parser.parse_args()
    reference = args.reference.resolve()
    if not (reference / "summary.csv").is_file():
        parser.error("Provide the original four-arm directory using --reference.")
    root = RESULTS / datetime.now(MSK).strftime("madelon-all-sizes-%Y%m%d-%H%M%S-%f")
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "protocol.json", dict(dataset="madelon", seed=SEED, width=[100, 500],
                      methods=METHODS, candidate_sizes="1 through all remaining columns; nested gain prefixes",
                      criterion="maximize pooled validation AUC; ties choose the smallest k",
                      gate=None, compression_penalty=None, hard_limit_s=LIMIT_S, reference=str(reference),
                      cold_start=True, previous_scores_used_for_selection=False,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows = []
    with (root / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        def live(message):
            print(message, file=log, flush=True)
            print(message, flush=True)

        live(f"Output: {root}")
        for width in (100, 500):
            for method in METHODS:
                arm = f"{width}-{method}"
                out = root / arm
                out.mkdir()
                live(f"{arm} starting, limit {LIMIT_S}s")
                process = multiprocessing.get_context("spawn").Process(
                    target=worker, args=(width, method, reference / arm, out))
                try:
                    outcome = supervise(process, out, LIMIT_S, live, heartbeat_s=10.,
                        heartbeat_formatter=lambda p,t,left: f"{arm} {p.get('phase','starting')} "
                        f"{p.get('checked',0)}/{p.get('total','?')} elapsed={t:.1f}s "
                        f"scan ETA={p.get('scan_eta_s',0):.1f}s deadline in {left:.1f}s")
                finally:
                    process.close()
                result = outcome["result"]
                row = dict(width=width, method=method, status=outcome["status"], wall_s=outcome["elapsed_s"],
                           checked=outcome["progress"].get("checked", 0))
                for key in ("candidates", "retained", "negative", "outside_top100", "validation_auc", "auc",
                            "fits", "ranking_fits", "scan_fits", "final_fits", "ranking_s", "scan_s"):
                    row[key] = result.get(key)
                rows.append(row)
                write_summary(root, rows)
                live(str(row))
    return int(any(row["status"] != "completed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
