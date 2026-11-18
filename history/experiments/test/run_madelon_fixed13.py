"""Diagnose a fixed gain-ranked size, independently of primary decisions."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.test.thesis_madelon_alpha_data import load_arm, load_heldout
from experiments.thesis_sampling_worker import subset_validation_scores, validation_halves
from minppi_mofs.experiment_protocol import AuditLog, JobSpec, atomic_write_json, canonical_hash
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


TEST_RESULTS = Path(__file__).resolve().parents[2] / "results/test/thesis_sampling"
ARMS = [(width, method) for width in (100, 500) for method in ("hard_negative_then_gain", "direct_gain")]


def read(path):
    return json.loads(path.read_bytes())


def run(primary, output):
    if (output / "frozen.json").exists():
        raise FileExistsError("The diagnostic is already frozen. Use a fresh directory for a rerun.")
    summary = pd.read_csv(primary / "summary.csv")
    assert len(summary) == 4 and (summary.status == "completed").all()
    frozen, data, times = {}, {}, {}
    for width, method in ARMS:
        start = perf_counter()
        arm = f"{width}-{method}"
        main = primary / arm
        v = read(main / "validation.json")
        assert canonical_hash(v) == read(main / "result.json")["frozen_hash"]
        train, yt, val, yv, top, audit = load_arm(width, alpha=None)
        policy = {k: audit["policy"][k] for k in ("task", "seed", "backend", "model_params")}
        halves, halves_info = validation_halves(yv, "binary", seed=42)
        assert audit == v["data"] and top == v["top100"] and halves_info == v["half_identity"]
        assert policy["model_params"]["n_estimators"] == 100 and policy["seed"] == 42
        assert policy["backend"] == "cuda" and policy["task"] == "binary"
        assert audit["policy"]["utility"] == "roc_auc" and v["margin"] == .005
        features = list(train.columns)
        ranked = v["remaining_gain_order"]
        assert features == v["features"] and set(ranked) == set(features) - set(v["negative"])
        assert len(ranked) == len(set(ranked))
        chosen_set = set(ranked[-13:])
        cols = [f for f in features if f in chosen_set]
        assert len(cols) == 13
        mask = hex(sum(1 << i for i, f in enumerate(features) if f in chosen_set))
        trial = next((t for t in v["trials"] if mask == t["mask"]), None)
        can_reuse_test = mask == v["selected"]
        if can_reuse_test:
            assert trial is not None and cols == v["selected_features"]
        fit_start = perf_counter()
        scores = trial["scores"] if trial else list(subset_validation_scores(
            train, yt, val, yv, cols, halves, utility="roc_auc", **policy))
        val_fit_s = 0 if trial else perf_counter() - fit_start
        deltas = [float(a - b) for a, b in zip(scores, v["full_validation"], strict=True)]
        gate = min(deltas) >= -.005 - 1e-12
        if trial:
            assert deltas == trial["deltas"] and gate == trial["passes"]
        frozen[arm] = dict(width=width, method=method, columns=cols, mask=mask,
                           full_validation=v["full_validation"], validation=scores, deltas=deltas,
                           gate_pass=gate, outside_top100=sorted(chosen_set - set(top)),
                           reused_validation=bool(trial), reuse_selected_test=can_reuse_test,
                           source_validation_hash=canonical_hash(v), test_index_sha256=audit["index_sha256"]["test"])
        data[arm] = (train, yt, val, yv, policy, audit)
        times[arm] = dict(val_fit_s=val_fit_s, selection_and_validation_s=perf_counter()-start)
    atomic_write_json(output / "frozen.json", frozen)
    frozen_sha = hashlib.sha256((output / "frozen.json").read_bytes()).hexdigest()
    assert len(set(d["test_index_sha256"] for d in frozen.values())) == 1
    rows, reference_y = [], None
    heldout_predictions = {}
    for arm, part in read(output / "frozen.json").items():
        start = perf_counter()
        main = primary / arm
        train, yt, val, yv, policy, audit = data[arm]
        cols = part["columns"]
        ident = canonical_hash([frozen_sha, arm])
        job = JobSpec(ident, audit["protocol_hash"], "removal", "madelon", 42,
                      "fixed13_" + part["method"], dict(mask=part["mask"]), (), "results.json")
        log = AuditLog.from_manifest(output / f"{arm}-test-access.json", [job])
        event = log.append_test_access(job_id=ident, stage="removal", dataset_id="madelon", seed=42,
                                       reason="final_removal_evaluation", decision="allowed")
        assert event["decision"] == "allowed"
        if part["reuse_selected_test"]:
            with np.load(main / "predictions.npz", allow_pickle=False) as source:
                pred, ye = source["selected"].copy(), source["y"].copy()
            test_fit_s = 0
        else:
            test, ye = load_heldout(part["width"], audit)
            both = pd.concat([train, val], ignore_index=True)
            labels = np.concatenate([yt, yv])
            fit_start = perf_counter()
            pred = fit_predict(both[cols], labels, test[cols], **policy)
            test_fit_s = perf_counter() - fit_start
        if reference_y is None:
            reference_y = ye
        assert np.array_equal(reference_y, ye) and len(ye) == 520 and pred.shape == (520, 2)
        auc = float(compute_utility(ye, pred, metric="roc_auc"))
        if part["reuse_selected_test"]:
            assert auc == read(main / "result.json")["auc"]
        heldout_predictions[arm] = pred
        val_fit_s = times[arm]["val_fit_s"]
        work_s = times[arm]["selection_and_validation_s"] + perf_counter() - start
        rows.append(dict(width=part["width"], method=part["method"], retained=len(cols),
                         outside_top100=len(part["outside_top100"]), gate=part["gate_pass"],
                         val=part["validation"][0], val_a=part["validation"][1], val_b=part["validation"][2],
                         worst_delta=min(part["deltas"]), auc=auc,
                         reused_validation=part["reused_validation"], reused_test=part["reuse_selected_test"],
                         extra_fits=int(not part["reused_validation"]) + int(not part["reuse_selected_test"]),
                         extra_fit_s=val_fit_s+test_fit_s, extra_worker_s=work_s))
    np.savez_compressed(output / "predictions.npz", y=reference_y, **heldout_predictions)
    atomic_write_json(output / "results.json", dict(primary=primary.name, frozen_sha256=frozen_sha, rows=rows))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True, help="Completed four-arm width experiment directory")
    parser.add_argument("--worker-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    primary = args.primary.resolve()
    if args.worker_directory is not None:
        run(primary, args.worker_directory)
        return 0
    if not (primary / "summary.csv").is_file():
        parser.error("The primary directory must contain a completed four-arm summary.csv.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = TEST_RESULTS / ("madelon-fixed13-widths-" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "plan.json", dict(
        primary=str(primary), dataset="madelon", seed=42, retained=13,
        rule="Use the 13 highest frozen gains, restore input column order, and reuse identical primary evaluations.",
        validation="Report pooled and both half validation differences versus the respective full model; tolerance 0.005.",
        test="Freeze all four masks and validation outcomes before any held-out prediction. Use 100-tree train+val fits.",
        safety="Only a supplementary equal-size comparison. No primary write or adaptive choice.",
        limit_s=900,
    ))
    print(f"Fixed-size diagnostic: {output}", flush=True)
    t = perf_counter()
    try:
        worker = [sys.executable, "-B", str(Path(__file__).resolve()), "--primary", str(primary),
                  "--worker-directory", str(output)]
        status = subprocess.run(worker, timeout=900, check=False).returncode
    except subprocess.TimeoutExpired:
        status = 124
    atomic_write_json(output / "watchdog.json", dict(limit_s=900, wall_s=perf_counter()-t, exit_code=status))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
