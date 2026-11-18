"""Compare two XGBoost prediction recipes on already frozen seed-42 feature sets."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.thesis_sampling_worker import DATASETS, SEED, SOURCE, prediction_scores, validation_halves
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict_checkpoints, make_model
from minppi_mofs.utility import compute_utility


def adopt_blend(original, blend):
    deltas = np.subtract(blend, original)
    return bool(np.min(deltas) >= -1e-12 and np.mean(deltas[1:]) > 1e-12)


def late_rounds(original_rounds):
    return [(3 * original_rounds + 1) // 2, 2 * original_rounds]


def rounds_predictions(train, labels, evaluation, *, task, model_params, backend, seed):
    """Fit a single 2B booster and return the B and mean(1.5B, 2B) predictions."""
    original = model_params["n_estimators"]
    first, second = late_rounds(original)
    by_round = fit_predict_checkpoints(train, labels, evaluation, task=task,
                                       model_params=dict(model_params, n_estimators=second), backend=backend,
                                       seed=seed, checkpoints=sorted({original, first, second}))
    return {"original": by_round[original], "blend": (by_round[first] + by_round[second]) * .5}


def run(dataset, output, mask_manifest, *, recipe="column_blend"):
    if recipe not in {"column_blend", "rounds"}:
        raise ValueError("Invalid predictor recipe")
    start = time.monotonic()
    output, mask_manifest = Path(output), Path(mask_manifest)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Use a new output directory for each experiment")
    state = dict(dataset=dataset, seed=SEED, phase="loading", predictor_recipe=recipe, validation_fits=0, test_fits=0)

    def publish(**values):
        state.update(values, elapsed_s=time.monotonic()-start)
        atomic_write_json(output / "progress.json", state)

    publish()
    with (SOURCE / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        row, = (r for r in csv.DictReader(handle) if r["dataset_id"] == dataset and int(r["seed"]) == SEED)
    for part in ("train", "val", "test"):
        row[f"{part}_row_count"] = int(row[f"{part}_row_count"])
    train, yt, val, yv = load_prepared_train_validation(SOURCE, row)
    frozen_protocol = yaml.safe_load((SOURCE / "resolved_protocol.yaml").read_text(encoding="utf-8"))
    policy = dict(task=row["task"], seed=SEED, backend=frozen_protocol["backend"],
                  model_params=frozen_protocol["model_params_by_dataset"][dataset])
    if recipe == "rounds":
        original_rounds = policy["model_params"]["n_estimators"]
        blend_checkpoints = late_rounds(original_rounds)
    metric = row["utility"]
    features = tuple(train.columns)
    payload = mask_manifest.read_bytes()
    manifest = json.loads(payload)
    frozen = manifest["datasets"][dataset]
    if manifest["seed"] != SEED or frozen["feature_order_sha256"] != canonical_hash(features):
        raise ValueError("Frozen guarded mask seed or feature-order digest differs")
    full = (1 << len(features)) - 1
    guarded = int(frozen["selected_mask"], 16)
    if not 0 < guarded <= full or not 0 < len(features):
        raise ValueError("Guarded manifest retained mask is invalid")
    masks = dict(full=full, guarded=guarded)
    halves, split_meta = validation_halves(yv, policy["task"])
    digest = hashlib.sha256(payload).hexdigest()
    publish(features=features, guarded_mask=hex(guarded), mask_manifest_sha256=digest,
            policy=dict(policy, utility=metric), validation_halves=split_meta,
            partition_sha256={part:row[f"{part}_file_sha256"] for part in ("train","val","test")})

    def predictions(mask, fitted_on, labels, evaluation, phase):
        cols = [f for i, f in enumerate(features) if mask & (1 << i)]
        if recipe == "rounds":
            publish(phase=phase, fit_mask=hex(mask), fit_recipe="rounds")
            by_recipe = rounds_predictions(fitted_on.loc[:, cols], labels, evaluation.loc[:, cols], **policy)
            state[f"{phase}_fits"] += 1
            return by_recipe
        safe = [f"f{i}" for i in range(len(cols))]
        X = fitted_on.loc[:, cols].set_axis(safe, axis=1)
        Z = evaluation.loc[:, cols].set_axis(safe, axis=1)
        outputs = {}
        for name in ("original", "subsample"):
            publish(phase=phase, fit_mask=hex(mask), fit_recipe=name)
            model = make_model(**policy)
            if name == "subsample":
                model.set_params(colsample_bynode=.5)
            model.fit(X, labels)
            predict = model.predict if policy["task"] == "regression" else model.predict_proba
            values = np.asarray(predict(Z), dtype=float)
            expected = (len(Z),) if policy["task"] == "regression" else (len(Z), len(np.unique(labels)))
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError("Invalid predictor output")
            outputs[name] = values
            state[f"{phase}_fits"] += 1
        return {"original": outputs["original"], "blend": (outputs["original"] + outputs["subsample"]) * .5}

    validation = {}
    for mask in dict.fromkeys(masks.values()):
        validation[mask] = {k:prediction_scores(yv, p, halves, metric) for k,p in
                            predictions(mask, train, yt, val, "validation").items()}
    decisions = {}
    for role, mask in masks.items():
        baseline, blend = validation[mask]["original"], validation[mask]["blend"]
        archived = frozen["full_validation"] if role == "full" else frozen["official_validation"][frozen["selected_source"]]
        if not np.allclose(baseline, archived, atol=1e-10, rtol=0):
            raise ValueError(f"Original validation scores differ from frozen {role} baseline")
        decisions[role] = dict(mask=hex(mask), original_scores=baseline, blend_scores=blend,
                               validation_deltas=np.subtract(blend, baseline).tolist(), adopt_blend=adopt_blend(baseline, blend))
    params = dict(mask_manifest_sha256=digest, masks={r:hex(m) for r,m in masks.items()}, seed=SEED,
                  blend_weights=[.5,.5], adopt_blend={r:d["adopt_blend"] for r,d in decisions.items()})
    if recipe == "rounds":
        params.update(predictor_recipe=recipe, original_rounds=original_rounds, blend_checkpoints=blend_checkpoints)
    else:
        params["colsample_bynode"] = .5
    job_id = canonical_hash(dict(dataset=dataset, seed=SEED, protocol=row["protocol_hash"], parameters=params))
    publish(phase="test_access", decisions=decisions, parameters=params, job_id=job_id)
    job = JobSpec(job_id, row["protocol_hash"], "removal", dataset, SEED, "fixed_predictor", params, (), "result.json")
    audit = AuditLog.from_manifest(output / "test_access.json", [job])
    paths = PreparedSplitPaths(dataset, SEED, row["target"], *(SOURCE / row[f"{p}_relpath"] for p in ("train","val","test")))
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=job_id, audit=audit).load_test(
        reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
    if tuple(test.columns) != features or len(ye) != row["test_row_count"]:
        raise ValueError("Test partition shape differs from frozen manifest")
    final_train = pd.concat([train, val], ignore_index=True)
    final_y = np.concatenate([yt,yv])
    test_preds = {m:predictions(m, final_train, final_y, test, "test") for m in dict.fromkeys(masks.values())}
    methods, archive = {}, {"y": ye}
    for role, mask in masks.items():
        for name in ("original", "blend", "chosen"):
            selected_recipe = name if name != "chosen" else ("blend" if decisions[role]["adopt_blend"] else "original")
            prediction = test_preds[mask][selected_recipe]
            key = role + "_" + name
            methods[key] = dict(mask=hex(mask), recipe=selected_recipe, scores=validation[mask][selected_recipe],
                                test_utility=compute_utility(ye, prediction, metric=metric))
            archive[key] = prediction
    filename = "paired_test_predictions.npz"
    np.savez_compressed(output / filename, **archive)
    publish(phase="complete", method_results=methods, test_training_partitions=["train","val"],
            paired_test_predictions=filename,
            paired_test_predictions_sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest())
    atomic_write_json(output / "result.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--recipe", choices=("column_blend","rounds"), default="column_blend")
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.mask_manifest, recipe=args.recipe)
