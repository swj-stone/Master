"""Replicate the frozen boosting-round recipe on three parts of the original training CSV."""

import argparse
import csv
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from experiments.calibrate_icomp2026_models import _safe_partition_path
from experiments.thesis_fixed_predictor import adopt_blend
from experiments.thesis_sampling_worker import DATASETS, SEED, SOURCE, prediction_scores, validation_halves
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import _validated_target, fit_predict_checkpoints
from minppi_mofs.utility import compute_utility


def training_partitions(labels, task):
    stratified = task != "regression"
    development, evaluation = train_test_split(np.arange(len(labels)), test_size=.2, random_state=SEED,
                                              stratify=labels if stratified else None)
    development = np.sort(development)
    fit, validation = train_test_split(development, test_size=.25, random_state=SEED,
                                       stratify=labels[development] if stratified else None)
    parts = tuple(np.sort(p) for p in (fit, validation, evaluation))
    info = dict(index_space="zero-based positions in frozen training CSV", order=["fit","validation","evaluation"],
                seed=SEED, stratified=stratified, sizes=[len(p) for p in parts],
                index_sha256=[hashlib.sha256(p.astype("<i8").tobytes()).hexdigest() for p in parts])
    return parts, info


def run(dataset, output, mask_manifest):
    output, mask_manifest = Path(output), Path(mask_manifest)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"frozen_decision.json").exists() or (output/"result.json").exists():
        raise FileExistsError("Use a fresh directory for each training replication")
    start = time.monotonic()
    state = dict(dataset=dataset, seed=SEED, source_partition="train", predictor_recipe="rounds",
                 phase="loading", validation_fits=0, test_fits=0)

    def publish(**changes):
        state.update(changes, elapsed_s=time.monotonic()-start)
        atomic_write_json(output/"progress.json", state)

    publish()
    with (SOURCE/"data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        row, = (r for r in csv.DictReader(stream) if r["dataset_id"] == dataset and int(r["seed"]) == SEED)
    raw = _safe_partition_path(SOURCE, row["train_relpath"]).read_bytes()
    train_digest = hashlib.sha256(raw).hexdigest()
    if train_digest != row["train_file_sha256"]:
        raise ValueError("Frozen training partition checksum differs")
    frame = pd.read_csv(BytesIO(raw))
    features = json.loads(row["selected_features_json"])
    if (list(frame.columns) != [*features, row["target"]] or len(frame) != int(row["train_row_count"])
            or not np.isfinite(frame.to_numpy(dtype=float)).all()):
        raise ValueError("Invalid frozen training rows or columns")
    task, metric = row["task"], row["utility"]
    labels = _validated_target(frame.pop(row["target"]).to_numpy(), row_count=len(frame), task=task)
    mask_blob = mask_manifest.read_bytes()
    locked = json.loads(mask_blob)
    selection = locked["datasets"][dataset]
    mask = int(selection["selected_mask"],16)
    if (locked["seed"] != SEED or canonical_hash(features) != selection["feature_order_sha256"]
            or not 0 < mask < 1 << len(features)):
        raise ValueError("Frozen guarded feature manifest differs")
    selected = [f for i,f in enumerate(features) if mask & (1 << i)]
    X = frame.loc[:, selected]
    protocol = yaml.safe_load((SOURCE/"resolved_protocol.yaml").read_text(encoding="utf-8"))
    model_params = protocol["model_params_by_dataset"][dataset]
    policy = dict(task=task, model_params=model_params, backend=protocol["backend"], seed=SEED)
    B = model_params["n_estimators"]
    late, end = (3*B+1)//2, 2*B
    parameters = dict(original_rounds=B, blend_checkpoints=[late,end], blend_weights=[.5,.5])
    (fit, validation, evaluation), split = training_partitions(labels, task)
    halves, half_info = validation_halves(labels[validation], task)
    half_info["index_space"] = "zero-based positions within sorted internal training validation subset"
    mask_digest = hashlib.sha256(mask_blob).hexdigest()
    publish(features=features, guarded_mask=hex(mask), mask_manifest_sha256=mask_digest,
            source_partition_sha256=train_digest, training_split=split, validation_halves=half_info,
            parameters=parameters, policy=dict(policy, utility=metric))

    def predict(fitting, scoring):
        current = dict(policy, model_params=dict(model_params, n_estimators=end))
        ps = fit_predict_checkpoints(X.iloc[fitting], labels[fitting], X.iloc[scoring], **current,
                                     checkpoints=sorted({B,late,end}))
        return {"original":ps[B], "blend":(ps[late] + ps[end]) * .5}

    publish(phase="internal_validation")
    valid_preds = predict(fit, validation)
    valid_scores = {k:prediction_scores(labels[validation], pred, halves, metric) for k,pred in valid_preds.items()}
    original, blend = valid_scores["original"], valid_scores["blend"]
    use_blend = adopt_blend(original, blend)
    decision = dict(mask=hex(mask), original_scores=original, blend_scores=blend, adopt_blend=use_blend,
                    validation_deltas=np.subtract(blend, original).tolist())
    lock = dict(dataset=dataset, seed=SEED, source_partition_sha256=train_digest, mask_manifest_sha256=mask_digest,
                training_split=split, validation_halves=half_info, parameters=parameters, decisions={"guarded": decision})
    atomic_write_json(output/"frozen_decision.json", lock)
    publish(phase="internal_evaluation", validation_fits=1, decisions=lock["decisions"],
            decision_sha256=hashlib.sha256((output/"frozen_decision.json").read_bytes()).hexdigest())
    predictions = predict(np.sort(np.r_[fit,validation]), evaluation)
    y = labels[evaluation]
    archive, methods = {"y":y}, {}
    utilities = {k:compute_utility(y, pred, metric=metric) for k,pred in predictions.items()}
    for name in ("original", "blend", "chosen"):
        selected_recipe = name if name != "chosen" else ("blend" if use_blend else "original")
        key = "guarded_" + name
        archive[key] = predictions[selected_recipe]
        methods[key] = dict(mask=hex(mask), recipe=selected_recipe, scores=valid_scores[selected_recipe],
                            test_utility=utilities[selected_recipe], test_delta_vs_original=utilities[selected_recipe]-utilities["original"])
    filename = "paired_test_predictions.npz"
    np.savez_compressed(output/filename, **archive)
    publish(phase="complete", test_fits=1, method_results=methods, paired_test_predictions=filename,
            paired_test_predictions_sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest())
    atomic_write_json(output/"result.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.mask_manifest)


if __name__ == "__main__":
    main()
