"""Time-limited Madelon pilot with two balanced sampling rounds."""

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.test.thesis_madelon_alpha_data import load_arm, load_heldout
from experiments.test.thesis_madelon_alpha_noise import check_positive_noise
from experiments.thesis_nested_k_worker import _guarded_selection
from experiments.thesis_sampling_worker import (
    full_validation_and_gain, sample, subset_validation_scores, validation_halves,
)
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import _empty_subset_predictions, fit_predict
from minppi_mofs.utility import compute_utility


ROUNDS = 2


def run(width, output, *, alpha=.99):
    """Run one arm in a fresh output directory under the pilot's parent watchdog."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("progress.json", "endpoints.sqlite", "setup.json", "result.json", "test_access.json", "k-1", "k-2"):
        if (output / name).exists():
            raise FileExistsError(f"{name} already exists in {output}")
    start = time.monotonic()
    last_write = start - 1
    state = dict(dataset="madelon", seed=42, width=width, alpha=alpha, requested_rounds=ROUNDS,
                 phase="loading", sample_fits=0, validation_fits=0, noise_fits=0, test_fits=0,
                 full_validation_fits=0, total_fits=0, sampled_draws=0, completed_rounds=0,
                 new_negative_by_round={})

    def publish(*, force=False, **changes):
        nonlocal last_write
        important = any(key in changes and changes[key] != state.get(key)
                        for key in ("phase", "round", "witness_count", "completed_rounds"))
        state.update(changes)
        now = time.monotonic()
        state["elapsed_s"] = now - start
        state["total_fits"] = sum(state[key] for key in (
            "full_validation_fits", "sample_fits", "validation_fits", "noise_fits", "test_fits"))
        if force or important or now - last_write >= .5:
            atomic_write_json(output / "progress.json", state)
            last_write = now

    publish(force=True)
    db = None
    try:
        train, yt, val, yv, historical_top100, audit = load_arm(width, alpha=alpha)
        features = tuple(train.columns)
        if len(features) != width or tuple(val.columns) != features:
            raise ValueError("Madelon arm feature order does not match the requested width")
        top100 = set(historical_top100)
        if len(top100) != 100 or not top100.issubset(features):
            raise ValueError("Frozen historical top-100 feature set is inconsistent")
        policy, margin = audit["policy"], audit["classification_margin"]
        if policy["seed"] != 42 or policy["utility"] != "roc_auc" or policy["model_params"]["n_estimators"] != 100:
            raise ValueError("Expected the frozen Madelon seed-42 fixed 100-tree AUC protocol")
        halves, half_metadata = validation_halves(yv, policy["task"], seed=policy["seed"])
        identifier = canonical_hash(dict(audit=audit, features=features, top100=historical_top100,
                                         alpha=alpha, rounds=ROUNDS, validation_halves=half_metadata))
        publish(phase="full_validation", n_features=width, planned_trials=width * ROUNDS)
        full_scores, gain, shap = full_validation_and_gain(train, yt, val, yv, halves, include_shap=True, **policy)
        atomic_write_json(output / "setup.json", dict(input_fingerprint=identifier, audit=audit,
                          features=features, top100=historical_top100, validation_halves=half_metadata,
                          full_scores=full_scores, gain=gain, shap=shap))
        threshold = alpha * full_scores[0]
        publish(phase="sampling", threshold=threshold, full_validation_auc=full_scores[0],
                full_validation_fits=1)
        full = (1 << width) - 1
        db = sqlite3.connect(output / "endpoints.sqlite")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS scores (mask TEXT PRIMARY KEY, pooled REAL, first REAL, second REAL)")
        cached = {full:tuple(full_scores)}
        sampled_masks = set()
        gate_masks = set()

        def columns(mask):
            return [f for i, f in enumerate(features) if mask & (1 << i)]

        def evaluate(mask):
            if mask not in cached:
                scores = subset_validation_scores(train, yt, val, yv, columns(mask), halves, **policy)
                # Commit each completed endpoint so a killed process loses at most the active fit.
                db.execute("INSERT INTO scores VALUES (?, ?, ?, ?)", (hex(mask), *scores))
                db.commit()
                cached[mask] = scores
            return cached[mask]

        def pooled(mask):
            fresh = mask not in sampled_masks and mask not in (0, full)
            scores = evaluate(mask)
            if fresh:
                sampled_masks.add(mask)
                publish(sample_fits=len(sampled_masks))
            return scores[0]

        previous_witnesses = set()
        round_records = {}

        def finish_round(k, evidence):
            if evidence["resolved_trials"] != k * width or evidence["partial_pair"] is not None:
                raise ValueError(f"Unresolved MinPI samples at round {k}")
            witnesses = evidence["witnesses"]
            new = [f for f in features if f in witnesses and f not in previous_witnesses]
            by_round = {**state["new_negative_by_round"], str(k):new}
            record = dict(evidence, k=k, width=width, seed=42, alpha=alpha, threshold=threshold,
                          full_validation_auc=full_scores[0], sample_fits=len(sampled_masks),
                          new_negative_features=new, new_negative_by_round=by_round,
                          new_negative_outside_top100=[f for f in new if f not in top100],
                          input_fingerprint=identifier)
            atomic_write_json(output / f"k-{k}" / "sampling.json", record)
            previous_witnesses.update(witnesses)
            round_records[str(k)] = {key:record[key] for key in (
                "witness_count", "sample_fits", "sampled_draws", "sample_elapsed_s", "new_negative_features")}
            publish(force=True, completed_rounds=k, new_negative_by_round=by_round)

        evidence = sample(features, ROUNDS, None, threshold, pooled, publish, seed=policy["seed"], on_prefix=finish_round)
        if list(round_records) != ["1", "2"]:
            raise RuntimeError("Both Madelon sampling rounds must complete before feature selection")
        positives = [f for f, m in evidence["sample_minima"].items() if m == 1]
        publish(phase="positive_noise_check", positive_count=len(positives))

        def noise_progress(event):
            publish(noise_fits=event["fit_count"], noise_progress=event)

        noise = check_positive_noise(train, yt, positives, policy=policy, margin=margin, publish=noise_progress)
        publish(force=True, noise_fits=noise["fit_count"], phase="validation_gate")

        def gate_scores(mask):
            scores = evaluate(mask)
            if mask not in (0, full) and mask not in sampled_masks:
                gate_masks.add(mask)
                publish(validation_fits=len(gate_masks))
            return scores

        choice, nominees, trials = _guarded_selection(train, evidence, gain, shap, gate_scores, full_scores, margin, publish)
        selected = choice["selected_mask"]
        retained = columns(selected)
        removed = choice["removed_features"]
        outside = [f for f in retained if f not in top100]
        frozen = dict(dataset="madelon", seed=42, width=width, alpha=alpha, k=ROUNDS, threshold=threshold,
                      input_fingerprint=identifier, policy=policy, validation_halves=half_metadata,
                      model_rounds=policy["model_params"]["n_estimators"], n_features=width,
                      full_validation_auc=full_scores[0], selected_validation_auc=choice["selected_scores"][0],
                      guarded_update=choice, selection_trials=trials, proposed_features=nominees,
                      removed_count=len(removed), removed_features=removed, retained_count=len(retained),
                      retained_features=retained, retained_outside_top100=outside, retained_outside_top100_count=len(outside),
                      new_negative_by_round=state["new_negative_by_round"], rounds=round_records,
                      sample_minima=evidence["sample_minima"], witnesses=evidence["witnesses"],
                      sampled_draws=evidence["sampled_draws"], sample_elapsed_s=evidence["sample_elapsed_s"],
                      sample_fits=len(sampled_masks), positive_count=len(positives), positive_features=positives,
                      noise_fits=noise["fit_count"], noise_check=noise)
        atomic_write_json(output / "validation.json", frozen)
        test_id = canonical_hash(dict(input_fingerprint=identifier, full_mask=hex(full), selected_mask=hex(selected),
                                      predictor=policy, audit_method="fixed_100_after_validation"))
        publish(phase="test_access")
        atomic_write_json(output / "test_access.json", dict(job_id=test_id, input_fingerprint=identifier,
                          full_mask=hex(full), selected_mask=hex(selected), reason="final_frozen_evaluation"))
        test, ye = load_heldout(width, audit)
        if tuple(test.columns) != features:
            raise ValueError("Held-out feature order differs from the frozen training input")
        fit_train, fit_y = pd.concat([train, val], ignore_index=True), np.concatenate([yt, yv])
        test_scores = {}
        test_predictions = {}
        for mask in dict.fromkeys([full, selected]):
            publish(phase="heldout_evaluation", test_mask=hex(mask))
            cols = columns(mask)
            if cols:
                pred = fit_predict(fit_train.loc[:, cols], fit_y, test.loc[:, cols], task=policy["task"],
                                   model_params=policy["model_params"], backend=policy["backend"], seed=policy["seed"])
            else:
                pred = _empty_subset_predictions(fit_y, task=policy["task"], row_count=len(test))
            test_predictions[mask] = np.asarray(pred)
            test_scores[mask] = float(compute_utility(ye, pred, metric=policy["utility"]))
            publish(force=True, test_fits=state["test_fits"] + int(bool(mask)))
        packet = output / "paired_test_predictions.npz"
        np.savez_compressed(packet, y=ye, full=test_predictions[full], selected=test_predictions[selected])
        result = dict(frozen, phase="complete", full_test_auc=test_scores[full], selected_test_auc=test_scores[selected],
                      delta_test_auc=test_scores[selected]-test_scores[full], test_job_id=test_id,
                      test_fits=state["test_fits"], full_validation_fits=1, validation_fits=len(gate_masks),
                      total_fits=state["total_fits"], elapsed_s=time.monotonic()-start,
                      test_training_partitions=["train", "val"], paired_test_predictions=packet.name)
        atomic_write_json(output / "result.json", result)
        publish(force=True, phase="complete", full_test_auc=test_scores[full], selected_test_auc=test_scores[selected],
                retained_count=len(retained), retained_outside_top100_count=len(outside))
        return result
    except Exception as error:
        publish(force=True, phase="error", failed_phase=state["phase"], error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if db is not None:
            db.close()
