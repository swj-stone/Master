"""Evaluate three prefixes of the same MinPI trajectory on one frozen split."""

import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.thesis_fixed_predictor import late_rounds
from experiments.thesis_sampling_worker import (
    fit_rounds_for_masks, freeze_rounds, full_validation_and_gain, sample, subset_validation_scores, validation_halves,
)
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.guarded_update import guarded_update
from minppi_mofs.removal_selection import nominate_removals, select_robust_removals
from minppi_mofs.sign_aware_removal import select_sign_aware_removals
from minppi_mofs.utility import compute_utility


CHECKPOINTS = (1, 2, 3)


def _guarded_selection(train, evidence, gain, shap, evaluate, full_scores, margin, publish):
    """Run the same two selectors and Pareto gate used by guarded_rounds."""
    features = tuple(train.columns)
    witnesses, minima = evidence["witnesses"], evidence["sample_minima"]
    trials = {}

    def track(trial, name):
        mask = trial["mask"]
        record = trials.setdefault(mask, dict(trial, official_policies=[]))
        record["official_policies"].append(name)
        publish(validation_masks_checked=len(trials))

    signed = select_sign_aware_removals(features, witnesses, minima, gain, shap, evaluate, full_scores,
                                        margin, publish=lambda t: track(t, "sign_worst"), include_controls=False)
    nominations = nominate_removals(train, gain, witnesses, sample_minima=minima)
    structured = select_robust_removals(features, nominations, gain, evaluate, full_scores, margin,
                                        sample_minima=minima, shap_importance=shap, structured_only=True,
                                        publish=lambda t: track(t, "structured"))
    scores = {r["selected_mask"]: r["selected_scores"] for r in (structured, signed)}
    choice = guarded_update(features, structured["selected_mask"], signed["selected_mask"], scores, full_scores, margin)
    nominees = set(signed["candidate_features"]) | set(structured["candidate_features"])
    if not set(choice["removed_features"]) <= nominees or any(minima[f] not in (-1, 0) for f in nominees):
        raise ValueError("Invalid MinPI removal candidates")
    return choice, [f for f in features if f in nominees], list(trials.values())


def run(dataset, output, *, seed, source):
    """Evaluate K=1/2/3 on one draw sequence and freeze three decisions before testing.

    sample_elapsed_s excludes checkpoint analysis. sample_fits counts actual sampling
    fits; unique_sampling_fits also counts masks already fitted for an earlier selection.
    checkpoint_selection_s and checkpoint_test_s measure newly performed work with the
    shared caches. elapsed_s in the job result measures the entire nested run.
    """
    output, source = Path(output), Path(source)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("progress.json", "result.json", "test_access.json", "k-1", "k-2", "k-3")):
        raise FileExistsError("Use a new output directory for each nested experiment")
    start = time.monotonic()
    state = dict(dataset=dataset, seed=seed, run_mode="formal", phase="loading", active_k=1,
                 frozen_ks=[], sample_fits=0, validation_fits=0, test_fits=0)

    def publish(**changes):
        state.update(changes, elapsed_s=time.monotonic() - start)
        atomic_write_json(output / "progress.json", state)

    try:
        publish()
        with (source / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            row, = (r for r in csv.DictReader(stream) if r["dataset_id"] == dataset and int(r["seed"]) == seed)
        parts = ("train", "val", "test")
        for part in parts:
            row[f"{part}_row_count"] = int(row[f"{part}_row_count"])
        train, yt, val, yv = load_prepared_train_validation(source, row)
        protocol = yaml.safe_load((source / "resolved_protocol.yaml").read_text(encoding="utf-8"))
        features = tuple(train.columns)
        n, full = len(features), (1 << len(features)) - 1
        policy = dict(task=row["task"], utility=row["utility"], model_params=protocol["model_params_by_dataset"][dataset],
                      backend=protocol["backend"], seed=seed)
        halves, split = validation_halves(yv, policy["task"], seed=seed)
        margin_name = "regression" if policy["task"] == "regression" else "classification"
        margin = protocol["removal"][f"{margin_name}_noninferiority_margin"]
        publish(phase="full_validation", n_features=n, planned_trials=n * CHECKPOINTS[-1])
        full_scores, gain, shap = full_validation_and_gain(train, yt, val, yv, halves, include_shap=True, **policy)
        threshold = protocol["alpha_default"] * full_scores[0]
        common = dict(dataset=dataset, seed=seed, features=list(features), n_features=n, policy=policy,
                      run_mode="formal", sample_seconds=None, alpha=protocol["alpha_default"], threshold=threshold,
                      margin=margin, validation_halves=split, full_validation_scores=full_scores, source_run_id=source.name,
                      protocol_hash=row["protocol_hash"], partition_sha256={p:row[f"{p}_file_sha256"] for p in parts})
        scores = {full:full_scores}
        first_sample_lookups = set()
        selection_credits = 0
        valid_predictions = {}
        frozen = {}
        setup_s = time.monotonic() - start
        publish(phase="sampling", validation_fits=1, policy=policy)

        def evaluate(mask):
            if mask not in scores:
                cols = tuple(f for i, f in enumerate(features) if mask & (1 << i))
                scores[mask] = subset_validation_scores(train, yt, val, yv, cols, halves, **policy)
                state["validation_fits"] += int(bool(mask))
            return scores[mask]

        def pooled(mask):
            nonlocal selection_credits
            if mask and mask != full and mask not in first_sample_lookups:
                first_sample_lookups.add(mask)
                if mask in scores:
                    selection_credits += 1
                else:
                    state["sample_fits"] += 1
            return evaluate(mask)[0]

        def freeze_prefix(k, evidence):
            began = time.monotonic()
            if evidence["partial_pair"] is not None or evidence["planned_trials"] != n*k or evidence["resolved_trials"] != n*k:
                raise RuntimeError(f"Incomplete sampling prefix for K={k}")
            prior = frozen.get(str(k-1), {})
            sampling = dict(**evidence, sample_fits=state["sample_fits"], sampling_cache_hits_from_selection=selection_credits,
                            unique_sampling_fits=len(first_sample_lookups), sampling_stop=evidence["stop"],
                            incremental_sample_s=evidence["sample_elapsed_s"] - prior.get("sample_elapsed_s", 0.),
                            incremental_sample_fits=state["sample_fits"] - prior.get("sample_fits", 0))
            folder = output / f"k-{k}"
            atomic_write_json(folder / "sampling.json", dict(dataset=dataset, seed=seed, k=k, **sampling))
            publish(phase="validation_gate", active_k=k, validation_masks_checked=0)
            before_fits = state["validation_fits"]
            choice, proposed, trials = _guarded_selection(train, evidence, gain, shap, evaluate, full_scores, margin, publish)
            mask = choice["selected_mask"]
            masks = {"full":full, "guarded":mask}
            new_masks = {name:m for name,m in masks.items() if m not in valid_predictions}
            valid_predictions.update(fit_rounds_for_masks(train, yt, val, new_masks, features, policy,
                                                          "predictor_validation", state, publish))
            decisions, methods = freeze_rounds(masks, valid_predictions, yv, halves, policy["utility"], scores)
            record = dict(common, **sampling)
            record.update(k=k, requested_rounds=k, phase="validation_frozen", predictor_decisions=decisions,
                          guarded_update=choice, selected_mask=mask, selected_method="guarded_chosen",
                          selected_source=choice["selected_source"], retained_count=mask.bit_count(),
                          removed_count=choice["removed_count"], removed_features=choice["removed_features"],
                          removed_witness_count=len(set(evidence["witnesses"]).intersection(choice["removed_features"])),
                          proposed_features=proposed, proposed_count=len(proposed), selection_trials=trials,
                          selected_validation_utility=methods["guarded_chosen"]["scores"][0], method_results=methods,
                          method_masks={name:r["mask"] for name,r in methods.items()},
                          checkpoint_selection_fits=state["validation_fits"]-before_fits,
                          checkpoint_selection_s=time.monotonic()-began)
            atomic_write_json(folder / "validation.json", record)
            frozen[str(k)] = record
            latest = {name:record[name] for name in ("k", "removed_count", "sampled_draws", "sample_elapsed_s",
                                                     "selected_validation_utility")}
            publish(phase="sampling", active_k=min(k+1, CHECKPOINTS[-1]),
                    frozen_ks=[int(key) for key in frozen], latest_checkpoint=latest)

        sample(features, CHECKPOINTS[-1], None, threshold, pooled, publish, seed=seed, on_prefix=freeze_prefix)
        if list(frozen) != list(map(str, CHECKPOINTS)):
            raise RuntimeError("Missing frozen K checkpoint before test access")
        original_rounds = policy["model_params"]["n_estimators"]
        params = dict(checkpoints=list(CHECKPOINTS), sample_seconds=None, selection_policy="guarded_rounds",
                      original_rounds=original_rounds, blend_checkpoints=late_rounds(original_rounds),
                      blend_weights=[.5,.5], prefixes={
                          k: dict(method_masks=record["method_masks"], adopt_blend={
                              role:d["adopt_blend"] for role,d in record["predictor_decisions"].items()})
                          for k,record in frozen.items()})
        job_id = canonical_hash(dict(dataset=dataset, seed=seed, protocol=row["protocol_hash"], parameters=params))
        publish(phase="test_access", job_id=job_id, parameters=params)
        job = JobSpec(job_id, row["protocol_hash"], "removal", dataset, seed, "nested_minpi", params, (), "result.json")
        audit = AuditLog.from_manifest(output / "test_access.json", [job])
        paths = PreparedSplitPaths(dataset, seed, row["target"], *(source / row[f"{p}_relpath"] for p in parts))
        test, ye = AuditedSplitAccess(paths, stage="removal", job_id=job_id, audit=audit).load_test(
            reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
        if tuple(test.columns) != features or len(ye) != row["test_row_count"]:
            raise ValueError("Test partition shape does not match frozen manifest")
        fit_train = pd.concat([train, val], ignore_index=True)
        fit_y = np.concatenate([yt,yv])
        fitted_test_masks = set()
        test_utilities = {}
        archive = {"y":ye}
        results = {}
        for key, locked in frozen.items():
            k = int(key)
            before = time.monotonic()
            fits_before = state["test_fits"]
            publish(phase="predictor_test", active_k=k)
            needed = {role:m for role,m in (("full",full),("guarded",locked["selected_mask"]))
                      if m not in fitted_test_masks}
            added = fit_rounds_for_masks(fit_train, fit_y, test, needed, features, policy, "predictor_test", state, publish)
            fitted_test_masks.update(added)
            for mask, predictions in added.items():
                for recipe, array in predictions.items():
                    archive[f"{hex(mask)}_{recipe}"] = array
                    test_utilities[mask, recipe] = compute_utility(ye, array, metric=policy["utility"])
            methods = {}
            prediction_keys = {}
            for name, record in locked["method_results"].items():
                mask, recipe = record["mask"], record["recipe"]
                methods[name] = test_utilities[mask, recipe]
                prediction_keys[name] = f"{hex(mask)}_{recipe}"
            results[key] = dict(locked, phase="complete", method_test_utilities=methods, job_id=job_id,
                                method_results={name:dict(record, test_utility=methods[name])
                                                for name,record in locked["method_results"].items()},
                                paired_prediction_keys=prediction_keys, checkpoint_test_fits=state["test_fits"]-fits_before,
                                checkpoint_test_s=time.monotonic()-before)
        packet = output/"paired_test_predictions.npz"
        np.savez_compressed(packet, **archive)
        digest = hashlib.sha256(packet.read_bytes()).hexdigest()
        for key, record in results.items():
            record.update(paired_test_predictions="../paired_test_predictions.npz", paired_test_predictions_sha256=digest)
            atomic_write_json(output/f"k-{key}"/"result.json", record)
        top = dict(common, phase="complete", checkpoints=results, setup_s=setup_s, sample_fits=state["sample_fits"],
                   validation_fits=state["validation_fits"], predictor_validation_fits=len(valid_predictions),
                   test_fits=state["test_fits"], job_id=job_id, parameters=params,
                   training_rankings={"gain":gain, "mean_abs_treeshap":shap}, test_training_partitions=["train","val"],
                   paired_test_predictions=packet.name, paired_test_predictions_sha256=digest, elapsed_s=time.monotonic()-start)
        atomic_write_json(output/"result.json", top)
        publish(phase="complete")
    except Exception as error:
        publish(phase="error", failed_phase=state["phase"], error=f"{type(error).__name__}: {error}")
        raise
