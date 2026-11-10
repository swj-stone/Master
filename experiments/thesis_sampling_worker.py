"""Run one cold, balanced-round MinPI experiment on a frozen seed-42 split."""

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.icomp2026_stages import _removal_safety_gate
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json,
    canonical_hash,
)
from minppi_mofs.minimum_sampling import sample_random_round, update_sample_minimum
from minppi_mofs.modeling import evaluate_fixed_subset, fit_predict
from minppi_mofs.utility import compute_utility


SOURCE = Path(__file__).resolve().parents[1] / "results/icomp2026_formal/icomp2026-formal-v1-s42-123"
DATASETS = ("abalone", "madelon", "spambase", "gisette", "isolet", "credit_card_fraud")
SEED = 42


class SamplingDeadline(Exception):
    pass


def subset_utility(train, yt, eval_frame, ye, selected, **policy):
    if not selected:
        return evaluate_fixed_subset(train, yt, eval_frame, ye, selected, **policy)
    # The frozen loader already validated all columns. Each fit only needs the selected ones.
    columns = list(selected)
    prediction = fit_predict(train.loc[:, columns], yt, eval_frame.loc[:, columns], task=policy["task"],
                             model_params=policy["model_params"], backend=policy["backend"], seed=policy["seed"])
    return float(compute_utility(ye, prediction, metric=policy["utility"]))


def sample(features, rounds, seconds, threshold, evaluate, publish, *, clock=time.monotonic):
    n = len(features)
    rng = np.random.default_rng(SEED)
    minima = {f: None for f in features}
    counts = dict.fromkeys(features, 0)
    witnesses = {}
    skipped = completed = 0
    active_round = 0
    partial = None
    start = clock()
    deadline = start + seconds
    deadline_utc = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
    planned = n * rounds

    def snapshot():
        draws = sum(counts.values())
        resolved = draws + sum(rounds - counts[f] for f in witnesses)
        elapsed = clock() - start
        remaining = max(0.0, seconds - elapsed)
        eta = None if resolved == 0 else min(remaining, elapsed * (planned - resolved) / resolved)
        return dict(round=active_round, completed_rounds=completed, coverage=sum(x > 0 for x in counts.values()),
                    marginals=draws, sampled_draws=draws, observed_marginals=draws - skipped,
                    proposed_count=len(witnesses), skipped_after=skipped, resolved_trials=resolved,
                    planned_trials=planned, sample_elapsed_s=elapsed, sampling_eta_s=0.0 if remaining == 0 else eta,
                    sampling_deadline_utc=deadline_utc)

    def check_time():
        if clock() >= deadline:
            raise SamplingDeadline

    stop = "round_limit"
    try:
        for active_round in range(1, rounds + 1):
            for i, before, after in sample_random_round(n, rng):
                feature = features[i]
                if feature in witnesses:
                    continue
                check_time()
                publish(**snapshot(), active_feature=feature, pair_state="before", partial_pair=None)
                u0 = evaluate(before)
                u1 = None

                def after_wins():
                    nonlocal u1, partial
                    partial = dict(feature=feature, before_mask=hex(before), after_mask=hex(after), before_utility=u0)
                    check_time()
                    publish(**snapshot(), active_feature=feature, pair_state="after", partial_pair=partial)
                    u1 = evaluate(after)
                    partial = None
                    return u1 >= threshold

                prior = 1 if minima[feature] is None else minima[feature]
                value = update_sample_minimum(prior, bool(u0 >= threshold), after_wins)
                if u1 is None:
                    skipped += 1
                if value == -1:
                    assert u1 is not None and u0 >= threshold > u1
                    witnesses[feature] = dict(round=active_round, before_mask=hex(before), after_mask=hex(after),
                                              before_utility=u0, after_utility=u1)
                minima[feature] = value
                counts[feature] += 1
                publish(**snapshot(), active_feature=feature, pair_state="complete", partial_pair=None,
                        sample_minima=minima, draws_per_feature=counts, witnesses=witnesses)
            completed = active_round
            if len(witnesses) == n:
                stop = "all_witnessed"
                break
    except SamplingDeadline:
        stop = "sample_time_limit"
    return dict(**snapshot(), stop=stop, sample_minima=minima, draws_per_feature=counts,
                witnesses=witnesses, partial_pair=partial)


def run(dataset, output, rounds, seconds):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"A completed result already exists in {output}")
    start = time.monotonic()
    state = dict(dataset=dataset, seed=SEED, phase="loading", round=0, n_features=0, coverage=0,
                 marginals=0, sampled_draws=0, observed_marginals=0, sampling_eta_s=None, sampling_deadline_utc=None,
                 proposed_count=0, skipped_after=0, resolved_trials=0, planned_trials=0,
                 validation_fits=0, sample_fits=0, test_fits=0, cache_hits=0, sample_elapsed_s=0.0)

    def publish(**changes):
        state.update(changes, elapsed_s=time.monotonic() - start, updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(output / "progress.json", state)

    try:
        publish()
        with (SOURCE / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
            row, = (r for r in csv.DictReader(file) if r["dataset_id"] == dataset and int(r["seed"]) == SEED)
        for part in ("train", "val", "test"):
            row[f"{part}_row_count"] = int(row[f"{part}_row_count"])
        train, yt, val, yv = load_prepared_train_validation(SOURCE, row)
        protocol = yaml.safe_load((SOURCE / "resolved_protocol.yaml").read_text(encoding="utf-8"))
        policy = dict(task=row["task"], utility=row["utility"], model_params=protocol["model_params_by_dataset"][dataset],
                      backend=protocol["backend"], seed=SEED)
        features = tuple(train.columns)
        n = len(features)
        full = (1 << n) - 1
        cache = {}

        def utility(mask, *, testing=False):
            if mask in cache:
                state["cache_hits"] += 1
                return cache[mask]
            selected = tuple(f for i, f in enumerate(features) if mask & (1 << i))
            publish(fit_started_utc=datetime.now(timezone.utc).isoformat(), fit_mask=hex(mask), fit_is_model=bool(mask))
            frames = (train_final, y_final, test, ye) if testing else (train, yt, val, yv)
            value = subset_utility(*frames, selected, **policy)
            if mask:
                state["test_fits" if testing else "validation_fits"] += 1
                if state["phase"] == "sampling":
                    state["sample_fits"] += 1
            state["fit_started_utc"] = None
            cache[mask] = value
            return value

        publish(phase="full_validation", n_features=n, planned_trials=n * rounds)
        full_val = utility(full)
        threshold = protocol["alpha_default"] * full_val
        publish(phase="sampling", full_validation_utility=full_val, threshold=threshold)
        sampled = sample(features, rounds, seconds, threshold, utility, publish)
        proposed = [f for f in features if f in sampled["witnesses"]]
        raw_mask = sum(1 << i for i, f in enumerate(features) if f not in sampled["witnesses"])
        publish(phase="validation_gate", **sampled, sampling_stop=sampled["stop"])
        fits_pre_gate = state["validation_fits"]
        raw_val = utility(raw_mask)
        gate = _removal_safety_gate(task=policy["task"], full_validation_utility=full_val,
                                    raw_prefilter_validation_utility=raw_val, raw_harmful_features=proposed,
                                    classification_margin=protocol["removal"]["classification_noninferiority_margin"],
                                    regression_margin=protocol["removal"]["regression_noninferiority_margin"],
                                    validation_model_fit_count=state["validation_fits"] - fits_pre_gate)
        retained = raw_mask if gate["accepted"] else full
        removed = gate["safe_harmful_features"]
        retained_val = raw_val if gate["accepted"] else full_val
        # Freeze the validation decision in progress before opening the held-out split.
        publish(phase="test_full", gate_passed=gate["accepted"], removed_count=len(removed),
                proposed_features=proposed, removed_features=removed, retained_validation_utility=retained_val)
        paths = PreparedSplitPaths(dataset, SEED, row["target"],
                                   *(SOURCE / row[f"{p}_relpath"] for p in ("train", "val", "test")))
        params = dict(rounds=rounds, sample_seconds=seconds, retained_mask=hex(retained))
        job_id = canonical_hash(dict(dataset=dataset, seed=SEED, protocol=row["protocol_hash"], parameters=params))
        job = JobSpec(job_id, row["protocol_hash"], "removal", dataset, SEED, "balanced_minpi",
                      params, (), "result.json")
        audit = AuditLog.from_manifest(output / "test_access.json", [job])
        test, ye = AuditedSplitAccess(paths, stage="removal", job_id=job_id, audit=audit).load_test(
            reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
        if tuple(test.columns) != features or len(ye) != row["test_row_count"]:
            raise ValueError("Test partition shape does not match frozen manifest")
        train_final = pd.concat([train, val], ignore_index=True)
        y_final = np.concatenate([yt, yv])
        cache = {}
        full_test = utility(full, testing=True)
        publish(phase="test_retained", full_test_utility=full_test)
        retained_test = utility(retained, testing=True)
        result = dict(state)
        result.update(sampled, phase="complete", elapsed_s=time.monotonic() - start, n_features=n,
                      requested_rounds=rounds, sample_seconds=seconds, features=features, policy=policy,
                      alpha=protocol["alpha_default"], threshold=threshold, gate=gate, proposed_validation_utility=raw_val,
                      validation_delta=retained_val - full_val, retained_test_utility=retained_test,
                      test_delta=retained_test - full_test, retained_count=retained.bit_count(),
                      test_training_partitions=["train", "val"], source_run_id=SOURCE.name,
                      partition_sha256={p: row[f"{p}_file_sha256"] for p in ("train", "val", "test")},
                      test_fits=state["test_fits"], cache_hits=state["cache_hits"], fit_started_utc=None)
        atomic_write_json(output / "result.json", result)
        publish(phase="complete", retained_test_utility=retained_test, test_delta=result["test_delta"])
    except Exception as error:
        publish(phase="error", failed_phase=state["phase"], error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sample-seconds", required=True, type=float)
    args = parser.parse_args()
    if args.rounds <= 0 or not np.isfinite(args.sample_seconds) or args.sample_seconds <= 0:
        parser.error("rounds and sample-seconds must be positive and finite")
    run(args.dataset, args.output_dir, args.rounds, args.sample_seconds)
