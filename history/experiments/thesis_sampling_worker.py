"""Run one balanced-round MinPI experiment on a frozen prepared split."""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.icomp2026_stages import _removal_safety_gate
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json,
    canonical_hash,
)
from minppi_mofs.minimum_sampling import sample_random_round, update_sample_minimum
from minppi_mofs.modeling import _empty_subset_predictions, evaluate_fixed_subset, fit_predict, make_model
from minppi_mofs.utility import compute_utility


SOURCE = Path(__file__).resolve().parents[1] / "results/icomp2026_formal/icomp2026-formal-v1-s42-123"
DATASETS = ("abalone", "madelon", "spambase", "gisette", "isolet", "credit_card_fraud")
SEED = 42
SELECTION_POLICIES = ("structured", "wide", "witness_only", "sign_aware", "confirmed", "guarded", "local_refine", "fold_consensus", "guarded_rounds")


class SamplingDeadline(Exception):
    pass


def subset_utility(train, yt, eval_frame, ye, selected, *, capture_prediction=None, **policy):
    if not selected:
        return evaluate_fixed_subset(train, yt, eval_frame, ye, selected, **policy)
    # The frozen loader already validated all columns. Each fit only needs the selected ones.
    columns = list(selected)
    prediction = fit_predict(train.loc[:, columns], yt, eval_frame.loc[:, columns], task=policy["task"],
                             model_params=policy["model_params"], backend=policy["backend"], seed=policy["seed"])
    if capture_prediction is not None:
        capture_prediction(prediction)
    return float(compute_utility(ye, prediction, metric=policy["utility"]))


def validation_halves(y, task, *, seed=SEED):
    indices = np.arange(len(y))
    halves = train_test_split(indices, test_size=0.5, random_state=seed,
                              stratify=None if task == "regression" else y)
    halves = tuple(np.sort(half) for half in halves)
    metadata = dict(seed=seed, index_space="zero-based positions in frozen validation CSV",
                    stratified=task != "regression", sizes=[len(h) for h in halves],
                    index_sha256=[hashlib.sha256(h.astype("<i8").tobytes()).hexdigest() for h in halves])
    return halves, metadata


def prediction_scores(y, prediction, halves, metric):
    return (float(compute_utility(y, prediction, metric=metric)),
            *(float(compute_utility(y[h], prediction[h], metric=metric)) for h in halves))


def full_validation_and_gain(train, yt, val, yv, halves, *, include_shap, **policy):
    # Match fit_predict's canonical XGBoost column names. Gain uses only the training fit.
    safe_names = [f"f{i}" for i in range(train.shape[1])]
    model = make_model(**{k: policy[k] for k in ("task", "model_params", "backend", "seed")})
    safe_train = train.set_axis(safe_names, axis=1)
    model.fit(safe_train, yt)
    prediction = (model.predict if policy["task"] == "regression" else model.predict_proba)(
        val.set_axis(safe_names, axis=1))
    scores = prediction_scores(yv, np.asarray(prediction, dtype=float), halves, policy["utility"])
    gains = model.get_booster().get_score(importance_type="gain")
    gain = {feature: float(gains.get(name, 0.0)) for feature, name in zip(train.columns, safe_names)}
    if not include_shap:
        return scores, gain, None
    from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance

    shap = fitted_tree_shap_importance(model, safe_train, seed=policy["seed"], max_rows=1024)
    return scores, gain, {feature: float(v) for feature, v in zip(train.columns, shap, strict=True)}


def subset_validation_scores(train, yt, val, yv, selected, halves, **policy):
    if selected:
        cols = list(selected)
        prediction = fit_predict(train.loc[:, cols], yt, val.loc[:, cols], task=policy["task"],
                                 model_params=policy["model_params"], backend=policy["backend"], seed=policy["seed"])
    else:
        prediction = _empty_subset_predictions(yt, task=policy["task"], row_count=len(val))
    return prediction_scores(yv, prediction, halves, policy["utility"])


def fit_rounds_for_masks(train, labels, evaluation, masks, features, policy, phase, state, publish):
    """Fit each distinct column set once, including the case where guarded retains all columns."""
    from experiments.thesis_fixed_predictor import rounds_predictions

    predictions = {}
    unique = list(dict.fromkeys(masks.values()))
    counter = "validation_fits" if phase == "predictor_validation" else "test_fits"
    for done, mask in enumerate(unique):
        roles = "+".join(role for role, value in masks.items() if value == mask)
        publish(phase=phase, predictor_role=roles, predictor_mask=hex(mask), predictor_fits_done=done,
                predictor_fits_planned=len(unique), fit_started_utc=datetime.now(timezone.utc).isoformat())
        cols = [f for i, f in enumerate(features) if mask & (1 << i)]
        predictions[mask] = rounds_predictions(
            train.loc[:, cols], labels, evaluation.loc[:, cols], task=policy["task"],
            model_params=policy["model_params"], backend=policy["backend"], seed=policy["seed"])
        publish(**{counter: state[counter] + 1}, predictor_fits_done=done + 1, fit_started_utc=None)
    return predictions


def freeze_rounds(masks, validation_predictions, labels, halves, metric, original_scores):
    from experiments.thesis_fixed_predictor import adopt_blend

    decisions, methods = {}, {}
    for role, mask in masks.items():
        scores = {recipe: prediction_scores(labels, prediction, halves, metric)
                  for recipe, prediction in validation_predictions[mask].items()}
        if not np.allclose(scores["original"], original_scores[mask], atol=1e-10, rtol=0):
            raise ValueError(f"{role} B-round validation scores do not reproduce the feature selection baseline")
        use_blend = adopt_blend(scores["original"], scores["blend"])
        decisions[role] = dict(mask=hex(mask), original_scores=scores["original"], blend_scores=scores["blend"],
                               validation_deltas=np.subtract(scores["blend"], scores["original"]).tolist(),
                               adopt_blend=use_blend)
        for variant in ("original", "blend", "chosen"):
            recipe = ("blend" if use_blend else "original") if variant == "chosen" else variant
            methods[f"{role}_{variant}"] = dict(mask=mask, recipe=recipe, scores=scores[recipe])
    return decisions, methods


def confirm_on_training(train, yt, features, structured, signed, full_scores, margin, publish, **policy):
    """Verify two frozen candidate masks on a deterministic split of the training partition."""
    from minppi_mofs.removal_confirmation import select_confirmed_removal

    candidates = {"structured": structured, "sign_worst": signed}
    masks = {name: result["selected_mask"] for name, result in candidates.items()}
    official_scores = {result["selected_mask"]: result["selected_scores"] for result in candidates.values()}
    official_scores[(1 << len(features)) - 1] = full_scores
    stratified = policy["task"] != "regression"
    fit_rows, check_rows = (np.sort(rows) for rows in train_test_split(
        np.arange(len(yt)), test_size=0.25, random_state=SEED, stratify=yt if stratified else None))
    split = dict(seed=SEED, index_space="zero-based positions in frozen training CSV", stratified=stratified,
                 sizes=[len(fit_rows), len(check_rows)],
                 index_sha256=[hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() for rows in (fit_rows, check_rows)])
    training, checking = train.iloc[fit_rows], train.iloc[check_rows]
    labels = np.asarray(yt)
    training_y, checking_y = labels[fit_rows], labels[check_rows]
    completed = {}
    publish(phase="training_confirmation", confirmation_candidate_masks=masks, confirmation_split=split, confirmation_fits=0)

    def evaluate_confirmation(mask):
        selected = tuple(f for i, f in enumerate(features) if mask & (1 << i))
        publish(fit_started_utc=datetime.now(timezone.utc).isoformat(), fit_mask=hex(mask), fit_is_model=True)
        score = subset_utility(training, training_y, checking, checking_y, selected, **policy)
        completed[hex(mask)] = score
        publish(confirmation_fits=len(completed), confirmation_scores=completed, fit_started_utc=None)
        return score

    decision = select_confirmed_removal(features, masks, official_scores, margin, evaluate_confirmation)
    decision.update(candidate_masks=masks, split=split)
    publish(confirmation=decision)
    return decision


def refinement_budget(gate_fit_seconds, max_test_fits, max_candidates):
    """Reserve the held-out refits using measured official validation-fit durations."""
    deadline_text = os.environ.get("MINPPI_DATASET_DEADLINE")
    deadline = float(deadline_text) if deadline_text is not None else None
    fit_s = max(1.0, float(np.mean(gate_fit_seconds)) if gate_fit_seconds else 1.0)
    reserve_s = 5.0 + max_test_fits * 1.25 * fit_s
    def allow_next():
        return deadline is None or deadline - time.monotonic() >= reserve_s + fit_s

    return allow_next, dict(has_dataset_deadline=deadline is not None, estimated_fit_s=fit_s,
                            held_out_reserve_s=reserve_s, max_local_masks=max_candidates)


def refine_guarded(train, yt, minima, gain, shap, incumbent, evaluate, full_scores, margin, publish,
                   completed_trial, gate_fit_seconds, method, **policy):
    """Build training-only rankings, then select changes on the frozen validation split."""
    features = tuple(train.columns)
    mask = incumbent["selected_mask"]
    scores = incumbent["selected_scores"]
    base = incumbent["methods"]["guarded"]
    if method == "fold_consensus":
        from minppi_mofs.fold_consensus import select_fold_consensus

        allow_next, budget = refinement_budget(gate_fit_seconds, max_test_fits=3, max_candidates=5)
        publish(phase="consensus_ranking", local_incumbent_mask=mask, local_incumbent_scores=scores,
                local_budget=budget, extra_ranking_fits=0)
        def ranking_progress(completed, planned):
            phase = "consensus_validation" if completed == planned else "consensus_ranking"
            publish(phase=phase, extra_ranking_fits=completed, planned_ranking_fits=planned)

        consensus = select_fold_consensus(train, yt, minima, mask, evaluate, scores, full_scores, margin,
                                          policy=policy, publish=lambda trial: completed_trial(trial, method), allow_next=allow_next,
                                          publish_ranking=ranking_progress)
        publish(fold_consensus=consensus, extra_ranking_fits=consensus["ranking_fits"])
        candidates = set(incumbent["candidate_features"]) | set(consensus["candidate_features"])
        candidates = [f for f in features if f in candidates]
        return dict(consensus, candidate_features=candidates, all_candidate_features=candidates,
                    methods={**consensus["methods"], "guarded_incumbent": base}, trials=incumbent["trials"])

    from minppi_mofs.local_refinement import select_local_refinement
    from minppi_mofs.residual_ranking import rank_residual_recovery

    allow_next, budget = refinement_budget(gate_fit_seconds, max_test_fits=6, max_candidates=8)
    publish(phase="residual_ranking", local_incumbent_mask=mask, local_incumbent_scores=scores, local_budget=budget)
    skip_ranking = not allow_next()
    if not skip_ranking:
        publish(fit_started_utc=datetime.now(timezone.utc).isoformat(), fit_mask=hex(mask), fit_is_model=True)
        recovery, retained_gain, metadata = rank_residual_recovery(train, yt, features, mask, minima, **policy)
    else:
        recovery, retained_gain = None, None
        metadata = {"fit_count": 0, "stage": "skipped_before_ranking", "split": None}
    publish(phase="local_refinement", fit_started_utc=None, extra_ranking_fits=metadata["fit_count"],
            residual_ranking=dict(metadata=metadata, recovery_scores=recovery, retained_gain=retained_gain))
    previous = tuple(dict(trial, official_policies=list(trial["official_policies"])) for trial in incumbent["trials"])
    local = select_local_refinement(features, minima, mask, gain, shap, recovery, retained_gain, evaluate,
                                    scores, full_scores, margin,
                                    publish=lambda trial: completed_trial(trial, "local_refine"), allow_next=allow_next,
                                    previous_trials=previous, skip_ranking=skip_ranking)
    publish(local_refinement=local)
    candidates = set(incumbent["candidate_features"]) | set(local["candidate_features"])
    candidates = [f for f in features if f in candidates]
    return dict(local, candidate_features=candidates, all_candidate_features=candidates,
                methods={**local["methods"], "guarded_incumbent": base}, trials=incumbent["trials"])


def load_sampling(path, dataset, features, row, policy, alpha, full_val, rounds, seconds, evaluate):
    """Read only sampling/provenance fields. Replay schedule structure and verify witnesses on validation."""
    path = Path(path)
    raw = path.read_bytes()
    old = json.loads(raw)

    def require(condition, field):
        if not condition:
            raise ValueError(f"Reusable sampling evidence conflicts with {field}")

    # A supervised source retains an independent copy at process exit. Compare
    # only sampling evidence, including neutral minima which carry no -1 witness.
    watchdog = path.with_name("watchdog.json")
    if path.name == "result.json" and watchdog.is_file():
        original = json.loads(watchdog.read_bytes())
        require(original["status"] == "completed", "source watchdog status")
        for field in ("draws_per_feature", "sample_minima", "witnesses", "partial_pair", "skipped_after",
                      "sample_fits", "sample_elapsed_s", "round", "completed_rounds", "stop"):
            require(old[field] == original["result"][field], f"watchdog sampling evidence: {field}")

    expected = dict(dataset=dataset, seed=SEED, features=list(features), requested_rounds=rounds,
                    sample_seconds=seconds, policy=policy, alpha=alpha, source_run_id=SOURCE.name, phase="complete")
    for field, value in expected.items():
        require(old[field] == value, field)
    for part in ("train", "val"):
        require(old["partition_sha256"][part] == row[f"{part}_file_sha256"], f"{part} hash")
    threshold = alpha * full_val
    for field, value in (("full_validation_utility", full_val), ("threshold", threshold)):
        require(math.isclose(old[field], value, abs_tol=1e-10, rel_tol=0), field)
    minima, counts, witnesses = (old[k] for k in ("sample_minima", "draws_per_feature", "witnesses"))
    n = len(features)
    require(set(minima) == set(counts) == set(features), "feature sequence")
    require(set(witnesses) == {f for f in features if minima[f] == -1}, "witness labels")
    for f in features:
        require(type(counts[f]) is int and 0 <= counts[f] <= rounds, "draw counts")
        require((minima[f] is None and counts[f] == 0) or
                (counts[f] > 0 and type(minima[f]) is int and minima[f] in (-1, 0, 1)), "minimum")
    draws = sum(counts.values())
    skipped = old["skipped_after"]
    require(type(skipped) is int and 0 <= skipped <= draws, "skipped endpoints")
    expected_counts = dict(marginals=draws, coverage=sum(c > 0 for c in counts.values()),
                           resolved_trials=draws + sum(rounds-counts[f] for f in witnesses),
                           planned_trials=n * rounds)
    for field, value in expected_counts.items():
        require(old[field] == value, field)
    require(old.get("observed_marginals", draws-skipped) == draws-skipped and
            old.get("sampled_draws", draws) == draws, "observed counts")
    require(old["stop"] in ("round_limit", "sample_time_limit", "all_witnessed")
            and 0 <= old["completed_rounds"] <= old["round"] <= rounds, "stop")
    if old["stop"] == "round_limit":
        require(old["completed_rounds"] == rounds and old["resolved_trials"] == n * rounds, "completed rounds")
    if old["stop"] == "all_witnessed":
        require(len(witnesses) == n, "all witnessed")
    elapsed = old["sample_elapsed_s"]
    require(math.isfinite(elapsed) and elapsed >= 0, "sampling time")
    if old["stop"] == "sample_time_limit":
        require(elapsed >= seconds, "sampling deadline")
    source_fits = old.get("inherited_sample_fits", 0) + old["sample_fits"]
    require(type(source_fits) is int and 0 <= source_fits <= 2*draws+1, "fit count")

    rng = np.random.default_rng(SEED)
    verified = dict.fromkeys(features, 0)
    checked_witnesses = {}
    remaining = draws
    next_trial = None
    for r in range(1, rounds+1):
        for i, before, after in sample_random_round(n, rng):
            f = features[i]
            if f in checked_witnesses:
                continue
            if remaining == 0:
                next_trial = (f, before, after)
                break
            verified[f] += 1
            remaining -= 1
            if f in witnesses and r == witnesses[f]["round"]:
                witness = witnesses[f]
                require(witness["before_mask"] == hex(before) and witness["after_mask"] == hex(after), "witness schedule")
                u0, u1 = evaluate(before), evaluate(after)
                require(u0 >= threshold > u1 and math.isclose(u0, witness["before_utility"], rel_tol=0, abs_tol=1e-10)
                        and math.isclose(u1, witness["after_utility"], rel_tol=0, abs_tol=1e-10), "witness utilities")
                checked_witnesses[f] = dict(round=r, before_mask=hex(before), after_mask=hex(after),
                                            before_utility=u0, after_utility=u1)
        if next_trial is not None:
            break
    require(verified == counts and set(checked_witnesses) == set(witnesses), "draw prefix/witnesses")
    partial = None
    if old.get("partial_pair") is not None:
        supplied = old["partial_pair"]
        require(old["stop"] == "sample_time_limit" and next_trial is not None, "partial pair")
        f, before, after = next_trial
        require((supplied["feature"], supplied["before_mask"], supplied["after_mask"])
                == (f, hex(before), hex(after)), "partial pair schedule")
        u0 = evaluate(before)
        require(math.isclose(u0, supplied["before_utility"], abs_tol=1e-10, rel_tol=0), "partial pair utility")
        partial = dict(feature=f, before_mask=hex(before), after_mask=hex(after), before_utility=u0)
    sampled = dict(**expected_counts, round=old["round"], completed_rounds=old["completed_rounds"],
                   sample_minima={f: minima[f] for f in features}, draws_per_feature=verified, witnesses=checked_witnesses,
                   proposed_count=len(witnesses), sampled_draws=draws, observed_marginals=draws-skipped,
                   skipped_after=skipped, stop=old["stop"], sample_elapsed_s=elapsed, sampling_eta_s=0.0,
                   partial_pair=partial)
    return sampled, hashlib.sha256(raw).hexdigest(), source_fits


def sample(features, rounds, seconds, threshold, evaluate, publish, *, clock=time.monotonic, seed=SEED, on_prefix=None):
    """Sample one trajectory; optionally hand off independent evidence for each complete K."""
    if on_prefix is not None and seconds is not None:
        raise ValueError("Sampling prefix checkpoints require an unbounded run")
    n = len(features)
    rng = np.random.default_rng(seed)
    minima = {f: None for f in features}
    counts = dict.fromkeys(features, 0)
    witnesses = {}
    skipped = completed = 0
    active_round = 0
    partial = None
    start = clock()
    deadline = None if seconds is None else start + seconds
    deadline_utc = None if seconds is None else (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def snapshot(k=rounds, sampled_at=None):
        draws = sum(counts.values())
        planned = n * k
        resolved = draws + sum(k - counts[f] for f in witnesses)
        elapsed = (clock() if sampled_at is None else sampled_at) - start
        eta = elapsed * (planned - resolved) / resolved if resolved else None
        if seconds is not None:
            remaining = max(0.0, seconds - elapsed)
            if remaining == 0 or eta is not None:
                eta = 0.0 if remaining == 0 else min(remaining, eta)
        return dict(round=active_round, completed_rounds=completed, coverage=sum(x > 0 for x in counts.values()),
                    marginals=draws, sampled_draws=draws, observed_marginals=draws - skipped,
                    witness_count=len(witnesses), proposed_count=len(witnesses), skipped_after=skipped, resolved_trials=resolved,
                    planned_trials=planned, sample_elapsed_s=elapsed, sampling_eta_s=eta,
                    sampling_deadline_utc=deadline_utc)

    def evidence(k, stop, sampled_at=None):
        return dict(**snapshot(k, sampled_at), stop=stop, sample_minima=minima.copy(),
                    draws_per_feature=counts.copy(), witnesses={f: w.copy() for f, w in witnesses.items()},
                    partial_pair=partial.copy() if partial is not None else None)

    def check_time():
        if deadline is not None and clock() >= deadline:
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
            if on_prefix is not None:
                sampled_at = clock()
                # A -1 certificate resolves all future draws for that feature. If
                # all are certified, emit the remaining logical budgets without fitting.
                last_k = rounds if stop == "all_witnessed" else completed
                for k in range(completed, last_k + 1):
                    on_prefix(k, evidence(k, stop, sampled_at))
                start += clock() - sampled_at  # exclude downstream checkpoint processing
            if stop == "all_witnessed":
                break
    except SamplingDeadline:
        if seconds is None:
            raise
        stop = "sample_time_limit"
    return evidence(rounds, stop)


def run(dataset, output, rounds, seconds, *, selection_policy="structured", reuse_sampling=None, seed=SEED, source=None):
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError("Unknown selection policy")
    combined = selection_policy == "guarded_rounds"
    if seconds is None and not combined:
        raise ValueError("A sampling time limit is required for trial policies")
    if combined and reuse_sampling is not None:
        raise ValueError("guarded_rounds requires cold sampling on the run's own prepared partition")
    if not combined and (seed != SEED or (source is not None and Path(source).resolve() != SOURCE.resolve())):
        raise ValueError("Custom prepared sources and seeds require guarded_rounds")
    source = SOURCE if source is None else Path(source)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"A completed result already exists in {output}")
    start = time.monotonic()
    state = dict(dataset=dataset, seed=seed, selection_policy=selection_policy, phase="loading", round=0, n_features=0, coverage=0,
                 marginals=0, sampled_draws=0, observed_marginals=0, sampling_eta_s=None, sampling_deadline_utc=None,
                 proposed_count=0, skipped_after=0, resolved_trials=0, planned_trials=0,
                 validation_fits=0, sample_fits=0, test_fits=0, cache_hits=0, sample_elapsed_s=0.0,
                 witness_count=0, ranking_fits=0, extra_ranking_fits=0, sampling_reused=reuse_sampling is not None,
                 sampling_source_sha256=None, inherited_sample_fits=0, inherited_sampling_s=0.0, new_sampling_s=0.0)

    if combined:
        state["run_mode"] = "formal" if seconds is None else "test"

    def publish(**changes):
        state.update(changes, elapsed_s=time.monotonic() - start, updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(output / "progress.json", state)

    try:
        publish()
        with (source / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
            row, = (r for r in csv.DictReader(file) if r["dataset_id"] == dataset and int(r["seed"]) == seed)
        for part in ("train", "val", "test"):
            row[f"{part}_row_count"] = int(row[f"{part}_row_count"])
        train, yt, val, yv = load_prepared_train_validation(source, row)
        protocol = yaml.safe_load((source / "resolved_protocol.yaml").read_text(encoding="utf-8"))
        policy = dict(task=row["task"], utility=row["utility"], model_params=protocol["model_params_by_dataset"][dataset],
                      backend=protocol["backend"], seed=seed)
        features = tuple(train.columns)
        n = len(features)
        full = (1 << n) - 1
        cache = {}

        compare_methods = selection_policy == "wide"
        refine_incumbent = selection_policy in {"local_refine", "fold_consensus"}
        compare_incumbents = refine_incumbent or combined or selection_policy in {"confirmed", "guarded"}
        use_sign_aware = selection_policy == "sign_aware" or compare_incumbents
        use_shap = compare_methods or use_sign_aware
        use_robust_gate = selection_policy != "witness_only"
        halves, split = validation_halves(yv, row["task"], seed=seed) if use_robust_gate else (None, None)
        gate_fit_seconds = []
        held_out_predictions = {}

        def evaluate(mask, *, testing=False):
            if mask in cache:
                state["cache_hits"] += 1
                return cache[mask]
            selected = tuple(f for i, f in enumerate(features) if mask & (1 << i))
            publish(fit_started_utc=datetime.now(timezone.utc).isoformat(), fit_mask=hex(mask), fit_is_model=bool(mask))
            frames = (train_final, y_final, test, ye) if testing else (train, yt, val, yv)
            if use_robust_gate and not testing:
                gate_start = time.monotonic()
                value = subset_validation_scores(*frames, selected, halves, **policy)
                if refine_incumbent and mask and state["phase"] == "validation_gate":
                    gate_fit_seconds.append(time.monotonic() - gate_start)
            elif testing and refine_incumbent:
                def capture(prediction):
                    held_out_predictions[mask] = np.asarray(prediction)
                value = (subset_utility(*frames, selected, capture_prediction=capture, **policy),)
            else:
                value = (subset_utility(*frames, selected, **policy),)
            if mask:
                state["test_fits" if testing else "validation_fits"] += 1
                if state["phase"] == "sampling":
                    state["sample_fits"] += 1
            state["fit_started_utc"] = None
            cache[mask] = value
            return value

        publish(phase="full_validation", features=features, n_features=n, planned_trials=n * rounds)
        if use_robust_gate:
            publish(fit_started_utc=datetime.now(timezone.utc).isoformat(), fit_mask=hex(full), fit_is_model=True)
            cache[full], gain, shap = full_validation_and_gain(train, yt, val, yv, halves, include_shap=use_shap, **policy)
            state.update(fit_started_utc=None, ranking_fits=1, validation_fits=1,
                         ranking_source="same full XGBoost fit on training partition",
                         treeshap_train_rows=min(len(train), 1024) if use_shap else 0,
                         treeshap_seed=seed if use_shap else None, treeshap_max_rows=1024 if use_shap else 0)
        full_scores = evaluate(full)
        full_val = full_scores[0]
        threshold = protocol["alpha_default"] * full_val
        phase = "verify_sampling" if reuse_sampling is not None else "sampling"
        publish(phase=phase, full_validation_utility=full_val, threshold=threshold)
        def pooled(mask):
            return evaluate(mask)[0]

        if reuse_sampling is None:
            sampled = sample(features, rounds, seconds, threshold, pooled, publish, seed=seed)
            state["new_sampling_s"] = sampled["sample_elapsed_s"]
        else:
            sampled, digest, inherited_fits = load_sampling(reuse_sampling, dataset, features, row, policy,
                                                            protocol["alpha_default"], full_val, rounds, seconds, pooled)
            state.update(sampling_source_sha256=digest, sampling_source=str(Path(reuse_sampling).resolve()),
                         inherited_sampling_s=sampled["sample_elapsed_s"], inherited_sample_fits=inherited_fits)
        if seconds is None and (sampled["stop"] not in {"round_limit", "all_witnessed"}
                                or sampled["partial_pair"] is not None
                                or sampled["resolved_trials"] != n * rounds):
            raise RuntimeError("Formal sampling ended before all scheduled trials were resolved")
        witness_names = [f for f in features if f in sampled["witnesses"]]
        state["sampling_evidence_verification_fits"] = state["validation_fits"] - 1 if reuse_sampling is not None else 0
        sampled["witness_count"] = len(witness_names)
        publish(phase="validation_gate", **sampled, sampling_stop=sampled["stop"])
        classification_margin = protocol["removal"]["classification_noninferiority_margin"]
        regression_margin = protocol["removal"]["regression_noninferiority_margin"]
        margin = regression_margin if row["task"] == "regression" else classification_margin
        fits_pre_gate = state["validation_fits"]
        if use_robust_gate:
            partial_trials = []
            trials_by_mask = {}
            state.update(validation_halves=split, full_validation_scores=full_scores)
            tried_features = set()

            def completed_trial(trial, source="sign_worst"):
                if compare_incumbents:
                    mask = trial["mask"]
                    if mask not in trials_by_mask:
                        record = dict(trial, official_policies=[])
                        trials_by_mask[mask] = record
                        partial_trials.append(record)
                    trials_by_mask[mask]["official_policies"].append(source)
                else:
                    partial_trials.append(trial)
                if compare_incumbents or (use_sign_aware and trial["is_sign_candidate"]):
                    tried_features.update(trial["removed_features"])
                    state.update(proposed_count=len(tried_features),
                                 proposed_features=[f for f in features if f in tried_features])
                publish(selection_trials=partial_trials, last_selection_trial=trial)

            minima = sampled["sample_minima"]
            if use_sign_aware:
                from minppi_mofs.sign_aware_removal import select_sign_aware_removals

                sign_groups = {label: [f for f in features if minima[f] == value] for label, value in
                               (("negative", -1), ("zero", 0), ("positive", 1), ("unseen", None))}
                publish(sample_sign_groups=sign_groups, sampled_eligible_count=len(sign_groups["negative"])+len(sign_groups["zero"]),
                        proposed_count=0, proposed_features=[])
                selection = select_sign_aware_removals(features, sampled["witnesses"], minima, gain, shap, evaluate,
                                                      full_scores, margin, publish=completed_trial,
                                                      include_controls=not compare_incumbents)
                # The union of several rankings need not be a tested mask.
                diagnostic = selection["structured_pool"]
                diagnostic_features = diagnostic["removed_features"]
                raw_scores = diagnostic["scores"] if diagnostic_features else None
                gate_scope = "largest_examined_fused_prefix"
            if not use_sign_aware or compare_incumbents:
                from minppi_mofs.removal_selection import nominate_removals, select_robust_removals

                nomination = nominate_removals(train, gain, sampled["witnesses"], sample_minima=minima)
                structured_features = [record["feature"] for record in nomination]
                if not compare_incumbents:
                    publish(structured_proposed_count=len(structured_features), structured_proposed_features=structured_features)
                structured = select_robust_removals(features, nomination, gain, evaluate, full_scores, margin,
                                                    sample_minima=minima, shap_importance=shap,
                                                    structured_only=not compare_methods,
                                                    publish=lambda row: completed_trial(row, "structured"))
                if compare_incumbents:
                    if selection_policy == "confirmed":
                        decision = confirm_on_training(train, yt, features, structured, selection, full_scores, margin, publish, **policy)
                    else:
                        from minppi_mofs.guarded_update import guarded_update

                        scores_by_mask = {r["selected_mask"]: r["selected_scores"] for r in (structured, selection)}
                        decision = guarded_update(features, structured["selected_mask"], selection["selected_mask"],
                                                  scores_by_mask, full_scores, margin)
                        publish(guarded_update=decision)
                    candidate_names = set(selection["candidate_features"]) | set(structured["candidate_features"])
                    candidates = [f for f in features if f in candidate_names]
                    mask, scores = decision["selected_mask"], decision["selected_scores"]
                    record = dict(mask=mask, removed_count=decision["removed_count"], removed_features=decision["removed_features"],
                                  scores=scores, selected_source=decision["selected_source"])
                    decision_method = "guarded" if refine_incumbent or combined else selection_policy
                    selection = dict(candidate_features=candidates, all_candidate_features=candidates,
                                     selected_mask=mask, selected_scores=scores, selected_method=decision_method,
                                     removed_features=decision["removed_features"], validation_utility=scores[0],
                                     selected_source=decision["selected_source"], full_scores=full_scores,
                                     trials=partial_trials, methods={decision_method: record})
                    if refine_incumbent:
                        selection = refine_guarded(train, yt, minima, gain, shap, selection, evaluate, full_scores,
                                                  margin, publish, completed_trial, gate_fit_seconds, selection_policy, **policy)
                else:
                    selection = structured
                    diagnostic_features = structured_features
                    raw_mask = full ^ sum(1 << features.index(f) for f in diagnostic_features)
                    raw_scores = evaluate(raw_mask) if raw_mask else None
                    gate_scope = "raw_structured_pool"
            proposed = selection["candidate_features"]
            retained = selection["selected_mask"]
            removed = selection["removed_features"]
            retained_val = selection["validation_utility"]
            raw_val = raw_scores[0] if raw_scores else None
            def passes(scores):
                return all(sub >= ref-margin-1e-12 for sub, ref in zip(scores, full_scores, strict=True))
            gate = dict(accepted=passes(raw_scores) if raw_scores else False,
                        pooled_accepted=bool(raw_scores and raw_val >= full_val-margin-1e-12),
                        full_scores=full_scores, raw_pool_scores=raw_scores, noninferiority_margin=margin,
                        raw_pool_features=diagnostic_features, raw_pool_size=len(diagnostic_features), feasible=raw_scores is not None,
                        scope=gate_scope, validation_model_fit_count=state["validation_fits"]-fits_pre_gate)
            final_gate = passes(selection["selected_scores"])
            assert final_gate and set(removed).issubset(proposed)
            assert all(sampled["sample_minima"][f] in (-1, 0) for f in proposed)
            method_results = dict(selection["methods"])
            if "witness_only" in selection:
                comparison = selection["witness_only"]
                if compare_methods:
                    method_results["witness_only_three_gate"] = comparison if comparison.get("accepted") else selection["baseline"]
                selection["witness_only_pooled_accepted"] = bool(comparison.get("pooled_accepted"))
            selected_method = selection["selected_method"]
        else:
            proposed = witness_names
            state.update(structured_proposed_count=len(proposed), structured_proposed_features=proposed)
            raw_mask = full ^ sum(1 << features.index(f) for f in proposed)
            raw_val = pooled(raw_mask)
            gate = _removal_safety_gate(task=policy["task"], full_validation_utility=full_val,
                                        raw_prefilter_validation_utility=raw_val, raw_harmful_features=proposed,
                                        classification_margin=classification_margin, regression_margin=regression_margin,
                                        validation_model_fit_count=state["validation_fits"] - fits_pre_gate)
            retained = raw_mask if gate["accepted"] else full
            removed = gate["safe_harmful_features"]
            retained_val = raw_val if gate["accepted"] else full_val
            final_gate = retained_val >= full_val-margin-1e-12
            selected_method = "witness_only"
            method_results = {"witness_only": dict(mask=retained, removed_count=len(removed), scores=[retained_val])}
            selection = dict(nomination=[dict(feature=f, reason="witness") for f in proposed], trials=[gate])
        predictor_parameters = {}
        if combined:
            from experiments.thesis_fixed_predictor import late_rounds

            masks = {"full": full, "guarded": retained}
            predictions = fit_rounds_for_masks(train, yt, val, masks, features, policy, "predictor_validation", state, publish)
            decisions, method_results = freeze_rounds(masks, predictions, yv, halves, policy["utility"], cache)
            for record in method_results.values():
                record["removed_count"] = n - record["mask"].bit_count()
            del predictions
            original_rounds = policy["model_params"]["n_estimators"]
            predictor_parameters = dict(predictor_recipe="rounds", original_rounds=original_rounds,
                                        blend_checkpoints=late_rounds(original_rounds), blend_weights=[.5, .5],
                                        adopt_blend={role: d["adopt_blend"] for role, d in decisions.items()})
            selected_method = "guarded_chosen"
            publish(predictor_decisions=decisions, predictor_parameters=predictor_parameters)
        method_masks = {method: int(record["mask"] if "mask" in record else record["selected_mask"])
                        for method, record in method_results.items()}
        # Publish every frozen validation-selected mask before opening the test split.
        publish(phase="test_full", gate_passed=final_gate if use_robust_gate else gate["accepted"], raw_pool_gate_passed=gate["accepted"],
                selected_gate_passed=final_gate, removed_count=len(removed), proposed_count=len(proposed),
                proposed_features=proposed, removed_features=removed, retained_validation_utility=retained_val,
                witness_count=len(witness_names), selected_method=selected_method, method_masks=method_masks)
        paths = PreparedSplitPaths(dataset, seed, row["target"],
                                   *(source / row[f"{p}_relpath"] for p in ("train", "val", "test")))
        params = dict(rounds=rounds, sample_seconds=seconds, retained_mask=hex(retained),
                      selection_policy=selection_policy, method_masks=method_masks, **predictor_parameters)
        job_id = canonical_hash(dict(dataset=dataset, seed=seed, protocol=row["protocol_hash"], parameters=params))
        job = JobSpec(job_id, row["protocol_hash"], "removal", dataset, seed, "balanced_minpi",
                      params, (), "result.json")
        if combined:
            publish(phase="test_access", parameters=params, job_id=job_id, protocol_hash=row["protocol_hash"])
        audit = AuditLog.from_manifest(output / "test_access.json", [job])
        test, ye = AuditedSplitAccess(paths, stage="removal", job_id=job_id, audit=audit).load_test(
            reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
        if tuple(test.columns) != features or len(ye) != row["test_row_count"]:
            raise ValueError("Test partition shape does not match frozen manifest")
        train_final = pd.concat([train, val], ignore_index=True)
        y_final = np.concatenate([yt, yv])
        cache = {}
        if combined:
            predictions = fit_rounds_for_masks(train_final, y_final, test, masks, features, policy, "predictor_test", state, publish)
            archive = {name: predictions[record["mask"]][record["recipe"]] for name, record in method_results.items()}
            method_test = {name: compute_utility(ye, pred, metric=policy["utility"]) for name, pred in archive.items()}
            full_test, retained_test = method_test["full_chosen"], method_test["guarded_chosen"]
            filename = "paired_test_predictions.npz"
            np.savez_compressed(output / filename, y=ye, **archive)
            publish(full_test_utility=full_test, paired_test_predictions=filename,
                    feature_test_delta=method_test["guarded_original"] - method_test["full_original"],
                    predictor_test_deltas={role: method_test[f"{role}_chosen"] - method_test[f"{role}_original"]
                                           for role in masks},
                    overall_test_delta_vs_original_full=retained_test - method_test["full_original"],
                    paired_test_predictions_sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest())
        else:
            full_test = evaluate(full, testing=True)[0]
            publish(phase="test_retained", full_test_utility=full_test)
            retained_test = evaluate(retained, testing=True)[0]
            method_test = {method: evaluate(mask, testing=True)[0] for method, mask in method_masks.items()}
        if refine_incumbent:
            filename = "paired_test_predictions.npz"
            np.savez_compressed(output / filename, y=ye, full_set=held_out_predictions[full],
                                **{method: held_out_predictions[mask] for method, mask in method_masks.items()})
            state["paired_test_predictions"] = filename
            state["paired_test_predictions_sha256"] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        method_results = {method: dict(record, test_utility=method_test[method], test_delta=method_test[method]-full_test)
                          for method, record in method_results.items()}
        result = dict(state)
        new_wall = time.monotonic() - start
        result.update(sampled, phase="complete", elapsed_s=new_wall, new_work_wall_s=new_wall, n_features=n,
                      requested_rounds=rounds, sample_seconds=seconds, features=features, policy=policy,
                      proposed_count=len(proposed), witness_count=len(witness_names), selected_method=selected_method,
                      raw_structured_pool=gate if use_robust_gate and not use_sign_aware else None,
                      external_candidate_features=selection.get("external_candidate_features", []),
                      all_candidate_features=selection.get("all_candidate_features", proposed),
                      removed_witness_count=sum(f in sampled["witnesses"] for f in removed),
                      training_rankings=({"gain": gain, "mean_abs_treeshap": shap} if use_shap
                                         else {"gain": gain} if use_robust_gate else None),
                      validation_halves=split, selection=selection, selection_trials=selection["trials"],
                      method_results=method_results, method_test_utilities=method_test,
                      alpha=protocol["alpha_default"], threshold=threshold, gate=gate, proposed_validation_utility=raw_val,
                      validation_delta=retained_val - full_val, retained_test_utility=retained_test,
                      test_delta=retained_test - full_test, retained_count=retained.bit_count(),
                      test_training_partitions=["train", "val"], source_run_id=source.name,
                      partition_sha256={p: row[f"{p}_file_sha256"] for p in ("train", "val", "test")},
                      test_fits=state["test_fits"], cache_hits=state["cache_hits"], fit_started_utc=None)
        atomic_write_json(output / "result.json", result)
        publish(phase="complete", retained_test_utility=retained_test, test_delta=result["test_delta"])
    except Exception as error:
        publish(phase="error", failed_phase=state["phase"], error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sample-seconds", type=float, help="Sampling limit for trials; omit for a full guarded_rounds run.")
    parser.add_argument("--selection-policy", choices=SELECTION_POLICIES, default="structured")
    parser.add_argument("--reuse-sampling", type=Path)
    args = parser.parse_args()
    if args.dataset not in DATASETS and args.selection_policy != "guarded_rounds":
        parser.error("Choose a legacy dataset or use guarded_rounds with a prepared source.")
    if args.sample_seconds is None and args.selection_policy != "guarded_rounds":
        parser.error("Trial policies require --sample-seconds.")
    if args.rounds <= 0 or (args.sample_seconds is not None and
                            (not np.isfinite(args.sample_seconds) or args.sample_seconds <= 0)):
        parser.error("rounds and any supplied sample-seconds must be positive and finite")
    run(args.dataset, args.output_dir, args.rounds, args.sample_seconds,
        selection_policy=args.selection_policy, reuse_sampling=args.reuse_sampling, seed=args.seed, source=args.source)
