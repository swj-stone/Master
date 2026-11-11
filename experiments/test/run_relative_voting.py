"""Run the frozen 96-path relative-vote experiment and matched Gain control."""

import argparse
import csv
import hashlib
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from xgboost import DMatrix

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.thesis_prepared_catalog import FORMAL_SEEDS, audit_prepared_catalog
from experiments.thesis_sampling_worker import prediction_scores, subset_validation_scores, validation_halves
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.utility import compute_utility


PATHS = 96
MARGIN = .005
DATASETS = ("madelon", "spambase")
METHODS = ("relative_vote", "gain")
LIMIT_S = 900
PARTITIONS = ("train", "val", "test")


def relative_votes(phi, labels, seed):
    """Aggregate local prediction-rank changes relative to other additions at the same position."""
    n, d = phi.shape
    positive = np.asarray(labels, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = n - n_positive
    if len(positive) != n or min(n_positive, n_negative) == 0 or d < 3:
        raise ValueError("Relative AUC voting needs both outcome classes and at least three features.")
    rng = np.random.default_rng(seed)
    orders = np.array([rng.permutation(d) for _ in range(PATHS)])
    deltas = np.empty((PATHS, d))
    for k, order in enumerate(orders):
        margins = np.cumsum(phi[:, order], axis=1)
        ranks = rankdata(margins, method="average", axis=0)
        aucs = (ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
        deltas[k] = np.diff(np.insert(aucs, 0, .5))

    reference = deltas.mean(axis=0)
    excess = deltas - reference
    votes = (excess > MARGIN).astype(int) - (excess < -MARGIN).astype(int)
    counts = np.zeros((3, d), dtype=int)
    totals = np.zeros_like(counts)
    for tier, positions in enumerate((slice(0, d//3), slice(d//3, 2*d//3), slice(2*d//3, d))):
        np.add.at(counts[tier], orders[:, positions], 1)
        np.add.at(totals[tier], orders[:, positions], votes[:, positions])
    if np.any(counts == 0):
        raise ValueError("A feature has no observation in a coalition-size tier.")
    means = totals / counts
    signs = np.select([np.any(3*totals < -counts, axis=0), np.all(3*totals > counts, axis=0)], [-1, 1], 0)
    return dict(orders=orders, deltas_by_position=deltas, baseline_by_position=reference, tier_counts=counts,
                tier_means=means, minima=means.min(axis=0), mean_across_tiers=means.mean(axis=0), signs=signs)


def _vote_order(booster, validation, labels, seed, features, output):
    matrix = DMatrix(validation, nthread=1)
    raw = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
    if raw.shape != (len(validation), 1, len(features)+1):
        raise ValueError("Unexpected shape for binary TreeSHAP contributions.")
    parts = raw[:, 0, :].astype(float)
    margin = booster.predict(matrix, output_margin=True).reshape(-1)
    error = float(np.max(np.abs(parts.sum(axis=1) - margin)))
    if np.ptp(parts[:, -1]) >= 1e-5 or error >= 1e-4:
        raise ValueError("TreeSHAP contributions do not reproduce the full model margins.")
    phi = parts[:, :-1]
    game = relative_votes(phi, labels, seed)
    repeat_started = time.monotonic()
    repeat = relative_votes(phi, labels, seed+100003)
    repeat_s = time.monotonic()-repeat_started
    signs = game["signs"]
    classes = (-1, 0, 1)
    counts = [int(np.sum(signs == c)) for c in classes]
    switch = [[int(np.sum((signs==a) & (repeat["signs"]==b))) for b in classes] for a in classes]
    np.savez_compressed(output/"game_evidence.npz", columns=features, y=labels, phi=phi, **game,
                        repeat_orders=repeat["orders"], repeat_tier_means=repeat["tier_means"],
                        repeat_minima=repeat["minima"], repeat_signs=repeat["signs"])

    def priority(i):
        return game["minima"][i], game["mean_across_tiers"][i], i

    order = [i for c in (-1, 0) for i in sorted(np.flatnonzero(signs==c).tolist(), key=priority)]
    stats = dict(negative=counts[0], neutral=counts[1], positive=counts[2],
                 discrete_seed_agreement=float(np.mean(signs==repeat["signs"])),
                 class_switch_matrix=switch, additivity_error=error, repeat_diagnostic_s=repeat_s)
    return order, stats


def _choose_subset(train, yt, val, yv, halves, policy, base, order, sizes):
    features = list(train.columns)
    d = len(features)
    full = (1<<d)-1
    trials = {}
    for size in sizes:
        removed = order[:min(size, d-1)]
        mask = full ^ sum(1<<i for i in removed)
        if mask in trials:
            continue
        columns = [f for i, f in enumerate(features) if mask & (1<<i)]
        scores = (base if mask == full else
                  subset_validation_scores(train, yt, val, yv, columns, halves, utility="roc_auc", **policy))
        changes = [a-b for a, b in zip(scores, base, strict=True)]
        worst = min(changes[1:])
        trials[mask] = dict(mask=hex(mask), retained=len(columns), scores=list(scores), deltas=changes,
                            passes=min(changes)>=-MARGIN-1e-12, worst_slack=worst+MARGIN,
                            score=len(removed)/d-max(0., -worst)/MARGIN)
    eligible = (m for m, trial in trials.items() if trial["passes"])
    chosen = max(eligible, key=lambda m: (trials[m]["score"], trials[m]["worst_slack"]))
    return chosen, trials


def run_arm(dataset, seed, method, source, output):
    start = time.monotonic()

    def progress(phase):
        atomic_write_json(output/"progress.json", dict(phase=phase, elapsed_s=time.monotonic()-start))

    progress("loading")
    with (source/"data_manifest.csv").open(encoding="utf-8-sig") as stream:
        row, = (r for r in csv.DictReader(stream) if r["dataset_id"] == dataset and int(r["seed"]) == seed)
    for part in PARTITIONS:
        row[part+"_row_count"] = int(row[part+"_row_count"])
    protocol = yaml.safe_load((source/"resolved_protocol.yaml").read_bytes())
    train, yt, val, yv = load_prepared_train_validation(source, row)
    features = list(train.columns)
    d = len(features)
    full = (1<<d)-1
    policy = dict(task=row["task"], seed=seed, model_params=protocol["model_params_by_dataset"][dataset],
                  backend=protocol["backend"])
    if (canonical_hash(protocol) != row["protocol_hash"] or row["utility"] != "roc_auc"
            or policy["task"] != "binary" or policy["backend"] != "cuda"
            or policy["model_params"]["n_estimators"] != 100
            or protocol["removal"]["classification_noninferiority_margin"] != MARGIN):
        raise ValueError("The archived protocol does not match the fixed binary AUC comparison.")
    halves, split = validation_halves(yv, policy["task"], seed=seed)
    safe = [f"f{i}" for i in range(d)]
    valid = val.set_axis(safe, axis=1)
    progress("full_model")
    model = make_model(**policy)
    model.fit(train.set_axis(safe, axis=1), yt)
    base = prediction_scores(yv, model.predict_proba(valid), halves, "roc_auc")
    sizes = [0, 1, (d+3)//4, (d+1)//2, (3*d+3)//4, (7*d)//8]
    start_rank = time.monotonic()
    progress("ranking")
    if method == "relative_vote":
        order, diagnostics = _vote_order(model.get_booster(), valid, yv, seed, features, output)
        sizes.append(diagnostics["negative"])
    elif method == "gain":
        gain = model.get_booster().get_score(importance_type="gain")
        order = sorted(range(d), key=lambda i: gain.get(safe[i], 0.))
        diagnostics = {}
    else:
        raise ValueError("Unknown ranking method.")
    game_s = time.monotonic()-start_rank-diagnostics.get("repeat_diagnostic_s", 0.)
    progress("subset_validation")
    chosen, trials = _choose_subset(train, yt, val, yv, halves, policy, base, order, sizes)
    names = [name for i, name in enumerate(features) if chosen & (1<<i)]
    frozen = dict(dataset=dataset, seed=seed, arm=method, feature_order=features, selected_mask=hex(chosen),
                  selected_features=names, selected_val=trials[chosen]["scores"], full_val=base,
                  proposals=list(trials.values()), rank_order=[features[i] for i in order], evidence=diagnostics,
                  game_s=game_s, validation_fits=len(trials), policy=policy, margin=MARGIN, halves=split,
                  partitions={p:row[p+"_file_sha256"] for p in PARTITIONS}, protocol_hash=row["protocol_hash"])
    atomic_write_json(output/"validation.json", frozen)
    identity = canonical_hash(frozen)
    progress("test_access")
    job = JobSpec(identity, row["protocol_hash"], "removal", dataset, seed, method,
                  dict(mask=hex(chosen)), (), "result.json")
    audit = AuditLog.from_manifest(output/"test_access.json", [job])
    paths = PreparedSplitPaths(dataset, seed, row["target"], *(source/row[p+"_relpath"] for p in PARTITIONS))
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=audit).load_test(
        reason="final_removal_evaluation", expected_sha256=row["test_file_sha256"])
    if list(test.columns) != features or len(ye) != row["test_row_count"]:
        raise ValueError("The held-out partition does not match the frozen selection.")
    progress("final_refits")
    both = pd.concat([train, val], ignore_index=True)
    labels = np.concatenate([yt, yv])
    preds = {}
    for mask in dict.fromkeys([full, chosen]):
        columns = [f for i, f in enumerate(features) if mask & (1<<i)]
        preds[mask] = fit_predict(both.loc[:, columns], labels, test.loc[:, columns], **policy)
    np.savez_compressed(output/"paired_test_predictions.npz", y=ye, full=preds[full], selected=preds[chosen])
    full_auc = float(compute_utility(ye, preds[full], metric="roc_auc"))
    auc = float(compute_utility(ye, preds[chosen], metric="roc_auc"))
    result = dict(dataset=dataset, seed=seed, arm=method, retained=len(names), full_test_auc=full_auc,
                  selected_test_auc=auc, delta=auc-full_auc, selected_validation_auc=trials[chosen]["scores"][0],
                  full_validation_auc=base[0], game_s=game_s, validation_fits=len(trials), total_fits=len(trials)+len(preds),
                  val_sha256=identity, worker_s=time.monotonic()-start, **diagnostics)
    atomic_write_json(output/"result.json", result)
    progress("complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=["madelon"])
    parser.add_argument("--seeds", choices=FORMAL_SEEDS, type=int, nargs="+", default=[42])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    args = parser.parse_args(argv)
    for values in (args.datasets, args.seeds, args.methods):
        if len(values) != len(set(values)):
            parser.error("Specify each dataset, seed, and method at most once.")
    jobs, audit = audit_prepared_catalog()
    root = ROOT/"results/test/thesis_sampling"/datetime.now(MSK).strftime("relative-vote-%Y%m%d-%H%M%S-%f")
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root/"protocol.json", dict(datasets=args.datasets, seeds=args.seeds, methods=args.methods,
                      permutations=PATHS, min_vote_threshold=1/3, auc_margin=MARGIN, limit_s=LIMIT_S,
                      prepared_catalog_sha256=canonical_hash(audit),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(f"Output: {root}", flush=True)
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for method in args.methods:
                out = root/f"{dataset}-s{seed}-{method}"
                out.mkdir()
                process = multiprocessing.get_context("spawn").Process(
                    target=run_arm, args=(dataset, seed, method, jobs[dataset, seed], out))
                print(f"{out.name} started with a {LIMIT_S}s limit", flush=True)
                try:
                    outcome = supervise(process, out, LIMIT_S, lambda s: print(f"{out.name} {s}", flush=True),
                                        heartbeat_formatter=lambda p,t,r: f"phase={p.get('phase', 'starting')} elapsed={t:.1f}s remaining={r:.1f}s")
                finally:
                    process.close()
                result = outcome.get("result") or {}
                row = dict(dataset=dataset, seed=seed, method=method, status=outcome["status"], wall_s=outcome["elapsed_s"])
                for key in ("retained", "selected_test_auc", "full_test_auc", "delta", "validation_fits", "total_fits",
                            "negative", "neutral", "positive", "discrete_seed_agreement", "repeat_diagnostic_s", "game_s"):
                    row[key] = result.get(key)
                rows.append(row)
                write_summary(root, rows)
                print(row, flush=True)
    return int(any(row["status"] != "completed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
