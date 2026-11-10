r"""Compare two finite-budget MinPI samplers on the frozen Madelon seed-42 game.

From the thesis worktree: .\thesis-python.ps1 experiments/thesis_sampling_pilot.py
"""

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from minppi_mofs.minimum_sampling import sample_random_round, update_sample_minimum
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/icomp2026_formal/icomp2026-formal-v1-s42-123"
DATASET, SEED = "madelon", 42
FIT_BUDGET, ARM_SECONDS = 384, 240
MOSCOW = timezone(timedelta(hours=3))


def sample_round(method, n, rng):
    """Yield (feature index, before mask, after mask) for a complete round."""
    if method == "independent_round_robin":
        yield from sample_random_round(n, rng)
        return
    if method != "permutation_chain":
        raise ValueError(f"Unknown pilot sampler {method}")
    prefix = 0
    for raw_feature in rng.permutation(n):
        i = int(raw_feature)
        before = prefix
        prefix |= 1 << i
        yield i, before, prefix


def run_arm(method, features, utility, anchors, threshold, log, arm_number, start, *, eager=False):
    rng = np.random.default_rng(SEED)
    cached = dict(anchors)
    scores = [None] * len(features)
    counts = [0] * len(features)
    witnesses = {}
    rounds = []
    arm_start = time.perf_counter()
    last_report = arm_start
    oracle_seconds = 0.0
    new_fits = 0
    cache_hits = 0
    skipped_after = 0

    class BudgetReached(Exception):
        pass

    def evaluate(mask):
        nonlocal new_fits, oracle_seconds, cache_hits
        if time.perf_counter() - arm_start >= ARM_SECONDS:
            raise BudgetReached("time limit")
        if mask in cached:
            cache_hits += 1
            return cached[mask]
        if new_fits >= FIT_BUDGET:
            raise BudgetReached("fit limit")
        tick = time.perf_counter()
        cached[mask] = utility(mask)
        oracle_seconds += time.perf_counter() - tick
        new_fits += 1
        return cached[mask]

    log(f"arm={arm_number}/2 {method} | fit budget={FIT_BUDGET} | ETA calibrating")
    try:
        while len(witnesses) < len(features):
            for i, before, after in sample_round(method, len(features), rng):
                if features[i] in witnesses:
                    continue  # Still present in other features' sampled contexts.
                u0 = evaluate(before)
                u1 = None

                def after_wins():
                    nonlocal u1
                    u1 = evaluate(after)
                    return u1 >= threshold

                current = 1 if scores[i] is None else scores[i]
                if eager:
                    scores[i] = min(current, int(after_wins()) - int(u0 >= threshold))
                else:
                    scores[i] = update_sample_minimum(current, u0 >= threshold, after_wins)
                skipped_after += int(u1 is None)
                counts[i] += 1
                if scores[i] == -1:
                    assert u1 is not None
                    witnesses[features[i]] = {
                        "before_mask": hex(before), "after_mask": hex(after),
                        "before_auc": u0, "after_auc": u1,
                    }
                now = time.perf_counter()
                if now - last_report >= 10 and new_fits:
                    elapsed = now - arm_start
                    seconds_per_fit = elapsed / new_fits
                    arm_eta = max(0, min(ARM_SECONDS - elapsed,
                                        (FIT_BUDGET - new_fits) * seconds_per_fit))
                    # Both arms have the same fit budget and learner. The
                    # second arm's projection is replaced by its own rate on entry.
                    eta = arm_eta + (2 - arm_number) * min(ARM_SECONDS, FIT_BUDGET * seconds_per_fit)
                    finish = datetime.now(MOSCOW) + timedelta(seconds=eta)
                    log(f"{method} fits={new_fits}/{FIT_BUDGET} | coverage={sum(c > 0 for c in counts)}/{len(counts)}"
                        f" | marginals={sum(counts)-skipped_after} | skipped={skipped_after} | witnessed={len(witnesses)}"
                        f" | elapsed={now-start:.0f}s | arm ETA~{arm_eta:.0f}s"
                        f" | sampling ETA~{eta:.0f}s | projected total~{now-start+eta:.0f}s"
                        f" | finish~{finish:%H:%M:%S} MSK")
                    last_report = now
            rounds.append({"round": len(rounds) + 1, "fits": new_fits, "witnessed": len(witnesses)})
        stop = "all features witnessed"
    except BudgetReached as end:
        stop = str(end)

    elapsed = time.perf_counter() - arm_start
    log(f"finish {method} | {stop} | {new_fits} fits | {sum(counts)-skipped_after} marginals"
        f" | {len(witnesses)} witnesses | arm duration={elapsed:.2f}s")
    return {
        "method": method, "stop": stop, "new_fits": new_fits, "cache_hits": cache_hits,
        "elapsed_s": elapsed, "oracle_s": oracle_seconds, "completed_rounds": rounds,
        "marginals": sum(counts)-skipped_after, "resolved_trials": sum(counts), "skipped_after": skipped_after,
        "draws_per_feature": dict(zip(features, counts)),
        "sample_minima": dict(zip(features, scores)), "witnesses": witnesses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager", action="store_true", help="Evaluate both endpoints as in the original pilot.")
    args = parser.parse_args()
    start = time.perf_counter()
    output = ROOT / "results/thesis_sampling" / datetime.now(MOSCOW).strftime("madelon-s42-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)

    def log(message):
        line = f"{datetime.now(MOSCOW):%H:%M:%S} MSK | {message}"
        print(line, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            print(line, file=handle)

    log(f"start Madelon seed=42 | output={output.name} | reading frozen partitions")
    with (SOURCE / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        [row] = [r for r in csv.DictReader(handle) if r["dataset_id"] == DATASET and int(r["seed"]) == SEED]
    for key in ("train_row_count", "val_row_count"):
        row[key] = int(row[key])
    train, y_train, val, y_val = load_prepared_train_validation(SOURCE, row)
    protocol = yaml.safe_load((SOURCE / "resolved_protocol.yaml").read_text(encoding="utf-8"))
    params = protocol["model_params_by_dataset"][DATASET]
    features = list(train.columns)
    full = (1 << len(features)) - 1

    def utility(mask):
        if mask == 0:
            return 0.5  # Constant training-prior predictions yield binary ROC AUC 0.5.
        columns = [i for i in range(len(features)) if mask & (1 << i)]
        preds = fit_predict(train.iloc[:, columns], y_train, val.iloc[:, columns],
                            task=row["task"], model_params=params, backend=protocol["backend"], seed=SEED)
        return compute_utility(y_val, preds, metric=row["utility"])

    full_auc = utility(full)
    threshold = protocol["alpha_default"] * full_auc
    log(f"shared full model AUC={full_auc:.9f} | quota={threshold:.9f} | features={len(features)}")
    arms = []
    for index, method in enumerate(("independent_round_robin", "permutation_chain"), start=1):
        arms.append(run_arm(method, features, utility, {0: 0.5, full: full_auc}, threshold, log, index, start,
                            eager=args.eager))

    for arm in arms:
        retained = sum(1 << i for i, name in enumerate(features) if name not in arm["witnesses"])
        value = full_auc if retained == full else utility(retained)
        arm["retained_validation_auc"] = value
        arm["validation_delta"] = value - full_auc
        margin = protocol["removal"]["classification_noninferiority_margin"]
        arm["passes_original_gate"] = bool(value >= full_auc - margin)
        log(f"validation {arm['method']} | retained={retained.bit_count()}/{len(features)}"
            f" | AUC={value:.9f} | delta={value - full_auc:+.6f} | gate={arm['passes_original_gate']}")

    result = {
        "dataset": DATASET, "seed": SEED, "eager": args.eager, "source": str(SOURCE.relative_to(ROOT)), "features": features,
        "train_sha256": row["train_file_sha256"], "val_sha256": row["val_file_sha256"],
        "model_params": params, "backend": protocol["backend"], "alpha": protocol["alpha_default"],
        "full_auc": full_auc, "threshold": threshold, "fit_budget_per_arm": FIT_BUDGET,
        "seconds_cap_per_arm": ARM_SECONDS, "arms": arms, "elapsed_s": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    log(f"complete | actual total={result['elapsed_s']:.2f}s | summary.json")


if __name__ == "__main__":
    main()
