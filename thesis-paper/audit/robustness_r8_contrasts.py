"""Recalculate R8 source-level contrasts from its archived CSVs using Python's standard library."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
BOOTSTRAP_SEED = 49837
BOOTSTRAP_DRAWS = 30_000
METHODS = ("slack", "vote")
PRECISIONS = ("2", "4", "8", "full")


def read_rows(relative_path):
    with (EVIDENCE / relative_path).open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def guarded_returns(relative_path, pipeline):
    """Key returns by panel, task, fold and draw; the original pipeline has one draw."""
    records = defaultdict(dict)
    for row in read_rows(relative_path):
        if pipeline == "original":
            if (row["voter"], row["receiver"]) != ("forest", "forest"):
                continue
            draw = 1
        else:
            if row["delivery"] != "guarded":
                continue
            draw = int(row["draw"])
        method = row["method"]
        if method not in METHODS:
            continue
        key = (row["panel"], row["dataset"], int(row["fold"]), draw)
        if method in records[key]:
            raise ValueError(f"Duplicate return for {pipeline}, {key}, {method}")
        records[key][method] = float(row["u"])
    return records


def by_task_and_pipeline(records, expected_draws):
    by_fold = defaultdict(dict)
    for (panel, task, fold, draw), methods in records.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"Missing policy on {panel}, {task}, {fold}, {draw}")
        by_fold[panel, task, fold][draw] = 100 * (methods["slack"] - methods["vote"])
    by_task = defaultdict(list)
    for (panel, task, fold), draws in by_fold.items():
        if set(draws) != expected_draws:
            raise ValueError(f"Incomplete draws on {panel}, {task}, {fold}")
        by_task[panel, task].append(mean(draws.values()))
    return {key: mean(values) for key, values in by_task.items()}, set(by_fold)


def interpolated_percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    weight = position - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[lower + 1]


def paired_interval(task_differences):
    # Each draw resamples complete tasks with their two pipelines already paired.
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(task_differences)
    replicates = sorted(
        sum(rng.choices(task_differences, k=n)) / n for _ in range(BOOTSTRAP_DRAWS)
    )
    return tuple(interpolated_percentile(replicates, p) for p in (.025, .975))


def print_pipeline_interaction():
    original, original_folds = by_task_and_pipeline(
        guarded_returns("forest_cross/policies.csv", "original"), {1}
    )
    local, local_folds = by_task_and_pipeline(
        guarded_returns("neighborhood_replicates/policies.csv", "local"), {1, 2, 3}
    )
    if original_folds != local_folds or len(local_folds) != 46:
        raise ValueError("Compared pipelines must use the same 46 folds.")
    print("panel,task,original_full_minus_vote_pp,local_full_minus_vote_pp,local_minus_original_pp")
    for panel in sorted({panel for panel, _ in original}):
        differences = []
        for key in sorted(key for key in original if key[0] == panel):
            delta = local[key] - original[key]
            differences.append(delta)
            print(f"{panel},{key[1]},{original[key]:.6f},{local[key]:.6f},{delta:.6f}")
        low, high = paired_interval(differences)
        print(f"{panel} mean {mean(differences):.6f} pp; paired 95% interval [{low:.6f}, {high:.6f}]")


def print_precision_comparisons():
    counts = defaultdict(Counter)
    for row in read_rows("score_resolution/proposals.csv"):
        if row["observer"] == "code":
            key = (row["dataset"], row["precision"], int(row["draw"]))
            counts[key][row["disposition"]] += 1
    tasks = sorted({key[0] for key in counts})
    print("bits,task,proposals,genuine_witness,genuine_unresolved,min_random_witness,strict_separation")
    for precision in PRECISIONS:
        separated = 0
        for task in tasks:
            genuine = counts[task, precision, 0]
            draws = [counts[task, precision, draw] for draw in range(1, 6)]
            if not genuine or any(draw.total() != genuine.total() for draw in draws):
                raise ValueError(f"Incomplete matched comparisons on {task}, {precision}")
            random_lower = min(draw["witness"] for draw in draws)
            genuine_upper = genuine["witness"] + genuine["unresolved"]
            above = random_lower > genuine_upper
            separated += above
            print(f"{precision},{task},{genuine.total()},{genuine['witness']},"
                  f"{genuine['unresolved']},{random_lower},{above}")
        print(f"{precision} bits: strict separation on {separated}/{len(tasks)} tasks")


if __name__ == "__main__":
    print_pipeline_interaction()
    print_precision_comparisons()
