"""Enumerate ordered tail pairs covered by the restoration masks saved for each fold."""
import argparse
import csv
import json
from itertools import permutations
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "results/thesis_benchmark/prospective-pmlb-contexts-r1"
OUTPUT = Path(__file__).with_name("restoration-coverage.csv")
ARMS = ("min", "antichain", "random_contexts")


def records(study):
    with (study / "per-fold.csv").open(encoding="utf-8") as stream:
        folds = sorted({(row["dataset"], int(row["fold"])) for row in csv.DictReader(stream)})
    for dataset, fold in folds:
        path = study / dataset / f"fold-{fold}" / "selection.json"
        selection = json.loads(path.read_text(encoding="utf-8"))
        tail = selection["order"][selection["anchor_k"]:]
        if len(tail) != len(set(tail)):
            raise ValueError(f"Repeated tail index in {path}")
        for arm in ARMS:
            masks = [set(mask) for mask in selection["traces"][arm]["context_extensions"]]
            if any(not mask.issubset(tail) for mask in masks):
                raise ValueError(f"Invalid tail extension for {arm} in {path}")
            total = len(tail) * (len(tail) - 1)
            covered = sum(any(i in mask and j not in mask for mask in masks)
                          for i, j in permutations(tail, 2))
            counts = [sum(i in mask for mask in masks) for i in tail]
            yield dict(dataset=dataset, fold=fold, arm=arm, tail=len(tail),
                       context_count=len(masks), covered_pairs=covered, possible_pairs=total,
                       fraction=covered / total if total else None,
                       min_inclusions=min(counts, default=None),
                       max_inclusions=max(counts, default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = list(records(args.study))
    if not rows:
        raise ValueError(f"No fold records in {args.study}")
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for arm in ARMS:
        informative = [row for row in rows if row["arm"] == arm and row["possible_pairs"]]
        if informative:
            complete = sum(row["covered_pairs"] == row["possible_pairs"] for row in informative)
            balanced = sum(row["min_inclusions"] == row["max_inclusions"] for row in informative)
            fraction = mean(row["fraction"] for row in informative)
            print(f"{arm}: {len(informative)} folds with pairs, {fraction:.4%} mean coverage, "
                  f"{complete} complete, {balanced} balanced across tail features")


if __name__ == "__main__":
    main()
