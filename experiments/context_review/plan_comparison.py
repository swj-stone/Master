"""Bound a study's achievable effective-compression contrast before outer scoring."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from experiments.context_review.decision_opportunities import EVIDENCE, RUNS, envelope
from experiments.context_review import budgeted_guard as bg


def support(columns):
    if not isinstance(columns, list) or not columns or not all(isinstance(x, str) for x in columns):
        raise ValueError("A support must be a non-empty list of column names")
    if len(set(columns)) != len(columns):
        raise ValueError("Repeated feature within a support")
    return frozenset(columns)


def lottery(entries, anchor):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Each policy needs at least one possible return")
    mass = Counter()
    for item in entries:
        cols = support(item["columns"])
        p = Fraction(str(item["probability"]))
        if p < 0 or not cols <= anchor:
            raise ValueError("Invalid probability or support outside its anchor")
        mass[cols] += p
    if sum(mass.values()) != 1:
        raise ValueError("Return probabilities must sum to one; write exact fractions such as 1/3")
    return mass


def assess(folds, target_pp=None):
    """Equal-source, equal-within-source bound from distributions after the guard."""
    if not isinstance(folds, list) or not folds:
        raise ValueError("Supply at least one fold")
    keys = [(fold["source"], fold["fold"]) for fold in folds]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate source/fold")
    sizes = Counter(key[0] for key in keys)
    lower = upper = Fraction(0)
    unequal = 0
    for fold in folds:
        anchor = support(fold["anchor"])
        left, right = (lottery(fold[side], anchor) for side in ("left", "right"))
        cols = list(left.keys() | right.keys())
        savings = np.array([1 - Fraction(len(s), len(anchor)) for s in cols])
        lo, hi = envelope(np.array([left[s] for s in cols]), np.array([right[s] for s in cols]), savings)
        weight = Fraction(1, len(sizes) * sizes[fold["source"]])
        lower += weight * lo
        upper += weight * hi
        unequal += int(left != right)
    result = dict(sources=len(sizes), folds=len(folds), differing_return_distributions=unequal,
                  lower_pp=float(100 * lower), upper_pp=float(100 * upper))
    if target_pp is not None:
        target = Fraction(str(target_pp))
        result.update(target_pp=float(target), target_ruled_out=100 * upper <= target)
    return result


def original_wide():
    folds = []
    seals = bg.read(EVIDENCE / "wide/prediction-lock.json")["folds"]
    for key, receipt in seals.items():
        path = RUNS / "wide-uci" / key / "selection.json"
        if bg.digest(path) != receipt["selection.json"]:
            raise ValueError(f"Changed selection receipt for {key}")
        choice = bg.read(path)["choices"]
        name, f = key.split("/")
        def names(cols):
            return [str(c) for c in cols]
        fold = dict(source=name, fold=f, anchor=names(choice["gain"]["anchor"]))
        for side, arm in [("left", "slack"), ("right", "point")]:
            fold[side] = [dict(columns=names(choice[arm]["selected"]), probability=1)]
        folds.append(fold)
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="JSON fold records with left and right guarded returns")
    group.add_argument("--original-wide", action="store_true", help="Recompute the published wide-panel ceiling")
    parser.add_argument("--target-pp", type=Fraction, help="Desired strictly greater improvement in percentage points")
    args = parser.parse_args()
    folds = original_wide() if args.original_wide else json.loads(args.input.read_text(encoding="utf8"))
    result = assess(folds, args.target_pp)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
