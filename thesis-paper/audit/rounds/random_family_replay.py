"""Check matched restoration masks and recompute held-out scores for repeated random families."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from statistics import mean

import numpy as np
from scipy.stats import rankdata

from restoration_coverage import ROOT, STUDY


def binary_auc(y, scores):
    """Mann–Whitney AUC, with average ranks for ties."""
    positive = np.asarray(y, dtype=bool)
    npos = positive.sum()
    nneg = len(y) - npos
    if not npos or not nneg:
        raise ValueError("AUC requires examples from both classes")
    return (rankdata(scores)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def prediction_auc(y, probabilities):
    if probabilities.shape[1] == 2:
        return binary_auc(y == 1, probabilities[:, 1])
    return mean(binary_auc(y == c, probabilities[:, c]) for c in range(probabilities.shape[1]))


def verify_geometry(masks, code, tail, row, arm):
    if (len(masks) != len(set(masks)) or any(not m <= tail for m in masks)
            or [len(m) for m in masks] != [len(m) for m in code]
            or masks[0] or masks[-1] != tail):
        raise ValueError(f"Invalid size-matched design for {row}")
    total = len(tail) * (len(tail) - 1)
    covered = sum(any(i in m and j not in m for m in masks)
                  for i in tail for j in tail if i != j)
    if arm != "random":
        actual = [frozenset(j for j, mask in enumerate(masks) if i in mask) for i in tail]
        original = [frozenset(j for j, mask in enumerate(code) if i in mask) for i in tail]
        if list(map(len, actual)) != list(map(len, original)) or len(set(map(len, actual))) > 1:
            raise ValueError(f"Unequal feature inclusion counts for {row}")
        if arm == "permute" and Counter(actual) != Counter(original):
            raise ValueError(f"Codewords changed under feature relabeling in {row}")
        missed = sum(n * (n - 1) for n in Counter(actual).values())
        if covered != total - missed or covered != int(row["covered"]) or total != int(row["eligible"]):
            raise ValueError(f"Pair enumeration, signature collisions and receipts differ for {row}")
    return total, covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-margins", action="store_true",
                        help="Verify switch and permutation controls with fixed row and column sums.")
    fixed = parser.parse_args().fixed_margins
    name = "fixed-margins" if fixed else "random-families"
    study = ROOT / "results/context_study" / name
    with (study / "per-fold.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    folds = {(r["dataset"], r["fold"]) for r in rows}
    with (STUDY / "per-fold.csv").open(encoding="utf-8") as stream:
        primary = {(r["dataset"], r["fold"]): r for r in csv.DictReader(stream)
                   if r["arm"] == "antichain"}
    spec = ROOT / "configs/context_study" / (name.replace("-", "_") + ".json")
    offsets = json.loads(spec.read_text(encoding="utf-8-sig"))["offsets" if fixed else "seed_offsets"]
    arms = ("switch", "permute") if fixed else ("random",)
    observed = {(r["dataset"], r["fold"], int(r["family_offset"]), r.get("arm", "random"))
                for r in rows}
    expected = {(d, f, o, a) for d, f in primary for o in offsets for a in arms}
    if observed != expected or len(observed) != len(rows):
        raise ValueError("Incomplete or duplicated repeated-family grid")
    coverage = defaultdict(list)
    errors = []
    for dataset, fold in sorted(folds):
        folder = study / dataset / f"fold-{fold}"
        source = STUDY / dataset / f"fold-{fold}"
        receipt = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        primary_design = json.loads((source / "selection.json").read_text(encoding="utf-8"))
        code = primary_design["traces"]["antichain"]["context_extensions"]
        tail = set(primary_design["order"][primary_design["anchor_k"]:])
        with np.load(folder / "predictions.npz") as pred, np.load(source / "predictions.npz") as base:
            y = pred["y"]
            np.testing.assert_array_equal(y, base["y"])
            np.testing.assert_array_equal(pred["row_id"], base["row_id"])
            for row in (r for r in rows if (r["dataset"], r["fold"]) == (dataset, fold)):
                offset = row["family_offset"]
                arm = row.get("arm", "random")
                tag = f"{arm}_{offset}" if fixed else offset
                masks = [frozenset(m) for m in receipt["families"][tag]]
                total, covered = verify_geometry(masks, code, tail, row, arm)
                support = receipt["selected"][tag]
                if (len(set(support)) != int(row["k"]) or len(support) != int(row["k"])
                    or not set(support) <= set(primary_design["order"][:primary_design["anchor_k"]])):
                    raise ValueError(f"Invalid support: {folder}, {offset}")
                if total:
                    coverage[arm].append((total, covered))
                p = pred[tag]
                if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] != len(np.unique(y)):
                    raise ValueError(f"Prediction shape or class count differs for {folder}, {tag}")
                if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError(f"Invalid probabilities for {folder}, {tag}")
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)
                auc = prediction_auc(y, p)
                accuracy = np.mean(np.argmax(p, axis=1) == y)
                errors.extend([abs(auc - float(row["auc"])), abs(accuracy - float(row["accuracy"]))])
    if not np.isfinite(errors).all() or max(errors) > 1e-12:
        raise ValueError(f"AUC or accuracy differs by {max(errors)}")
    print(f"Verified {len(rows)} held-out scores per metric. Maximum absolute error {max(errors):.3g}.")
    for arm, sizes in sorted(coverage.items()):
        rates = [covered / total for total, covered in sizes]
        weighted = sum(covered for _, covered in sizes) / sum(total for total, _ in sizes)
        print(f"{arm}: {len(sizes)} eligible designs, mean coverage {mean(rates):.4%}, "
              f"pooled-pair coverage {weighted:.4%}, {sum(z == 1 for z in rates)} complete.")


if __name__ == "__main__":
    main()
