"""Summarize registered universe comparisons and independently check the mask-mixture argument."""
from fractions import Fraction
from itertools import combinations
from math import comb
import json

import numpy as np
import pandas as pd

from experiments.minpi_rework.predictive_universe import OUT


def exact_witness():
    # Only active features {0,1} present and 2 absent produce a positive loss improvement.
    improvement = Fraction(591, 10000)
    values = {}
    for p in (3, 4):
        curve = []
        for size in range(1, p+1):
            hits = sum({0, 1}.issubset(mask) and 2 not in mask
                       for mask in combinations(range(p), size))
            curve.append(improvement * Fraction(p*hits, size*comb(p, size)))
        values[p] = curve
    assert max(values[3]) == Fraction(591, 20000)
    assert max(values[4]) == Fraction(197, 10000)
    # Independently verify the mixing relation for every stratum, with exact signed set losses.
    for m in range(1, 6):
        active = set(range(m))
        for p in range(m, 9):
            def loss(mask):
                members = active.intersection(mask)
                return Fraction((1+sum(members))**2 * (-1)**len(members)) if members else Fraction(0)
            def score(k, s):
                total = sum(loss(mask) for mask in combinations(range(k), s))
                return total * Fraction(k, s*comb(k, s))
            for s in range(1, p+1):
                mixed = sum(Fraction(comb(m, j)*comb(p-m, s-j), comb(p, s)) *
                            Fraction(p*j, s*m) * score(m, j)
                            for j in range(max(1, s-p+m), min(s, m)+1))
                assert mixed == score(p, s)


def main():
    exact_witness()
    frame = pd.read_csv(OUT/"scores.csv")
    fixed = frame[frame.arm.str.startswith("original")]
    indexed = fixed.pivot(index=["panel", "dataset", "seed", "arm", "stage"],
                          columns="law", values=["worst", "fail"])
    delta = (indexed["worst"]["active"]-indexed["worst"]["declared"]).astype(float)
    summary = pd.DataFrame(dict(increase=delta, materially_changed=delta > 1e-9,
        declared_fail=indexed["fail"]["declared"].astype(bool),
        active_fail=indexed["fail"]["active"].astype(bool)))
    summary["new_fail"] = summary.active_fail & ~summary.declared_fail
    assert not (summary.declared_fail & ~summary.active_fail).any()
    assert len(summary) == 220
    fixed_summary = summary.groupby(["panel", "arm", "stage"]).agg(
        jobs=("increase", "size"), changed=("materially_changed", "sum"), max_increase=("increase", "max"),
        mean_increase=("increase", "mean"), original_fail=("declared_fail", "sum"),
        active_fail=("active_fail", "sum"), new_fail=("new_fail", "sum"))
    fixed_summary.to_csv(OUT/"fixed-summary.csv")
    print(fixed_summary.to_string())
    print("Largest changes", summary.nlargest(4, "increase"), sep="\n")
    # Check original seven-source outer values and original wide intact forecasts.
    prior = pd.read_csv(OUT.parent/"predictive-confirmation"/"scores.csv")
    for arm, method in (("original_mdi", "mdi_min"), ("original_portfolio", "mdi_plus_min_min")):
        ix = ["dataset", "seed"]
        a = fixed.query("panel=='seven' and stage=='outer' and law=='declared' and arm==@arm").set_index(ix)
        b = prior[prior.method==method].set_index(ix).loc[a.index]
        np.testing.assert_allclose(a.worst, b.outer_worst, atol=2e-10)
    wide_prior = pd.read_csv(OUT.parent/"wide-veto"/"scores.csv")
    priors = {"original_mdi": "mdi", "original_portfolio": "portfolio"}
    for arm, method in priors.items():
        ix = ["dataset", "seed"]
        a = fixed.query("panel=='wide' and stage=='outer' and law=='declared' and arm==@arm").set_index(ix)
        b = wide_prior[wide_prior.method==method].set_index(ix).loc[a.index]
        np.testing.assert_allclose(a[["auc","accuracy","r2","mae"]], b[["auc","accuracy","r2","mae"]],
                                   equal_nan=True, atol=2e-10)
    rec = json.loads((OUT/"supports.json").read_bytes())
    wide = frame.query("panel=='wide' and stage=='outer' and law=='active'").set_index(["dataset","seed","arm"])
    report = []
    for (source, seed), block in wide.groupby(level=[0, 1]):
        block = block.droplevel([0, 1])
        original, repeat = block.loc["original_portfolio"], block.loc["active_portfolio"]
        ids = rec[f"{source}-{seed}"]["supports"]
        row = dict(dataset=source, seed=seed, p=int(original.p), m=int(original.m),
            original_width_ratio=original.reachable/block.loc["original_mdi"].reachable,
            active_width_ratio=repeat.reachable/block.loc["active_mdi"].reachable)
        for arm in ("mdi", "portfolio"):
            old = block.loc["original_"+arm]
            new = block.loc["active_"+arm]
            row.update({arm+"_changed": ids["original_"+arm] != ids["active_"+arm],
                        arm+"_old_width":old.reachable,arm+"_new_width":new.reachable,
                        arm+"_old_fail":old.fail,arm+"_new_fail":new.fail,
                        arm+"_worst_delta":new.worst-old.worst})
        for metric in ("auc", "accuracy", "r2", "scaled_mae"):
            row[metric+"_vs_mdi"] = repeat[metric]-block.loc["active_mdi"][metric]
        report.append(row)
    result = pd.DataFrame(report)
    result.to_csv(OUT/"reselection.csv", index=False)
    numeric = result.groupby("dataset").mean(numeric_only=True)
    ratios = np.exp(np.log(result[["active_width_ratio","original_width_ratio"]]).groupby(result.dataset).mean())
    print(numeric.to_string(), ratios.to_string(), "equal-source ratios", np.exp(np.log(ratios).mean()),
          "summary", result.sum(numeric_only=True), sep="\n")


if __name__ == "__main__":
    main()
