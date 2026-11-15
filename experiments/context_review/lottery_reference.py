"""Exact finite-slate distribution of guarded savings under independent uniform nominations."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pandas as pd


EVIDENCE = Path(__file__).resolve().parents[2] / "thesis-paper" / "evidence"


def ratio(value):
    # All observed widths and candidate counts are below 1000. Refuse a
    # value this bound cannot recover to the precision of the source CSV.
    recovered = Fraction(str(value)).limit_denominator(1000)
    if abs(float(recovered) - value) > 1e-12:
        raise ValueError(f"Unrecognised finite-slate fraction {value}")
    return recovered


def reference(folds, sizes, receiver, panel):
    uniform = folds.loc[folds.method.eq("uniform")].set_index(["dataset", "fold"])
    if not uniform.index.is_unique or set(uniform.index) != set(sizes.index):
        raise ValueError("Candidate and lottery fold identifiers do not match")
    task_folds = uniform.groupby(level="dataset").size()
    weights = {key: Fraction(1, len(task_folds) * int(task_folds[key[0]])) for key in uniform.index}
    dist = {Fraction(0): Fraction(1)}
    for key, row in uniform.iterrows():
        ceiling = ratio(row.joint_ceiling)
        expected = ratio(row.u)
        if not ceiling:
            assert expected == 0
            continue
        p = expected / ceiling
        assert 0 < p <= 1 and (p * int(sizes[key])).denominator == 1
        new = defaultdict(Fraction)
        for total, mass in dist.items():
            if p < 1:
                new[total] += mass * (1 - p)
            new[total + weights[key] * ceiling] += mass * p
        dist = new
    assert sum(dist.values()) == 1
    mean = sum(u * mass for u, mass in dist.items())
    assert mean == sum(weights[key] * ratio(row.u) for key, row in uniform.iterrows())
    rows = []
    for arm in ("gain", "vote", "point", "slack"):
        chosen = folds.loc[folds.method.eq(arm)].set_index(["dataset", "fold"])
        assert chosen.index.is_unique and set(chosen.index) == set(weights)
        total = sum(weights[key] * ratio(row.u) for key, row in chosen.iterrows())
        ties = dist.get(total, Fraction(0))
        exceeds = sum(mass for u, mass in dist.items() if u > total)
        rows.append(dict(receiver=receiver, panel=panel, policy=arm, u=float(total),
                         lottery_expected_u=float(mean), lottery_matches=float(ties),
                         lottery_exceeds=float(exceeds), lottery_at_least=float(exceeds + ties)))
    return rows


def main():
    logistic = pd.read_csv(EVIDENCE / "receiver/folds.csv")
    results = []
    for panel in ("first", "wide"):
        tree = pd.read_csv(EVIDENCE / "chance" / f"{panel}-folds.csv")
        tree = tree.loc[tree.tolerance.eq(.03)]
        sizes = tree.loc[tree.method.eq("uniform")].set_index(["dataset", "fold"]).candidates
        for name, folds in (("xgb", tree), ("logistic", logistic.loc[logistic.panel.eq(panel)])):
            results.extend(reference(folds, sizes, name, panel))
    frame = pd.DataFrame(results)
    frame.to_csv(EVIDENCE / "chance/finite-lottery.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
