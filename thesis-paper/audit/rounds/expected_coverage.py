"""Exact chance that random distinct masks of specified sizes expose an ordered pair."""
import csv
import json
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import comb, prod
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2] / "results/thesis_benchmark/prospective-pmlb-contexts-r1"


def expected(m, sizes):
    if m < 2:
        raise ValueError("An ordered pair needs two distinct features")
    miss = Fraction(1)
    for size, r in Counter(s for s in sizes if 0 < s < m).items():
        pool = comb(m, size)
        expose = comb(m-2, size-1)
        miss *= Fraction(comb(pool-expose, r), comb(pool, r))
    return 1 - miss


def exhaustive(m, sizes):
    choices = [list(combinations(list(combinations(range(m), size)), r))
               for size, r in Counter(sizes).items()]
    covered = sum(any(0 in mask and 1 not in mask for group in family for mask in group)
                  for family in product(*choices))
    return Fraction(covered, prod(len(c) for c in choices))


def main():
    for m, sizes in [(2, [1,1]), (3,[1,1,2]), (4,[1,2,2]), (5,[2,2,3,3])]:
        assert expected(m, sizes) == exhaustive(m, sizes)
    rows = []
    for path in sorted(STUDY.glob("*/fold-*/selection.json")):
        record = json.loads(path.read_bytes())
        tail = record["order"][record["anchor_k"]:]
        if len(tail) < 2:
            continue
        masks = record["traces"]["antichain"]["context_extensions"]
        value = expected(len(tail), list(map(len, masks)))
        rows.append(dict(dataset=path.parents[1].name, fold=int(path.parent.name[5:]),
                         tail=len(tail), expected=float(value), numerator=value.numerator,
                         denominator=value.denominator))
    if len(rows) != 41:
        raise ValueError("Unexpected number of folds with an ordered pair")
    with (HERE/"expected-random-coverage.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} eligible folds: expected equal-fold coverage {mean(r['expected'] for r in rows):.8%}")


if __name__ == "__main__":
    main()
