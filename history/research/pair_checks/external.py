"""Retrospective equal-width enumeration on all twelve external test folds."""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.lowdim.method import fast_support, pair_support
from research.pair_checks.ablation import HERE, KEY
from research.pair_checks.reference import Refits, mechanism_supports, standardize
from research.pair_selection.benchmark import external_cases


def run():
    summaries = []
    details = []
    for case in external_cases():
        key = dict(zip(KEY, case.key))
        X = case.X
        seed = key["seed"]
        old, old_info = fast_support(X, case.y, task=case.task, seed=seed)
        pair, audit = pair_support(X, case.y, task=case.task, seed=seed, audit=True)
        ref = Refits(*standardize(X, case.y, case.task, seed))
        k = len(old)
        p = X.shape[1]
        assert len(pair) == k and p <= 11
        paths = mechanism_supports(ref, old, old_info["rank"])
        np.testing.assert_array_equal(pair, np.sort(paths["pair_forward"]))
        if k < p:
            np.testing.assert_array_equal(paths["pair_forward"], audit["selection_order"])
        selected = {name: tuple(sorted(path)) for name, path in paths.items()}
        metric = "AUC" if case.task == "classification" else "R2"

        # Every candidate arm is fixed before the held-out distribution is constructed.
        by_subset = {}
        for subset in combinations(range(p), k):
            value = case.score(*case.predict(np.asarray(subset, dtype=int)))
            by_subset[subset] = value
            details.append(key | dict(metric=metric, p=p, k=k,
                                      columns=" ".join(map(str, subset)), score=value))
        values = np.array(list(by_subset.values()))
        assert len(values) == comb(p, k)
        for name, subset in selected.items():
            value = by_subset[subset]
            tied = np.abs(values-value) <= 1e-12
            better = int((values > value+1e-12).sum())
            summaries.append(key | dict(
                metric=metric, p=p, k=k, reduced=k < p, arm=name,
                columns=" ".join(map(str, subset)), score=value,
                same_width_subsets=len(values), random_median=np.median(values),
                random_mean=values.mean(), min_score=values.min(), max_score=values.max(),
                rank_min=better+1, rank_max=better+int(tied.sum()),
            ))
    result = pd.DataFrame(summaries)
    assert len(result) == 48 and not result.duplicated(KEY + ["arm"]).any()
    result.to_csv(HERE / "external-ranks.csv", index=False)
    pd.DataFrame(details).to_csv(HERE / "external-all-subsets.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
