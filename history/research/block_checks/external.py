"""Exact retrospective equal-width controls on external folds at each realized block width."""
from itertools import chain, combinations
from math import comb

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.block_checks.ablation import HERE, KEY
from research.block_selection.fresh import fresh_cases
from research.pair_selection.benchmark import external_cases


def run():
    arms = pd.read_csv(HERE/"cases.csv", keep_default_na=False).set_index(KEY + ["arm"])
    records = []
    summaries = []
    for case in chain(external_cases(), fresh_cases()):
        key = case.key
        meta = dict(zip(KEY, key))
        block = arms.loc[key+("block",)]
        k, p = int(block.k), int(block.p)
        metric = block.metric
        subsets = {}
        for subset in combinations(range(p), k):
            score = case.score(*case.predict(np.array(subset)))
            subsets[subset] = score
            records.append(meta | dict(metric=metric, p=p, k=k,
                                       columns=" ".join(map(str, subset)), score=score))
        scores = np.array(list(subsets.values()))
        assert len(scores) == comb(p, k)
        for name in ["block", "pair_matched", "anchor_matched", "vote_matched"]:
            row = arms.loc[key+(name,)]
            selected = tuple(int(j) for j in str(row["columns"]).split())
            score = subsets[selected]
            np.testing.assert_allclose(score, row.score, atol=1e-12, rtol=1e-12)
            strictly_better = int(np.sum(scores > score+1e-12))
            ties = int(np.sum(np.abs(scores - score) <= 1e-12))
            summaries.append(meta | dict(metric=metric, p=p, k=k, arm=name,
                                         score=score, random_median=np.median(scores),
                                         random_mean=scores.mean(), total=len(scores),
                                         rank_min=1+strictly_better, rank_max=strictly_better+ties))
    assert len(summaries) == 4*18
    pd.DataFrame(summaries).to_csv(HERE/"external-ranks.csv", index=False)
    pd.DataFrame(records).to_csv(HERE/"external-subsets.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
