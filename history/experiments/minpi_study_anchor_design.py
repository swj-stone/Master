"""Compare the exact column work for every possible anchor of the same game."""
import argparse
from math import ceil, lcm
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study_order_diagnostics import observed
from minppi_mofs.integer_minpi import _layers, _rational_scores


def examine(votes, h):
    p = votes.shape[1]
    counts = sorted({min(p - 1, k) for k in (0, 4, 16)})
    layers = _layers(votes, counts, None)
    unit = lcm(*(layer[0] for layer in layers))
    scores = [_rational_scores(votes, [layer]) * (unit // layer[0]) for layer in layers]
    weakest = np.minimum.reduce(scores)
    target = np.sort(np.argsort(-weakest, kind="stable")[:h])
    for k, anchor in zip(counts, scores):
        seed = np.argsort(-anchor, kind="stable")[:h]
        threshold = min(weakest[seed])
        eligible = np.flatnonzero(anchor >= threshold)
        recovered = eligible[np.argsort(-weakest[eligible], kind="stable")[:h]]
        np.testing.assert_array_equal(target, np.sort(recovered))
        yield dict(k=k, p=p, h=h, eligible=len(eligible), fraction=len(eligible)/p)


def run(output):
    rows = []
    for dataset, seed, votes, _ in observed():
        for result in examine(votes, ceil(.1 * votes.shape[1])):
            rows.append(dict(dataset=dataset, seed=seed, **result))
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "candidates.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    with threadpool_limits(1):
        run(parser.parse_args().output)

