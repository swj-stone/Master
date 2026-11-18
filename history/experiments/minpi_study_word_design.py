"""Benchmark the signed-word alternative on the same training arrays and ballots."""
import argparse
from functools import partial
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study_raw_cost import panels, sizes, integer_anchor_pruning
from minppi_mofs.certified_minpi import top_from_votes, _validate
from minppi_mofs.integer_minpi import rational_top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale, word_support


def full_with_dispatch(votes, h):
    votes = _validate(votes, h)
    w, p = votes.shape
    tiers = sizes(p)
    scale = native_scale(p, tiers, w)
    if scale:
        return word_support(votes, h, tiers, scale, prune=False)[0]
    return rational_top_from_votes(votes, h, prune=False)[0]


def anchor_with_dispatch(votes, h):
    votes = _validate(votes, h)
    w, p = votes.shape
    tiers = sizes(p)
    scale = native_scale(p, tiers, w)
    if scale:
        return word_support(votes, h, tiers, scale, prune=True)[0]
    return integer_anchor_pruning(votes, h)[0]


def run(output):
    rng = np.random.default_rng(91831)
    trials, supports = [], []
    for source, seed, build, _, reference, _ in panels():
        v = build()
        w, p = v.shape
        h = ceil(.1*p)
        if reference is not None:
            np.testing.assert_array_equal(reference, v)
        native = bool(native_scale(p, sizes(p), w))
        target = rational_top_from_votes(v, h, prune=False)[0]
        supports.append(dict(source=source, seed=seed, native=native, w=w, p=p, h=h,
                             support=" ".join(map(str, target))))
        # Policies that run the same engine on a panel share that observation.
        methods = {"dispatch": lambda x: exact_top_from_votes(x, h)[0],
                   "full": partial(full_with_dispatch, h=h),
                   "float": lambda x: top_from_votes(x, h)[0]}
        if native:
            methods["dyadic"] = lambda x: rational_top_from_votes(x, h)[0]
            shared = "anchor"
        else:
            methods["anchor"] = partial(anchor_with_dispatch, h=h)
            shared = "dyadic"
        for repeat in range(-1, 5):
            for position, name in enumerate(rng.permutation(list(methods))):
                start = perf_counter_ns()
                fresh = build()
                middle = perf_counter_ns()
                cols = methods[name](fresh)
                stop = perf_counter_ns()
                np.testing.assert_array_equal(v, fresh)
                if name != "float":
                    np.testing.assert_array_equal(target, cols)
                if repeat < 0:
                    continue
                trial = dict(source=source, seed=seed, native=native, repeat=repeat, position=position,
                             kernel=name, same_support=np.array_equal(target, cols),
                             build_ms=(middle-start)/1e6, score_ms=(stop-middle)/1e6,
                             select_ms=(stop-start)/1e6)
                trials.append(dict(trial, method=name))
                if name == "dispatch":
                    trials.append(dict(trial, method=shared))
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(supports).to_csv(output/"supports.csv", index=False)
    pd.DataFrame(trials).to_csv(output/"trials.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(1):
        run(parser.parse_args().output)

