"""Development runs for conditional forward votes; shares guard with backward paths."""
import argparse
from unittest.mock import patch

from threadpoolctl import threadpool_limits
from experiments.minpi_rework import path_pilot
from experiments.minpi_compression import benchmark, chain_validation_sources, witness_transfer_sources
from minppi_mofs.forward_minpi import rank_forward


RULES = ["min", "mean", "current", "fit"]
prior_rank = path_pilot.rank
base_split = benchmark.split


def development_split(name, seed):
    if name in chain_validation_sources.SOURCES:
        return chain_validation_sources.split(name, seed)
    if name in witness_transfer_sources.SOURCES:
        return witness_transfer_sources.split(name, seed)
    return base_split(name, seed)


def rank(method, x, y, task, idx, seed):
    if method not in RULES:
        return prior_rank(method, x, y, task, idx, seed)
    fit, vote = idx["fit"], idx["vote"]
    prep = benchmark.transform(x, fit)
    order = rank_forward(prep.transform(x[fit]), y[fit], prep.transform(x[vote]),
                         y[vote], task, method)
    return order, {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", default=benchmark.DESIGN["confirmation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--run", default="forward-development")
    parser.add_argument("--methods", nargs="+", default=[*RULES, "coefficient", "rfe", "full"])
    parser.add_argument("--gain", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--dense", action="store_true")
    with threadpool_limits(limits=1), patch.object(path_pilot, "rank", rank):
        with patch.object(benchmark, "split", development_split):
            path_pilot.pilot(parser.parse_args())
