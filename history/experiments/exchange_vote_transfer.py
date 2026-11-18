"""Evaluate minPI exchange votes and the exact constant-column control on the frozen panels."""
import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_transfer as transfer
from experiments import exchange_vote_benchmark as benchmark
from experiments.stable_minpi_stress import ROOT, file_hash
from minppi_mofs.exchange_vote_minpi import exchange_vote_top_from_training
from minppi_mofs.sketch_minpi import reduced_width


BASE = ("exchange_auto", "random_exchange", "F_exchange", "constant", "full")
FIXED = ("exchange_75", "random_75", "F_75")


def methods(panel):
    return BASE + FIXED if panel.family.startswith("previous") else BASE


def select(method, panel, widths):
    if method == "exchange_auto" or method == "exchange_75":
        count = reduced_width(panel.X.shape[1], .75) if method == "exchange_75" else None
        return exchange_vote_top_from_training(panel.X, panel.y, count, task=panel.task,
                                               random_state=panel.seed, groups=panel.groups,
                                               ordered_groups=panel.ordered)[0]
    return benchmark.select(method, panel, widths)[0]


def protocol():
    state = transfer.protocol()
    paths = [Path(__file__), ROOT/"src/minppi_mofs/exchange_vote_minpi.py",
             ROOT/"experiments/exchange_vote_benchmark.py", ROOT/"experiments/moment_minpi_benchmark.py"]
    state["hashes"].update({p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths})
    return dict(state, methods=list(BASE), previous_panels_extra_methods=list(FIXED),
                constant_control="Remove every exactly constant column, verified on all outer training rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        transfer.run(args.output, args.verify, make_protocol=protocol, get_methods=methods,
                     chooser=select, adaptive_names={"exchange_auto": "exchange"})
