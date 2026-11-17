"""Matched trimmed additive receivers for exact single/pair deletion paths."""
import argparse
from unittest.mock import patch

from threadpoolctl import threadpool_limits

from experiments.minpi_rework import additive
from minppi_mofs.pair_vote_path import rank_pair


def rank(x,y,z,truth,task,kind,rule):
    chooser = rank_pair if rule.startswith(("pair_","double_","single_")) else ORIGINAL
    return chooser(x,y,z,truth,task,kind,rule)


ORIGINAL = additive.rank_additive


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run",required=True)
    cli.add_argument("--seeds",nargs="+",type=int,default=[42])
    cli.add_argument("--kind",choices=["linear","spline"],default="spline")
    cli.add_argument("--receiver",choices=["ridge","trim"],default="trim")
    cli.add_argument("--names",nargs="+",required=True)
    cli.add_argument("--methods",nargs="+",default=["pair_min","pair_point","pair_pooled","pair_energy","min","point","pooled","energy","full"])
    cli.add_argument("--zero",action="store_true")
    with threadpool_limits(1), patch.object(additive,"rank_additive",rank):
        additive.develop(cli.parse_args())
