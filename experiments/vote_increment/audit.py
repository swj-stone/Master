"""Paired incremental-value gate. The unit of a claim is a registered dataset."""
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ARMS=("min","mean","gain")
METRICS=("k","fits","seconds","primary")


def checks(frame,names,seeds,methods=DEFAULT_ARMS):
    """Accept one independently timed, complete row per arm, seed and collection.

    If there are repeated timings, the caller first verifies support, predictions
    and fit-count parity and supplies their median. A fit includes the final
    deployment fit and every trained selector/surrogate when applicable.
    """
    if len(methods)!=3 or len(set(methods))!=3:
        raise ValueError("Expected distinct minimum, arithmetic-mean and direct Gain arms")
    keys=["dataset","seed","method"]
    data=frame.loc[frame.method.isin(methods)].copy()
    expected=pd.MultiIndex.from_product([names,seeds,methods], names=keys)
    received=pd.MultiIndex.from_frame(data[keys])
    if (not expected.is_unique or not received.is_unique or
            len(expected.difference(received)) or len(received.difference(expected))):
        raise ValueError("Missing, repeated or unregistered dataset/seed/arm")
    if not np.isfinite(data[list(METRICS)].to_numpy(float)).all() or not (data.seconds>0).all():
        raise ValueError("Nonfinite score, count or wall time")
    if not all((data[c]>=1).all() and np.equal(data[c],np.floor(data[c])).all() for c in ("k","fits")):
        raise ValueError("Expected positive integer retained widths and fit counts")

    index=["dataset","seed"]
    mi=data[data.method==methods[0]].set_index(index).sort_index()
    out=pd.DataFrame(index=mi.index)
    out["pass_both"]=True
    out["pass_both_pareto"]=True
    out["pass_both_five_pct"]=True
    for method in methods[1:]:
        control=data[data.method==method].set_index(index).reindex(mi.index)
        quality=mi.primary>=control.primary-.01-1e-14
        fewer=mi.k<control.k
        fits=mi.fits<control.fits
        fast=mi.seconds<control.seconds
        comparable=mi.k-control.k<=np.maximum(1.,.05*control.k)
        saved=fits&fast&comparable
        passes=quality&(fewer|saved)
        out[f"primary_diff_{method}"]=mi.primary-control.primary
        out[f"width_diff_{method}"]=mi.k-control.k
        out[f"fits_diff_{method}"]=mi.fits-control.fits
        out[f"time_ratio_{method}"]=mi.seconds/control.seconds
        out[f"narrower_{method}"]=quality&fewer
        out[f"saved_refits_{method}"]=quality&saved
        out[f"pass_{method}"]=passes
        out["pass_both"] &= passes
        out["pass_both_five_pct"] &= quality&(fewer|(fits&comparable&(mi.seconds<=.95*control.seconds)))
        out["pass_both_pareto"] &= (quality&(mi.k<=control.k)&(mi.fits<=control.fits)&
                                   (mi.seconds<=control.seconds)&(fewer|fits))
        for extra in ("accuracy","mae"):
            if extra in data:
                out[f"{extra}_diff_{method}"]=mi[extra]-control[extra]
    paired=out.reset_index()
    src=out.groupby(level="dataset")[["pass_both","pass_both_five_pct","pass_both_pareto"]].sum()
    src=src.add_suffix("_seeds").astype(int)
    src["seeds"]=len(seeds)
    src["pass"]=2*src.pass_both_seeds>src.seeds
    src["pass_pareto"]=2*src.pass_both_pareto_seeds>src.seeds
    return paired,src.reset_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file",type=Path)
    parser.add_argument("--datasets",required=True,nargs="+")
    parser.add_argument("--seeds",required=True,type=int,nargs="+")
    parser.add_argument("--methods",nargs=3,default=DEFAULT_ARMS)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    paired,sources=checks(pd.read_csv(args.file),args.datasets,args.seeds,args.methods)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        paired.to_csv(args.out/"gate-seeds.csv",index=False)
        sources.to_csv(args.out/"gate-sources.csv",index=False)
    print(sources.to_string(index=False))
    passed=int(sources["pass"].sum())
    minimum=ceil(.8*len(sources))
    print(f"Passed {passed}/{len(sources)} datasets, required {minimum}; "
          f"same-seed passes {paired.pass_both.sum()}/{len(paired)}")


if __name__=="__main__":
    main()
