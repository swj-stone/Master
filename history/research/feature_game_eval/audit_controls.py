"""Check archived F supports directly using between-group or regression sums of squares."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.feature_game_eval import panels as bench


def effect_size(X,y,task):
    """Bounded monotone transform of univariate F, directly in float64."""
    Z=np.array(X,dtype=float,copy=True,order="C")
    Z -= Z.mean(axis=0)
    total=np.einsum("ij,ij->j",Z,Z)
    if task=="regression":
        centered=y-y.mean(dtype=float)
        numerator=(centered @ Z)**2 / (centered @ centered)
    else:
        numerator=np.zeros(X.shape[1])
        for label in np.unique(y):
            subset=Z[y==label]
            numerator+=subset.sum(axis=0)**2 / len(subset)
    return np.divide(numerator,total,out=np.full_like(total,-np.inf if task=="classification" else 0.),where=total>0)


def run(archive,out):
    rows=[]
    registry = archive / 'arrays.csv'
    locations = {} if not registry.exists() else pd.read_csv(registry).set_index('panel').source.to_dict()
    for cohort,panel,_,_,_ in bench.panels():
        X=panel.X
        if panel.name in bench.deployment.ADDITIONAL:
            X,_=bench._impute_validation(X,X[:0])
        eta=effect_size(X,panel.y,panel.task)
        p=len(eta)
        id=f"{cohort}-{panel.name}-{panel.fold}-{panel.seed}"
        location = bench.ROOT / locations[id] if locations else archive/(id+".npz")
        with np.load(location) as f:
            cols=f["F_columns"]
            k=len(cols)
            omitted=np.setdiff1d(np.arange(p),cols)
            violation=0. if len(omitted)==0 else max(0.,np.max(eta[omitted])-np.min(eta[cols]))
            for draw in ["random",*(f"random_{i}" for i in range(bench.VARIATES))]:
                np.testing.assert_array_equal(bench.select(draw,panel,k),f[draw+"_columns"])
        rows.append(dict(case=id,task=panel.task,width=k,p=p,max_violation_bounded_F=violation))
    frame=pd.DataFrame(rows)
    frame.to_csv(out,index=False,mode="x")
    print(frame.sort_values("max_violation_bounded_F",ascending=False).head(10).to_string(index=False))
    if len(frame)!=126 or (frame.max_violation_bounded_F>1e-5).any():
        raise ValueError("Selected column below boundary; inspect magnitude and arithmetic")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.archive,args.out)
