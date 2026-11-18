"""Recover the input fingerprints from each original formal split for cross-panel deduplication."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.minpi_study import P, load_dataset
from research.feature_game_eval.panels import digest
from research.lowdim.formal_study import SOURCES, inputs


def recover(completed, out):
    rows=[]
    for dataset in SOURCES:
        raw, y, task, _, colmap, _, _=load_dataset(dataset)
        for fold, seed in enumerate(P["formal_seeds"],1):
            X,target,Xt,truth,tr,te,_,_=inputs(dataset,seed,raw,y,colmap,task)
            with np.load(completed/f"{dataset}-{seed}.npz") as saved:
                for name,vals in (("truth",truth),("train",tr),("test",te)):
                    np.testing.assert_array_equal(saved[name],vals)
            rows.append(dict(dataset=dataset,fold=fold,seed=seed,n_train=len(target),n_test=len(truth),
                             classes=len(np.unique(target)) if task=="classification" else 0,
                             train_sha=digest(X),target_sha=digest(target),test_sha=digest(Xt),truth_sha=digest(truth)))
    result=pd.DataFrame(rows)
    assert len(result)==40 and not result.duplicated(["dataset","fold","seed"]).any()
    result.to_csv(out,mode="x",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("completed",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    recover(args.completed,args.out)
