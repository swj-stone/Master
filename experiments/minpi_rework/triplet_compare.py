"""Development pilot with common prefix widths and separately trained receivers."""
import argparse
from unittest.mock import patch

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain
from experiments.minpi_rework.forward import development_split
from minppi_mofs.triplet_path import rank_triplet_path


def neighbor_predict(x, y, task, train, query, order, widths):
    prep = b.transform(x, train)
    a, z = prep.transform(x[train]), prep.transform(x[query])
    cls = KNeighborsClassifier if task == "classification" else KNeighborsRegressor
    outputs = {}
    for k in widths:
        cols = np.sort(order[:k])
        model = cls(n_neighbors=min(7,len(train)), weights="distance", algorithm="brute", p=1)
        model.fit(a[:,cols],y[train])
        outputs[k] = (model.predict_proba if task == "classification" else model.predict)(z[:,cols])
    return outputs


base_rank = b.ranked


def ranked(method,x,y,task,idx,seed):
    if method in ("lazy","minimum","pooled","current","margin"):
        a,z = idx["fit"],idx["vote"]
        return rank_triplet_path(x[a],y[a],x[z],y[z],task=task,seed=seed,rule=method,path_fraction=.75)
    if method == "gain":
        return legacy_gain.gain_rank(method,x,y,task,idx,seed)
    return base_rank(method,x,y,task,idx,seed)


def run(args):
    from experiments.minpi_rework import path_pilot
    with patch.object(path_pilot,"rank",ranked),patch.object(b,"split",development_split):
        if args.receiver=="knn":
            with patch.object(b,"predict",neighbor_predict):
                path_pilot.pilot(args)
        else:
            path_pilot.pilot(args)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--names",nargs="+",default=["glass","raisin","wifi","biodeg","magic","wall","musk1","aquatic","power","yacht","slump"])
    cli.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    cli.add_argument("--receiver",choices=["knn","ridge"],default="knn")
    cli.add_argument("--run",required=True)
    cli.add_argument("--methods",nargs="+",default=["lazy","pooled","current","margin","coefficient","rfe","full"])
    cli.add_argument("--gain",action="store_true")
    cli.add_argument("--repeats",type=int,default=1)
    cli.add_argument("--cpu",type=int)
    cli.set_defaults(dense=False)
    with threadpool_limits(1):
        run(cli.parse_args())
