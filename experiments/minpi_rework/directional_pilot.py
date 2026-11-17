"""Development comparison for projection ballots in the minimum faithful-width game."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework.exchange_benchmark import load_dev
from experiments.minpi_rework.quota_compress import greedy,prefix
from minppi_mofs.additive_minpi import fit_additive
from minppi_mofs.directional_quota import directional_search,gram_loss,FidelityGame


def game(x,y,task,idx):
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    model=fit_additive(x[pool],y[pool],task,"linear")
    columns=model.parts(x[idx["width"]])-model.center
    matrix=columns.transpose(0,2,1).reshape(-1,x.shape[1])
    return model,FidelityGame.from_factor(matrix)


def run(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    jobs=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    directional_search(FidelityGame.from_factor(np.eye(4)),np.ones(4,bool),shared=True,cache=True)
    for name in args.names:
        for seed in args.seeds:
            x,y,task,idx,truth,mask=load_dev(name,seed,jobs)
            start=perf_counter()
            model,objective=game(x,y,task,idx)
            g=objective.gram
            setup=perf_counter()-start
            greedy_start=perf_counter()
            initial,_=greedy(g)
            if not objective.feasible(initial,.2):
                initial=np.ones(len(initial),bool)
            greedy_s=perf_counter()-greedy_start
            keep,info=directional_search(objective,initial,max_seconds=args.seconds,shared=args.shared,cache=args.cache)
            energy=prefix(g,np.argsort(-g.diagonal(),kind="stable"))
            for method,chosen,taken in (("dual",keep,info["solver_s"]+greedy_s),("greedy",initial,greedy_s),("energy",energy,0.),("full",np.ones(len(g),bool),0.)):
                cols=np.flatnonzero(chosen)
                pred=model.restrict(cols).predict(x[np.ix_(idx["test"],cols)])
                metrics=b.metrics(truth[mask],pred[mask] if task=="classification" else pred[mask].ravel(),task,
                                  max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12))
                rows.append(dict(dataset=name,seed=seed,task=task,p=len(g),method=method,k=len(cols),
                                 selection_s=setup+taken,deployment_s=setup+taken,error=gram_loss(g,chosen),**metrics,
                                 **(info if method=="dual" else {})))
            pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
            print(f"{name}/{seed}: {initial.sum()} -> {keep.sum()}, LB {info['lb']}, {info['nodes']} nodes, {info['solver_s']:.3f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--names",nargs="+",default=legacy.CONFIG["datasets"])
    cli.add_argument("--run",required=True)
    cli.add_argument("--seeds",nargs="+",type=int,default=[42])
    cli.add_argument("--seconds",type=float,default=3.)
    cli.add_argument("--shared",action="store_true")
    cli.add_argument("--cache",action="store_true")
    with threadpool_limits(1):
        run(cli.parse_args())
