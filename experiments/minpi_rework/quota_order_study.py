"""Historical CPU-only seed-42 comparisons of cardinality-interval exchange restrictions."""
import argparse
import ctypes
import json
import os
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import quota_compress as q,quota_compress_run as dev
from experiments.minpi_rework.quota_compress_control import from_feasible,improve
from experiments.minpi_rework.quota_order import solve,order_edges,width_votes


def incumbent(g):
    order=np.lexsort((np.arange(len(g)),-g.diagonal()))
    a=from_feasible(g,q.prefix(g,order))
    b=from_feasible(g,np.ones(len(g),bool))
    z=np.zeros(len(g),bool)
    while not q.feasible(g,z,q.RHO):
        out=np.flatnonzero(~z)
        change=2*g[:,out].sum(axis=1)-g.diagonal()
        z[out[np.argmax(change[out])]]=True
        z=improve(g,z)
    c=from_feasible(g,z)
    return min((a,b,c),key=lambda m:(int(m.sum()),q.loss(g,m),tuple(np.flatnonzero(m))))


def run(args):
    folder=dev.b.ROOT/"results/minpi_rework"/args.run
    folder.mkdir(parents=True,exist_ok=False)
    if os.name=="nt":
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<args.cpu)):
            raise ctypes.WinError(ctypes.get_last_error())
    else: os.sched_setaffinity(0,{args.cpu})
    g=np.eye(4)
    order_edges(g,np.arange(4,dtype=float),1,2,"range")
    width_votes(g,np.arange(4,dtype=float),1,2)
    jobs=dev.legacy.jobs()
    rows=[]
    for name in args.names:
        start=perf_counter()
        x,y,task,idx=dev.load(name,jobs)
        train_clock=perf_counter()
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        teacher=dev.fit_additive(x[pool],y[pool],task,"linear")
        a=teacher.parts(x[idx["width"]])-teacher.center
        matrix=a.transpose(0,2,1).reshape(-1,x.shape[1])
        g=(matrix.T@matrix)/max(1e-100,np.sum(matrix.sum(axis=1)**2))
        teacher_s=perf_counter()-train_clock
        shared=perf_counter()
        initial=incumbent(g); _,path=q.greedy(g)
        upper_s=perf_counter()-shared
        methods=[(backend,rule) for backend in ("cone","ballot") for rule in args.rules]
        perm=np.random.default_rng(2505+dev.NAMES.index(name)).permutation(len(methods))
        for repetition in range(args.repeats):
            for i in np.roll(perm,repetition):
                backend,rule=methods[i]
                best,info=solve(g,initial,path,backend,rule,args.seconds)
                rows.append(dict(dataset=name,task=task,seed=42,p=len(g),backend=backend,rule=rule,k=int(best.sum()),
                    repeat=repetition,teacher_s=teacher_s,upper_s=upper_s,relative_error=np.sqrt(max(0.,q.loss(g,best))),
                    support=" ".join(map(str,np.flatnonzero(best))),**info))
                pd.DataFrame(rows).to_csv(folder/"scores.csv",index=False)
        print(f"{name}: completed {len(methods)} arms in {perf_counter()-start:.2f}s",flush=True)
    (folder/"settings.json").write_text(json.dumps(vars(args),indent=2),encoding="utf8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--names",nargs="+",default=list(dev.NAMES),choices=dev.NAMES)
    cli.add_argument("--seconds",type=float,default=3.)
    cli.add_argument("--cpu",type=int,default=6)
    cli.add_argument("--run",required=True)
    cli.add_argument("--repeats",type=int,default=1)
    cli.add_argument("--rules",nargs="+",choices=["none","free","range","hot_none","hot_range"],
                     default=["none","free","range"])
    with threadpool_limits(1): run(cli.parse_args())
