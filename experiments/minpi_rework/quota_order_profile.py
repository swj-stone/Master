"""Count global and cardinality-specific implications on the historic development sources."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import quota_order_study as run
from experiments.minpi_rework.quota_order import order_edges,width_votes


def profile():
    dev,q=run.dev,run.q
    jobs=dev.legacy.jobs()
    rows=[]
    for name in dev.NAMES:
        x,y,task,idx=dev.load(name,jobs)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        model=dev.fit_additive(x[pool],y[pool],task,"linear")
        a=model.parts(x[idx["width"]])-model.center
        matrix=a.transpose(0,2,1).reshape(-1,x.shape[1])
        g=matrix.T@matrix/max(1e-100,np.sum(matrix.sum(1)**2))
        d=2*g.sum(1)-g.diagonal()
        best=run.incumbent(g); _,path=q.greedy(g)
        w,rhs=q.initial_ballots(g,path,q.RHO)
        lo,hi=q.lower_bound(w,rhs),int(best.sum())
        allr=order_edges(g,d,lo,hi,"range")
        ar=width_votes(g,d,lo,hi)
        counts=ar.sum(axis=(1,2))
        rows.append(dict(name=name,lo=lo,hi=hi,global_pairs=int(allr.sum()),conditional_union=int(ar.any(0).sum()),
             at_lower=int(counts[0]),at_upper=int(counts[-1]),mean_per_width=float(counts.mean())))
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(1): profile()
