"""A development run on the first historical ISOLET and Digits configurations."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from experiments.shared_grouped_development import panels
from experiments.grouped_solver_cost_audit import fit
from experiments.compare_predictive_minpi import choose
from research.conditional_dense.method import conditional_support
from research.comparators.mrmr_mid import mrmr_mid_order

def run():
    rows=[]
    with threadpool_limits(1):
        for q in panels():
            n,p=q.X.shape
            for mode in ('half','auto'):
                k=(p+1)//2 if mode=='half' else None
                own,info=conditional_support(q.X,q.y,k,seed=q.seed)
                k=len(own)
                for name in ('conditional','full','grouped','f','random','mrmr'):
                    if name=='mrmr' and q.name!='digits':continue
                    t=time.perf_counter()
                    if name=='conditional':cols=conditional_support(q.X,q.y,k if mode=='half' else None,seed=q.seed)[0]
                    elif name=='mrmr':cols=np.sort(mrmr_mid_order(q.X,q.y,k)[0])
                    else:cols=choose(name,q.X,q.y,q.task,q.seed,k)
                    sel=time.perf_counter()
                    pred,_,_=fit(q,cols,'cholesky')
                    end=time.perf_counter()
                    rows.append(dict(data=q.name,mode=mode,method=name,k=len(cols),ba=balanced_accuracy_score(q.gold,pred),sel_ms=1000*(sel-t),total_ms=1000*(end-t)))
    result=pd.DataFrame(rows)
    print(result.to_string(index=False))
    result.to_csv(Path(__file__).with_name('development.csv'),index=False,mode='x')

if __name__=='__main__':run()
