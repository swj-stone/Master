"""Development-only screen from archived native validation refits. No timings or test scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_replication.direct_gain import fit_native,gain_order,predict_native,validation_utility
from experiments.split_vote.run import ORIGINAL,ROOT,load
from experiments.vote_increment.path import bracket,search


def run():
    rows=[]
    dest=ROOT/'thesis-paper/evidence/vote_increment/path/validation'
    dest.mkdir(parents=True,exist_ok=True)
    for name in ORIGINAL:
        for seed in (42,123,456,789,1024):
            x,y,task,idx=load(name,seed)
            tr=np.sort(idx['fit'])
            va=np.sort(np.r_[idx['vote'],idx['width']])
            model=fit_native(x[tr],y[tr],task,seed)
            order=gain_order(model,x.shape[1])
            saved=json.loads((ROOT/f'results/split_vote/six-paired/{name}-{seed}/exhaustive/receipt.json').read_bytes())['record']
            if saved['gain_order']!=order:
                raise ValueError(f'Gain order mismatch {name}/{seed}')
            curve=saved['width_curve']
            full=validation_utility(y[va],predict_native(model,x[va],task),task)
            np.testing.assert_allclose(full,curve[-1],atol=1e-10,rtol=0)
            for arm in ('min','mean','gain'):
                profile=search.loss_profile(model,order,arm)
                for version,selector in [('endpoints',search.search),('interior',bracket.search)]:
                    k,trace=selector(profile,lambda w:curve[w-1])
                    rows.append(dict(dataset=name,seed=seed,version=version,method=arm,k=k,
                                     trace=json.dumps(trace),fits=len(trace)+1,validation=curve[k-1]))
            print(name,seed,flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(dest/'decisions.csv',index=False)
    for version,r in df.groupby('version'):
        widths=r.pivot(index=['dataset','seed'],columns='method',values='k')
        fits=r.pivot(index=['dataset','seed'],columns='method',values='fits')
        possible=np.logical_and.reduce([(widths['min']<widths[c])|((fits['min']<fits[c])&(widths['min']<=widths[c]+np.maximum(1,.05*widths[c]))) for c in ('mean','gain')])
        print(version, pd.Series(possible,index=widths.index).groupby(level=0).sum().to_dict())


if __name__=='__main__':
    with threadpool_limits(1):
        run()
