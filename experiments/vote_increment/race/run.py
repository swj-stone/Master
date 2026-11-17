"""Evaluate prospective block proposals with native refits and two matched controls."""
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.anchored_vote.audit import score
from experiments.minpi_design.study import load
from experiments.minpi_replication import direct_gain as native
from experiments.split_vote.run import ORIGINAL, ROOT
from experiments.vote_increment import audit
from experiments.vote_increment.race.game import block_votes, nominate


def select(x,y,task,idx,seed,arm,design):
    start=perf_counter()
    fit=np.sort(idx['fit'])
    vote=np.sort(idx['vote'])
    valid=np.sort(idx['width'])
    dev=np.sort(np.r_[fit,vote,valid])
    full=np.arange(x.shape[1])
    p=len(full)
    def trained(rows,cols):
        return native.fit_native(x[np.ix_(rows,cols)],y[rows],task,seed,'cpu')
    def predict(model,rows,cols):
        return native.predict_native(model,x[np.ix_(rows,cols)],task)
    model=trained(fit,full)
    order=np.array(native.gain_order(model,p))
    full_score=native.validation_utility(y[valid],predict(model,valid,full),task)
    floor=full_score-.005
    scores={p:full_score}
    def observe(k):
        if k not in scores:
            cols=np.sort(order[:k])
            scores[k]=native.validation_utility(y[valid],predict(trained(fit,cols),valid,cols),task)
        return scores[k]
    tick=perf_counter()
    if arm=='gain':
        g=model.get_booster().get_score(importance_type='total_gain')
        mass=np.array([g.get(f'f{i}',0.) for i in order])
        u=native.validation_utility(y[vote],predict(model,vote,full),task)
        scale=max(u-(.5 if task=='classification' else 0.),0.)
        losses=scale*(1-(mass.cumsum()/mass.sum() if mass.sum() else np.arange(1,p+1)/p))
    else:
        rates=block_votes(model,x[vote],y[vote],task,order,seed)
        losses=rates.min(axis=0) if arm=='min' else rates.mean(axis=0)
    timing=perf_counter()-tick
    if design=='guard':
        cutoff=.005
        width=nominate(losses,cutoff,1,p)
        if observe(width)>=floor-1e-14:
            if width>1:
                observe(nominate(losses,4*cutoff,1,width-1))
        elif width<p-1:
            observe(nominate(losses,cutoff/4,width+1,p-1))
    else:
        raise ValueError(design)
    k=min(k for k,u in scores.items() if u>=floor-1e-14)
    cols=np.sort(order[:k])
    pred=predict(trained(dev,cols),idx['test'],cols)
    return dict(k=k,cols=cols.tolist(),fits=len(scores)+1,seconds=perf_counter()-start,vote_s=timing,
                guard=floor,trace=scores,validation=scores[k],losses=losses.tolist()),pred


def run(args):
    rawdir=ROOT/'results/vote_increment/race'/args.tag
    out=ROOT/'thesis-paper/evidence/vote_increment/race'/args.tag
    rawdir.mkdir(parents=True,exist_ok=False)
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i,name in enumerate(args.names):
        for seed in args.seeds:
            x,y,task,idx,held=load(name,seed)
            x=np.asarray(x,dtype=np.float32)
            frozen=[]
            for method in np.roll(['min','mean','gain'],i+args.seeds.index(seed)):
                rec,pred=select(x,y,task,idx,seed,method,args.design)
                f=rawdir/f'{name}-{seed}-{method}.npz'
                np.savez_compressed(f,indices=idx['test'],prediction=pred)
                rec.update(task=task,dataset=name,seed=seed,method=str(method),sha256=sha256(f.read_bytes()).hexdigest())
                f.with_suffix('.json').write_text(json.dumps(rec,allow_nan=False),encoding='utf8')
                frozen.append((rec,f))
                print(name,seed,method,rec['k'],rec['fits'],round(rec['seconds'],2),flush=True)
            truth=held()
            for rec,f in frozen:
                assert sha256(f.read_bytes()).hexdigest()==rec['sha256']
                with np.load(f) as saved:
                    np.testing.assert_array_equal(saved['indices'],idx['test'])
                    values=score(truth,saved['prediction'],task)
                rows.append({k:v for k,v in rec.items() if k not in ('cols','losses','trace')}|values)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'seeds.csv',index=False)
    pairs,sources=audit.checks(frame,args.names,args.seeds)
    pairs.to_csv(out/'gate.csv',index=False)
    print(sources.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--names',nargs='+',default=list(ORIGINAL))
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--tag',required=True)
    p.add_argument('--design',default='guard')
    args=p.parse_args()
    if os.name=='nt' and not ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.c_void_p(-1),1<<8):
        raise RuntimeError('Could not pin native timing to CPU 8')
    with threadpool_limits(1):
        run(args)
