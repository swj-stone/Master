"""Frozen, train-only conditional-vote evaluation across dense real sources."""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.grouped_solver_cost_audit import fit
from experiments.minpi_backward_confirmation import evaluate
from experiments.minpi_stable_shift import get_shift
from experiments import budget_minpi_deployments as deploy, fast_minpi_five_source as five
from experiments.signal_vote_sources import ASSETS
from research.conditional_dense.shared_method import conditional_support
from research.comparators.mrmr_mid import mrmr_mid_order

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=('conditional','full','F','random','grouped','zero','F_half','random_half')
SEEDS=(1129,2027,3469)

@dataclass
class Panel:
    name:str
    fold:int
    seed:int
    X:np.ndarray
    y:np.ndarray
    Xt:np.ndarray
    gold:np.ndarray
    units:np.ndarray|None
    task:str='classification'


def panels():
    for fold,(seed,X,y,groups,Xt,gold,units,task,_) in enumerate(deploy.configurations("isolet"),1):
        if fold>1:
            yield Panel('isolet',fold,seed,X,y,Xt,gold,None)
    for name in ('har','gas'):
        X,y,groups,Xt,gold,units,task=get_shift(name)
        assert not np.intersect1d(groups,units).size
        for seed in SEEDS:
            yield Panel(name,1,seed,X,y,Xt,gold,units)
    path=ROOT/'data/raw/uci-multiple-features.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ASSETS[path.name]
    with ZipFile(path) as z:
        parts=[]
        for name in ('fac','fou','kar','mor','pix','zer'):
            with z.open('mfeat-'+name) as stream:parts.append(np.loadtxt(stream,dtype=np.float32))
    X=np.column_stack(parts)
    y=np.repeat(np.arange(10),200)
    assert X.shape==(2000,649)
    unique=~pd.Series([five.digest(row) for row in X]).duplicated(keep=False).to_numpy()
    for fold,(tr,te) in enumerate(StratifiedKFold(3,shuffle=True,random_state=18112026).split(X,y),1):
        tr,te=tr[unique[tr]],te[unique[te]]
        yield Panel('multiple_features',fold,SEEDS[fold-1],X[tr],y[tr],X[te],y[te],None)
    X,y,task,groups,source,duplicate=five.inputs('digits')
    for fold,(tr,te) in enumerate(five.folds(X,y,task,groups),1):
        if fold==1:continue
        with np.load(five.ARCHIVE/f'digits-{fold}-split.npz') as z:
            np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['test'],te)
        assert not np.intersect1d(duplicate[tr],duplicate[te]).size
        yield Panel('digits',fold,1129,X[tr],y[tr],X[te],y[te],None)


def zero_vote(X,y,k,seed):
    # Same sampling scheme and scale as conditional_support; zero peers.
    rng=np.random.default_rng(seed)
    classes,ids=np.unique(y,return_inverse=True)
    if len(y)>768:
        first=[rng.choice(np.flatnonzero(ids==i)) for i in range(len(classes))]
        pool=np.setdiff1d(np.arange(len(y)),first,assume_unique=True)
        rows=np.sort(np.r_[first,rng.choice(pool,768-len(first),replace=False)])
    else: rows=np.arange(len(y))
    Z=np.array(X[rows],dtype=np.float32,order='C')
    Z-=Z.mean(axis=0,dtype=np.float64)
    norm=np.linalg.norm(Z,axis=0)
    Z/=np.where(norm>0,norm/np.sqrt(np.float32(len(rows))),1.)
    Y=(ids[rows,None]==np.arange(len(classes))).astype(np.float32)
    Y-=Y.mean(axis=0)
    gain=((Z.T@Y)**2).sum(axis=1)/((Z**2).sum(axis=0)+np.sqrt(len(rows)))
    return np.sort(np.argsort(-gain,kind='stable')[:k])


def run_arm(panel,name,k):
    X,y,seed=panel.X,panel.y,panel.seed
    start=perf_counter_ns()
    if name=='conditional':
        columns,info=conditional_support(X,y,seed=seed)
    elif name=='mrmr':
        columns=np.sort(mrmr_mid_order(X,y,k)[0])
    elif name=='zero':
        columns=zero_vote(X,y,k,seed)
    elif name in ('F','random','F_half','random_half','grouped','full'):
        method=name.removesuffix('_half').lower()
        columns=choose(method,X,y,'classification',seed,(X.shape[1]+1)//2 if name.endswith('_half') else k)
    else:raise ValueError(name)
    selected=perf_counter_ns()
    pred,decision,_=fit(panel,columns,'cholesky')
    elapsed=(perf_counter_ns()-start)/1e6
    return columns,pred,decision,(selected-start)/1e6,elapsed


def main(out,pass_id):
    out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),HERE/'shared_method.py',HERE/'protocol.md',ROOT/'experiments/grouped_solver_cost_audit.py',
           ROOT/'experiments/compare_predictive_minpi.py', ROOT/'research/comparators/mrmr_mid.py']
    state=dict(pass_id=pass_id,seed=SEEDS,threads=1,raw_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},panels=[])
    rows=[];random_quality=[]
    for panel in panels():
        k=len(conditional_support(panel.X,panel.y,seed=panel.seed)[0])
        key=f'{panel.name}-{panel.fold}-{panel.seed}'
        methods=BASE+(('mrmr',) if panel.name=='digits' else ())
        data=dict(panel=key,n=len(panel.y),p=panel.X.shape[1],target_domain=(len(np.unique(panel.y))>=3 and
                  128<=panel.X.shape[1]<=1024 and len(panel.y)>=2*panel.X.shape[1]))
        state['panels'].append(dict(**data,X=five.digest(panel.X),y=five.digest(panel.y),Xt=five.digest(panel.Xt),gold=five.digest(panel.gold)))
        reference={};cache={}
        for name in methods:
            cols,pred,decision,_,_=run_arm(panel,name,k)
            reference[name]=(cols,pred,decision)
            cache[name+'_cols']=cols;cache[name+'_pred']=pred;cache[name+'_decision']=decision
        np.savez_compressed(out/(key+'.npz'),truth=panel.gold,**cache)
        order=np.random.default_rng([panel.seed,pass_id,14783]).permutation(methods)
        repeats=3 if pass_id else 1
        for repeat in range(repeats):
            for name in np.roll(order,repeat):
                cols,pred,decision,sel,total=run_arm(panel,name,k)
                for a,b in zip((cols,pred,decision),reference[name]):np.testing.assert_array_equal(a,b)
                score,_=evaluate(panel.name,panel.gold,pred,panel.units,'classification')
                rows.append(dict(**data,method=name,k=len(cols),score=score,select_ms=sel,total_ms=total,repeat=repeat))
        if pass_id==0:
            for draw in range(20):
                rng=np.random.default_rng([54319,panel.seed,panel.fold,draw])
                cols=np.sort(rng.permutation(panel.X.shape[1])[:k])
                pred,_,_=fit(panel,cols,'cholesky')
                score,_=evaluate(panel.name,panel.gold,pred,panel.units,'classification')
                random_quality.append(dict(panel=key,draw=draw,k=k,score=score,support_hash=five.digest(cols)))
        print(key,k,flush=True)
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    if pass_id==0: pd.DataFrame(random_quality).to_csv(out/'random-quality.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--pass-id',required=True,type=int,choices=(0,1,2))
    args=parser.parse_args()
    with threadpool_limits(1): main(args.out,args.pass_id)
