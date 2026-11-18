"""Independent UCI spoken-Arabic-digit validation for the frozen dense selector."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from research.conditional_dense.confirmation import BASE, HERE, ROOT, SEEDS, Panel, fit, run_arm
from research.conditional_dense.shared_method import conditional_support

ARCHIVE=ROOT/'data/raw/uci-spoken-arabic-digit.zip'
SOURCE='https://archive.ics.uci.edu/static/public/195/spoken+arabic+digit.zip'
SHA256='3c8c83404289c49631605421830872be337d92d887e61798385aacbe9423010d'


def utterances(raw,per_class):
    blocks=re.split(rb'\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*', raw.strip())
    assert len(blocks)==per_class*10,(len(blocks),per_class)
    output=np.empty((len(blocks),208),dtype=np.float32)
    for i,block in enumerate(blocks):
        frames=np.fromstring(block,sep=' ',dtype=np.float64).reshape(-1,13)
        positions=np.linspace(0,len(frames)-1,16)
        lo=positions.astype(int);hi=np.minimum(lo+1,len(frames)-1)
        fraction=(positions-lo)[:,None]
        output[i]=((1-fraction)*frames[lo]+fraction*frames[hi]).ravel()
    assert np.isfinite(output).all()
    return output,np.repeat(np.arange(10),per_class)


def run(out,pass_id):
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==SHA256
    out.mkdir(parents=True,exist_ok=False)
    t=perf_counter()
    with ZipFile(ARCHIVE) as archive:
        X,y=utterances(archive.read('Train_Arabic_Digit.txt'),660)
        Xt,truth=utterances(archive.read('Test_Arabic_Digit.txt'),220)
    preparation=perf_counter()-t
    rows=[];extra=[]
    for seed in SEEDS:
        panel=Panel('spoken_arabic_digit',1,seed,X,y,Xt,truth,None)
        key=f'{panel.name}-1-{seed}'
        k=len(conditional_support(X,y,seed=seed)[0])
        store={}
        for arm in BASE:
            cols,pred,decision,_,_=run_arm(panel,arm,k)
            store[arm+'_cols']=cols;store[arm+'_pred']=pred;store[arm+'_decision']=decision
        np.savez_compressed(out/(key+'.npz'),truth=truth,**store)
        order=np.random.default_rng([seed,pass_id,14783]).permutation(BASE)
        for repeat in range(3 if pass_id else 1):
            for arm in np.roll(order,repeat):
                cols,pred,decision,select,total=run_arm(panel,arm,k)
                for suffix,res in zip(('cols','pred','decision'),(cols,pred,decision)):
                    np.testing.assert_array_equal(res,store[arm+'_'+suffix])
                rows.append(dict(panel=key,n=len(y),p=208,target_domain=True,method=arm,k=len(cols),
                    score=balanced_accuracy_score(truth,pred),select_ms=select,total_ms=total,repeat=repeat))
        if not pass_id:
            for draw in range(20):
                rng=np.random.default_rng([54319,seed,1,draw])
                cols=np.sort(rng.permutation(X.shape[1])[:k]);pred,_,_=fit(panel,cols,'cholesky')
                extra.append(dict(panel=key,draw=draw,k=k,score=balanced_accuracy_score(truth,pred),
                                  support_hash=hashlib.sha256(cols.tobytes()).hexdigest()))
        print(key,k,flush=True)
    pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
    if not pass_id:pd.DataFrame(extra).to_csv(out/'random-quality.csv',index=False)
    paths=[Path(__file__),HERE/'shared_method.py',HERE/'confirmation.py',HERE/'protocol.md',HERE/'speech-protocol.md']
    state=dict(pass_id=pass_id,source_url=SOURCE,source_sha256=SHA256,
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},panels=[
        dict(panel=f'spoken_arabic_digit-1-{s}',target_domain=True,p=208,train=6600,test=2200) for s in SEEDS])
    (out/'protocol.json').write_text(json.dumps(state,indent=2))
    (out/'preparation.json').write_text(json.dumps(dict(shared_extraction_seconds=preparation)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pass-id',type=int,required=True,choices=(0,1,2))
    args=parser.parse_args()
    with threadpool_limits(1):run(args.out,args.pass_id)
