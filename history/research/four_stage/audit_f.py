"""Check the actual F supports against a stable independent variance decomposition."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.four_stage.benchmark import ROOT,DATASETS,SEEDS,load
from research.feature_game_eval.audit_controls import effect_size
from research.conditional_dense.confirmation import panels
from research.conditional_dense.speech import utterances,ARCHIVE

RUNS=ROOT/"results/minpi-study"


def compare(effects, array, arms, key):
    rows=[]
    order=np.argsort(-effects,kind="stable")
    for arm in arms:
        chosen=np.sort(array[arm])
        k=len(chosen)
        correct=np.sort(order[:k])
        out=np.setdiff1d(correct,chosen)
        inside=np.setdiff1d(chosen,correct)
        gap=float(effects[out].max()-effects[inside].min()) if len(out) else 0.
        rows.append(dict(panel=key,arm=arm,k=k,n_different=len(out),max_effect_gap=gap))
    return rows


def run(breadth_folder):
    rows=[]
    for source in DATASETS:
        X,y,task,_=load(source)
        for seed in SEEDS:
            key=f"{source}-{seed}"
            with np.load(breadth_folder/(key+".npz")) as a:
                tr=a["train"]
                x=X[tr]
                if not np.isfinite(x).all():
                    mask=np.isfinite(x)
                    x=np.where(mask,x,np.where(mask,x,0.).sum(axis=0)/np.maximum(mask.sum(axis=0),1))
                rows.extend(compare(effect_size(x,y[tr],task),a,("F80","F_match"),key))
    for panel in panels():
        key=f"{panel.name}-{panel.fold}-{panel.seed}"
        with np.load(RUNS/"conditional-dense-quality-20261118a"/(key+".npz")) as a:
            rows.extend(compare(effect_size(panel.X,panel.y,"classification"),a,("F_cols","F_half_cols"),key))
    with ZipFile(ARCHIVE) as z:
        X,y=utterances(z.read("Train_Arabic_Digit.txt"),660)
    effects=effect_size(X,y,"classification")
    for f in sorted((RUNS/"conditional-speech-quality-20261118a").glob("*.npz")):
        with np.load(f) as a:
            rows.extend(compare(effects,a,("F_cols","F_half_cols"),f.stem))
    table=pd.DataFrame(rows)
    print(table[table.n_different>0].to_string(index=False))
    print("Verified",len(table),"support comparisons")
    table.to_csv(breadth_folder/"F-boundary-audit.csv",index=False)
    if (table.max_effect_gap>1e-9).any():
        raise ValueError("F support differs from the double-precision variance reference")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('breadth_folder',type=Path)
    with threadpool_limits(1):run(parser.parse_args().breadth_folder)
