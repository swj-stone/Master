"""Quality-only BNS comparison and random draws; no timing claim."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from sklearn.metrics import balanced_accuracy_score
from research.text_peer.bench import CACHE, HERE, original
from research.text_peer.bns import bns


def main():
    directory=original.ROOT/"results/minpi-study/text-peer-log-20261118a"
    records=[]
    for meta in json.loads((directory/"panels.json").read_text(encoding="utf8")):
        key=meta["key"]
        X=sparse.load_npz(CACHE/(key+"-X.npz"));Xt=sparse.load_npz(CACHE/(key+"-Xt.npz"))
        with np.load(CACHE/(key+"-labels.npz")) as z:y,gold=z["y"],z["gold"]
        with np.load(directory/(key+".npz")) as z:
            width=len(z["candidate_cols"]);score=balanced_accuracy_score(gold,z["candidate_pred"])
        for draw in range(-1,20):
            cols=bns(X,y,width) if draw<0 else np.sort(np.random.default_rng([708891,meta["seed"],draw]).choice(X.shape[1],width,replace=False))
            pred=original.fit_predict(X,y,Xt,cols,"classification",meta["seed"],False)
            records.append(dict(dataset=meta["dataset"],fold=meta["fold"],k=width,method="BNS" if draw==-1 else "random_draw",
                draw=draw,baseline_ba=balanced_accuracy_score(gold,pred),candidate_ba=score,
                support_hash=original.digest(cols),pred_hash=original.digest(pred)))
    result=pd.DataFrame(records); result.to_csv(HERE/"bns-random-quality.csv",index=False)
    b=result[result.method=="BNS"];print(b.groupby("dataset").agg(bns=("baseline_ba","mean"),candidate=("candidate_ba","mean")))


if __name__=="__main__":
    with threadpool_limits(1):main()
