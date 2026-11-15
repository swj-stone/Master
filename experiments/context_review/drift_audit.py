"""Freshly fit each forward support from scikit-learn's separate LIBSVM loader and audit scores."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import f_classif
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/context_review/drift-locked"
DATA=ROOT/"data/context_study/uci-gas-drift/uci224.zip"


def model():
    return XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.1,n_jobs=1,random_state=20421,
                         tree_method="hist",device="cpu")


def calculate(y,p):
    hard=p.argmax(axis=1)
    recall=[]
    auc=[]
    for c in np.unique(y):
        mask=y==c
        n=mask.sum()
        recall.append((hard[mask]==c).mean())
        if len(y)>n:
            auc.append((rankdata(p[:,c])[mask].sum()-n*(n+1)/2)/(n*(len(y)-n)))
    return np.array([np.mean(recall),(y==hard).mean(),np.mean(auc)])


def main():
    inputs={}
    with ZipFile(DATA) as archive:
        for b in range(1,11):
            f,y=load_svmlight_file(BytesIO(archive.read(f"Dataset/batch{b}.dat")),zero_based=False,n_features=128)
            inputs[b]=(f.toarray().astype(np.float32),y.astype(int)-1)
    xa=np.vstack([inputs[b][0] for b in range(1,8)])
    ya=np.concatenate([inputs[b][1] for b in range(1,8)])
    xe=np.vstack([inputs[b][0] for b in range(8,11)])
    assert np.isfinite(xa).all() and np.isfinite(xe).all() and len(xa)+len(xe)==13910
    rec=json.loads((OUT/"supports.json").read_bytes())
    cols=rec["supports"]
    f,_=f_classif(xa,ya)
    for k in (13,32):
        name="anova_at_10pct" if k==13 else "anova_at_25pct"
        assert sorted(np.argsort(f,kind="stable")[-k:])==cols[name]
    ranked=rec["rank_order"]
    assert cols["compact_gain"]==sorted(ranked[:len(cols["compact_gain"])])
    assert cols["gain_at_vote_width"]==sorted(ranked[:len(cols["vote"])])
    live=[c for c in ranked if c in cols["vote"]]
    assert cols["weak_gain"]==sorted(live[:-1] if len(live)>1 else live)
    # Independently replay RFE by eliminating the smallest squared feature importances.
    for name in ("rfe_at_vote_width","rfe_at_challenger_width"):
        k=len(cols[name])
        retained=np.arange(xa.shape[1])
        while len(retained)>k:
            importances=model().fit(xa[:,retained],ya).feature_importances_**2
            remove=min(int(.2*xa.shape[1]),len(retained)-k)
            retained=np.sort(retained[np.argsort(importances,kind="stable")[remove:]])
        np.testing.assert_array_equal(retained,cols[name])
    scores=pd.read_csv(OUT/"by-batch.csv").set_index(["arm","batch"])
    guard={}
    inner_train=np.vstack([inputs[b][0] for b in range(1,7)])
    inner_y=np.concatenate([inputs[b][1] for b in range(1,7)])
    with np.load(OUT/"guard-probabilities.npz") as ga:
        for arm in ("compact_gain","vote","maximin","weak_gain"):
            s=cols[arm]
            pg=model().fit(inner_train[:,s],inner_y).predict_proba(inputs[7][0][:,s])
            np.testing.assert_array_equal(pg,ga[arm])
            guard[arm]=calculate(inputs[7][1],pg)[1:]
    for arm in ("vote","maximin","weak_gain"):
        ref=guard["compact_gain"]
        ok=bool(np.all(ref-guard[arm] <= .03*np.abs(ref)+1e-12))
        assert ok==rec["gate_pass"][arm]
        fallback=cols["vote_guard"] if arm!="vote" else cols["compact_gain"]
        assert cols[arm+"_guard"]==(cols[arm] if ok else fallback)
    audited=[]
    by_support={}
    with np.load(OUT/"predictions.npz") as probs:
        for arm,s in cols.items():
            if tuple(s) not in by_support:
                by_support[tuple(s)]=model().fit(xa[:,s],ya).predict_proba(xe[:,s])
            prediction=by_support[tuple(s)]
            np.testing.assert_array_equal(prediction,probs[arm])
            for b in range(8,11):
                p=prediction[probs["batch"]==b]
                m=calculate(inputs[b][1],p)
                expected=scores.loc[(arm,b),["recall","accuracy","auc"]].to_numpy()
                delta=float(np.max(np.abs(expected-m)))
                assert delta<1e-12
                audited.append(dict(arm=arm,batch=b,rows=len(p),k=len(s),channels=len({i//8 for i in s}),
                                    max_probability_error=0,max_metric_error=delta))
    pd.DataFrame(audited).to_csv(OUT/"audit.csv",index=False)
    print(f"{len(audited)} arm-by-batch assessments, {len(by_support)} distinct fresh external refits, "
          f"maximum score error {max(r['max_metric_error'] for r in audited):.2g}")


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
