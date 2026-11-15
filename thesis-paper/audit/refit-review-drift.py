"""Independent model and metric checks for the frozen forward gas-drift test."""
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from rounds.pmlb_baselines import ROOT, compact_gain, fit, read, sha

OUT = ROOT / "results/context_review/drift-locked"
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/context_study/uci-gas-drift/uci224.zip"
SEED = 20421


def metrics(y, p):
    labels = np.unique(y)
    assert p.shape == (len(y), 6) and set(labels) == set(range(6))
    counts = confusion_matrix(y, p.argmax(axis=1), labels=labels)
    recall = np.diag(counts) / counts.sum(axis=1)
    return dict(accuracy=np.trace(counts)/len(y), auc=roc_auc_score(y,p,multi_class="ovr"),
                recall=recall.mean()), recall


def audit():
    assert sha(DATA) == read(OUT/"protocol.json")["archive_sha256"]
    with ZipFile(DATA) as archive:
        batches = []
        for b in range(1,11):
            with archive.open(f"Dataset/batch{b}.dat") as source:
                m,v = load_svmlight_file(source,n_features=128,dtype=np.float32,zero_based=False)
            matrix = m.toarray()
            assert np.isfinite(matrix).all()
            batches.append((matrix,v.astype(int)-1))
    x, y = (np.concatenate(items) for items in zip(*batches))
    batch = np.repeat(np.arange(1,11), [len(v) for _,v in batches])
    spec = read(OUT/"supports.json")
    supports = spec["supports"]
    order, k = compact_gain(fit(x[batch<=5],y[batch<=5],SEED),
                            x[batch<=5],x[batch==6],y[batch==6],SEED)
    assert order==spec["rank_order"] and sorted(order[:k])==supports["compact_gain"]
    for arm in ["rfe_at_vote_width","rfe_at_challenger_width","anova_at_10pct","anova_at_25pct"]:
        width = len(supports[arm])
        if arm.startswith("rfe"):
            model = XGBClassifier(n_jobs=1,random_state=SEED,n_estimators=100,learning_rate=.1,
                                  max_depth=4,tree_method="hist",device="cpu",importance_type="gain")
            selector = RFE(model,n_features_to_select=width,step=.20)
        else:
            selector = SelectKBest(f_classif,k=width)
        selector.fit(x[batch<=7],y[batch<=7])
        np.testing.assert_array_equal(supports[arm],selector.get_support(indices=True))
    @lru_cache(None)
    def predict(last, columns):
        valid = batch>7 if last==7 else batch==7
        return fit(x[batch<=last][:,columns],y[batch<=last],SEED).predict_proba(x[valid][:,columns])
    with np.load(OUT/"guard-probabilities.npz") as inner:
        guard = {}
        for arm in ["compact_gain","vote","maximin","weak_gain"]:
            fresh = predict(6,tuple(supports[arm]))
            np.testing.assert_array_equal(fresh,inner[arm])
            scores,_ = metrics(y[batch==7],fresh)
            guard[arm] = np.array([scores["auc"],scores["accuracy"]])
        for arm in ["vote","maximin","weak_gain"]:
            passed = bool(np.all(guard[arm]+1e-12 >= .97*guard["compact_gain"]))
            assert passed == spec["gate_pass"][arm]
            fallback = "vote" if arm!="vote" and spec["gate_pass"]["vote"] else "compact_gain"
            assert supports[arm+"_guard"] == supports[arm if passed else fallback]
    rows,classes = [],[]
    with np.load(OUT/"predictions.npz") as saved:
        future = batch[batch>7]
        np.testing.assert_array_equal(saved["batch"],future)
        for arm,columns in supports.items():
            p = predict(7,tuple(columns))
            np.testing.assert_array_equal(p,saved[arm])
            observed = y[batch>7]
            for b in (8,9,10):
                ix = future==b
                scores,rs = metrics(observed[ix],p[ix])
                rows.append(dict(arm=arm,batch=b,k=len(columns),n=int(ix.sum()),**scores))
                for c,r in enumerate(rs):
                    classes.append(dict(arm=arm,batch=b,archive_class=c+1,
                                        observations=int((observed[ix]==c).sum()),recall=float(r)))
    result = pd.DataFrame(rows).set_index(["arm","batch"]).sort_index()
    archived = pd.read_csv(OUT/"by-batch.csv").set_index(["arm","batch"]).sort_index()
    np.testing.assert_allclose(result,archived[result.columns],rtol=0,atol=1e-12)
    mean = result.groupby("arm").mean().drop(columns="n")
    archived = pd.read_csv(OUT/"summary.csv").set_index("arm").sort_index()
    np.testing.assert_allclose(mean,archived[mean.columns],rtol=0,atol=1e-12)
    pd.DataFrame(classes).to_csv(HERE/"refit-review-drift-classes.csv",index=False)
    print(f"Verified {len(rows)} batch summaries, {len(supports)} arms and {predict.cache_info().misses} distinct refits.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit()
