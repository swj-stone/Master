"""Audit original PMLB folds, ranks, sizes, selectors and test predictions without project imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/context_study/pmlb"
RUN = ROOT / "results/thesis_benchmark/prospective-pmlb-contexts-r1"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def auc(y, p):
    if p.shape[1] == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, average="macro", multi_class="ovr")


def fit(x, y, seed):
    return XGBClassifier(n_jobs=1, random_state=seed, n_estimators=100, learning_rate=.1,
                         max_depth=4, tree_method="hist", device="cpu").fit(x, y)


def lasso(xt, yt, xr, yr, seed):
    scaler = StandardScaler().fit(xt)
    options = []
    for c in (.01, .1, 1., 10.):
        model = OneVsRestClassifier(LogisticRegression(l1_ratio=1, C=c, solver="liblinear",
                                    random_state=seed, max_iter=2000)).fit(scaler.transform(xt), yt)
        nonzero = np.logical_or.reduce([np.ravel(m.coef_) != 0 for m in model.estimators_])
        cols = np.flatnonzero(nonzero).tolist() or [int(xt.var(axis=0).argmax())]
        score = auc(yr, model.predict_proba(scaler.transform(xr)))
        options.append((score, -len(cols), -c, cols))
    return max(options)


def compact_gain(model, xt, xr, yr, seed):
    scores = model.get_booster().get_score(importance_type="gain")
    ranked = sorted((int(name[1:]) for name in scores), key=lambda i: (-scores[f"f{i}"], i)) or [0]
    totals = np.array(list(model.get_booster().get_score(importance_type="total_gain").values()))
    effective = totals.sum()**2 / (totals @ totals) if len(totals) and totals.any() else 1
    m = len(ranked)
    bgrows = np.random.default_rng(seed).choice(len(xt), min(512,len(xt)), replace=False)
    med = np.median(xt[bgrows],axis=0)
    def vector(pred):
        vals = [auc(yr,pred), np.mean(pred.argmax(axis=1)==yr)]
        if pred.shape[1] == 2 and min(np.bincount(yr)) <= len(yr)*.05:
            vals.append(average_precision_score(yr,pred[:,1]))
        return np.asarray(vals)
    reference = vector(model.predict_proba(xr))
    evaluated = {m: True}
    def feasible(k):
        if k not in evaluated:
            xm = np.broadcast_to(med, xr.shape).copy()
            xm[:, ranked[:k]] = xr[:, ranked[:k]]
            evaluated[k] = bool(np.all(vector(model.predict_proba(xm))+1e-12 >= .92*reference))
        return evaluated[k]
    initial = min(m,max(1,int(np.ceil(effective))))
    low, high = 0,m
    if feasible(initial):
        high = initial
        while high > 1 and len(evaluated) <= 12:
            half = max(high//2,1)
            if not feasible(half):
                low = half
                break
            high = half
    else:
        low = initial
        while low<m and len(evaluated)<=12:
            double = min(2*low,m)
            if feasible(double):
                high = double
                break
            low = double
    while high-low>1 and len(evaluated)<=12:
        mid = (low+high)//2
        if feasible(mid):
            high = mid
        else:
            low = mid
    return ranked, min(k for k,ok in evaluated.items() if ok)


def audit(name, committed_sha, wait_seconds):
    table = DATA/name/f"{name}.tsv.gz"
    manifest = read(RUN/"sources.json")["sources"][name]
    for rel, record in manifest.items():
        assert sha(table.parent/rel) == record["sha256"]
    pointer = dict(line.split(maxsplit=1) for line in table.with_suffix(".lfs").read_text().splitlines())
    assert pointer["oid"] == "sha256:"+sha(table) and int(pointer["size"])==table.stat().st_size
    frame = pd.read_csv(table, sep="\t")
    label = frame.pop("target").to_numpy()
    # Exact raw parsed values, retaining all columns; a duplicate is allowed to have another label.
    fingerprint = pd.util.hash_pandas_object(frame,index=False).to_numpy()
    for idx in frame.groupby(list(frame.columns), dropna=False, sort=False).indices.values():
        assert len(set(fingerprint[idx]))==1
    assert len(set(fingerprint))==len(frame.drop_duplicates())
    x = frame.to_numpy(dtype=np.float32)
    audited, sensitivity, sizes = [], [], []
    for fold,(dev,test) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261115).split(x,label)):
        folder = RUN/name/f"fold-{fold}"
        limit = time.monotonic()+wait_seconds
        while not (folder/"completed.json").exists() and time.monotonic()<limit:
            time.sleep(2)
        done, chosen = read(folder/"completed.json"),read(folder/"selection.json")
        assert done["protocol_sha256"] == committed_sha
        assert done["selection_sha256"]==sha(folder/"selection.json")
        assert done["predictions_sha256"]==sha(folder/"predictions.npz")
        seed=20421+fold
        fitidx, rankidx = train_test_split(dev,test_size=.25,stratify=label[dev],random_state=seed)
        for key, value in (("train",fitidx),("rank",rankidx),("test",test)):
            np.testing.assert_array_equal(chosen[key],value)
        classes, counts = np.unique(label[fitidx],return_counts=True)
        if len(classes)==2 and counts[0]<counts[1]:
            classes=classes[::-1]
        np.testing.assert_array_equal(classes,chosen["class_order"])
        y = pd.Categorical(label,categories=classes).codes
        selection_imputer = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[fitidx])
        xt,xr = selection_imputer.transform(x[fitidx]),selection_imputer.transform(x[rankidx])
        ranked,k = compact_gain(fit(xt,y[fitidx],seed),xt,xr,y[rankidx],seed)
        assert ranked==chosen["order"] and k==chosen["anchor_k"], (name,fold,"Gain")
        l1_score,_,minus_c,l1 = lasso(xt,y[fitidx],xr,y[rankidx],seed)
        np.testing.assert_allclose(l1_score,chosen["traces"]["l1"]["ranking_auc"])
        assert -minus_c==chosen["traces"]["l1"]["chosen_C"]
        expected_supports = {"gain":sorted(ranked[:k]),"full":list(range(x.shape[1])),
                 "matched_gain":sorted(ranked[:len(chosen["selected"]["min"])]),"l1":l1}
        for arm, expected in expected_supports.items():
            assert expected == chosen["selected"][arm],(name,fold,arm)
        final_imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
        d,e = final_imp.transform(x[dev]),final_imp.transform(x[test])
        new = ~np.isin(fingerprint[test],fingerprint[dev])
        sizes.append(dict(dataset=name,fold=fold,rows=len(y),test_rows=len(test),cross_fold_duplicates=int(sum(~new))))
        memo = {}
        with np.load(folder/"predictions.npz") as stored:
            np.testing.assert_array_equal(stored["y"],y[test])
            np.testing.assert_array_equal(stored["row_id"],test)
            for row in done["records"]:
                arm = row["arm"]
                cols = tuple(chosen["selected"][arm])
                assert len(cols)==len(set(cols))==row["k"] and list(cols)==json.loads(row["selected"])
                assert np.isclose(row["selection_s"]+row["final_s"],row["total_s"]) and row["final_s"]>0
                if cols not in memo:
                    memo[cols] = fit(d[:,cols],y[dev],seed).predict_proba(e[:,cols])
                pred=memo[cols]
                np.testing.assert_allclose(stored[arm],pred,rtol=0,atol=1e-7)
                acc=np.mean(pred.argmax(1)==y[test])
                np.testing.assert_allclose([auc(y[test],pred),acc],[row["auc"],row["accuracy"]],rtol=0,atol=1e-10)
                audited.append(dict(dataset=name,fold=fold,arm=arm,k=len(cols),auc=row["auc"],accuracy=acc,
                        independent_support=arm in expected_supports,max_prediction_error=float(np.max(abs(stored[arm]-pred)))))
                if len(set(y[test][new])) == len(classes):
                    sensitivity.append(dict(dataset=name,fold=fold,arm=arm,auc=auc(y[test][new],pred[new]),
                                           accuracy=np.mean(pred[new].argmax(1)==y[test][new]),test_rows=int(sum(new))))
    return audited, sensitivity, sizes


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds",type=int,default=0)
    opts=parser.parse_args()
    lock = read(RUN/"protocol-lock.json")
    for rel,digest in lock["code"].items():
        assert sha(ROOT/rel)==digest
    assert sha(ROOT/"configs/context_study/prospective.json")==lock["spec_sha256"]
    assert sha(RUN/"sources.json")==lock["sources_sha256"]
    outputs=[[],[],[]]
    with threadpool_limits(limits=1):
        for name in lock["design"]["datasets"]:
            findings=audit(name,sha(RUN/"protocol-lock.json"),opts.wait_seconds)
            for target, rows in zip(outputs,findings,strict=True):
                target.extend(rows)
            print(f"Verified {name}, {len(findings[0])} evaluations",flush=True)
    for rows, filename in zip(outputs,["pmlb-baselines.csv","pmlb-unique-test.csv","pmlb-duplicates.csv"],strict=True):
        pd.DataFrame(rows).to_csv(HERE/filename,index=False)
