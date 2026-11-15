"""Rebuild forest refits column by column and check masked metrics using sklearn scoring."""
from io import BytesIO
import json
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from experiments.context_review.forest_predict import DEST, ROOT, SOURCE
from experiments.context_review.receiver_audit import measure, passes
from experiments.context_study.benchmark import store
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_witness import _sample


def estimator(seed):
    return RandomForestClassifier(n_estimators=100, bootstrap=True, criterion="gini",
                                  max_features="sqrt", n_jobs=1, random_state=seed)


def weighted_metrics(y, p, weights, names):
    metrics = []
    for weight in weights:
        auc = roc_auc_score(y,p[:,1],sample_weight=weight) if p.shape[1] == 2 else roc_auc_score(
            y,p,multi_class="ovr",average="macro",sample_weight=weight)
        result = dict(accuracy=accuracy_score(y,p.argmax(axis=1),sample_weight=weight),auc=auc)
        if "ap" in names:
            result["ap"] = average_precision_score(y,p[:,1],sample_weight=weight)
        metrics.append([result[k] for k in names])
    return np.asarray(metrics)


def verify_masks(inputs, receipt, select, stored):
    supports = receipt["supports"]
    if len(supports)==1:
        assert stored.shape[1] == 0
        return 0, 0.
    x,y = inputs["x"],inputs["y"]
    train,proposal = inputs["train"],inputs["proposal"]
    seed = 20421+receipt["fold"]
    replacement = SimpleImputer(strategy="median",keep_empty_features=True)
    xt = replacement.fit_transform(x[train])
    xp = replacement.transform(x[proposal])
    m = estimator(seed).fit(xt,y[train])
    choices = np.random.default_rng(seed).choice(len(xt),min(512,len(xt)),replace=False)
    bg = np.median(xt[choices],axis=0)
    task = "binary" if len(m.classes_)==2 else "multiclass"
    rows,weights = _sample(y[proposal],task,np.random.SeedSequence([seed,659]))
    family = separating_contexts(select["choices"]["gain"]["order"][len(supports[0]):])
    expected = {}
    error = 0.
    for i,cols in enumerate(supports):
        for j,context in enumerate(family):
            keep = tuple(sorted(set(cols) | set(context)))
            if keep not in expected:
                visible = np.isin(np.arange(xt.shape[1]),keep)
                p = m.predict_proba(np.where(visible,xp[rows],bg))
                expected[keep] = weighted_metrics(y[proposal][rows],p,weights,receipt["metrics"])
            error = max(error,np.max(np.abs(stored[i,j]-expected[keep])))
    assert error<1e-12, error
    return len(expected), float(error)


def audit():
    started = perf_counter()
    lock = json.loads((DEST/"prediction-lock.json").read_bytes())
    data = pd.read_csv(DEST/"candidates.csv").query("voter=='forest' and receiver=='forest'").set_index(
        ["panel","dataset","fold","index"])
    count, metrics_count, masks, biggest, mask_error = 0,0,0,0.,0.
    with ZipFile(DEST/"predictions.zip") as pred, ZipFile(ROOT/"thesis-paper/evidence/receiver/outcomes.zip") as truth, \
         threadpool_limits(limits=1):
        for key, rec in lock["folds"].items():
            src = ROOT/"results/context_review"/SOURCE[rec["panel"]]/key.split("/",1)[1]
            select = json.loads((src/"selection.json").read_bytes())
            with np.load(src/"inputs.npz") as src_data, np.load(BytesIO(pred.read(key+".npz"))) as archived, \
                 np.load(BytesIO(truth.read(key+".npz"))) as target:
                inputs = {k:src_data[k] for k in src_data.files}
                x, y, outer = inputs["x"], inputs["y"], inputs["x_outer"]
                guard = inputs["guard"]
                inner = np.sort(np.r_[inputs["train"],inputs["proposal"]])
                endpoints = []
                for i, cols in enumerate(rec["supports"]):
                    pair = []
                    for phase, train, evaluate, labels in (
                            ("g",inner,x[guard],y[guard]),("o",np.arange(len(y)),outer,target["y"])):
                        pipe = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                                             estimator(20421+rec["fold"]))
                        pipe.fit(x[train][:,cols],y[train])
                        p = pipe.predict_proba(evaluate[:,cols])
                        err = float(np.abs(p-archived[f"{phase}{i}"]).max())
                        biggest = max(biggest,err)
                        assert err <= 1e-12,(key,phase,i,err)
                        pair.append(measure(labels,p,"ap" in rec["metrics"]))
                        count += 1
                    endpoints.append(pair)
                    if i:
                        row = data.loc[(rec["panel"],rec["dataset"],rec["fold"],i)]
                        go = passes(pair[0],endpoints[0][0])
                        oo = passes(pair[1],endpoints[0][1])
                        assert (go,oo,go and oo) == (row.guard,row.outer,row.joint)
                        metrics_count += 2
                n,err = verify_masks(inputs,rec,select,archived["tensor"])
                masks += n
                mask_error = max(mask_error,err)
            print(f"Audited {key}; {count} reference fits; elapsed {perf_counter()-started:.1f}s",flush=True)
    result = dict(folds=len(lock["folds"]), independently_rebuilt_support_fits=count,
                  max_probability_error=biggest, candidate_guard_outer_pass_decisions=metrics_count,
                  independently_scored_forest_masks=masks, max_weighted_metric_error=mask_error)
    store(DEST/"audit.json",result)
    print(result)


if __name__ == "__main__":
    audit()
