"""Independently verify the alternate ANOVA rankings, exchange slates, refits and masked metrics."""
from io import BytesIO
from itertools import product
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.filter_neighborhood import DEST, ROOT
from experiments.context_review.forest_audit import estimator, measure, passes, verify_masks
from experiments.context_review.receiver_transport import SOURCE


def verify_slate(x,y,rec):
    imputed = SimpleImputer(strategy="median",keep_empty_features=True).fit_transform(x).astype(np.float64)
    center = imputed.mean(axis=0)
    between = np.zeros(len(center))
    within = np.zeros(len(center))
    classes = np.unique(y)
    for label in classes:
        block = imputed[y==label]
        groupmean = block.mean(axis=0)
        between += len(block) * (groupmean-center)**2
        within += ((block-groupmean)**2).sum(axis=0)
    with np.errstate(invalid="ignore",divide="ignore"):
        f = (between/(len(classes)-1)) / (within/(len(y)-len(classes)))
    f[np.ptp(imputed,axis=0)==0] = np.nan
    order = np.array(rec["order"])
    ranked = f[order]
    valid = ~np.isnan(ranked)
    assert not np.any(np.diff(valid.astype(int))>0)
    left,right = ranked[:-1][valid[1:]],ranked[1:][valid[1:]]
    assert np.all((left>=right) | np.isclose(left,right,rtol=1e-8,atol=1e-9))
    # Independently enumerate actual column pairs, sample their offsets and
    # reconstruct the retained sets from the documented seed and membership.
    width = min(len(order), max(2,int(np.ceil(len(order)/2))))
    keep = width-max(1,width//4)
    anchor = order[:width].tolist()
    assert sorted(anchor)==rec["supports"][0]
    if width==1:
        assert len(rec["supports"])==1
        return
    prefix,unused = anchor[:keep],anchor[keep:]
    pairs = list(product(prefix,unused))
    draw = np.random.default_rng(np.random.SeedSequence([87201,rec["id"],rec["fold"]]))
    indices = range(len(pairs)) if len(pairs)<=7 else draw.choice(len(pairs),7,replace=False)
    reconstructed = {tuple(sorted(prefix))}
    for i in indices:
        a,b = pairs[i]
        reconstructed.add(tuple(sorted((set(prefix)-{a})|{b})))
    rank = {col:i for i,col in enumerate(anchor)}
    sorted_sets = sorted(reconstructed,key=lambda s:(sum(rank[c] for c in s),s))
    assert list(map(tuple,rec["supports"][1:]))==sorted_sets


def audit():
    lock = bg.read(DEST/"prediction-lock.json")
    candidates = pd.read_csv(DEST/"candidates.csv").set_index(["panel","dataset","fold","candidate"])
    fits, comparisons, mask_count = 0,0,0
    prediction_error = mask_error = 0.
    started = perf_counter()
    with ZipFile(DEST/"predictions.zip") as forecasts, ZipFile(ROOT/"thesis-paper/evidence/receiver/outcomes.zip") as truths, threadpool_limits(limits=1):
        for key,rec in lock["folds"].items():
            source = ROOT/"results/context_review"/SOURCE[rec["panel"]]/key.split("/",1)[1]
            with np.load(source/"inputs.npz") as loaded, np.load(BytesIO(forecasts.read(key+".npz"))) as test, np.load(BytesIO(truths.read(key+".npz"))) as truth:
                inputs = {k:loaded[k] for k in loaded.files}
                x,y = inputs["x"],inputs["y"]
                tr,pr,g = (inputs[k] for k in ("train","proposal","guard"))
                verify_slate(x[tr],y[tr],rec)
                endpoints = []
                for idx,cols in enumerate(rec["supports"]):
                    pair = []
                    for group,train,evaluation,labels in (("g",np.sort(np.r_[tr,pr]),x[g],y[g]),
                        ("o",np.arange(len(y)),inputs["x_outer"],truth["y"])):
                        pipeline = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                                                 estimator(20421+rec["fold"]))
                        model = pipeline.fit(x[train][:,cols],y[train])
                        probs = model.predict_proba(evaluation[:,cols])
                        error = float(np.abs(probs-test[f"{group}{idx}"]).max())
                        prediction_error = max(prediction_error,error)
                        assert error<=1e-12
                        pair.append(measure(labels,probs,"ap" in rec["metrics"]))
                        fits += 1
                    endpoints.append(pair)
                    if idx:
                        row = candidates.loc[(rec["panel"],rec["dataset"],rec["fold"],idx)]
                        assert (passes(pair[0],endpoints[0][0]),passes(pair[1],endpoints[0][1]))==(row.guard,row.outer)
                        comparisons += 2
                # The forest audit consumes only the supplied tail ordering.
                n,err = verify_masks(inputs,rec,{"choices":{"gain":{"order":rec["order"]}}},test["tensor"])
                mask_count += n
                mask_error = max(mask_error,float(err))
            print(f"Verified {key}, {fits} support fits, {perf_counter()-started:.1f}s",flush=True)
    bg.store(DEST/"audit.json",dict(folds=len(lock["folds"]),independent_support_fits=fits,pass_decisions=comparisons,
                                   max_probability_difference=prediction_error,independent_masks=mask_count,
                                   max_weighted_metric_difference=mask_error))


if __name__ == "__main__":
    audit()
