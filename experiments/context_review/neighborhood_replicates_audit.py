"""Check new exchange forecasts against fresh support-wise pipelines and mask scores."""
from io import BytesIO
from itertools import product
from zipfile import ZipFile

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import filter_neighborhood as first
from experiments.context_review.forest_audit import estimator, verify_masks
from experiments.context_review.neighborhood_replicates import DEST, ROOT
from experiments.context_review.receiver_transport import SOURCE


def audit():
    lock = bg.read(DEST/"prediction-lock.json")
    old = bg.read(first.DEST/"prediction-lock.json")
    fits = masks = 0
    pe = me = 0.
    with ZipFile(DEST/"predictions.zip") as new, threadpool_limits(limits=1):
        for key,rec in lock["folds"].items():
            base = old["folds"][key]
            order = base["order"]
            width = len(base["supports"][0])
            cut = len(base["supports"][1])
            prefix = order[:cut]
            pairs = list(product(prefix,order[cut:width]))
            for j,seed in enumerate(lock["seeds"]):
                rng = np.random.default_rng(np.random.SeedSequence([seed,rec["id"],rec["fold"]]))
                ids = np.arange(len(pairs)) if len(pairs)<=7 else rng.choice(len(pairs),7,replace=False)
                sets = {tuple(sorted((set(prefix)-{pairs[i][0]})|{pairs[i][1]})) for i in ids}
                sets.add(tuple(sorted(prefix)))
                assert sets=={tuple(rec["supports"][i]) for i in rec["slates"][j][1:]}
            src = ROOT/"results/context_review"/SOURCE[rec["panel"]]/key.split("/",1)[1]/"inputs.npz"
            with np.load(src) as source, np.load(BytesIO(new.read(key+".npz"))) as archive:
                inputs = {k:source[k] for k in source.files}
                x,y,g = inputs["x"],inputs["y"],inputs["guard"]
                inside = np.sort(np.r_[inputs["train"],inputs["proposal"]])
                for i in range(len(base["supports"]),len(rec["supports"])):
                    cols = rec["supports"][i]
                    for phase,train,rows in (("g",inside,x[g]),("o",np.arange(len(y)),inputs["x_outer"])):
                        est = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),estimator(20421+rec["fold"]))
                        p = est.fit(x[train][:,cols],y[train]).predict_proba(rows[:,cols])
                        pe = max(pe,float(np.max(np.abs(p-archive[f"{phase}{i}"]))))
                        fits += 1
                maskn,err = verify_masks(inputs,rec,{"choices":{"gain":{"order":order}}},archive["tensor"])
                masks += maskn
                me = max(me,float(err))
            print(f"Verified {key}, {fits} additional support fits",flush=True)
    assert pe<=1e-12 and me<=1e-12
    bg.store(DEST/"audit.json",dict(folds=len(lock["folds"]),draws=len(lock["seeds"]),
                                    new_support_fits=fits,unique_masks=masks,max_probability_error=pe,
                                    max_weighted_metric_error=me))


if __name__=="__main__":
    audit()
