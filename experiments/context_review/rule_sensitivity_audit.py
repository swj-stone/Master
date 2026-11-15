"""Independent raw-data checks for the new learner and expanded bootstrap panels."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_study.data import DATA, ROOT, OUT as PRIMARY, config

OUT = ROOT/"results/context_review/rule-sensitivity"
DONOR = ROOT/"results/context_review/refit-target"


def read(path):
    return json.loads(path.read_bytes())


def weights(y, seed):
    labels, counts = np.unique(y, return_counts=True)
    quota = np.zeros_like(counts)
    budget = min(256, len(y))
    for j, i in enumerate(np.argsort(counts, kind="stable")):
        quota[i] = min(counts[i], budget//(len(counts)-j))
        budget -= quota[i]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 659]))
    ids = np.sort(np.concatenate([rng.choice(np.flatnonzero(y==c), size=q, replace=False)
                                   for c,q in zip(labels,quota,strict=True)]))
    extra = np.random.default_rng(np.random.SeedSequence([seed, 659, 8181]))
    result = np.zeros((12, len(ids)))
    for c,size in zip(labels,counts,strict=True):
        loc = np.flatnonzero(y[ids]==c)
        q = len(loc)
        mass = size/len(y)
        result[0,loc] = mass/q
        for panel in range(1,12):
            generator = rng if panel < 4 else extra
            result[panel, loc] = mass*np.bincount(generator.choice(q, size=q), minlength=q)/q
    return ids, result


def linear(x, y, seed):
    scale = StandardScaler().fit(x.astype("float64"))
    model = LogisticRegression(C=1, max_iter=2000, solver="lbfgs", tol=.0001, random_state=seed)
    model.fit(scale.transform(x.astype("float64")), y)
    return lambda z: model.predict_proba(scale.transform(z.astype("float64")))


def scores(y, prob, w, rare):
    classes = prob.shape[1]
    if classes==2:
        auc = roc_auc_score(y, prob[:,1], sample_weight=w)
    else:
        auc = np.mean([roc_auc_score(y==k, prob[:,k], sample_weight=w) for k in range(classes)])
    vals = [auc, np.average(prob.argmax(axis=1)==y, weights=w)]
    if rare:
        vals.append(average_precision_score(y, prob[:,1], sample_weight=w))
    return np.array(vals)


def audit(name, fold):
    suffix = Path(name)/f"fold-{fold}"
    source = read(PRIMARY/suffix/"selection.json")
    design = read(DONOR/suffix/"selection.json")
    completed = read(OUT/suffix/"completed.json")
    table = pd.read_csv(DATA/name/f"{name}.tsv.gz", sep="\t")
    y = pd.Categorical(table.pop("target"), categories=source["class_order"]).codes
    raw = table.to_numpy(dtype=np.float32)
    train, rank, test = (np.array(source[k]) for k in ("train", "rank", "test"))
    dev = np.sort(np.concatenate((train, rank)))
    seed = 20421+fold
    rare = len(source["class_order"])==2 and min(np.bincount(y[rank]))<=.05*len(rank)
    supports = [design["anchor"], *design["supports"]]
    outer_error, mask_error = 0., 0.
    with np.load(OUT/suffix/"logistic-outer.npz") as archive:
        np.testing.assert_array_equal(archive["row_id"], test)
        np.testing.assert_array_equal(archive["y"], y[test])
        for i,columns in enumerate(supports):
            data = raw[:,columns]
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(data[dev])
            # The original column projection supplies Fortran-contiguous views to the scaler.
            fitted = linear(np.asfortranarray(imputer.transform(data[dev])), y[dev], seed)
            prob = fitted(np.asfortranarray(imputer.transform(data[test])))
            archived = archive["anchor" if i==0 else f"s{i-1}"]
            np.testing.assert_allclose(prob, archived, rtol=0, atol=1e-12)
            outer_error = max(outer_error,float(np.max(abs(prob-archived))))
            current = scores(y[test], prob, np.ones(len(test)), rare)
            if i==0:
                reference = current
                continue
            loss = reference-current
            np.testing.assert_allclose(loss, completed["outcomes"]["logistic"]["loss"][i-1], rtol=0, atol=1e-12)
            tol = np.array([.03,.03,.05][:len(loss)])
            assert bool(np.any(loss>tol*abs(reference)+1e-12))==completed["outcomes"]["logistic"]["failure"][i-1]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw[train])
    xt = imputer.transform(raw[train])
    ids, w = weights(y[rank], seed)
    xr = imputer.transform(raw[rank[ids]])
    bgrows = np.random.default_rng(seed).choice(len(xt), min(len(xt),512), replace=False)
    bg = np.median(xt[bgrows], axis=0)
    family = design["contexts"]["separating"]
    ext = family[1] if len(family)>2 else family[-1]
    with np.load(OUT/suffix/"votes.npz") as archive:
        np.testing.assert_array_equal(archive["rows"], rank[ids])
        np.testing.assert_array_equal(archive["weights"], w)
        for learner in ("xgb","logistic"):
            if learner=="xgb":
                model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1, device="cpu",
                                      tree_method="hist", n_jobs=1, random_state=seed).fit(xt,y[train])
                predict = model.predict_proba
            else:
                predict = linear(xt,y[train],seed)
            for s in (design["anchor"], design["supports"][0]):
                mask = sorted(set(s)|set(ext))
                masked = np.tile(bg, (len(xr),1))
                masked[:,mask] = xr[:,mask]
                p = predict(masked)
                actual = np.array([scores(y[rank[ids]],p,panel,rare) for panel in w])
                present = np.isin(np.arange(xt.shape[1]),mask)
                index, = np.flatnonzero((archive[learner+"_masks"]==present).all(axis=1))
                ref = archive[learner+"_scores"][index]
                np.testing.assert_allclose(actual,ref,rtol=0,atol=1e-12)
                mask_error = max(mask_error,float(np.max(abs(actual-ref))))
    return dict(dataset=name, fold=fold, logistic_outer_refits=len(supports),
                independently_scored_mask_panels=48, max_outer_probability_error=outer_error,
                max_rank_metric_error=mask_error)


def main():
    records = []
    with threadpool_limits(limits=1):
        for name, fold in product(config()["datasets"],range(3)):
            records.append(audit(name,fold))
            print(f"Verified {name} fold {fold}",flush=True)
    frame = pd.DataFrame(records)
    assert len(frame)==42 and frame.logistic_outer_refits.sum()==746
    frame.to_csv(OUT/"independent-audit.csv",index=False)
    print(frame[["logistic_outer_refits","independently_scored_mask_panels"]].sum().to_string())
    print(frame[["max_outer_probability_error","max_rank_metric_error"]].max().to_string())


if __name__=="__main__":
    main()
