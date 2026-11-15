"""Replay UCI HAR held-out scores, conventional supports and final fits without importing the study runner."""
import io
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"results/context_review/external-har-locked-r1"
RAW = ROOT/"data/context_study/uci-har/uci-har-240.zip"


def load():
    with ZipFile(RAW) as outer, ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as inner:
        arrays = {}
        for phase in ("train", "test"):
            for variable in ("X","y","subject"):
                with inner.open(f"UCI HAR Dataset/{phase}/{variable}_{phase}.txt") as f:
                    arrays[phase,variable] = np.loadtxt(f, dtype=np.float32 if variable=="X" else np.int64)
    return arrays


def fitted(x,y):
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1, n_jobs=1,
                         random_state=20421, tree_method="hist", device="cpu", importance_type="gain").fit(x,y)


def manual_rfe(x, y, k, cache):
    live = np.arange(x.shape[1])
    while len(live)>k:
        key = tuple(live)
        if key not in cache:
            model = fitted(x[:,live],y)
            importance = np.square(model.feature_importances_)
            cache[key] = np.argsort(importance,kind="stable")
        removable = min(max(1,int(.2*x.shape[1])),len(live)-k)
        live = np.delete(live, cache[key][:removable])
    return list(map(int,live))


def auc_from_ranks(y,p):
    auc = []
    for cls in range(p.shape[1]):
        mask = y==cls
        n = int(mask.sum())
        positive_rank_sum = rankdata(p[:,cls].astype(np.float64))[mask].sum(dtype=np.float64)
        auc.append((positive_rank_sum-n*(n+1)/2)/(n*(len(y)-n)))
    return np.mean(auc)


def main():
    if not (RUN/"scored.json").exists():
        raise ValueError("The primary run must finish before replay")
    arr = load()
    x, xt = arr["train","X"], arr["test","X"]
    y, yt = arr["train","y"]-1, arr["test","y"]-1
    g, gt = arr["train","subject"], arr["test","subject"]
    train, rank = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20421).split(x,y,g))
    archived_split = json.loads((RUN/"split.json").read_bytes())
    assert train.tolist()==archived_split["fit"] and rank.tolist()==archived_split["rank"]
    assert not (set(g[train])&set(g[rank]) or set(g)&set(gt))
    data = pd.read_csv(RUN/"scores.csv").set_index("arm")
    by_person = pd.read_csv(RUN/"by-subject.csv").set_index(["arm","subject"])
    k_min = int(data.loc["antichain","k"])
    support_file = lambda a: json.loads((RUN/a/"support.json").read_bytes())
    direct_start = perf_counter()
    train_model = fitted(x[train],y[train])
    inner_order = train_model.get_booster().get_score(importance_type="gain")
    inner_order = sorted((int(key[1:]) for key in inner_order), key=lambda j: (-inner_order[f"f{j}"],j))
    prefix = sorted(inner_order[:k_min])
    clean_select_s = perf_counter()-direct_start
    direct_start = perf_counter()
    clean_probability = fitted(x[:,prefix],y).predict_proba(xt[:,prefix])
    clean_refit_s = perf_counter()-direct_start
    assert prefix == support_file("gain_donor")["selected"]
    np.testing.assert_array_equal(clean_probability,np.load(RUN/"gain_donor/predictions.npz")["probabilities"])
    pd.DataFrame([dict(arm="gain_donor",k=k_min,selection_s=clean_select_s,refit_predict_s=clean_refit_s,
                       total_s=clean_select_s+clean_refit_s)]).to_csv(RUN/"timing-conditional-gain.csv",index=False)
    global_model = fitted(x,y)
    average_gain = global_model.get_booster().get_score(importance_type="gain")
    global_order = sorted(range(x.shape[1]), key=lambda j: (-average_gain.get(f"f{j}",0),j))
    final_cache, elimination_cache, output = {}, {}, []
    for arm, entry in data.iterrows():
        recorded = support_file(arm)["selected"]
        support_reconstructed = True
        if arm.startswith("budget_gain_"):
            k = ceil(x.shape[1]*int(arm.split("_")[-1])/100)
            reconstructed = sorted(global_order[:k])
        elif arm.startswith("rfe_"):
            k = k_min if arm=="rfe_donor" else ceil(x.shape[1]*int(arm.split("_")[-1])/100)
            reconstructed = manual_rfe(x,y,k,elimination_cache)
        elif arm=="gain_donor":
            reconstructed = sorted(inner_order[:k_min])
        elif arm=="full":
            reconstructed = list(range(x.shape[1]))
        else:
            support_reconstructed=False
            reconstructed = recorded
        assert recorded == reconstructed
        key = tuple(recorded)
        if key not in final_cache:
            clf = global_model if len(recorded)==x.shape[1] else fitted(x[:,recorded],y)
            final_cache[key] = clf.predict_proba(xt[:,recorded])
        p = final_cache[key]
        frozen = np.load(RUN/arm/"predictions.npz")["probabilities"]
        np.testing.assert_array_equal(p,frozen)
        correct = p.argmax(axis=1)==yt
        subject_acc = [correct[gt==s].mean() for s in np.unique(gt)]
        np.testing.assert_allclose(subject_acc, [by_person.loc[arm,s].accuracy for s in np.unique(gt)],atol=1e-14,rtol=0)
        values = [auc_from_ranks(yt,p), correct.mean(), np.mean(subject_acc)]
        saved = [entry.pooled_auc,entry.pooled_accuracy,entry.mean_subject_accuracy]
        error = np.max(np.abs(np.asarray(values)-saved))
        assert error < 1e-12
        output.append(dict(arm=arm,k=len(recorded),support_reconstructed=support_reconstructed,
                            max_probability_difference=float(np.max(np.abs(p-frozen))),max_metric_difference=error))
        print(f"Replayed {arm}",flush=True)
    pd.DataFrame(output).to_csv(RUN/"audit.csv",index=False)
    random_root = RUN.parent/"external-har-random-columns"
    if (random_root/"scores.csv").is_file():
        selection = json.loads((random_root/"supports.json").read_bytes())["supports"]
        outputs = pd.read_csv(random_root/"scores.csv").set_index("draw")
        anchor = json.loads((RUN/"gain/support.json").read_bytes())["selected"]
        flags = []
        with np.load(random_root/"predictions.npz") as pfile:
            for i, pool in enumerate(["all_original_columns","compact_gain_anchor"]):
                universe = np.arange(x.shape[1]) if i==0 else sorted(anchor)
                for d in range(20):
                    name=f"{pool}_{d:02}"
                    derived=list(map(int,sorted(np.random.default_rng(np.random.SeedSequence([918273,i,d])).choice(
                        universe,k_min,replace=False))))
                    assert derived==selection[name]
                    pp=pfile[name]
                    correct=pp.argmax(axis=1)==yt
                    independent=[auc_from_ranks(yt,pp),correct.mean(),np.mean([correct[gt==s].mean() for s in np.unique(gt)])]
                    original=outputs.loc[name,["pooled_auc","pooled_accuracy","mean_subject_accuracy"]].to_numpy(dtype=float)
                    error=np.max(np.abs(np.asarray(independent)-original))
                    assert error<1e-12
                    flags.append(dict(draw=name,reconstructed=True,max_metric_difference=error))
        pd.DataFrame(flags).to_csv(random_root/"audit.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
