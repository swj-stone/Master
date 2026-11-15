"""Independently refit RFECV and group-disjoint PMLB controls on their recorded folds."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from pmlb_baselines import DATA, HERE, ROOT, RUN, auc, compact_gain, fit, lasso, read, sha


def all_class_folds(ids, y, groups, n, seed):
    count = len(np.unique(y[ids]))
    for attempt in range(seed, seed + 128):
        cv = StratifiedGroupKFold(n, shuffle=True, random_state=attempt)
        folds = [(ids[a], ids[b]) for a, b in cv.split(ids, y[ids], groups[ids])]
        valid = [len(np.unique(y[a])) == count == len(np.unique(y[b])) for a, b in folds]
        if n == 3 and all(valid):
            return folds, attempt
        if n == 4 and any(valid):
            return folds[valid.index(True)], attempt
    raise ValueError("No valid partitions")


def replay_path(x, y, dev, test, fold, chosen):
    seed = 20421 + fold
    inner = list(StratifiedKFold(3, shuffle=True, random_state=31173 + fold).split(dev, y[dev]))
    for (a, b), (saved_a, saved_b) in zip(inner, chosen["inner"], strict=True):
        np.testing.assert_array_equal(dev[a], saved_a)
        np.testing.assert_array_equal(dev[b], saved_b)
        assert set(test).isdisjoint(dev[a]) and set(test).isdisjoint(dev[b])
    wrapper = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                            XGBClassifier(random_state=seed, n_estimators=100, learning_rate=.1, max_depth=4,
                                          tree_method="hist", n_jobs=1, device="cpu", importance_type="gain"))
    metric = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr"
    assert metric == chosen["scoring"]
    search = RFECV(wrapper, step=.2, min_features_to_select=1, cv=inner, scoring=metric, n_jobs=1,
                   importance_getter="named_steps.xgbclassifier.feature_importances_").fit(x[dev], y[dev])
    for key, scores in chosen["cv_results"].items():
        np.testing.assert_allclose(search.cv_results_[key], scores, atol=1e-12, rtol=0)
    return np.flatnonzero(search.support_).tolist()


def audit(study):
    source = {"rfecv":"rfecv", "grouped":"duplicate-groups", "antichain_matched":"antichain-matched"}[study]
    other = RUN.parent / f"prospective-pmlb-{source}-r1"
    lock = read(other / "protocol-lock.json")
    assert lock["primary_sha256"] == sha(RUN / "protocol-lock.json")
    assert lock.get("source_sha256", lock.get("code_sha256")) == sha(ROOT / f"experiments/context_study/{study}.py")
    spec = ROOT / f"configs/context_study/{study.replace('_','-')}.json"
    assert lock.get("specification_sha256", lock.get("design_sha256")) == sha(spec)
    assert read(other / "failures.json") == []
    datasets = read(RUN / "protocol-lock.json")["design"]["datasets"]
    if study == "grouped":
        original = pd.read_csv(HERE / "pmlb-duplicates.csv").groupby("dataset").sum(numeric_only=True)
        datasets = [n for n in datasets if original.at[n, "cross_fold_duplicates"] > .1 * original.at[n, "test_rows"]]
        assert datasets == lock["tasks"]
    rows = []
    for name in datasets:
        src = DATA / name / f"{name}.tsv.gz"
        assert sha(src) == read(RUN / "sources.json")["sources"][name][src.name]["sha256"]
        frame = pd.read_csv(src, sep="\t")
        label = frame.pop("target").to_numpy()
        groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        assert len(np.unique(groups)) == len(frame.drop_duplicates())
        x = frame.to_numpy(dtype=np.float32)
        idx = np.arange(len(x))
        if study != "grouped":
            outer = StratifiedKFold(3, shuffle=True, random_state=20261115).split(x, label)
        else:
            outer, outer_seed = all_class_folds(idx, label, groups, 3, 20261115)
        tested = []
        for fold, (dev, test) in enumerate(outer):
            tested.extend(test)
            seed = 20421 + fold
            folder = other / name / f"fold-{fold}"
            done, chosen = read(folder / "completed.json"), read(folder / "selection.json")
            assert sha(folder / "selection.json") == done["selection_sha256"]
            assert sha(folder / "predictions.npz") == done["predictions_sha256"]
            assert sha(other / "protocol-lock.json") == done["protocol_sha256"]
            np.testing.assert_array_equal(test, chosen["test"])
            assert set(test).isdisjoint(dev) and set(test).union(dev) == set(idx)
            if study != "grouped":
                primary = RUN / name / f"fold-{fold}" / "selection.json"
                assert sha(primary) == chosen.get("primary_selection_sha256", chosen.get("donor_selection_sha256"))
                stored = read(primary)
                classes = stored["class_order"]
                y = pd.Categorical(label, categories=classes).codes
                assert set(dev) == set(stored["train"]).union(stored["rank"])
                if study == "rfecv":
                    s = replay_path(x, y, dev, test, fold, chosen)
                else:
                    donor = stored["selected"]["antichain"]
                    assert donor == chosen["donor"]
                    s = sorted(stored["order"][:len(donor)])
                supports = {done["record"]["arm"]: s}
                assert s == chosen["selected"]
                records = [done["record"]]
            else:
                (train, rank), inner_seed = all_class_folds(dev, label, groups, 4, seed)
                for vals, key in [(rank, "rank"), (train, "train")]:
                    np.testing.assert_array_equal(vals, chosen[key])
                for a, b in [(train, rank), (rank, test), (train, test)]:
                    assert not np.intersect1d(groups[a], groups[b]).size
                assert (outer_seed, inner_seed) == (chosen["outer_seed"], chosen["inner_seed"])
                classes, counts = np.unique(label[train], return_counts=True)
                if len(classes) == 2 and counts[0] < counts[1]:
                    classes = classes[::-1]
                np.testing.assert_array_equal(classes, chosen["class_order"])
                y = pd.Categorical(label, categories=classes).codes
                imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
                a, b = imp.transform(x[train]), imp.transform(x[rank])
                order, k = compact_gain(fit(a, y[train], seed), a, b, y[rank], seed)
                assert order == chosen["order"] and k == chosen["anchor_k"]
                _, _, minus_c, l1 = lasso(a, y[train], b, y[rank], seed)
                assert -minus_c == chosen["traces"]["l1"]["chosen_C"]
                supports = {"gain": sorted(order[:k]), "l1": l1, "full": list(range(x.shape[1]))}
                for arm, s in supports.items():
                    assert s == chosen["selected"][arm]
                records = done["records"]
            final = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[dev])
            d, e = final.transform(x[dev]), final.transform(x[test])
            with np.load(folder / "predictions.npz") as saved:
                np.testing.assert_array_equal(y[test], saved["y"])
                np.testing.assert_array_equal(test, saved["row_id"])
                for record in records:
                    arm = record["arm"]
                    sel = supports.get(arm)
                    independent = sel is not None
                    if not independent:
                        sel = chosen["selected"][arm]
                    assert len(sel) == len(set(sel)) == record["k"] and json.loads(record["selected"]) == sel
                    if study == "antichain_matched":
                        assert record["conditional_refit_s"] > 0
                    else:
                        assert np.isclose(record["selection_s"] + record["final_s"], record["total_s"]) and record["final_s"] > 0
                    p = fit(d[:, sel], y[dev], seed).predict_proba(e[:, sel])
                    err = np.abs(p - saved[arm]).max()
                    np.testing.assert_allclose(p, saved[arm], rtol=0, atol=1e-7)
                    score, acc = auc(y[test], p), np.mean(p.argmax(1) == y[test])
                    np.testing.assert_allclose([score, acc], [record["auc"], record["accuracy"]], rtol=0, atol=1e-10)
                    rows.append(dict(dataset=name, fold=fold, arm=arm, k=len(sel), auc=score, accuracy=acc,
                                     independent_support=independent, max_prediction_error=err))
        np.testing.assert_array_equal(np.sort(tested), idx)
        print(f"Verified {study} {name}", flush=True)
    merged = pd.DataFrame(rows).merge(pd.read_csv(other/"per-fold.csv"), on=["dataset", "fold", "arm"],
                                      suffixes=("", "_record"), validate="one_to_one")
    assert len(rows) == len(merged) == (84 if study == "grouped" else 42)
    for key in ["k", "auc", "accuracy"]:
        np.testing.assert_allclose(merged[key], merged[key+"_record"], rtol=0, atol=1e-10)
    pd.DataFrame(rows).to_csv(HERE / f"{study}-baselines.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["rfecv", "grouped", "antichain_matched"])
    with threadpool_limits(limits=1):
        audit(parser.parse_args().study)
