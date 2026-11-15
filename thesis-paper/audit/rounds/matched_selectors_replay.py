"""Reconstruct the two donated-width baselines without importing either selector driver."""
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from pmlb_baselines import DATA, HERE, ROOT, RUN, auc, fit, read, sha


OUTPUT = ROOT / "results/context_study/matched-selectors"


def greedy_rfe(train, target, seed, width):
    """Reproduce fractional-step RFE and the stable original-index tie order."""
    kept = np.arange(train.shape[1])
    step = max(1, int(.2 * kept.size))
    while kept.size > width:
        fitted = fit(train[:, kept], target, seed)
        strength = np.square(fitted.feature_importances_)
        drop = np.argsort(strength, kind="stable")[:min(step, kept.size - width)]
        kept = np.delete(kept, drop)
    return kept


def best_f(train, target, width):
    # Native F statistic; replay the public selector's stable score ordering explicitly.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Features .* are constant")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores, _ = f_classif(train, target)
    finite_floor = np.finfo(scores.dtype).min
    scores = np.where(np.isnan(scores), finite_floor, scores)
    return np.sort(np.argsort(scores, kind="stable")[-width:])


def checked(source, name, field):
    path = source/name
    assert sha(path) == read(source/"completed.json")[field], path
    return path


def one(dataset, source_manifest):
    table = DATA/dataset/f"{dataset}.tsv.gz"
    assert sha(table) == source_manifest[dataset][table.name]["sha256"]
    frame = pd.read_csv(table, sep="\t")
    labels = frame.pop("target").to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    records = []
    for fold, (dev, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=20261115).split(x, labels)):
        seed = 20421 + fold
        fit_idx, rank_idx = train_test_split(dev, test_size=.25, stratify=labels[dev], random_state=seed)
        classes, counts = np.unique(labels[fit_idx], return_counts=True)
        if classes.size == 2 and counts[0] < counts[1]:
            classes = classes[::-1]
        y = pd.Categorical(labels, categories=classes).codes
        donor_dir = RUN/dataset/f"fold-{fold}"
        donor_file = checked(donor_dir, "selection.json", "selection_sha256")
        donor = read(donor_file)
        for name, ids in [("train", fit_idx), ("rank", rank_idx), ("test", test), ("class_order", classes)]:
            np.testing.assert_array_equal(donor[name], ids)
        coded = donor["selected"]["antichain"]
        k = len(coded)
        imputation = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[dev])
        train = imputation.transform(x[dev])
        valid = imputation.transform(x[test])
        supports = {"rfe_same_k": greedy_rfe(train, y[dev], seed, k), "anova_same_k": best_f(train, y[dev], k)}
        baseline = OUTPUT/dataset/f"fold-{fold}"
        deadline = time.monotonic() + 3600
        while not (baseline/"completed.json").exists() and time.monotonic() < deadline:
            time.sleep(1)
        selected = read(checked(baseline, "selection.json", "supports"))
        assert selected["donor"] == sha(donor_file)
        np.testing.assert_array_equal(selected["development"], dev)
        np.testing.assert_array_equal(selected["test"], test)
        receipt = read(baseline/"completed.json")
        assert receipt["protocol"] == sha(OUTPUT/"protocol.json")
        scores = {row["arm"]: row for row in receipt["rows"]}
        with np.load(checked(baseline, "predictions.npz", "probabilities")) as pred:
            np.testing.assert_array_equal(pred["row_id"], test)
            np.testing.assert_array_equal(pred["y"], y[test])
            for arm, cols in supports.items():
                np.testing.assert_array_equal(cols, selected["selected"][arm])
                assert len(cols) == k == scores[arm]["k"]
                refit = fit(train[:, cols], y[dev], seed)
                probs = refit.predict_proba(valid[:, cols])
                error = float(np.max(np.abs(probs-pred[arm])))
                np.testing.assert_array_equal(probs, pred[arm])
                a, acc = auc(y[test], probs), accuracy_score(y[test], probs.argmax(axis=1))
                np.testing.assert_allclose([a, acc], [scores[arm]["auc"], scores[arm]["accuracy"]], atol=1e-12, rtol=0.)
                records.append(dict(dataset=dataset, fold=fold, arm=arm, k=k, width=x.shape[1],
                                    shared_with_coded=len(set(cols).intersection(coded)), test_rows=len(test),
                                    probabilities_error=error, auc_error=abs(a-scores[arm]["auc"]),
                                    accuracy_error=abs(acc-scores[arm]["accuracy"]), auc=a, accuracy=acc))
    return records


def main():
    protocol = read(OUTPUT/"protocol.json")
    assert protocol["matched_selectors.json"] == sha(ROOT/"configs/context_study/matched_selectors.json")
    data = read(RUN/"sources.json")["sources"]
    names = read(ROOT/"configs/context_study/prospective.json")["datasets"]
    rec = []
    for dataset in names:
        rec.extend(one(dataset, data))
        print(f"{dataset}: {len(rec)} / 84 independently checked", flush=True)
    out = pd.DataFrame(rec)
    assert len(out)==84 and not out.duplicated(["dataset","fold","arm"]).any()
    out.to_csv(HERE/"matched-selectors-replay.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

