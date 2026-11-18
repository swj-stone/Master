"""Evaluate fixed feature budgets on the official, subject-disjoint UCI HAR split."""
import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from minppi_mofs.exact_minpi import pivotal_rates, witness_margins


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/subject-transfer/uci-har-240.zip"
DIGEST = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
OUTPUT = ROOT / "results/minpi-study/subject-transfer"
PROTOCOL = ROOT / "docs/minpi-study/transfer-protocol.md"
SEEDS = (17, 29, 43)
BUDGETS = (1, 5, 10, 25, 50)
TIERS = (0, 4, 16)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_har(check_only):
    if sha256(ARCHIVE) != DIGEST:
        raise ValueError("UCI HAR archive SHA-256 mismatch")
    with zipfile.ZipFile(ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as data:
            def load(file, dtype):
                with data.open("UCI HAR Dataset/" + file) as stream:
                    return np.loadtxt(stream, dtype=dtype)
            features = data.read("UCI HAR Dataset/features.txt").decode().splitlines()
            if [int(row.split()[0]) for row in features] != list(range(1, 562)):
                raise ValueError("Unexpected feature schema")
            parts = {}
            for split, size in (("train", 7352), ("test", 2947)):
                y = load(f"{split}/y_{split}.txt", np.int8)
                subjects = load(f"{split}/subject_{split}.txt", np.int8)
                if len(y) != size or len(subjects) != size:
                    raise ValueError("Unexpected official split size")
                for subject in np.unique(subjects):
                    if set(y[subjects == subject]) != set(range(1, 7)):
                        raise ValueError("Subject does not contain all six activities")
                X = None if check_only else load(f"{split}/X_{split}.txt", np.float32)
                if X is not None and (X.shape != (size, 561) or not np.isfinite(X).all()):
                    raise ValueError("Unexpected UCI HAR design matrix")
                parts[split] = X, y, subjects
    training = set(parts["train"][2])
    test = set(parts["test"][2])
    if len(training) != 21 or len(test) != 9 or training & test or training | test != set(range(1, 31)):
        raise ValueError("Expected distinct sets of 21 training and 9 test subjects")
    return parts, features


def rank(scores):
    return np.argsort(-np.asarray(scores), kind="stable")


def rankings(X, y, seed):
    margins = witness_margins(X, y, "classification", seed)
    rates = pivotal_rates(np.sign(margins).astype(np.int8), TIERS)
    score_f = f_classif(X.astype(np.float64), y)[0]
    trees = ExtraTreesClassifier(n_estimators=64, max_features="sqrt", n_jobs=1, random_state=seed)
    trees.fit(X, y)
    return {
        "exact": rank(rates.min(axis=0)),
        "mean": rank(rates.mean(axis=0)),
        "k16": rank(rates[-1]),
        "vote": rank(rates[0]),
        "margin": rank(margins.mean(axis=0)),
        "f": rank(np.nan_to_num(score_f, nan=-np.inf)),
        "forest": rank(trees.feature_importances_),
        "random": np.random.default_rng(seed).permutation(X.shape[1]),
    }, rates


def model_predictions(X, Xt, y, cols):
    training = np.ascontiguousarray(X[:, cols])
    test = np.ascontiguousarray(Xt[:, cols])
    ridge = RidgeClassifier(alpha=1., solver="cholesky")
    knn = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="brute", n_jobs=1)
    return {name: model.fit(training, y).predict(test).astype(np.int8)
            for name, model in (("ridge", ridge), ("knn5", knn))}


def results(gold, predicted, group, base):
    records = []
    for unit in [0, *np.unique(group)]:
        sel = np.ones(len(group), dtype=bool) if unit == 0 else group == unit
        correct = (predicted == gold) & sel
        counts = np.bincount(gold[sel], minlength=7)[1:]
        hits = np.bincount(gold[correct], minlength=7)[1:]
        rec = dict(**base, subject=int(unit), n=int(sel.sum()), accuracy=correct.sum()/sel.sum(),
                   balanced_accuracy=float(np.mean(hits/counts)))
        for activity, (hit, count) in enumerate(zip(hits, counts, strict=True), 1):
            rec[f"correct_activity_{activity}"] = int(hit)
            rec[f"n_activity_{activity}"] = int(count)
        records.append(rec)
    return records


def fingerprint():
    owned = [Path(__file__), PROTOCOL, ROOT/"src/minppi_mofs/exact_minpi.py",
             ROOT/"src/minppi_mofs/piv2.py", ROOT/"src/minppi_mofs/piv3_compact_filter.py"]
    return {p.resolve().relative_to(ROOT).as_posix(): sha256(p) for p in owned}


def run_seed(seed, X, y, scaled, scaled_test, truth, people, code_hashes):
    final = OUTPUT / f"seed-{seed}"
    if final.exists():
        previous = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
        if previous["code_sha256"] != code_hashes or previous["archive_sha256"] != DIGEST:
            raise ValueError(f"Completed seed {seed} has a different source or protocol")
        return
    order, rates = rankings(X, y, seed)
    supports = {}
    predictions = {}
    rows = []
    cache = {}
    for method, ranking in [*order.items(), ("full", np.arange(X.shape[1]))]:
        for budget in ((100,) if method == "full" else BUDGETS):
            count = (X.shape[1] * budget + 99) // 100
            cols = tuple(sorted(map(int, ranking[:count])))
            key = f"{method}_{budget}"
            supports[key] = [c+1 for c in cols]
            if cols not in cache:
                cache[cols] = model_predictions(scaled, scaled_test, y, list(cols))
            for learner, pred in cache[cols].items():
                predictions[f"{key}_{learner}"] = pred
                rows.extend(results(truth, pred, people, dict(seed=seed, method=method,
                                    percent=budget, width=count, learner=learner)))
    tie = np.isclose(rates, rates.min(axis=0), rtol=0., atol=1e-12)
    shares = (tie / tie.sum(axis=0)).sum(axis=1)
    meta = dict(seed=seed, archive_sha256=DIGEST, code_sha256=code_hashes, numpy=np.__version__,
                sklearn=sklearn.__version__, tier_sizes=TIERS, min_ties=int((tie.sum(axis=0)>1).sum()),
                fractional_argmin_counts=shares.tolist(), fractional_argmin_shares=(shares/len(tie[0])).tolist())
    with tempfile.TemporaryDirectory(prefix=f".seed-{seed}-", dir=OUTPUT) as tmp:
        stage = Path(tmp)/"completed"
        stage.mkdir()
        pd.DataFrame(rows).to_csv(stage/"metrics.csv", index=False)
        (stage/"supports.json").write_text(json.dumps(supports, separators=(",", ":")), encoding="utf-8")
        np.savez_compressed(stage/"predictions.npz", gold=truth, subject=people, tier_rates=rates, **predictions)
        (stage/"metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        stage.rename(final)
    print(f"Saved {final.name}", flush=True)


def jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def summarize():
    frames = []
    overlaps = []
    all_supports = {}
    for seed in SEEDS:
        path = OUTPUT/f"seed-{seed}"
        frames.append(pd.read_csv(path/"metrics.csv"))
        all_supports[seed] = json.loads((path/"supports.json").read_text(encoding="utf-8"))
        supports = all_supports[seed]
        for name, cols in supports.items():
            method, budget = name.split("_")
            if method != "full":
                overlaps.append(dict(seed=seed, comparison="exact", other=method, percent=int(budget),
                                     jaccard=jaccard(supports[f"exact_{budget}"], cols)))
    disagreement = []
    for seed in SEEDS:
        with np.load(OUTPUT/f"seed-{seed}"/"predictions.npz") as z:
            subjects, truth = z["subject"], z["gold"]
            for budget in BUDGETS:
                for learner in ("ridge", "knn5"):
                    focal = z[f"exact_{budget}_{learner}"]
                    for other in ("mean", "k16", "vote", "margin", "f", "forest", "random", "full"):
                        other_budget = 100 if other == "full" else budget
                        comp = z[f"{other}_{other_budget}_{learner}"]
                        for subject in np.unique(subjects):
                            rows = subjects == subject
                            activity = truth[rows]
                            def balanced(pred):
                                correct = (pred[rows] == activity)
                                hits = np.bincount(activity, weights=correct, minlength=7)[1:]
                                return float((hits/np.bincount(activity, minlength=7)[1:]).mean())
                            disagreement.append(dict(seed=seed, percent=budget, learner=learner,
                                other=other, subject=int(subject), disagreement=(focal[rows] != comp[rows]).mean(),
                                exact_minus_comparator_ba=balanced(focal)-balanced(comp)))
    pd.DataFrame(disagreement).to_csv(OUTPUT/"subject_differences.csv", index=False)
    for first, second in combinations(SEEDS, 2):
        for name, cols in all_supports[first].items():
            method, budget = name.split("_")
            overlaps.append(dict(seed=f"{first}-{second}", comparison="seed_pair", other=method,
                                 percent=int(budget), jaccard=jaccard(cols, all_supports[second][name])))
    pd.DataFrame(overlaps).to_csv(OUTPUT/"overlap.csv", index=False)
    df = pd.concat(frames)
    keys = ["method", "learner", "percent", "width"]
    # Each seed and test subject receives the same weight.
    by_subject = df[df.subject != 0].groupby([*keys, "subject"], as_index=False)[["balanced_accuracy", "accuracy"]].mean()
    by_subject.to_csv(OUTPUT/"by_subject.csv", index=False)
    summary = by_subject.groupby(keys, as_index=False).agg(subject_balanced_accuracy=("balanced_accuracy","mean"),
                  subject_accuracy=("accuracy","mean"), lowest_subject_ba=("balanced_accuracy","min"))
    pooled = df[df.subject == 0].groupby(keys, as_index=False)[["balanced_accuracy","accuracy"]].mean()
    summary = summary.merge(pooled.rename(columns={"balanced_accuracy":"pooled_ba","accuracy":"pooled_accuracy"}), on=keys)
    summary.to_csv(OUTPUT/"summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true", help="verify identity and subject partitions")
    args = parser.parse_args()
    data, features = load_har(args.check_data)
    X, y, train_subjects = data["train"]
    Xt, truth, test_subjects = data["test"]
    print(f"training subjects {np.unique(train_subjects).tolist()}, test subjects {np.unique(test_subjects).tolist()}", flush=True)
    if args.check_data:
        return
    OUTPUT.mkdir(exist_ok=True, parents=True)
    (OUTPUT/"features.txt").write_text("\n".join(features)+"\n", encoding="utf-8")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    scaled_test = scaler.transform(Xt)
    frozen = fingerprint()
    with threadpool_limits(1):
        for seed in SEEDS:
            run_seed(seed, X, y, scaled, scaled_test, truth, test_subjects, frozen)
    summarize()


if __name__ == "__main__":
    main()


