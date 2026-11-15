"""Identify refit decisions under permutations of a fixed multiset of outer feature rows."""
import argparse
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import rule_sensitivity as rules
from experiments.context_review.refit_target import metrics
from experiments.context_study.benchmark import fit
from experiments.context_study.data import DATA, OUT, ROOT, config, digest
from experiments.test.minpi_convergence_metrics import make_scorer


DEST = ROOT / "thesis-paper/evidence/transcript_permutations"
PROTOCOL = ROOT / "configs/context_review/transcript_permutations.json"


def partition(rows):
    """Two outer rows have the same integer code exactly when their vectors agree."""
    return np.unique(rows, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int32)


def extract(name, fold):
    folder = f"{name}/fold-{fold}"
    base, target, grid = OUT / folder, rules.DONOR / folder, rules.OUT / folder
    source = rules.read(base / "selection.json")
    design = rules.read(target / "selection.json")
    origin = rules.read(base / "completed.json")
    fitted = rules.read(target / "completed.json")
    assert design["donor"] == origin["selection_sha256"] == digest(base / "selection.json")
    assert fitted["selection"] == digest(target / "selection.json")
    assert fitted["predictions"] == digest(target / "predictions.npz")
    assert rules.read(grid / "voting.json")["votes"] == digest(grid / "votes.npz")
    data = DATA / name / f"{name}.tsv.gz"
    assert digest(data) == origin["data_sha256"]
    frame = pd.read_csv(data, sep="\t")
    y = frame.pop("target").map({v: i for i, v in enumerate(source["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train, ranking, outer = (np.asarray(source[k]) for k in ("train", "rank", "test"))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xr, xo = [imputer.transform(x[ids]) for ids in (train, ranking, outer)]
    seed = 20421 + fold
    model = fit(xt, y[train], seed)
    donors = np.random.default_rng(seed).choice(len(xt), min(512, len(xt)), replace=False)
    bg = np.median(xt[donors], axis=0)
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    ids, weights = rules.panels(y[ranking], task, seed)
    scorer = make_scorer(y[ranking][ids], weights, task, "ovr_auc", False)
    with np.load(grid / "votes.npz") as observed, np.load(target / "predictions.npz") as preds:
        np.testing.assert_array_equal(outer, preds["row_id"])
        np.testing.assert_array_equal(y[outer], preds["y"])
        names = observed["metrics"].tolist()
        assert set(names) == {"auc", "accuracy"}
        mask_ids = {}
        codes = []
        for i, mask in enumerate(observed["xgb_masks"]):
            mask_ids[frozenset(np.flatnonzero(mask))] = i
            together = np.concatenate((xr[ids], xo), axis=0)
            together[:, ~mask] = bg[~mask]
            prob = model.predict_proba(together)
            stored = np.array([[row[m] for m in names] for row in scorer(prob[:len(ids)])])
            np.testing.assert_allclose(stored, observed["xgb_scores"][i], rtol=0, atol=2e-13)
            codes.append(partition(prob[len(ids):]))
        full = partition(model.predict_proba(xo))
        anchor = partition(preds["anchor"])
        extensions = {frozenset(e) for family in design["contexts"].values() for e in family}
        ref = frozenset(design["anchor"])
        data_out = {k: preds[k] for k in preds.files}
        data_out.update(codes=np.array(codes), initial=full, anchor_code=anchor)
    joined = []
    for cols in design["supports"]:
        cand = frozenset(cols)
        all_masks = sorted({mask_ids[s | e] for e in extensions for s in (ref, cand)})
        joined.append(dict(columns=cols, point=[mask_ids[ref], mask_ids[cand]], recorded=all_masks))
    meta = dict(dataset=name, fold=fold, proposals=joined, source=origin["data_sha256"],
                selection=fitted["selection"], predictions=fitted["predictions"],
                votes=digest(grid / "votes.npz"))
    out = BytesIO()
    np.savez_compressed(out, **data_out)
    return meta, out.getvalue()


def freeze():
    DEST.mkdir(parents=True, exist_ok=True)
    records = {}
    path = DEST / "row-partitions.zip"
    temp = path.with_suffix(".tmp")
    with ZipFile(temp, "w") as archive, threadpool_limits(limits=1):
        for name in config()["datasets"]:
            for fold in range(3):
                key = f"{name}/fold-{fold}"
                records[key], data = extract(name, fold)
                archive.writestr(ZipInfo(key+".npz"), data, compress_type=ZIP_DEFLATED)
            print(f"Verified continuous scores and outer partitions for {name}", flush=True)
    assert len(records) == 42 and sum(len(m["proposals"]) for m in records.values()) == 704
    temp.replace(path)
    manifest = dict(protocol=digest(PROTOCOL), archive=digest(path), folds=records)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")


def score_columns(y, probs):
    """AUC and accuracy contributions by assigned label at each fixed score vector."""
    n, classes = probs.shape
    auc = np.zeros((n, classes))
    targets = [1] if classes == 2 else range(classes)
    for c in targets:
        pos = np.count_nonzero(y == c)
        auc[:, c] = rankdata(probs[:, c]) / (len(targets) * pos * (n-pos))
    accuracy = (probs.argmax(axis=1)[:, None] == np.arange(classes)) / n
    return auc, accuracy


def blocks(y, probs, codes):
    groups = []
    for indices in np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1]):
        if len(indices) > 1 and np.any(y[indices] != y[indices[0]]) and np.any(probs[indices] != probs[indices[0]]):
            groups.append(indices)
    return groups


def extremizer(y, objective, groups, maximize):
    perm = np.arange(len(y))
    for g in groups:
        labels = y[g]
        classes = np.unique(labels)
        if len(classes) == 2:
            low, high = classes
            benefit = objective[g, high] - objective[g, low]
            order = np.argsort(benefit if maximize else -benefit, kind="stable")
            perm[g[np.argsort(labels, kind="stable")]] = g[order]
        else:
            scores = objective[g][:, labels].T
            dest, source = linear_sum_assignment(scores, maximize=maximize)
            perm[g[dest]] = g[source]
    return perm


def evaluate(y, probs, codes, floors):
    auc, acc = score_columns(y, probs)
    groups = blocks(y, probs, codes)
    varying_scores = np.unique(np.column_stack((codes, partition(probs))), axis=0).shape[0] > np.unique(codes).size
    mixed_labels = np.unique(np.column_stack((codes, y)), axis=0).shape[0] > np.unique(codes).size
    measured = metrics(y, probs, False)
    base = np.array([measured["auc"], measured["accuracy"]])
    original = bool(np.all(base >= floors))
    gain = []
    order = np.arange(len(y))
    low, high = base.copy(), base.copy()
    witness = None
    for j, objective in enumerate((auc, acc)):
        for maximize in (False, True):
            p = extremizer(y, objective, groups, maximize)
            assert np.array_equal(codes[p], codes) and np.array_equal(np.sort(p), order)
            difference = [float((a[p, y]-a[order, y]).sum()) for a in (auc, acc)]
            v = base + difference
            (high if maximize else low)[j] = v[j]
            gain.append((p, v))
    for p, v in gain:
        if np.all(v >= floors) != original:
            direct = metrics(y, probs[p], False)
            exact = np.array([direct["auc"], direct["accuracy"]])
            np.testing.assert_allclose(v, exact, rtol=0, atol=1e-13)
            if bool(np.all(exact >= floors)) != original:
                witness = p
                break
    # Only certify decisions separated from numerical comparison uncertainty.
    certainty = np.any(high < floors-1e-13) or np.all(low >= floors+1e-13)
    disposition = "witness" if witness is not None else "invariant" if certainty or not groups else "unresolved"
    return dict(groups=len(np.unique(codes)), row_score_can_change=varying_scores,
                labels_can_exchange=mixed_labels, exchange_groups=len(groups), permutation_rows=sum(map(len, groups)),
                original_pass=int(original), auc=base[0], accuracy=base[1],
                min_auc=low[0], max_auc=high[0], min_accuracy=low[1], max_accuracy=high[1],
                disposition=disposition), witness


def replay():
    manifest = rules.read(DEST / "manifest.json")
    path = DEST / "row-partitions.zip"
    assert digest(path) == manifest["archive"] and digest(PROTOCOL) == manifest["protocol"]
    rows, witnesses = [], []
    with ZipFile(path) as archive:
        for key, record in manifest["folds"].items():
            with np.load(BytesIO(archive.read(key+".npz"))) as data:
                y = data["y"]
                anchor = metrics(y, data["anchor"], False)
                floors = .97 * np.array([anchor["auc"], anchor["accuracy"]]) - 1e-12
                common = np.column_stack((data["initial"], data["anchor_code"]))
                masks = data["codes"].T
                for i, proposal in enumerate(record["proposals"]):
                    probs = data[f"s{i}"]
                    for variant in ("point", "recorded"):
                        transcript = np.column_stack((common, masks[:, proposal[variant]]))
                        codes = partition(transcript)
                        row, perm = evaluate(y, probs, codes, floors)
                        ident = dict(dataset=record["dataset"], fold=record["fold"], proposal=i, observer=variant)
                        rows.append(dict(**ident, **row))
                        if perm is not None:
                            np.testing.assert_array_equal(transcript[perm], transcript)
                            changed = np.flatnonzero(perm != np.arange(len(y)))
                            witnesses.append(dict(**ident, positions=changed.tolist(), donors=perm[changed].tolist()))
    results = pd.DataFrame(rows)
    assert len(results) == 1408
    results.to_csv(DEST / "proposals.csv", index=False)
    (DEST / "witnesses.json").write_text(json.dumps(witnesses, indent=2)+"\n", encoding="utf8")
    summary = results.assign(variable_auc=results.max_auc-results.min_auc > 1e-13,
                             variable_accuracy=results.max_accuracy-results.min_accuracy > 1e-13,
                             mixed_label_and_score=results.exchange_groups > 0)
    counts = summary.groupby(["observer","disposition"]).agg(proposals=("proposal","size"),
                variable_row_scores=("row_score_can_change","sum"), mixed_labels=("labels_can_exchange","sum"),
                mixed_label_and_score=("mixed_label_and_score","sum"), variable_auc=("variable_auc","sum"),
                variable_accuracy=("variable_accuracy","sum")).reset_index()
    counts.to_csv(DEST/"summary.csv", index=False)
    print(counts.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "replay"])
    if parser.parse_args().mode == "freeze":
        freeze()
    else:
        replay()
