"""Fit the fixed forest masker and receivers, and seal predictions before outer scoring."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as primary
from experiments.context_review.receiver_transport import SOURCE, frozen_folds, member


ROOT = bg.ROOT
DEST = ROOT / "thesis-paper/evidence/forest_cross"
RUN = ROOT / "results/context_review/forest-cross"
DESIGN = ROOT / "configs/context_review/forest_cross.json"
SCORES = ("vote", "slack", "point")


def new_forest(seed):
    return RandomForestClassifier(n_estimators=100, criterion="gini", max_features="sqrt",
                                  bootstrap=True, n_jobs=1, random_state=seed)


def transform(x, rows, evaluate):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(x[rows]), imputer.transform(evaluate)


def receiver(x, y, train, evaluate, supports, seed):
    start = perf_counter()
    xt, xv = transform(x, train, evaluate)
    probabilities, fit_seconds = [], []
    for cols in supports:
        model = new_forest(seed)
        before = perf_counter()
        model.fit(xt[:, cols], y[train])
        np.testing.assert_array_equal(model.classes_, np.arange(len(model.classes_)))
        probabilities.append(model.predict_proba(xv[:, cols]))
        fit_seconds.append(perf_counter() - before)
    return probabilities, fit_seconds, perf_counter() - start


def forest_masker(x, y, train, prop, choice, supports, seed):
    start = perf_counter()
    xt, xp = transform(x, train, x[prop])
    model = new_forest(seed).fit(xt, y[train])
    fit_s = perf_counter() - start
    np.testing.assert_array_equal(model.classes_, np.arange(len(model.classes_)))
    anchor = supports[0]
    context = primary.separating_contexts(choice["order"][len(anchor):])
    metric_names = ["auc", "accuracy"] + (["ap"] if choice["rare"] else [])
    checks, winners = {}, {}
    for method in SCORES:
        begin = perf_counter()
        ballot = primary.build_ballot(model, xt, xp, y[prop], seed, anchor, choice["rare"], point=method=="point")
        family = (frozenset(),) if method == "point" else context
        best, winner = -np.inf, 0
        for i, cols in enumerate(supports[1:], 1):
            val = primary.rating(ballot, cols, family, method, best=best if method=="vote" else None)
            if val > best:
                best, winner = val, i
            if method == "vote" and best == 4:
                break
        winners[method] = winner
        checks[method] = dict(masks=len(ballot.scores), rows=len(ballot.scores)*len(ballot.rows),
                              seconds=perf_counter()-begin)
    before = perf_counter()
    ballot = primary.build_ballot(model, xt, xp, y[prop], seed, anchor, choice["rare"])
    tensor = np.zeros((len(supports), len(context), 4, len(metric_names)), dtype=np.float64)
    if len(supports) > 1:
        for i, cols in enumerate(supports):
            for j, restored in enumerate(context):
                metrics = ballot.score(frozenset(cols) | restored)
                tensor[i,j] = [[m[k] for k in metric_names] for m in metrics]
    else:
        # A singleton has no candidate. It has no measured mask transcript.
        tensor = tensor[:, :0]
    timing = dict(voter_fit_s=fit_s, diagnostic_masks=len(ballot.scores),
                  diagnostic_rows=len(ballot.scores)*len(ballot.rows),
                  diagnostic_s=perf_counter()-before, nomination=checks)
    return tensor, metric_names, winners, timing


def predict_fold(panel, key, selection_sha, campaign):
    src = ROOT / "results/context_review" / SOURCE[panel] / key
    assert bg.digest(src/"selection.json") == selection_sha
    sel, layout = bg.read(src/"selection.json"), bg.read(src/"layout.json")
    assert bg.digest(src/"layout.json") == sel["layout"]
    assert bg.digest(src/"inputs.npz") == layout["inputs_sha256"]
    choice = sel["choices"]["gain"]
    supports = [choice["anchor"]] + [r["columns"] for r in sel["pool"]]
    with np.load(src/"inputs.npz") as data:
        x, y, outer, train, prop, guard, ids = (data[k] for k in
            ("x","y","x_outer","train","proposal","guard","test_row"))
    seed = 20421 + layout["fold"]
    transcript, names, winners, timing = forest_masker(x, y, train, prop, choice, supports, seed)
    inner = np.sort(np.r_[train,prop])
    np.testing.assert_array_equal(np.sort(np.r_[inner,guard]), np.arange(len(y)))
    gv, guard_times, timing["all_guard_s"] = receiver(x, y, inner, x[guard], supports, seed)
    ov, outer_times, timing["all_outer_s"] = receiver(x, y, np.arange(len(y)), outer, supports, seed)
    timing.update(guard_support_s=guard_times, outer_support_s=outer_times)
    data = BytesIO()
    np.savez_compressed(data, tensor=transcript, guard=guard, guard_y=y[guard], test_row=ids,
                        **{f"g{i}":p for i,p in enumerate(gv)}, **{f"o{i}":p for i,p in enumerate(ov)})
    rec = dict(panel=panel, dataset=layout["dataset"], fold=layout["fold"], supports=supports, metrics=names,
               winners=winners, campaign=campaign, selection_sha=selection_sha, layout_sha=sel["layout"],
               inputs_sha=layout["inputs_sha256"], targets_sha=layout["targets_sha256"],
               member_sha=sha256(data.getbuffer()).hexdigest(), timings=timing)
    return rec, data.getvalue()


def deadline(key):
    print(f"Aborting: {key} reached the 3600-second limit.", flush=True)
    os._exit(124)


def fit():
    DEST.mkdir(exist_ok=True, parents=True)
    jobs = frozen_folds()
    design_sha = bg.digest(DESIGN)
    campaign = dict(design_sha=design_sha, source_sha=bg.digest(Path(__file__)), sklearn=sklearn.__version__)
    records = {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for n, (panel, key, selection_sha) in enumerate(jobs, 1):
            full = f"{panel}/{key}"
            folder = RUN / full
            folder.mkdir(exist_ok=True, parents=True)
            path, receipt = folder/"probabilities.npz", folder/"receipt.json"
            before = perf_counter()
            if not receipt.exists():
                timer = Timer(3600, deadline, args=(full,))
                timer.daemon = True
                timer.start()
                try:
                    rec, data = predict_fold(panel, key, selection_sha, campaign)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(path)
                    bg.store(receipt, rec)
                finally:
                    timer.cancel()
            rec = bg.read(receipt)
            assert rec["campaign"] == campaign and rec["selection_sha"] == selection_sha
            assert bg.digest(path) == rec["member_sha"]
            records[full] = rec
            elapsed = perf_counter() - start
            print(f"[{n:02}/{len(jobs)}] {full} fold {perf_counter()-before:.1f}s, elapsed {elapsed:.1f}s, "
                  f"ETA {elapsed*(len(jobs)-n)/n:.1f}s", flush=True)
    temp = DEST / "predictions.tmp"
    with ZipFile(temp, "w") as archive:
        for full in records:
            member(archive, full+".npz", (RUN/full/"probabilities.npz").read_bytes())
    path = DEST / "predictions.zip"
    temp.replace(path)
    bg.store(DEST/"prediction-lock.json", dict(**campaign, archive_sha=bg.digest(path), folds=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    fit()
