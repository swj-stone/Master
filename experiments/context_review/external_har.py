"""Locked subject-held-out external transport check on the original UCI HAR task."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
from math import ceil
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import digest
from experiments.context_study.selection import select_vote
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import matched_random_contexts, separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_tree import select as select_anchor
from minppi_mofs.piv3_compact_witness import _sample


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT/"configs/context_review/external_har.json"
OUT = ROOT/"results/context_review/external-har-locked-r1"
DATA = ROOT/"data/context_study/uci-har/uci-har-240.zip"
SOURCE = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
EXPECTED = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
SEED = 20421


def read_json(path):
    return json.loads(path.read_bytes())


def acquire():
    if not DATA.is_file():
        DATA.parent.mkdir(exist_ok=True, parents=True)
        part = DATA.with_suffix(".part")
        with urlopen(SOURCE, timeout=120) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out)
        if digest(part) != EXPECTED:
            raise ValueError("UCI archive differs from recorded original")
        part.replace(DATA)
    if digest(DATA) != EXPECTED:
        raise ValueError("UCI archive differs from recorded original")


@contextmanager
def original_archive():
    with ZipFile(DATA) as wrapper:
        with ZipFile(io.BytesIO(wrapper.read("UCI HAR Dataset.zip"))) as original:
            yield original


def load_member(archive, split, kind):
    dtype = np.float32 if kind == "X" else np.int64
    name = f"UCI HAR Dataset/{split}/{kind}_{split}.txt"
    with archive.open(name) as handle:
        return np.loadtxt(handle, dtype=dtype)


def say(message):
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (OUT/"run.log").open("a", encoding="utf8") as f:
        print(line, file=f, flush=True)


def bindings():
    paths = [Path(__file__), SPEC, ROOT/"experiments/context_study/benchmark.py",
             ROOT/"experiments/context_study/selection.py", ROOT/"experiments/context_study/working_set.py",
             ROOT/"src/minppi_mofs/context_design.py", ROOT/"src/minppi_mofs/context_vote.py",
             ROOT/"src/minppi_mofs/piv3_compact_witness.py", ROOT/"src/minppi_mofs/piv3_compact_tree.py",
             ROOT/"experiments/test/minpi_convergence_metrics.py", ROOT/"src/minppi_mofs/utility.py"]
    return {"source": SOURCE, "archive_sha256": digest(DATA),
            "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}}


def masks_for_seed(model, xt, xr, yr, order, k, seed):
    rows, weights = _sample(yr, "multiclass", np.random.SeedSequence([SEED, 659]))
    bg = np.random.default_rng(SEED).choice(len(xt), min(512,len(xt)), replace=False)
    tail = order[k:]
    family = matched_random_contexts(separating_contexts(tail), tail, seed=seed)
    ballot, cache = oracle(model, xr[rows], np.median(xt[bg],axis=0), yr[rows], weights, frozenset(order[:k]), False)
    chosen, trace = search(order[:k], family, ballot)
    trace.update(mask_calls=len(cache), rows=rows.tolist(), n_panels=len(weights))
    return sorted(chosen), trace


def estimator():
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1, tree_method="hist",
                         device="cpu", n_jobs=1, random_state=SEED, importance_type="gain")


def choose(arm, x, y, fit_ids, rank_ids, donor_k):
    width = x.shape[1]
    if arm == "full":
        return list(range(width)), {}
    if arm.startswith("rfe_") or arm.startswith("budget_gain_"):
        k = donor_k if arm == "rfe_donor" else ceil(width*int(arm.split("_")[-1])/100)
        if arm.startswith("rfe_"):
            selector = RFE(estimator(), n_features_to_select=k, step=.20).fit(x,y)
            return list(map(int,np.flatnonzero(selector.support_))), {}
        model = fit(x,y,SEED)
        gains = model.get_booster().get_score(importance_type="gain")
        order = sorted(range(width), key=lambda j: (-gains.get(f"f{j}",0), j))
        return sorted(order[:k]), {}
    xt, xr = x[fit_ids], x[rank_ids]
    model = fit(xt, y[fit_ids], SEED)
    names = [f"c{i}" for i in range(width)]
    _, anchor = select_anchor(model, pd.DataFrame(xt,columns=names), y[fit_ids],
                              pd.DataFrame(xr,columns=names), y[rank_ids],
                              task="multiclass", utility="ovr_auc", seed=SEED, aggregate="gain")
    order = [int(key[1:]) for key in anchor["order"]]
    k = anchor["k"]
    if arm in {"gain","gain_donor"}:
        selected, trace = sorted(order[:k if arm=="gain" else donor_k]), {}
    elif arm in {"antichain","min","random_77531"}:
        rule = "random_contexts" if arm=="random_77531" else arm
        selected, trace = select_vote(rule, model, xt, xr, y[rank_ids], order, k, 0)
    else:
        selected, trace = masks_for_seed(model, xt, xr, y[rank_ids], order, k, int(arm.split("_")[1]))
    trace.update(anchor=anchor)
    return selected, trace


def worker():
    spec = read_json(SPEC)
    with original_archive() as archive:
        x = load_member(archive,"train","X")
        y = load_member(archive,"train","y")-1
        subject = load_member(archive,"train","subject")
        xtest = load_member(archive,"test","X")
        test_subject = load_member(archive,"test","subject")
    if (x.shape!=(7352,561) or xtest.shape!=(2947,561) or set(subject)&set(test_subject)
            or len(set(subject))!=21 or len(set(test_subject))!=9):
        raise ValueError("Unexpected subject identity or dimension")
    if not (np.isfinite(x).all() and np.isfinite(xtest).all()):
        raise ValueError("Nonfinite archive measurements")
    groups = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
    train, rank = next(groups.split(x, y, subject))
    assert not set(subject[train])&set(subject[rank])
    assert set(y[train])==set(y[rank])==set(range(6))
    arms = ["full","gain","antichain","min", *[f"random_{s}" for s in spec["random_seeds"]],
            "gain_donor","rfe_donor"]
    for budget in spec["fixed_budgets"]:
        arms += [f"budget_gain_{round(100*budget)}",f"rfe_budget_{round(100*budget)}"]
    splits = dict(train_subjects=sorted(set(map(int,subject[train]))),
                  rank_subjects=sorted(set(map(int,subject[rank]))),
                  test_subjects=sorted(set(map(int,test_subject))), fit=train.tolist(), rank=rank.tolist())
    store(OUT/"split.json",splits)
    lock = digest(OUT/"lock.json")
    donor_k = None
    for arm in arms:
        folder = OUT/arm
        folder.mkdir(exist_ok=True)
        if not (folder/"done.json").exists():
            say(f"starting {arm}")
            start = perf_counter()
            cols, trace = choose(arm,x,y,train,rank,donor_k)
            selector_s = perf_counter()-start
            if not cols or len(set(cols))!=len(cols):
                raise ValueError("Invalid selected feature set")
            store(folder/"support.json",dict(selected=cols,trace=trace,lock_sha256=lock))
            start = perf_counter()
            probs = fit(x[:,cols], y, SEED).predict_proba(xtest[:,cols])
            refit_predict_s = perf_counter()-start
            np.savez_compressed(folder/"predictions.npz", probabilities=probs)
            store(folder/"done.json",dict(arm=arm,k=len(cols),selection_s=selector_s,
                  refit_predict_s=refit_predict_s,mask_calls=trace.get("mask_calls",0),
                  support_sha256=digest(folder/"support.json"),p_sha256=digest(folder/"predictions.npz"),
                  lock_sha256=lock))
            say(f"frozen {arm}: k={len(cols)}, select={selector_s:.2f}s refit/predict={refit_predict_s:.2f}s")
        done = read_json(folder/"done.json")
        if done["lock_sha256"]!=lock or done["support_sha256"]!=digest(folder/"support.json") or done["p_sha256"]!=digest(folder/"predictions.npz"):
            raise ValueError(f"Stale result on {arm}")
        if arm=="antichain":
            donor_k=done["k"]
    store(OUT/"frozen.json", dict(arms=arms,lock_sha256=lock,split_sha256=digest(OUT/"split.json"),
          done={a:digest(OUT/a/"done.json") for a in arms}))
    say(f"all {len(arms)} supports and prediction arrays frozen; no test outcome member read")


def evaluate():
    lock = read_json(OUT/"lock.json")
    receipt = read_json(OUT/"frozen.json")
    if lock!=bindings() or receipt["lock_sha256"]!=digest(OUT/"lock.json") or receipt["split_sha256"]!=digest(OUT/"split.json"):
        raise ValueError("Protocol or split identity changed")
    armlist = receipt["arms"]
    for a in armlist:
        folder=OUT/a
        done=read_json(folder/"done.json")
        if (digest(folder/"done.json")!=receipt["done"][a] or digest(folder/"support.json")!=done["support_sha256"]
                or digest(folder/"predictions.npz")!=done["p_sha256"]):
            raise ValueError(f"Frozen result changed on {a}")
    say("opening official test outcome member after validating all frozen predictions")
    with original_archive() as archive:
        y = load_member(archive,"test","y")-1
        subject = load_member(archive,"test","subject")
    cases, scores = [], []
    for a in armlist:
        with np.load(OUT/a/"predictions.npz") as source:
            p = source["probabilities"]
        assert p.shape==(len(y),6) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)
        rows = [dict(arm=a,subject=s,n=int(sum(subject==s)),accuracy=accuracy_score(y[subject==s],p[subject==s].argmax(axis=1))) for s in np.unique(subject)]
        cases.extend(rows)
        done = read_json(OUT/a/"done.json")
        scores.append(dict(arm=a,k=done["k"],retained=done["k"]/561,
            mean_subject_accuracy=np.mean([r["accuracy"] for r in rows]),pooled_accuracy=accuracy_score(y,p.argmax(axis=1)),
            pooled_auc=roc_auc_score(y,p,multi_class="ovr",average="macro"), masks=done["mask_calls"],
            selection_s=done["selection_s"],refit_predict_s=done["refit_predict_s"],
            total_s=done["selection_s"]+done["refit_predict_s"]))
    table = pd.DataFrame(scores)
    table.to_csv(OUT/"scores.csv",index=False)
    byperson = pd.DataFrame(cases)
    byperson.to_csv(OUT/"by-subject.csv", index=False)
    paired = byperson.pivot(index="subject", columns="arm", values="accuracy")
    contrasts = pd.DataFrame({a:paired.antichain-paired[a] for a in ["rfe_donor","gain_donor","gain","min"]})
    contrasts.to_csv(OUT/"paired-subject-differences.csv")
    store(OUT/"scored.json",dict(frozen_sha256=digest(OUT/"frozen.json"),scores_sha256=digest(OUT/"scores.csv")))
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["fetch","run","worker","score"],default="run")
    args=parser.parse_args()
    acquire()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.stage=="fetch":
        return
    if args.stage=="worker":
        with threadpool_limits(limits=1):
            return worker()
    if args.stage=="score":
        return evaluate()
    locked=bindings()
    path=OUT/"lock.json"
    if path.exists() and read_json(path)!=locked:
        raise ValueError("Frozen configuration changed; make a new run")
    store(path,locked)
    subprocess.run([sys.executable,"-m","experiments.context_review.external_har","--stage","worker"],check=True,timeout=3600)
    evaluate()


if __name__ == "__main__":
    main()
