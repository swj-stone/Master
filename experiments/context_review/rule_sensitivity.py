"""Replay frozen deletion proposals with prespecified vote settings and two refit targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost

from experiments.context_review.refit_target import metrics
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.piv3_compact_witness import _sample

DESIGN = ROOT / "configs/context_review/rule_sensitivity.json"
DONOR = ROOT / "results/context_review/refit-target"
OUT = ROOT / "results/context_review/rule-sensitivity"
LEARNERS = ("xgb", "logistic")
METRICS = ("auc", "accuracy", "ap")
AXES = ("family", "aggregation", "panels", "quorum", "vote_tolerance_pct")


def read(path):
    return json.loads(path.read_bytes())


def rule_grid():
    design = read(DESIGN)
    return list(product(*(design[k] for k in ("families", "aggregation", "panels", "quorums",
                                                "vote_tolerance_pct"))))


def panels(y, task, seed):
    ids, original = _sample(y, task, np.random.SeedSequence([seed, 659]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 659, 8181]))
    extra = np.zeros((8, len(ids)))
    for label, size in zip(*np.unique(y, return_counts=True), strict=True):
        group = np.flatnonzero(y[ids] == label)
        for w in extra:
            hits = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
            w[group] = (size / len(y)) * hits / len(group)
    weights = np.concatenate((original, extra))
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-14, rtol=0)
    return ids, weights


def logistic(x, y, seed, where, warning_log):
    estimator = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, solver="lbfgs", tol=1e-4, max_iter=2000, random_state=seed))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x.astype(np.float64), y)
    warning_log.extend(dict(fit=where, category=w.category.__name__, message=str(w.message)) for w in seen)
    return estimator


def required(n, quorum):
    return {"half_or_more": (n+1)//2, "strict_majority": n//2+1,
            "three_quarters_or_more": (3*n+3)//4, "unanimous": n}[quorum]


def family_contexts(records):
    groups = {name: [frozenset(e) for e in group] for name, group in records.items()}
    separating = groups["separating"]
    groups["endpoints"] = list(dict.fromkeys((separating[0], separating[-1])))
    groups["empty"] = [separating[0]]
    return groups


def ballots(x, xt, y_rank, y_train, ids, weights, task, rare, seed, design, learner, warning_log):
    model = fit(xt, y_train, seed) if learner == "xgb" else logistic(xt, y_train, seed, "masking", warning_log)
    rows = np.random.default_rng(seed).choice(len(xt), min(512, len(xt)), replace=False)
    bg = np.median(xt[rows], axis=0)
    ranking = x[ids]
    scorer = make_scorer(y_rank[ids], weights, task, "ovr_auc", rare)
    anchor = frozenset(design["anchor"])
    supports = [frozenset(s) for s in design["supports"]]
    groups = family_contexts(design["contexts"])
    labels = METRICS if rare else METRICS[:2]
    masks = sorted({tuple(sorted(s | ext)) for s in [anchor, *supports]
                    for extensions in groups.values() for ext in extensions})
    scores = []
    for mask in masks:
        xx = np.broadcast_to(bg, ranking.shape).copy()
        xx[:, mask] = ranking[:, mask]
        pred = model.predict_proba(xx if learner == "xgb" else xx.astype(np.float64))
        values = scorer(pred)
        scores.append([[v[label] for label in labels] for v in values])
    scores = np.asarray(scores)
    mask_ids = {frozenset(s): i for i, s in enumerate(masks)}

    @lru_cache(maxsize=None)
    def votes(family, tol):
        ext = groups[family]
        ref = scores[[mask_ids[anchor | e] for e in ext]]
        reduced = scores[[[mask_ids[s | e] for e in ext] for s in supports]]
        allowance = np.array([.05 if key == "ap" else tol/100 for key in labels])
        return np.all(ref-reduced <= allowance*np.abs(ref)+1e-12, axis=-1)

    if learner == "xgb":
        for family in ("nested", "separating", "random"):
            known = np.asarray([g[family] for g in design["ballots"]], dtype=bool)
            np.testing.assert_array_equal(votes(family, 3)[:, :, :4], known)
    decisions = []
    for family, aggregation, count, quorum, tol in rule_grid():
        table = votes(family, tol)[:, :, :count]
        if aggregation == "minimum":
            passed = table.sum(axis=2).min(axis=1) >= required(count, quorum)
        else:
            passed = table.sum(axis=(1, 2)) >= required(count*table.shape[1], quorum)
        decisions.append(~passed)
    decisions = np.asarray(decisions).T
    if learner == "xgb":
        for arm, family, aggregation, count in [
            ("point", "empty", "minimum", 1), ("empty", "empty", "minimum", 4),
            ("endpoints", "endpoints", "minimum", 4), ("nested", "nested", "minimum", 4),
            ("separating", "separating", "minimum", 4), ("random", "random", "minimum", 4),
            ("pooled", "separating", "pooled", 4), ("one_panel_min", "separating", "minimum", 1)]:
            j = rule_grid().index((family, aggregation, count, "strict_majority", 3))
            np.testing.assert_array_equal(decisions[:, j], [d[arm] for d in design["decisions"]])
    mask_matrix = np.zeros((len(masks), xt.shape[1]), dtype=bool)
    for i, mask in enumerate(masks):
        mask_matrix[i, list(mask)] = True
    return dict(veto=decisions, scores=scores, masks=mask_matrix)


def worker(name, fold):
    started = perf_counter()
    suffix = Path(name)/f"fold-{fold}"
    target, primary, folder = DONOR/suffix, PRIMARY/suffix, OUT/suffix
    folder.mkdir(parents=True, exist_ok=True)
    protocol = read(OUT/"protocol.json")
    locked = protocol["sha256"]
    for local in [DATA/name/f"{name}.tsv.gz", primary/"selection.json", primary/"completed.json",
                  target/"selection.json", target/"completed.json", target/"predictions.npz"]:
        assert digest(local) == locked[local.relative_to(ROOT).as_posix()]
    design = read(target/"selection.json")
    source = read(primary/"selection.json")
    assert design["donor"] == digest(primary/"selection.json") == read(primary/"completed.json")["selection_sha256"]
    assert read(target/"completed.json")["selection"] == digest(target/"selection.json")
    assert read(target/"completed.json")["predictions"] == digest(target/"predictions.npz")
    table = pd.read_csv(DATA/name/f"{name}.tsv.gz", sep="\t")
    y = table.pop("target").map({v:i for i,v in enumerate(source["class_order"])}).to_numpy()
    x = table.to_numpy(dtype=np.float32)
    train, rank, test = [np.array(source[key]) for key in ("train", "rank", "test")]
    assert len(np.unique(np.r_[train, rank, test])) == len(x) == len(train)+len(rank)+len(test)
    dev = np.sort(np.r_[train, rank])
    seed = 20421+fold
    task = "binary" if len(source["class_order"]) == 2 else "multiclass"
    rare = task == "binary" and min(np.bincount(y[rank])) <= .05*len(rank)
    fields = METRICS if rare else METRICS[:2]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xr = imp.transform(x[train]), imp.transform(x[rank])
    ids, weights = panels(y[rank], task, seed)
    warning_log, ledger = [], dict(rows=rank[ids], y=y[rank][ids], weights=weights, metrics=fields)
    for learner in LEARNERS:
        result = ballots(xr, xt, y[rank], y[train], ids, weights, task, rare, seed, design, learner, warning_log)
        ledger.update({f"{learner}_{key}": value for key, value in result.items()})
    votefile = folder/"votes.npz"
    np.savez_compressed(votefile, **ledger)
    store(folder/"voting.json", dict(votes=digest(votefile), donor=digest(target/"selection.json"),
                                     protocol=digest(OUT/"protocol.json"), metrics=fields))
    frozen = digest(folder/"voting.json")
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[dev])
    xd, xe = imp.transform(x[dev]), imp.transform(x[test])
    columns = [("anchor", design["anchor"]), *((f"s{i}", s) for i,s in enumerate(design["supports"]))]
    pred = dict(row_id=test, y=y[test])
    for key, cols in columns:
        model = logistic(xd[:, cols], y[dev], seed, key, warning_log)
        pred[key] = model.predict_proba(xe[:, cols].astype(np.float64))
    np.savez_compressed(folder/"logistic-outer.npz", **pred)
    outcomes = {}
    for learner, archive in [("xgb", target/"predictions.npz"), ("logistic", folder/"logistic-outer.npz")]:
        with np.load(archive) as saved:
            np.testing.assert_array_equal(saved["row_id"], test)
            np.testing.assert_array_equal(saved["y"], y[test])
            values = np.array([[m[key] for key in fields] for m in
                               [metrics(y[test], saved[k], rare) for k,_ in columns]])
        loss = values[0]-values[1:]
        failure = np.any(loss > np.array([.05 if key == "ap" else .03 for key in fields])*abs(values[0])+1e-12, axis=1)
        outcomes[learner] = dict(reference=values[0].tolist(), loss=loss.tolist(), failure=failure.tolist())
        if learner == "xgb":
            for row in read(target/"completed.json")["rows"]:
                assert row["fail"] == failure[row["subset"]]
                np.testing.assert_allclose(loss[row["subset"]], [row["loss_"+key] for key in fields], rtol=0, atol=1e-12)
    assert frozen == digest(folder/"voting.json")
    store(folder/"completed.json", dict(outcomes=outcomes, logistic_predictions=digest(folder/"logistic-outer.npz"),
          voting=frozen, warnings=warning_log, elapsed_seconds=perf_counter()-started))


def lock_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__).resolve(), DESIGN, PRIMARY/"protocol-lock.json", DONOR/"protocol.json",
             ROOT/"configs/context_study/prospective.json", ROOT/"experiments/context_review/refit_target.py",
             ROOT/"experiments/context_study/benchmark.py", ROOT/"experiments/context_study/data.py",
             ROOT/"experiments/test/minpi_convergence_metrics.py", ROOT/"src/minppi_mofs/piv3_compact_witness.py"]
    for name in config()["datasets"]:
        files.append(DATA/name/f"{name}.tsv.gz")
        for fold in range(3):
            suffix = Path(name)/f"fold-{fold}"
            files.extend(PRIMARY/suffix/key for key in ("selection.json", "completed.json"))
            files.extend(DONOR/suffix/key for key in ("selection.json", "predictions.npz", "completed.json"))
    lock = dict(sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files}, numpy=np.__version__,
                sklearn=sklearn.__version__, xgboost=xgboost.__version__, python=sys.version)
    path = OUT/"protocol.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Locked inputs or software changed")
    store(path, lock)
    pd.DataFrame(rule_grid(), columns=AXES).rename_axis("rule_id").to_csv(OUT/"rules.csv")


def dispatch(pair):
    name, fold = pair
    folder = OUT/name/f"fold-{fold}"
    if (folder/"completed.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (folder/"worker.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "experiments.context_review.rule_sensitivity", "--worker",
                            name, str(fold)], timeout=3600, check=True, stdout=log, stderr=subprocess.STDOUT)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        store(folder/"failure.json", dict(dataset=name, fold=fold, error=str(error)))
        print(f"FAILED {name} fold {fold}: {error}", flush=True)
        return
    print(f"Completed {name} fold {fold}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            worker(args.worker[0], int(args.worker[1]))
        return
    lock_inputs()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(dispatch, product(config()["datasets"], range(3))))
    failures = [p for p in OUT.glob("*/fold-*/failure.json") if not (p.parent/"completed.json").exists()]
    if failures:
        raise RuntimeError(f"{len(failures)} incomplete folds; see failure.json and worker.log")


if __name__ == "__main__":
    main()
