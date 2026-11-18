"""Five-arm sparse Ridge comparison for all registered document and Farm Ads panels."""
import argparse
import csv
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from time import perf_counter_ns
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments import minpi_constant_transfer as farm
from experiments.stable_minpi_stress import digest
from research.conditional_dense import shared_method
from research.conditional_dense.vote_width import method as frozen
from research.lowdim import wide_csr
from research.text_peer import square_bench as docs

HERE = Path(__file__).resolve().parent
ARCH = farm.ROOT / "results/minpi-study"
KEY = ["cohort", "dataset", "fold", "seed"]
METHODS = ("new", "previous", "full", "F", "random")
REPEATS = 3


def cases():
    for data, archive in ((docs.prior_cases(), "text-square-recheck-20261118"),
                          (docs.new_source(), "text-square-bydate-20261118b")):
        for X, y, Xt, gold, meta in data:
            yield X, y, Xt, gold, meta, ARCH/archive/(meta["key"]+".npz")
    for name, fold, seed, X, y, Xt, gold, _, _ in islice(farm.panels(), 6):
        assert name.startswith("farm-ads-")
        key = f"{name}-{fold}"
        yield X, y, Xt, gold, dict(dataset=name, fold=fold, seed=seed, key=key), ARCH/"constant-transfer-20261117a"/(key+".npz")


def sparse_digest(X):
    fingerprint = sha256()
    for component in (np.array(X.shape, dtype=np.int64), X.indptr, X.indices, X.data):
        fingerprint.update(component.dtype.str.encode())
        fingerprint.update(np.ascontiguousarray(component).view(np.uint8))
    return fingerprint.hexdigest()


def choose(arm, X, y, seed, width):
    if arm in ("new", "previous"):
        return wide_csr.csr_vote_support(X, y, seed=seed)[0]
    if arm == "full":
        return np.arange(X.shape[1])
    if arm == "random":
        return np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:width])
    if arm == "F":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            scores = f_classif(X.astype(np.float64, copy=False), y)[0]
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        return np.sort(np.argsort(-scores, kind="stable")[:width])
    raise ValueError(arm)


def audit_f(X, y, cols):
    X = X.astype(np.float64, copy=False)
    sums = np.asarray(X.sum(axis=0)).ravel()
    means = sums / X.shape[0]
    total = np.asarray(X.multiply(X).sum(axis=0)).ravel() - means*sums
    between = np.zeros(X.shape[1])
    for label in np.unique(y):
        member = (y == label)
        group = np.asarray(X[member].sum(axis=0)).ravel()
        between += (group-member.sum()*means)**2/member.sum()
    effects = np.divide(between, total, out=np.full_like(total, -np.inf), where=total>0)
    outside = np.setdiff1d(np.arange(X.shape[1]), cols)
    gap = 0. if not len(outside) else max(0., float(effects[outside].max()-effects[cols].min()))
    assert gap <= 1e-9, gap
    exact = np.array_equal(cols, np.sort(np.argsort(-effects, kind="stable")[:len(cols)]))
    return gap, bool(exact)


def deploy(arm, X, y, Xt, seed, width):
    t = perf_counter_ns()
    cols = choose(arm, X, y, seed, width)
    selected = perf_counter_ns()
    pred = farm.fit_predict(X, y, Xt, cols, "classification", seed, False)
    end = perf_counter_ns()
    return cols, pred, (selected-t)/1e6, (end-t)/1e6


def fingerprints():
    files = [Path(__file__), HERE/"text-protocol.md", Path(wide_csr.__file__), Path(frozen.__file__),
             Path(shared_method.__file__), Path(docs.__file__), Path(farm.__file__), docs.CACHE/"panels.json"]
    return {str(f.relative_to(farm.ROOT)): farm.file_hash(f) for f in files}


def run(base, process):
    directory = base / f"process{process}"
    directory.mkdir(parents=True, exist_ok=False)
    source = fingerprints()
    first = base / "process1"
    if process == 2:
        assert json.loads((first/"complete.json").read_text())["sources"] == source
    timing = directory/"times.csv"
    inputs, checks = [], []
    with timing.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=KEY+["method", "repeat", "position", "p", "k",
                                                       "score", "select_ms", "total_ms"])
        writer.writeheader()
        for X, y, Xt, truth, meta, ref in cases():
            seed, name = meta["seed"], meta["dataset"]
            key = dict(cohort="text", dataset=name, fold=meta["fold"], seed=seed)
            p = X.shape[1]
            assert p > 32
            origin = ("20 Newsgroups bydate" if name.startswith("bydate_") else
                      "Farm Ads" if name.startswith("farm-ads-") else name)
            inputs.append(dict(**key, task="classification", metric="BA", classes=len(np.unique(y)),
                               family="sparse text", source_kind="observational", source=origin, n_train=len(y),
                               n_test=len(truth), train_sha=sparse_digest(X), target_sha=digest(y),
                               test_sha=sparse_digest(Xt), truth_sha=digest(truth)))
            arrays, scores, width = dict(truth=truth), {}, None
            for arm in METHODS:
                cols, pred, _, _ = deploy(arm, X, y, Xt, seed, width)
                assert np.array_equal(cols, np.unique(cols)) and 0 < len(cols) <= p
                if arm == "new":
                    width = len(cols)
                elif arm != "full":
                    assert len(cols) == width
                if arm == "F":
                    gap, exact = audit_f(X, y, cols)
                    checks.append(dict(**key, independent_F_gap=gap, exact_F_set=exact))
                arrays[arm+"_cols"], arrays[arm+"_pred"] = cols, pred
                scores[arm] = balanced_accuracy_score(truth, pred)
            with np.load(ref) as archived:
                np.testing.assert_array_equal(truth, archived["truth" if "truth" in archived else "gold"])
                np.testing.assert_array_equal(arrays["full_pred"], archived["full_pred"])
            for suffix in ("_cols", "_pred"):
                np.testing.assert_array_equal(arrays["new"+suffix], arrays["previous"+suffix])
            filename = meta["key"]+".npz"
            if process == 2:
                with np.load(first/filename) as replay:
                    for k, values in arrays.items():
                        np.testing.assert_array_equal(replay[k], values)
            np.savez_compressed(directory/filename, **arrays)
            order = np.random.default_rng([88351, meta["fold"], seed, process]).permutation(METHODS)
            for r in range(REPEATS):
                for pos, arm in enumerate(np.roll(order, r)):
                    cols, pred, selection, total = deploy(arm, X, y, Xt, seed, width)
                    np.testing.assert_array_equal(cols, arrays[arm+"_cols"])
                    np.testing.assert_array_equal(pred, arrays[arm+"_pred"])
                    writer.writerow(dict(**key, method=arm, repeat=r, position=pos, p=p, k=len(cols),
                                         score=scores[arm], select_ms=selection, total_ms=total))
            stream.flush()
            print(process, len(inputs), meta["key"], flush=True)
    assert len(inputs) == 38
    pd.DataFrame(inputs).to_csv(directory/"inputs.csv", index=False)
    pd.DataFrame(checks).to_csv(directory/"f-audit.csv", index=False)
    state = dict(process=process, sources=source, arms=METHODS, repeats=REPEATS, threads=1,
                 cases=len(inputs), packages=[sklearn.__version__, scipy.__version__, np.__version__])
    (directory/"complete.json").write_text(json.dumps(state, indent=2), encoding="utf8")


def summarize(base, out):
    frames = []
    for run_id in (1, 2):
        folder = base/f"process{run_id}"
        assert json.loads((folder/"complete.json").read_text())["cases"] == 38
        frame = pd.read_csv(folder/"times.csv")
        assert len(frame) == 38*len(METHODS)*REPEATS
        assert frame.groupby(KEY+["method"]).position.nunique().eq(REPEATS).all()
        assert frame.select_ms.ge(0).all() and frame.total_ms.ge(frame.select_ms).all()
        frames.append(frame.assign(process=run_id))
    for item in ("inputs.csv", "f-audit.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(base/"process1"/item), pd.read_csv(base/"process2"/item))
    raw = pd.concat(frames)
    assert raw.groupby(KEY+["method"]).score.nunique().eq(1).all()
    med = raw.groupby(KEY+["method","process"], as_index=False).agg(
        p=("p","first"), k=("k","first"), score=("score","first"),
        select_ms=("select_ms","median"), total_ms=("total_ms","median"))
    rows = []
    for key, group in med.groupby(KEY):
        case = dict(zip(KEY, key))
        a, b = (group[group.process==i].set_index("method") for i in (1,2))
        n = a.loc["new"]
        case.update(p=int(n.p), k=int(n.k), removed=bool(n.k<n.p))
        for arm in METHODS:
            s = a.loc[arm]
            case["score_"+arm], case["width_"+arm] = s.score, int(s.k)
            gap = n.score-s.score
            case["gap_"+arm] = gap
            for i, t in enumerate((a,b),1):
                case[f"ms{i}_{arm}"] = t.loc[arm,"total_ms"]
                case[f"select_ms{i}_{arm}"] = t.loc[arm,"select_ms"]
            case["ratio_"+arm] = max(t.loc["new","total_ms"]/t.loc[arm,"total_ms"] for t in (a,b))
            for prefix, metric in (("time","total_ms"), ("select","select_ms")):
                case[prefix+"_diff_ms_"+arm] = max(t.loc["new",metric]-t.loc[arm,metric] for t in (a,b))
            case["faster_"+arm] = case["ratio_"+arm] < 1
            case["better_"+arm] = gap > 1e-12
            case["both_"+arm] = case["better_"+arm] and case["faster_"+arm]
        case["cut_faster"] = case["removed"] and case["faster_full"]
        case["cut_faster_within_002"] = case["cut_faster"] and case["gap_full"]>=-.02-1e-12
        rows.append(case)
    result = pd.DataFrame(rows).merge(pd.read_csv(base/"process1/inputs.csv"), on=KEY, validate="one_to_one")
    assert len(result) == 38 and result.gap_previous.eq(0).all()
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out/"cases.csv", index=False)
    med.to_csv(out/"process-medians.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--process", type=int, choices=(1,2))
    action.add_argument("--summary", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.summary:
            summarize(args.base, args.summary)
        else:
            run(args.base, args.process)
