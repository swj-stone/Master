"""Supplementary transfer panel with raw splits outside the original registry.

Gasoline uses a shuffled 75:25 split by seed. Yarn uses its published train flag.
Olive uses the supplied train/test tables, excluding the row ID and C16:0 target.
Their predictors are the numeric spectral measurements, with no wavelength screening.
All three use train-standardized LSQR ridge and held-out R-squared. Real-sim uses its
supplied CSR split and binary labels, train-scaled uncentered LSQR ridge
classification and balanced accuracy. All estimators share the paper's alpha,
tolerance and iteration limit. Only Gasoline has three different test samples.

Run two separate invocations with OUTPUT --process 1 and OUTPUT --process 2,
followed by OUTPUT --summarize. Scores and timings are paired within each case.
"""
import argparse
import csv
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

from research.fast_single import runner
from research.fast_single.summarize import read_case, STABLE
from research.peer_crossover import runner as replay
from research.wide_blocks import panels
from research.wide_blocks.benchmark import audit_f, check_f, compare_arrays


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw"
SPECTRA = DATA / "spectral-landmarks"
REAL = DATA / "full-scale-trials/prepared/real_sim"
FILES = {"gasoline": [SPECTRA / "gasoline.json"], "yarn": [SPECTRA / "yarn.json"],
         "olive": [SPECTRA / "olive_train.csv", SPECTRA / "olive_test.csv"],
         "real_sim": [REAL / n for n in ("X.npz", "Xt.npz", "labels.npz")]}
SEEDS = (1129, 2027, 3469)
REPEATS = 3
REGISTRY = tuple((name, seed) for name in FILES for seed in SEEDS)
SUMMARY = Path(__file__).resolve().parent / "evidence/transfer.csv"


def arrays(name, seed):
    paths = FILES[name]
    if name == "real_sim":
        with np.load(paths[2]) as labels:
            return (sparse.load_npz(paths[0]).tocsr(), labels["y"],
                    sparse.load_npz(paths[1]).tocsr(), labels["yt"])
    if name == "olive":
        train, test = (pd.read_csv(path) for path in paths)
        if not train.columns.equals(test.columns):
            raise ValueError("Olive train and test wavelength columns differ")
        cols = train.columns.drop(["Unnamed: 0", "C16:0"])
        return (train[cols].to_numpy(dtype=np.float64), train["C16:0"].to_numpy(),
                test[cols].to_numpy(dtype=np.float64), test["C16:0"].to_numpy())
    rows = json.loads(paths[0].read_text(encoding="utf8"))
    X = np.asarray([r["NIR"] for r in rows], dtype=np.float64)
    if name == "gasoline":
        y = np.asarray([r["octane"] for r in rows], dtype=np.float64)
        a, b = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed)
    else:
        y = np.asarray([r["density"] for r in rows], dtype=np.float64)
        train = np.asarray([r["train"] for r in rows])
        if train.dtype != np.bool_:
            raise ValueError("Yarn train flag must be boolean")
        a, b = train, ~train
    return X[a], y[a], X[b], y[b]


def case_for(name, seed):
    X, y, Xt, truth = arrays(name, seed)
    if X.shape[0] != len(y) or Xt.shape != (len(truth), X.shape[1]):
        raise ValueError(f"Feature/response shape mismatch on {name}")
    if sparse.issparse(X):
        task, metric = "classification", "balanced_accuracy"
        values = (X.data, Xt.data, y, truth)
        predict = partial(panels.sparse_predict, X, y, Xt, seed)
        score = partial(panels.sparse_score, truth)
    else:
        task, metric = "regression", "r2"
        values = (X, Xt, y, truth)
        predict = partial(panels.prior.restricted_train, X, y, Xt, task, seed, False)
        score = partial(panels.prior.standard_score, truth, metric)
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError(f"Nonfinite transfer input on {name}")
    return panels.Case(("transfer", name, 0, seed), X, y, Xt, truth, task, metric,
                       FILES[name][0], "", predict, score)


def matched_reference(case, select_fast):
    p = case.X.shape[1]
    aliases, refs, stats, widths = {}, {}, {}, {}
    for arm in runner.ARMS:
        kind, _, owner = arm.partition("_")
        k = widths.get(owner)
        source = kind + "_fast" if owner == "single" and k == widths["fast"] else arm
        aliases[arm] = source
        if source != arm:
            continue
        fields, _, _ = runner.deploy(case, arm, k, select_fast)
        cols, pred, decision = fields
        np.testing.assert_array_equal(cols, np.unique(cols))
        if not (0 < len(cols) <= p and 0 <= cols[0] <= cols[-1] < p):
            raise ValueError(f"Invalid support on {case.key}, {arm}")
        if arm in ("fast", "single"):
            widths[arm] = len(cols)
        elif arm == "full":
            np.testing.assert_array_equal(cols, np.arange(p))
        else:
            if len(cols) != k:
                raise ValueError(f"Width mismatch on {case.key}, {arm}")
            if kind == "random":
                expected = np.random.default_rng(case.seed).permutation(p)[:k]
                np.testing.assert_array_equal(cols, np.sort(expected))
            elif sparse.issparse(case.X):
                audit_f(case.X, case.y, cols)
            else:
                check_f(case.X, case.y, case.task, cols)
        score = float(case.score(pred, decision))
        if not np.isfinite(score):
            raise ValueError(f"Score is nonfinite on {case.key}, {arm}")
        stats[arm] = dict(k=len(cols), score=score, **dict(zip(
            replay.HASHES, (panels.bench.digest(a) if a is not None else "" for a in fields))))
        refs[arm] = fields
    return aliases, refs, stats, widths


def checked_json(path, value, process):
    if path.exists():
        if json.loads(path.read_text(encoding="utf8")) != value:
            raise ValueError(f"Data, code or prediction fingerprint differs at {path}")
    elif process == 1:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2), encoding="utf8")
        os.replace(temp, path)
    else:
        raise ValueError(f"Missing pass-one reference {path}")


def measure(case, case_id, base, process, select_fast):
    out = base / f"pass{process}/case-{case_id:03}.csv"
    meta = base / f"input-{case_id:03}.json"
    aliases, refs, stats, widths = matched_reference(case, select_fast)
    info = dict(zip(panels.KEY, case.key), case_id=case_id, p=case.X.shape[1],
                task=case.task, metric=case.metric, n_train=len(case.y), n_test=len(case.truth))
    fingerprint = {f"{field}_{label}": value for label, source in aliases.items()
                   for field, value in stats[source].items()}
    checked_json(meta, {**info, **fingerprint, **dict(zip(
        panels.HASHES, panels.input_hashes(case)))}, process)
    # Every process independently obtains the same columns and predictions before timing.
    schedule = np.random.default_rng([216712, *case.key[2:]]).permutation(list(refs))
    temp = out.with_suffix(".tmp")
    with temp.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=replay.FIELDS)
        writer.writeheader()
        header = {k: v for k, v in info.items() if k in replay.FIELDS}
        for rep in range(REPEATS):
            order = np.roll(schedule, rep * len(schedule) // REPEATS)
            for pos, arm in enumerate(order if process == 1 else order[::-1]):
                k = widths.get(arm.partition("_")[2])
                result, select_ms, total_ms = runner.deploy(case, arm, k, select_fast)
                compare_arrays(result, refs[arm])
                for label, source in aliases.items():
                    if source == arm:
                        writer.writerow(dict(header, process=process, repeat=rep, position=pos,
                                             timed_arm=arm, arm=label, select_ms=select_ms,
                                             total_ms=total_ms, **stats[arm]))
    expected = pd.Series({**info, **fingerprint})
    read_case(temp, case_id, process, expected, REPEATS)
    os.replace(temp, out)


def run(base, process):
    from research.fast_single import selector

    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise ValueError("Expected single-threaded numerical runtimes")
    (base / f"pass{process}").mkdir(parents=True, exist_ok=True)
    spec = runner.protocol(REPEATS, selector)
    sources = [Path(__file__), ROOT / "configs/minpi-study.json",
               ROOT / "experiments/minpi_constant_transfer.py", *(p for files in FILES.values() for p in files)]
    for path in sources:
        spec["hashes"][path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    spec.update(seeds=list(SEEDS), panel="supplementary-transfer", preprocessing=__doc__,
                datasets=list(FILES))
    checked_json(base / "protocol.json", spec, process)
    for case_id, (name, seed) in enumerate(REGISTRY):
        filename = f"case-{case_id:03}.csv"
        if process == 2 and not (base / "pass1" / filename).is_file():
            raise ValueError(f"Missing first pass for {name}, {seed}")
        if not (base / f"pass{process}" / filename).exists():
            measure(case_for(name, seed), case_id, base, process, selector.select_fast)
            print(f"pass {process}, {case_id+1}/{len(REGISTRY)}, {name}, {seed}", flush=True)


def summarize(base, dest):
    records = []
    for i, (name, seed) in enumerate(REGISTRY):
        meta = pd.Series(json.loads((base / f"input-{i:03}.json").read_text(encoding="utf8")))
        if (meta.dataset, meta.seed) != (name, seed):
            raise ValueError(f"Case registry mismatch for {i}")
        a, b = (read_case(base / f"pass{proc}/case-{i:03}.csv", i, proc, meta, REPEATS)
                for proc in (1, 2))
        pd.testing.assert_frame_equal(a[STABLE], b[STABLE])
        for arm in runner.ARMS:
            cols = int(a.at[arm, "k"])
            record = dict(dataset=name, seed=seed, n_train=meta.n_train, n_test=meta.n_test,
                          metric=meta.metric, p=meta.p, arm=arm, k=cols,
                          score=a.at[arm, "score"], loss_vs_full=a.at["full", "score"]-a.at[arm, "score"])
            for proc, table in ((1, a), (2, b)):
                record[f"select_ms_{proc}"] = table.at[arm, "select_ms"]
                record[f"total_ms_{proc}"] = table.at[arm, "total_ms"]
            owner = arm if arm in ("single", "fast") else arm.partition("_")[2]
            comparisons = {"full": "full"}
            if owner:
                comparisons.update(random="random_"+owner, F="F_"+owner)
            for label, baseline in comparisons.items():
                ratios = [m.at[arm, "total_ms"] / m.at[baseline, "total_ms"] for m in (a, b)]
                record[f"worst_ratio_vs_{label}"] = max(ratios)
                record[f"delta_score_vs_{label}"] = a.at[arm, "score"] - a.at[baseline, "score"]
            record["three_way_002"] = (cols < meta.p and record["loss_vs_full"] <= .02 + 1e-12
                                       and record["worst_ratio_vs_full"] < 1.)
            records.append(record)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(dest, index=False)
    print(f"{len(records)} case-arm records written to {dest}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--process", choices=(1, 2), type=int)
    action.add_argument("--summarize", action="store_true")
    cli.add_argument("--out", type=Path, default=SUMMARY)
    args = cli.parse_args()
    if args.summarize:
        summarize(args.output, args.out)
    else:
        with threadpool_limits(1):
            run(args.output, args.process)
