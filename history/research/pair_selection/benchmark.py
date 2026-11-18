"""Replay paired selection-to-prediction timings on the archived benchmark inputs."""
import argparse
import csv
import json
import platform
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_info, threadpool_limits

from research.feature_game_eval import panels as bench
from research.lowdim import extra_study, formal_study, survey_inputs
from research.lowdim import method
from research.lowdim.build_native import library_path
from research.conditional_dense.original_tasks.paper_splits import f_support


HERE = Path(__file__).resolve().parent
INDEX = HERE.parent / "lowdim/evidence/all-cases.csv"
KEY = ["cohort", "dataset", "fold", "seed"]
ARMS = ("pair", "current", "F", "random", "full")
OLD = {"current": "new", "full": "full", "F": "F", "random": "random"}


@dataclass
class Case:
    key: tuple
    X: np.ndarray
    y: np.ndarray
    Xt: np.ndarray
    truth: np.ndarray
    task: str
    impute: bool
    predict: object
    score: object
    archive: Path


def restricted_train(X, y, Xt, task, seed, impute, cols):
    A, B = (X, Xt) if len(cols) == X.shape[1] else (X[:, cols], Xt[:, cols])
    if impute:
        A, B = bench._impute_validation(A, B)
    return formal_study.train("ridge", A, y, B, task, seed)


def four_predict(d, cols):
    A, B = (d.X, d.Xt) if len(cols) == d.X.shape[1] else (d.X[:, cols], d.Xt[:, cols])
    A, B = extra_study.four.encode(A, B, False)
    estimator = Ridge if d.task == "regression" else RidgeClassifier
    return estimator(alpha=1., solver="cholesky").fit(A, d.y).predict(B), None


def core_score(d, truth, units, pred, decision):
    return bench.deployment.score_predictions(d.name, truth, pred, units, d.task, decision)[0]


def standard_score(truth, metric, pred, decision):
    return formal_study.metrics(truth, pred, decision)[metric]


def four_score(d, pred, decision):
    return extra_study.quality(d, pred)


def standard_case(key, X, y, Xt, truth, task, impute, metric, archive):
    return Case(key, X, y, Xt, truth, task, impute,
                partial(restricted_train, X, y, Xt, task, key[-1], impute),
                partial(standard_score, truth, metric), archive)


def cases(wanted):
    root = bench.ARCHIVE
    for cohort, d, truth, units, _ in bench.panels():
        key = (cohort, d.name, d.fold, d.seed)
        yield Case(key, d.X, d.y, d.Xt, truth, d.task, d.name in bench.deployment.ADDITIONAL,
                   partial(bench.fit, d), partial(core_score, d, truth, units),
                   root / "lowdim-core-20261118a-pass1" / ("-".join(map(str, key)) + ".npz"))
    formal_names = sorted({k[1] for k in wanted if k[0] == "formal"})
    for name in formal_names:
        X, y, task, _, order, _, _ = formal_study.load_dataset(name)
        for fold, seed in enumerate(formal_study.P["formal_seeds"], 1):
            key = ("formal", name, fold, seed)
            A, b, B, t, *_ = formal_study.inputs(name, seed, X, y, order, task)
            yield standard_case(key, A, b, B, t, task, False,
                                "r2" if task == "regression" else "auc",
                                root / "lowdim-formal-20261118a-pass1" / f"{name}-{seed}.npz")
    for d, _, _ in extra_study.cases("four_stage_ridge"):
        key = ("four_stage_ridge", d.name, d.fold, d.seed)
        if key in wanted:
            archive = root / "lowdim-breadth-20261118a/four_stage_ridge1" / f"{d.name}-{d.fold}-{d.seed}.npz"
            yield Case(key, d.X, d.y, d.Xt, d.gold, d.task, not np.isfinite(d.X).all(),
                       partial(four_predict, d), partial(four_score, d), archive)
    for cohort, tag, loader in [("archive_supplement", "survey", survey_inputs.study_cases),
                                ("conference_fraud", "fraud", survey_inputs.fraud_cases)]:
        remaining = {k for k in wanted if k[0] == cohort}
        for d in loader():
            key = (cohort, d.dataset, d.fold, d.seed)
            if key in remaining:
                archive = root / f"lowdim-{tag}-20261118a-pass1" / f"{d.dataset}-{d.fold}-{d.seed}.npz"
                yield standard_case(key, d.X, d.y, d.Xt, d.truth, d.task, d.impute, d.metric, archive)
                remaining.remove(key)
                if not remaining:
                    break


def deploy(case, arm, width):
    start = perf_counter_ns()
    X, seed = case.X, case.key[-1]
    if case.impute and arm in ("pair", "current", "F"):
        X = bench._impute_validation(X, X[:0])[0]
    if arm == "pair":
        cols = method.pair_support(X, case.y, task=case.task, seed=seed)[0]
    elif arm == "current":
        cols = method.fast_support(X, case.y, task=case.task, seed=seed)[0]
    elif arm == "F":
        cols = f_support(X, case.y, case.task, width)
    elif arm == "random":
        cols = np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:width])
    elif arm == "full":
        cols = np.arange(X.shape[1])
    else:
        raise ValueError(arm)
    selected = perf_counter_ns()
    pred, decision = case.predict(cols)
    end = perf_counter_ns()
    return cols, pred, decision, (selected - start) / 1e6, (end - start) / 1e6


def check_repeat(reference, cols, pred, decision):
    np.testing.assert_array_equal(cols, reference[0])
    np.testing.assert_array_equal(pred, reference[1])
    if reference[2] is None:
        assert decision is None
    else:
        np.testing.assert_array_equal(decision, reference[2])


def external_cases():
    from research.pair_selection.external import external_panels
    result = []
    for d in external_panels():
        metric = "r2" if d.task == "regression" else "auc"
        result.append(Case(("external", d.name, d.fold, d.seed), d.X, d.y, d.Xt, d.truth,
                           d.task, False, partial(bench.fit, d),
                           partial(standard_score, d.truth, metric), None))
    return result


def source_hashes(external):
    sources = [Path(__file__), INDEX, Path(method.__file__), HERE.parent / "lowdim/native_votes.c",
               library_path(), Path(bench.__file__), Path(formal_study.__file__),
               Path(extra_study.__file__), Path(survey_inputs.__file__)]
    if external:
        sources.append(HERE / "external.py")
    return {p.relative_to(bench.ROOT).as_posix(): bench.checksum(p) for p in sources}


def replay(out, process, previous=None, external=False):
    manifest = pd.read_csv(INDEX, keep_default_na=False)
    manifest = manifest[(manifest.p <= 32) | manifest.cohort.isin(["original", "extra"])].set_index(KEY)
    assert manifest.index.is_unique and len(manifest) == 161 and (manifest.p <= 32).sum() == 56
    if external:
        inputs = external_cases()
        manifest = pd.DataFrame([
            dict(zip(KEY, c.key), p=c.X.shape[1], task="binary" if c.task == "classification" else "regression",
                 metric="AUC" if c.task == "classification" else "R2")
            for c in inputs]).set_index(KEY)
        assert len(manifest) == 12 and manifest.index.is_unique
    else:
        inputs = cases(set(manifest.index))
    hashes = source_hashes(external)
    if previous is not None:
        assert json.loads((previous / "complete.json").read_text())["source_sha256"] == hashes
    out.mkdir(parents=True, exist_ok=False)
    seen, accuracy, fingerprints = set(), [], []
    measures = KEY + ["method", "repeat", "position", "select_ms", "total_ms"]
    with (out / "times.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=measures)
        writer.writeheader()
        for case in inputs:
            assert case.key not in seen
            seen.add(case.key)
            record = manifest.loc[case.key]
            cfg = dict(zip(KEY, case.key))
            assert record.p == case.X.shape[1]
            signature = {}
            for array, field in zip([case.X, case.y, case.Xt, case.truth],
                                    ["train_sha", "target_sha", "test_sha", "truth_sha"]):
                signature[field] = bench.digest(array)
                if not external:
                    assert signature[field] == record[field], (case.key, field)
            fingerprints.append(dict(**cfg, **signature, n_train=len(case.y), n_test=len(case.truth)))
            expected, arrays, width = {}, {}, None
            with (np.load(case.archive) if case.archive is not None else nullcontext(None)) as archived:
                if archived is not None:
                    np.testing.assert_array_equal(case.truth, archived["truth"])
                for arm in ARMS:
                    cols, pred, decision, _, _ = deploy(case, arm, width)
                    if arm == "pair":
                        width = len(cols)
                        if not external:
                            assert width == record.k
                    assert len(cols) == (record.p if arm == "full" else width)
                    assert np.array_equal(cols, np.unique(cols)) and 0 <= cols[0] <= cols[-1] < record.p
                    score = case.score(pred, decision)
                    if archived is not None and arm in OLD:
                        prefix = OLD[arm]
                        np.testing.assert_array_equal(cols, archived[prefix + "_cols"])
                        np.testing.assert_array_equal(pred, archived[prefix + "_pred"])
                        if prefix + "_decision" in archived.files:
                            np.testing.assert_array_equal(decision, archived[prefix + "_decision"])
                        np.testing.assert_allclose(score, record["score_" + prefix], atol=1e-12, rtol=1e-12)
                    if arm == "current" and record.p > 32:
                        check_repeat(expected["pair"], cols, pred, decision)
                    expected[arm] = cols, pred, decision
                    arrays[arm + "_cols"], arrays[arm + "_pred"] = cols, pred
                    if decision is not None:
                        arrays[arm + "_decision"] = decision
                    row = dict(**cfg, task=record.task, metric=record.metric, p=record.p,
                               k=len(cols), method=arm, score=score, cols=" ".join(map(str, cols)))
                    if case.key[1] in ("credit_card_fraud", "secom"):
                        row["average_precision"] = average_precision_score(case.truth, decision)
                    accuracy.append(row)
            filename = "-".join(map(str, case.key)) + ".npz"
            if previous is None:
                np.savez_compressed(out / filename, **arrays)
            else:
                with np.load(previous / filename) as first:
                    assert set(first.files) == set(arrays)
                    for name, value in arrays.items():
                        np.testing.assert_array_equal(value, first[name])
            schedule = np.random.default_rng([59161, *case.key[2:], process]).permutation(ARMS)
            for repeat in range(len(ARMS)):
                for position, arm in enumerate(np.roll(schedule, repeat)):
                    cols, pred, decision, select_ms, total_ms = deploy(case, arm, width)
                    writer.writerow(dict(**cfg, repeat=repeat, position=position, method=arm,
                                         select_ms=select_ms, total_ms=total_ms))
                    check_repeat(expected[arm], cols, pred, decision)
            stream.flush()
            if len(seen) % 20 == 0 or len(seen) == len(manifest):
                print(f"process {process}: {len(seen)}/{len(manifest)}", flush=True)
    assert seen == set(manifest.index) and source_hashes(external) == hashes
    fingerprint_frame = pd.DataFrame(fingerprints)
    if previous is not None:
        pd.testing.assert_frame_equal(fingerprint_frame, pd.read_csv(previous / "inputs.csv", keep_default_na=False))
    fingerprint_frame.to_csv(out / "inputs.csv", index=False)
    pd.DataFrame(accuracy).to_csv(out / "quality.csv", index=False)
    state = dict(process=process, cases=len(seen), arms=ARMS, threads=threadpool_info(),
                 python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                 source_sha256=hashes)
    (out / "complete.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        replay(args.out, 2 if args.previous else 1, args.previous, args.external)
