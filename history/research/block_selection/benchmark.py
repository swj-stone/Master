"""Compare adaptive block selection with the archived selectors and new-width controls."""
import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_info, threadpool_limits

from research.conditional_dense.original_tasks.paper_splits import check_f
from research.pair_selection import benchmark as prior


HERE = Path(__file__).resolve().parent
ARCHIVE = prior.bench.ARCHIVE / "pair-completion-20261118a"
KEY = prior.KEY
ARMS = ("block", "pair", "current", "F", "random", "full")


def deploy(case, arm, width):
    if arm != "block":
        return prior.deploy(case, arm, width)
    start = perf_counter_ns()
    X = case.X
    if case.impute:
        X = prior.bench._impute_validation(X, X[:0])[0]
    cols = prior.method.block_support(X, case.y, task=case.task, seed=case.key[-1])[0]
    selected = perf_counter_ns()
    pred, decision = case.predict(cols)
    end = perf_counter_ns()
    return cols, pred, decision, (selected-start)/1e6, (end-start)/1e6


def source_hashes(suite):
    root = prior.bench.ROOT
    files = [Path(__file__), *sorted((HERE.parent / "lowdim").glob("*.c"))]
    if suite == "fresh":
        files.append(HERE/"fresh.py")
    return {**prior.source_hashes(suite == "external"), **{f.relative_to(root).as_posix():prior.bench.checksum(f)
                                               for f in files}}


def validate_control(case, arm, cols, pred, decision, old, width_changed):
    if arm == "block":
        if case.X.shape[1] <= prior.method.MAX_FEATURES:
            return
        old_arm = "current"
    else:
        old_arm = arm
    if (width_changed or old is None) and arm in ("F", "random"):
        if arm == "random":
            perm = np.random.default_rng(case.key[-1]).permutation(case.X.shape[1])
            np.testing.assert_array_equal(cols, np.sort(perm[:len(cols)]))
        else:
            X = prior.bench._impute_validation(case.X,case.X[:0])[0] if case.impute else case.X
            check_f(X, case.y, case.task, cols)
        return
    if old is None:
        return
    np.testing.assert_array_equal(cols, old[old_arm + "_cols"])
    np.testing.assert_array_equal(pred, old[old_arm + "_pred"])
    if old_arm + "_decision" in old.files:
        np.testing.assert_array_equal(decision, old[old_arm + "_decision"])
    else:
        assert decision is None


def training_record(case):
    arrays = (case.X, case.y, case.Xt, case.truth)
    fields = ("train_sha", "target_sha", "test_sha", "truth_sha")
    return dict(zip(KEY,case.key), n_train=len(case.X), n_test=len(case.Xt),
                **{f:prior.bench.digest(a) for f,a in zip(fields, arrays)})


def run(out, previous, suite):
    base = ARCHIVE / (suite + "-pass1")
    if suite == "fresh":
        from research.block_selection.fresh import fresh_cases
        cases = list(fresh_cases())
        old_inputs = pd.DataFrame(map(training_record, cases)).set_index(KEY)
        old_quality = None
        n = 6
    else:
        old_inputs = pd.read_csv(base / "inputs.csv", keep_default_na=False).set_index(KEY)
        old_quality = pd.read_csv(base / "quality.csv", keep_default_na=False).set_index(KEY+["method"])
        cases = prior.external_cases() if suite=="external" else prior.cases(set(old_inputs.index))
        n = 12 if suite=="external" else 161
    assert old_inputs.index.is_unique and len(old_inputs) == n
    process = 2 if previous else 1
    hashes = source_hashes(suite)
    if previous:
        assert json.loads((previous/"complete.json").read_text())["source_sha256"] == hashes
    out.mkdir(parents=True, exist_ok=False)
    if suite=="fresh":
        old_inputs.reset_index().to_csv(out/"inputs.csv", index=False)
        if previous:
            assert prior.bench.checksum(out/"inputs.csv") == prior.bench.checksum(previous/"inputs.csv")
    scores, done = [], set()
    with (out/"times.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=KEY+["method","repeat","position","select_ms","total_ms"])
        writer.writeheader()
        for case in cases:
            key = case.key
            assert key not in done
            done.add(key)
            cfg = dict(zip(KEY, key))
            fingerprint = old_inputs.loc[key]
            for array, field in zip([case.X, case.y, case.Xt, case.truth],
                                    ["train_sha", "target_sha", "test_sha", "truth_sha"]):
                assert prior.bench.digest(array) == fingerprint[field], (key, field)
            row = old_quality.loc[key+("current",)] if old_quality is not None else None
            task = row.task if row is not None else "binary" if case.task=="classification" else "regression"
            metric = row.metric if row is not None else "AUC" if task=="binary" else "R2"
            p = case.X.shape[1]
            width, reference = None, {}
            previous_case = base/("-".join(map(str,key))+".npz")
            with (np.load(previous_case) if row is not None else nullcontext(None)) as old:
                for arm in ARMS:
                    cols,pred,decision,_,_ = deploy(case,arm,width)
                    if arm == "block":
                        width = len(cols)
                    if arm=="full":
                        assert len(cols)==p
                    elif arm in ("current","pair"):
                        if row is not None:
                            assert len(cols)==row.k
                        if arm=="current":
                            assert len(cols)==len(reference["pair"][0])
                    else:
                        assert len(cols)==width
                    assert 1 <= len(cols) <= p
                    assert np.array_equal(cols,np.unique(cols)) and 0 <= cols[0] <= cols[-1] < p
                    width_changed = row is None or width != row.k
                    validate_control(case,arm,cols,pred,decision,old,width_changed)
                    score = case.score(pred,decision)
                    if row is not None and arm in prior.ARMS and not (width_changed and arm in ("F","random")):
                        np.testing.assert_allclose(score,old_quality.loc[key+(arm,), "score"],rtol=1e-12,atol=1e-12)
                    reference[arm] = cols,pred,decision
                    record = dict(**cfg, task=task, metric=metric, p=p,
                                  method=arm, k=len(cols), score=score, cols=" ".join(map(str,cols)),
                                  prediction_sha=prior.bench.digest(pred),
                                  decision_sha=prior.bench.digest(decision) if decision is not None else "")
                    if key[1] in ("credit_card_fraud","secom"):
                        record["average_precision"] = average_precision_score(case.truth,decision)
                    scores.append(record)
            schedule = np.random.default_rng([941093,*key[2:],process]).permutation(ARMS)
            for repeat in range(len(ARMS)):
                for position, arm in enumerate(np.roll(schedule,repeat)):
                    cols,pred,decision,select_ms,total_ms = deploy(case, arm, width)
                    writer.writerow(dict(**cfg,method=arm,repeat=repeat,position=position,
                                         select_ms=select_ms,total_ms=total_ms))
                    prior.check_repeat(reference[arm],cols,pred,decision)
            stream.flush()
            if len(done)%30 == 0 or len(done)==n:
                print(f"{suite}, process {process}: {len(done)}/{n}",flush=True)
    assert done == set(old_inputs.index) and hashes == source_hashes(suite)
    pd.DataFrame(scores).to_csv(out/"quality.csv",index=False)
    if previous:
        first = pd.read_csv(previous/"quality.csv",keep_default_na=False)
        now = pd.read_csv(out/"quality.csv",keep_default_na=False)
        pd.testing.assert_frame_equal(first,now,check_exact=True)
    info=dict(process=process,cases=n,arms=ARMS,threads=threadpool_info(),source_sha256=hashes,
              sklearn=prior.sklearn.__version__,numpy=np.__version__,python=prior.platform.python_version(),
              previous_run=base.name if suite!="fresh" else None,
              inputs_sha256=prior.bench.checksum((out if suite=="fresh" else base)/"inputs.csv"))
    (out/"complete.json").write_text(json.dumps(info,indent=2),encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--previous",type=Path)
    suite=p.add_mutually_exclusive_group()
    suite.add_argument("--external",dest="suite",action="store_const",const="external")
    suite.add_argument("--fresh",dest="suite",action="store_const",const="fresh")
    p.set_defaults(suite="archive")
    args=p.parse_args()
    with threadpool_limits(1):
        run(args.out,args.previous,args.suite)
