"""Replay all wide archived cases with paired warm selection-to-prediction timings."""
import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_info, threadpool_limits

from research.conditional_dense.original_tasks.paper_splits import check_f
from research.feature_game_eval import panels as core
from research.lowdim.build_native import library_path
from research.lowdim.text_study import audit_f


HERE = Path(__file__).resolve().parent
INDEX = HERE.parent / "lowdim/evidence/all-cases.csv"
KEY = ["cohort", "dataset", "fold", "seed"]
ARMS = ("wide", "cap", "old", "F", "random", "full")
ID = KEY + ["method"]
TIME_FIELDS = ID + ["repeat", "position", "select_ms", "total_ms"]
QUALITY_FIELDS = ID + ["p", "k", "score", "average_precision", "route", "screen_width",
                       "width_cap", "columns_sha", "prediction_sha", "decision_sha"]
META = ["task", "metric", "n_train", "n_test", "family", "source_kind", "source",
        "train_sha", "target_sha", "test_sha", "truth_sha"]


def manifest():
    index = pd.read_csv(INDEX, keep_default_na=False).set_index(KEY)
    index = index[index.p > 32]
    assert len(index) == 331 and index.index.is_unique
    return index


def signature():
    from research.wide_blocks.build_native import library_path as wide_library_path
    files = [INDEX, library_path(), HERE.parent/"lowdim/native_votes.c",
             HERE.parent/"lowdim/method.py", HERE.parent/"lowdim/wide_csr.py",
             HERE.parent/"lowdim/text_study.py", HERE.parent/"lowdim/panel_study.py",
             HERE.parent/"lowdim/extra_study.py", HERE.parent/"lowdim/formal_study.py",
             HERE.parent/"lowdim/survey_inputs.py", HERE.parent/"conditional_dense/shared_method.py",
             HERE.parent/"conditional_dense/vote_width/method.py",
             HERE.parent/"conditional_dense/original_tasks/paper_splits.py",
             HERE.parent/"conditional_dense/original_tasks/regression.py",
             HERE.parent/"pair_selection/benchmark.py", HERE.parent/"lowdim/verify_combined.py",
             HERE.parent/"four_stage/benchmark.py", core.ROOT/"experiments/minpi_study.py",
             core.ROOT/"experiments/grouped_solver_cost_audit.py",
             core.ROOT/"experiments/minpi_constant_transfer.py",
             Path(core.__file__), HERE/"benchmark.py", HERE/"panels.py", HERE/"method.py", HERE/"protocol.md",
             HERE/"build_native.py", HERE/"native.c", wide_library_path()]
    return {p.relative_to(core.ROOT).as_posix():core.checksum(p) for p in files}


def deploy(case, arm, k, selector):
    start = perf_counter_ns()
    info = None
    if arm in ("wide", "cap"):
        cols, info = selector(case.selection_X(), case.y, task=case.task,
                              seed=case.seed, fill_to_cap=arm=="cap")
    elif arm == "old":
        cols = case.select_old()
    elif arm == "F":
        cols = case.select_F(k)
    elif arm == "random":
        cols = case.select_random(k)
    elif arm == "full":
        cols = np.arange(case.X.shape[1])
    else:
        raise ValueError(arm)
    selected = perf_counter_ns()
    pred, decision = case.predict(cols)
    end = perf_counter_ns()
    return cols, pred, decision, info, (selected-start)/1e6, (end-start)/1e6


def compare_arrays(actual, ref):
    for a, b in zip(actual, ref, strict=True):
        if b is None:
            assert a is None
        else:
            np.testing.assert_array_equal(a, b)


def validate_control(case, arm, k, fields, archive, old):
    cols, pred, decision = fields
    if arm == "old":
        archive_arm = "new"
    elif arm == "full" or (arm in ("F","random") and k == int(old.k_new)):
        archive_arm = arm
    else:
        archive_arm = None
    if archive_arm:
        np.testing.assert_array_equal(cols, archive[archive_arm+"_cols"])
        np.testing.assert_array_equal(pred, archive[archive_arm+"_pred"])
        if archive_arm+"_decision" in archive:
            np.testing.assert_array_equal(decision, archive[archive_arm+"_decision"])
        np.testing.assert_allclose(case.score(pred,decision),float(old["score_"+archive_arm]),
                                   rtol=0,atol=2e-12)
    if arm == "random":
        expected = np.random.default_rng(case.seed).permutation(case.X.shape[1])
        np.testing.assert_array_equal(cols, np.sort(expected[:k]))
    elif arm == "F":
        X = case.selection_X()
        if sparse.issparse(X):
            audit_f(X, case.y, cols)
        else:
            check_f(X, case.y, case.task, cols)


def warmup(case, selector, old):
    width, reference, info_by_arm, quality = None, {}, {}, []
    with np.load(case.archive) as archive:
        for arm in ARMS:
            cols,pred,decision,info,_,_ = deploy(case,arm,width,selector)
            p = case.X.shape[1]
            assert 1 <= len(cols) <= p and 0 <= cols[0] <= cols[-1] < p
            np.testing.assert_array_equal(cols,np.unique(cols))
            if arm == "wide":
                width = len(cols)
            if arm in ("F","random"):
                assert len(cols) == width
            if arm == "full":
                assert len(cols) == p
            reference[arm] = (cols,pred,decision)
            validate_control(case,arm,width,reference[arm],archive,old)
            record = dict(zip(KEY,case.key), method=arm, p=p, k=len(cols),
                          score=case.score(pred,decision), columns_sha=core.digest(cols),
                          prediction_sha=core.digest(pred), decision_sha=core.digest(decision) if decision is not None else "")
            if case.key[1] in ("secom","credit_card_fraud"):
                record["average_precision"] = average_precision_score(case.truth,decision)
            if info is not None:
                no_signal = bool(info.get("no_signal",False))
                record["route"] = "full_no_signal" if no_signal else ("screened" if info["screened"] else "exact")
                record["screen_width"] = int(info["candidate_count"])
                record["width_cap"] = int(info["width_cap"])
                assert 1 <= record["screen_width"] <= p
                if no_signal:
                    assert record["width_cap"] == len(cols) == p
                else:
                    assert len(cols) <= record["width_cap"] <= record["screen_width"]
                assert int(info["selected_width"]) == len(cols)
                info_by_arm[arm] = (record["route"],record["screen_width"],record["width_cap"])
                if arm == "cap":
                    assert len(cols) == record["width_cap"] and info_by_arm["wide"] == info_by_arm[arm]
            quality.append(record)
    return width, reference, quality


def selected_arrays(reference):
    return {arm+"_"+suffix:v for arm in ("wide","cap","F","random")
            for suffix,v in zip(("cols","pred","decision"),reference[arm]) if v is not None}


def run(base, process):
    from research.wide_blocks.method import wide_support
    from research.wide_blocks.panels import cases
    frozen = manifest()
    wanted = set(frozen.index)
    folder = base/f"pass{process}"
    first = base/"pass1"
    hashes = signature()
    if process == 2:
        assert json.loads((first/"complete.json").read_text())["sources"] == hashes
    folder.mkdir(parents=True,exist_ok=False)
    seen = set()
    with (folder/"times.csv").open("x",newline="") as timing, (folder/"quality.csv").open("x",newline="") as output:
        times = csv.DictWriter(timing,fieldnames=TIME_FIELDS)
        quality = csv.DictWriter(output,fieldnames=QUALITY_FIELDS)
        times.writeheader()
        quality.writeheader()
        for case in cases(wanted):
            assert case.key in wanted and case.key not in seen
            seen.add(case.key)
            width, ref, rows = warmup(case, wide_support, frozen.loc[case.key])
            quality.writerows(rows)
            arrays = selected_arrays(ref)
            artifact = "-".join(map(str,case.key)) + ".npz"
            if process == 1:
                np.savez_compressed(first/artifact,**arrays)
            else:
                with np.load(first/artifact) as old:
                    assert set(old.files) == set(arrays)
                    for name,a in arrays.items():
                        np.testing.assert_array_equal(a,old[name])
            schedule = np.random.default_rng([1936207, *case.key[2:], process]).permutation(ARMS)
            for repeat in range(len(ARMS)):
                for position,arm in enumerate(np.roll(schedule,repeat)):
                    cols,pred,decision,_,sel_ms,total_ms = deploy(case,arm,width,wide_support)
                    times.writerow(dict(zip(KEY,case.key),method=arm,repeat=repeat,position=position,
                                        select_ms=sel_ms,total_ms=total_ms))
                    compare_arrays((cols,pred,decision),ref[arm])
            output.flush()
            timing.flush()
            if len(seen)%10 == 0 or len(seen) == len(wanted):
                print(f"pass {process} {len(seen)}/{len(wanted)} {case.key[0]} {case.key[1]}",flush=True)
    assert seen == wanted and signature() == hashes
    if process == 2:
        pd.testing.assert_frame_equal(read_quality(first),read_quality(folder),check_exact=True)
    info = dict(cases=len(seen),process=process,methods=ARMS,threads=threadpool_info(),sources=hashes,
                numpy=np.__version__,sklearn=sklearn.__version__,python=platform.python_version())
    assert all(x["num_threads"]==1 for x in info["threads"])
    (folder/"complete.json").write_text(json.dumps(info,indent=2),encoding="utf-8")


def read_quality(folder):
    return pd.read_csv(folder/"quality.csv",keep_default_na=False).set_index(ID).sort_index()


def summarize(base):
    frozen = manifest()
    quality = read_quality(base/"pass1")
    medians = []
    for process in (1,2):
        folder = base/f"pass{process}"
        info = json.loads((folder/"complete.json").read_text())
        assert info["cases"]==len(frozen) and info["process"]==process
        pd.testing.assert_frame_equal(quality,read_quality(folder),check_exact=True)
        raw = pd.read_csv(folder/"times.csv",keep_default_na=False)
        assert len(raw)==len(frozen)*len(ARMS)**2 and (raw.select_ms<=raw.total_ms).all()
        assert not raw.duplicated(ID+["position"]).any() and not raw.duplicated(ID+["repeat"]).any()
        assert not raw.duplicated(KEY+["repeat","position"]).any()
        medians.append(raw.groupby(ID)[["select_ms","total_ms"]].median().add_suffix(str(process)))
    assert quality.index.is_unique and len(quality) == len(frozen)*len(ARMS)
    joined = quality.join(medians,how="inner").reset_index().merge(
        frozen[META].reset_index(),on=KEY,validate="many_to_one")
    assert len(joined)==len(quality)
    wide = joined[joined.method=="wide"].set_index(KEY)
    for arm in ("F","random"):
        assert wide.k.equals(joined[joined.method==arm].set_index(KEY).k)
    out = HERE/"evidence"
    out.mkdir(exist_ok=True)
    joined.to_csv(out/"wide-method-cases.csv",index=False)
    print(f"Verified {len(joined)} method rows and {2*len(raw)} timed pipelines")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--process",type=int,choices=(1,2))
    action.add_argument("--summary",action="store_true")
    args=parser.parse_args()
    if args.summary:
        summarize(args.run_dir)
    else:
        with threadpool_limits(1):
            run(args.run_dir,args.process)
