"""Frozen half-width evaluation on additional cohorts and three prior shift holdouts."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.minpi_har_confirmation import full_fit_pfi as by_subject_pfi
from experiments.minpi_iteration_benchmark import full_fit_pfi
from experiments.minpi_stable_shift import SEEDS, digest, get_shift
from experiments.minpi_study import CONFIG, ROOT, file_hash, load_dataset, train
from experiments.minpi_study_device import RAW_SHA256
from experiments.minpi_study_new_cohorts import DATA, DIGESTS, batch_scores
from experiments.minpi_study_subjects import DIGEST
from experiments.pivotal_wide_comparison import held_permutation_scores
from minppi_mofs.backward_minpi import backward_top_from_training
from minppi_mofs.environment_minpi import environment_top_from_training
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training


SOURCES = ("mhealth", "diabetes", "isolet", "gas", "supercon", "har")
METHODS = ("backward", "forward", "local", "F", "PFI", "full")


def splits(source):
    if source in ("gas", "har", "supercon"):
        yield SEEDS[0], get_shift(source)
    elif source == "mhealth":
        with np.load(DATA/"mhealth-windows.npz") as z, np.load(ROOT/"results/minpi-study/device/records.npz") as d:
            assert z["raw_sha256"].item() == RAW_SHA256
            ids, windows = z["ids"], z["windows"]
            fit, test, sensors = d["dev"], d["test"], d["train_device"]
            np.testing.assert_array_equal(fit, np.flatnonzero(ids[:,0] <= 5))
            np.testing.assert_array_equal(test, np.flatnonzero(ids[:,0] >= 6))
            X = windows[fit, sensors]
            xt = np.concatenate([windows[test,0], windows[test,1]])
            test_y, test_groups = np.tile(ids[test,1], 2), np.tile(ids[test,0], 2)
            for seed in SEEDS:
                yield seed, (X, ids[fit,1], ids[fit,0], xt, test_y, test_groups, "classification")
    else:
        if source == "isolet":
            X, y, task, *_ = load_dataset("isolet")
            cuts = (train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=seed)
                    for seed in SEEDS)
        else:
            X, y = load_diabetes(return_X_y=True, scaled=False)
            task = "regression"
            cuts = KFold(3, shuffle=True, random_state=41437).split(X)
        for seed, (fit, test) in zip(SEEDS, cuts, strict=True):
            yield seed, (X[fit], y[fit], None, X[test], y[test], test, task)


def select(method, source, seed, X, y, groups, task):
    width = ceil(X.shape[1]/2)
    if method == "full":
        return np.arange(X.shape[1])
    if method in ("backward", "forward"):
        selector = backward_top_from_training if method == "backward" else environment_top_from_training
        return selector(X, y, width, task=task, random_state=seed, groups=groups,
                        ordered_groups=source == "gas")[0]
    if method == "local":
        return local_refit_fast_top_from_training(X,y,width,task=task,random_state=seed)[0]
    if method == "F":
        return choose("f",X,y,task,seed,width)
    if source in ("mhealth","har"):
        rank,_ = by_subject_pfi(X,y,groups,seed)
    elif source == "isolet":
        scores = held_permutation_scores(X,y,seed,large=True)
        rank = np.argsort(-scores,kind="stable")
    else:
        rank = full_fit_pfi(X,y,task,seed,groups)
    return np.sort(rank[:width])


def evaluate(source, truth, pred, test_groups, task):
    if source == "mhealth":
        y, replicated = np.split(truth, 2)
        np.testing.assert_array_equal(y, replicated)
        g, _ = np.split(test_groups,2)
        wrist, ankle = np.split(pred,2)
        wrist_scores, ankle_scores = batch_scores(y,wrist,g), batch_scores(y,ankle,g)
        units = {unit: min(wrist_scores[unit],ankle_scores[unit]) for unit in wrist_scores}
    elif source in ("gas", "har"):
        units = batch_scores(truth,pred,test_groups)
    elif task == "regression":
        return r2_score(truth,pred), {}
    else:
        return balanced_accuracy_score(truth,pred), {}
    return float(np.mean(list(units.values()))), units


def run(output, verify):
    if not verify:
        output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), CONFIG, ROOT/"src/minppi_mofs/backward_minpi.py",
             ROOT/"src/minppi_mofs/environment_minpi.py", ROOT/"src/minppi_mofs/local_refit_fast.py",
             ROOT/"experiments/minpi_stable_shift.py", ROOT/"experiments/minpi_study.py",
             ROOT/"experiments/minpi_iteration_benchmark.py", ROOT/"experiments/minpi_har_confirmation.py",
             ROOT/"experiments/pivotal_wide_comparison.py"]
    hashes = {str(path.relative_to(ROOT)):file_hash(path) for path in files}
    protocol = dict(methods=METHODS, sources=SOURCES, seeds=SEEDS, threads=1, repeats=2,
                    width="ceil(p/2) for all selectors, and all p for full baseline; no calibration or fallback",
                    validation="MHEALTH existing persons 1-5 vs 6-10 split and saved fit-device choices; score mean person lower wrist/ankle BA. Diabetes three KFold splits, seed 41437 and paired seeds, pooled held R2 per split. ISOLET paired stratified sample splits, test_size 25%, balanced accuracy; no speaker IDs. Gas, supercon, HAR previous outer splits, primary seed only.",
                    timing="Two rotated single-thread runs per fixed evaluation case. End-to-end includes finding columns, extracting them, fitting and predicting with the same standardized ridge. Timings are descriptive.",
                    sklearn=sklearn.__version__, numpy=np.__version__, source_hashes=hashes,
                    data={**DIGESTS, "har":DIGEST, "mhealth_raw":RAW_SHA256,
                          "mhealth_windows":file_hash(DATA/"mhealth-windows.npz"),
                          "mhealth_device_assignment":file_hash(ROOT/"results/minpi-study/device/records.npz"),
                          "isolet":file_hash(ROOT/"data/raw/isolet.csv")})
    if verify:
        assert json.loads((output/"protocol.json").read_text(encoding="utf-8")) == json.loads(json.dumps(protocol))
    else:
        (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    records, units_records = [], []
    for source in SOURCES:
        for seed, (X,y,groups,xt,truth,units,task) in splits(source):
            if groups is not None:
                assert not np.intersect1d(groups,units).size
            saved = {"truth":truth,"test_units":units}
            if groups is not None:
                saved["fit_groups"] = groups
            methods = np.random.default_rng(seed).permutation(METHODS)
            for repeat in range(1 if verify else 2):
                for method in (("backward",) if verify else np.roll(methods, repeat*3)):
                    start = perf_counter_ns()
                    cols = select(method,source,seed,X,y,groups,task)
                    middle = perf_counter_ns()
                    A,B = (X,xt) if method=="full" else (X[:,cols],xt[:,cols])
                    pred,_ = train("ridge",A,y,B,task,seed)
                    stop = perf_counter_ns()
                    assert len(cols) == (X.shape[1] if method == "full" else ceil(X.shape[1]/2))
                    np.testing.assert_array_equal(np.unique(cols),cols)
                    score, per_unit = evaluate(source,truth,pred,units,task)
                    for key, value in ((method+"-support",cols), (method+"-pred",pred)):
                        if repeat:
                            np.testing.assert_array_equal(value,saved[key])
                        else:
                            saved[key] = value
                    records.append(dict(source=source,seed=seed,repeat=repeat,method=method,width=len(cols),p=X.shape[1],
                                        score=score,select_ms=(middle-start)/1e6,total_ms=(stop-start)/1e6,
                                        support_sha256=digest(cols),prediction_sha256=digest(pred)))
                    if not repeat:
                        units_records.extend(dict(source=source,seed=seed,method=method,unit=k,score=v)
                                             for k,v in per_unit.items())
            path = output/f"{source}-{seed}.npz"
            if verify:
                with np.load(path) as old:
                    for key,value in saved.items():
                        np.testing.assert_array_equal(old[key],value)
            else:
                np.savez_compressed(path,**saved)
            print(f"Finished {source} seed={seed} p={X.shape[1]}",flush=True)
    df = pd.DataFrame(records)
    if verify:
        df.drop(columns=["select_ms","total_ms"]).to_csv(output/"fresh-process-audit.csv",index=False)
        return
    df.to_csv(output/"trials.csv",index=False)
    pd.DataFrame(units_records).to_csv(output/"unit_scores.csv",index=False)
    cases = df.groupby(["source","seed","method"],as_index=False).agg(
        score=("score","first"),width=("width","first"),select_ms=("select_ms","median"),total_ms=("total_ms","median"))
    cases.to_csv(output/"cases.csv",index=False)
    summary = cases.groupby(["source","method"],as_index=False).agg(score=("score","mean"),width=("width","first"),
                    min_score=("score","min"),max_score=("score","max"),select_ms=("select_ms","median"),
                    total_ms=("total_ms","median"))
    summary.to_csv(output/"summary.csv",index=False)
    print(summary.round(4).to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.verify)
