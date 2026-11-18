"""Compare matched-width screens before ExtraTrees on ten numeric datasets."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings
import zipfile
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.minpi_study import P, ROOT, SOURCES, file_hash, forest, load_dataset, metrics, train
from minppi_mofs.exact_minpi import ExactMinPI


OUT = ROOT/"results/minpi-study/chain"
MAIN = ROOT/"results/minpi-study/formal"
DIGESTS = {"cnae": "458957518ec4339e0fc456837a96d44419ef81ae61425a6cdf2451112f18fbaa",
           "secom": "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"}
DATASETS = [*P["primary_sources"], *DIGESTS]
ARMS = ("none", "exact", "f", "random")
PROTOCOL = ROOT/"docs/minpi-study/chain-protocol.md"


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [*SOURCES, Path(__file__), PROTOCOL]}


def dataset(name):
    if name not in DIGESTS:
        x, y, task, _, order, _, source = load_dataset(name)
        return x, y, task, order, source, None
    path = ROOT/f"data/chain/{name}.zip"
    digest = file_hash(path)
    if digest != DIGESTS[name]:
        raise ValueError(f"Unexpected {name} archive digest")
    times = None
    with zipfile.ZipFile(path) as z:
        if name == "cnae":
            with z.open("CNAE-9.data") as file:
                raw = np.loadtxt(file, delimiter=",", dtype=np.float32)
            x, y = raw[:, 1:], raw[:, 0].astype(int)-1
            if x.shape != (1080, 856) or set(y) != set(range(9)) or not np.isfinite(x).all():
                raise ValueError("Unexpected CNAE shape, features or labels")
        else:
            with z.open("secom.data") as file:
                x = np.loadtxt(file, dtype=np.float32)
            with z.open("secom_labels.data") as file:
                labels = pd.read_csv(file, sep=" ", header=None)
            times = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S", errors="raise").to_numpy()
            if x.shape != (1567, 590) or set(labels[0]) != {-1, 1} or len(times) != len(x):
                raise ValueError("Unexpected SECOM shape, timestamps or labels")
            y = (labels[0] == 1).to_numpy(dtype=int)
    order = np.random.default_rng(P["column_permutation"]).permutation(x.shape[1])
    return np.ascontiguousarray(x[:, order]), y, "classification", order, digest, times


def split(name, y, task, times, seed):
    rows = np.arange(len(y))
    if name == "secom":
        ordered = np.argsort(times, kind="stable")
        cut = 3*len(y)//4
        a, b = ordered[:cut], ordered[cut:]
        if not times[a[-1]] < times[b[0]]:
            raise ValueError("Temporal holdout boundary is not strictly separated")
    else:
        a, b = train_test_split(rows, test_size=P["test_fraction"], random_state=seed,
                                stratify=y if task=="classification" else None)
    if name in P["primary_sources"]:
        with np.load(MAIN/name/f"s{seed}"/"predictions.npz") as main:
            np.testing.assert_array_equal(a, main["train_indices"])
            np.testing.assert_array_equal(b, main["test_indices"])
    return a, b


def impute(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(a, axis=0)
    medians = np.nan_to_num(medians, nan=0.)
    return np.where(np.isnan(a), medians, a), np.where(np.isnan(b), medians, b)


def initial_screen(arm, x, y, task, seed):
    p = x.shape[1]
    m = ceil(P["prefilter_fraction"]*p)
    if arm == "none":
        return np.arange(p)
    if arm == "exact":
        return ExactMinPI(task=task, random_state=seed, fraction=P["prefilter_fraction"]).fit(x, y).get_support(indices=True)
    if arm == "random":
        return np.sort(np.random.default_rng(seed).permutation(p)[:m])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        f = (f_classif if task == "classification" else f_regression)(x, y)[0]
    return np.sort(np.argsort(-np.nan_to_num(f, nan=-np.inf), kind="stable")[:m])


def compute(name, seed, data):
    x, y, task, order, source, times = data
    a, b = split(name, y, task, times, seed)
    raw_train, raw_test = x[a], x[b]
    yt, yv = y[a], y[b]
    k = ceil(P["selection_fraction"]*x.shape[1])
    archive = dict(train_indices=a, test_indices=b, y_test=yv, original_column_indices=order)
    meta = dict(dataset=name, seed=seed, n=len(y), p=x.shape[1], task=task, source=source,
                n_train=len(a), n_test=len(b), source_sha256=source_hashes(), arms={})
    if name == "secom":
        meta.update(last_train=str(times[a[-1]]), first_test=str(times[b[0]]), failures=int(yv.sum()),
                    prevalence=float(yv.mean()), empty_train_columns=int(np.isnan(raw_train).all(axis=0).sum()))
    rng = np.random.default_rng(seed+7219)
    rows = []
    with threadpool_limits(1):
        for arm in rng.permutation(ARMS):
            started = perf_counter()
            fit_x, test_x = impute(raw_train, raw_test) if name == "secom" else (raw_train, raw_test)
            ready = perf_counter()
            survivors = initial_screen(arm, fit_x, yt, task, seed)
            screened = perf_counter()
            ranks = forest(task, seed).fit(fit_x if len(survivors)==x.shape[1] else fit_x[:,survivors], yt).feature_importances_
            cols = np.sort(survivors[np.argsort(-ranks, kind="stable")[:k]])
            selected = perf_counter()
            meta["arms"][arm] = dict(screen_indices=survivors.tolist(), selected_indices=cols.tolist())
            for learner in rng.permutation(P["learners"]):
                begin = perf_counter()
                pred, score = train(learner, fit_x[:,cols], yt, test_x[:,cols], task, seed)
                duration = perf_counter()-begin
                row = dict(dataset=name, seed=seed, arm=arm, learner=learner, p=x.shape[1],
                           screened=len(survivors), final=len(cols), imputation_s=ready-started,
                           screen_s=screened-ready, selector_s=selected-screened, predict_fit_s=duration,
                           total_s=selected-started+duration, **metrics(yv,pred,score))
                if name == "secom":
                    row["ap"] = average_precision_score(yv, score)
                rows.append(row)
                archive[f"{arm}_{learner}_pred"] = pred
                if score is not None:
                    archive[f"{arm}_{learner}_score"] = score
    return meta, archive, rows


def run_seed(name, seed, data):
    parent = OUT / name
    parent.mkdir(exist_ok=True, parents=True)
    final = parent / f"s{seed}"
    if final.exists():
        meta = json.loads((final/"record.json").read_text(encoding="utf-8"))
        if meta["source_sha256"] != source_hashes() or meta["source"] != data[4]:
            raise ValueError(f"Invalid resume fingerprint at {final}")
        return
    meta, archive, rows = compute(name, seed, data)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".scratch-") as scratch:
        stage = Path(scratch)/"complete"
        stage.mkdir()
        (stage/"record.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
        np.savez_compressed(stage/"predictions.npz",**archive)
        pd.DataFrame(rows).to_csv(stage/"metrics.csv",index=False)
        stage.rename(final)
    print(f"Saved {name}/s{seed}",flush=True)


def summarize():
    frames = [pd.read_csv(OUT/name/f"s{seed}"/"metrics.csv") for name in DATASETS for seed in P["formal_seeds"]]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT/"metrics.csv",index=False)
    index = ["dataset","seed","learner"]
    differences = []
    for base in ("none", "f", "random"):
        just_base = df[df.arm==base].set_index(index)
        for alternative in (set(ARMS)-{base} if base == "none" else ["exact"]):
            comp = df[df.arm==alternative].set_index(index)
            calc = pd.DataFrame(index=comp.index)
            calc["arm"], calc["comparator"] = alternative, base
            calc["time_ratio"] = comp.total_s/just_base.total_s
            for value in ("auc","accuracy","balanced_accuracy","r2","mae","ap"):
                calc[f"delta_{value}"] = comp[value]-just_base[value]
            calc["meets_quality"] = np.where(calc.delta_r2.isna(),
                    (calc.delta_auc>=-.02)&(calc.delta_accuracy>=-.03),calc.delta_r2>=-.03)
            calc["speed_and_quality"] = (calc.time_ratio<=.8)&calc.meets_quality
            differences.append(calc.reset_index())
    paired = pd.concat(differences,ignore_index=True).sort_values([*index,"arm","comparator"])
    paired.to_csv(OUT/"paired.csv",index=False)
    group = ["dataset","learner","arm","comparator"]
    stats = {f"mean_{m}":(f"delta_{m}","mean") for m in ("auc","accuracy","balanced_accuracy","r2","mae","ap")}
    summary = paired.groupby(group,as_index=False).agg(median_time_ratio=("time_ratio","median"),
              min_time_ratio=("time_ratio","min"),max_time_ratio=("time_ratio","max"),
              quality_count=("meets_quality","sum"),speed_quality_count=("speed_and_quality","sum"),**stats)
    summary.to_csv(OUT/"summary.csv",index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs",action="store_true")
    parser.add_argument("--case",choices=DATASETS)
    parser.add_argument("--seed",type=int,choices=P["formal_seeds"])
    args=parser.parse_args()
    if args.check_inputs:
        for name in DATASETS:
            data = dataset(name)
            for seed in P["formal_seeds"]:
                a,b = split(name,data[1],data[2],data[-1],seed)
            print(f"{name}: X={data[0].shape}, train={len(a)}, test={len(b)}")
        return
    if args.case:
        if args.seed is None:
            parser.error("--case needs --seed")
        return run_seed(args.case,args.seed,dataset(args.case))
    if args.seed:
        parser.error("--seed needs --case")
    for seed in P["formal_seeds"]:
        for name in DATASETS:
            # A separate process bounds a single dataset/seed and releases arrays between cases.
            subprocess.run([sys.executable,"-B","-m","experiments.minpi_study_chain","--case",name,"--seed",str(seed)],
                   check=True,timeout=3600,env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
    summarize()


if __name__ == "__main__":
    main()

