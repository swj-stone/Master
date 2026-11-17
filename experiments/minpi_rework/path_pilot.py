"""Development-only comparison of backward quota paths on previously used splits."""
import argparse
import ctypes
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import (
    benchmark as bench, chain_validation_sources, legacy_gain as legacy, witness_transfer_sources)
from experiments.minpi_rework import path_guard
from minppi_mofs.local_vote_path import path_order

ORIGINAL_RANK = bench.ranked
MODES = ["integral", "point", "minimum", "mean", "full_marginal", "train_cost",
         "signed", "signed_eager", "signed_train", "negative_only"]
DEFAULT = ["integral", "point", "minimum", "mean", "full_marginal", "train_cost", "coefficient", "rfe", "full"]


def rank(method, x, y, task, idx, seed):
    if method == "gain":
        return legacy.gain_rank(method, x, y, task, idx, seed)
    if method == "pooled_coefficient":
        rows = np.sort(np.r_[idx["fit"],idx["vote"]])
        prep = bench.transform(x,rows)
        z = prep.transform(x[rows])
        from sklearn.linear_model import Ridge
        out = bench.targets(y[rows],task,len(np.unique(y[rows])))
        coeff = np.atleast_2d(Ridge(alpha=1.).fit(z,out).coef_)
        return np.lexsort((np.arange(z.shape[1]),-np.linalg.norm(coeff, axis=0))), {}
    rule, _, schedule = method.partition("@")
    if rule.removesuffix("_gcv") not in MODES:
        return ORIGINAL_RANK(method, x, y, task, idx, seed)
    fit, vote = idx["fit"], idx["vote"]
    prep = bench.transform(x, fit)
    return path_order(prep.transform(x[fit]), y[fit], prep.transform(x[vote]), y[vote],
                      task=task, seed=seed, mode=rule, schedule=schedule or "quarter")


def pilot(args):
    out = bench.ROOT / "results/minpi_rework" / args.run
    out.mkdir(parents=True, exist_ok=True)
    if (out/"settings.json").exists():
        raise FileExistsError(f"Existing run at {out}, choose a new --run")
    if args.cpu is not None:
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1 << args.cpu)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.sched_setaffinity(0, {args.cpu})
    from minppi_mofs import local_vote_path
    signature = {p.name:bench.digest(p) for p in map(Path, [
        __file__,path_guard.__file__,local_vote_path.__file__,bench.__file__])}
    (out/"settings.json").write_text(json.dumps(dict(arguments=vars(args),sha256=signature),indent=2),encoding="utf8")
    manifest = legacy.jobs() if set(args.names) & set(legacy.CONFIG["datasets"]) else {}
    rows = []
    start = perf_counter()
    for name in args.names:
        original = name in legacy.CONFIG["datasets"]
        for seed in args.seeds:
            job_start = perf_counter()
            if original:
                directory, row = manifest[name, seed]
                x, y, task, idx = legacy.read_split(directory, row, seed)
            else:
                split = (chain_validation_sources.split if name in chain_validation_sources.SOURCES else
                         witness_transfer_sources.split if name in witness_transfer_sources.SOURCES else bench.split)
                x, labels, task, idx = split(name, seed)
                y = labels.copy()
                y[idx["test"]] = -1
            # Rotate order across timing repetitions. All entrants share the guard and refit.
            records, forecasts = {}, {}
            methods = args.methods + (["gain"] if original and args.gain else [])
            ordered = np.random.default_rng(seed).permutation(methods)
            for repeat in range(args.repeats):
                for method in np.roll(ordered, repeat):
                    if args.dense:
                        rec, pred = path_guard.select(method,x,y,task,idx,seed,rank,zero_tolerance=args.zero)
                    else:
                        with patch.object(bench, "ranked", rank):
                            rec, pred = bench.select_one(method, x, y, task, idx, seed)
                    if repeat:
                        np.testing.assert_array_equal(pred, forecasts[method])
                        assert records[method][0]["order"] == rec["order"]
                    records.setdefault(method, []).append(rec)
                    forecasts[method] = pred
            job = out / f"{name}-{seed}"
            job.mkdir(exist_ok=True)
            (job/"selection.json").write_text(json.dumps(records), encoding="utf8")
            np.savez_compressed(job/"predictions.npz", **forecasts)
            dev = np.r_[idx["fit"], idx["vote"], idx["width"]]
            scale = np.std(y[dev]) or 1.
            if original:
                test = pd.read_csv(directory / row["test_relpath"], usecols=[row["target"]])[row["target"]].to_numpy()
                with np.load(legacy.DEST / "novel-input-rows.npz") as masks:
                    mask = masks[f"{name}-{seed}"].copy()
            else:
                test = labels[idx["test"]]
                mask = np.ones(len(test), dtype=bool)
            for method, pred in forecasts.items():
                reps = records[method]
                first = reps[0]
                scores = bench.metrics(test[mask], pred[mask], task, scale)
                info = {k:np.median([r[k] for r in reps]) for k in ("rank_s","selection_s","deployment_s")}
                work = {k:first.get(k) for k in ("skipped_edges","deletion_edges","factorizations","checked","auc_calls")}
                rows.append(dict(dataset=name, seed=seed, task=task, method=method,
                    original=original, n_test=int(np.sum(mask)), p=x.shape[1], k=first["k"],
                    retained=first["k"]/x.shape[1], selected=" ".join(map(str, sorted(first["order"][:first["k"]]))),
                    **info, **work, **scores))
            pd.DataFrame(rows).to_csv(out/"scores.csv", index=False)
            print(f"{name}/{seed} finished in {perf_counter()-job_start:.1f}s; total {perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", default=bench.DESIGN["confirmation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--run", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT)
    parser.add_argument("--gain", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--cpu", type=int)
    with threadpool_limits(limits=1):
        pilot(parser.parse_args())
