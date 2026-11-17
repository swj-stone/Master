"""Retrospective paired test of cross-coalition exclusion on the frozen ridge tasks."""
from functools import partial
from hashlib import sha256
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from minppi_mofs.dictator_minpi import rank_bounds
from minppi_mofs.sampled_linear_minpi import rank_features


def jobs():
    for dataset in b.DESIGN["confirmation"]:
        for seed in b.DESIGN["seeds"]:
            yield "confirmation", dataset, seed, b.split(dataset, seed)
    for folder in sorted((b.ROOT/"results/context_review/wide-uci").glob("*/fold-*")):
        inputs = folder/"inputs.npz"
        assert b.digest(inputs) == json.loads((folder/"layout.json").read_bytes())["inputs_sha256"]
        with np.load(inputs) as data:
            x = np.concatenate([data["x"][:, :100], data["x_outer"][:, :100]]).astype(float)
            y = np.r_[data["y"], np.full(len(data["x_outer"]), -1, dtype=int)]
            idx = dict(fit=data["train"], vote=data["proposal"], width=data["guard"],
                       test=np.arange(len(data["y"]),len(y)))
        seed = 42 + int(folder.name.split("-")[-1])
        yield "wide", folder.parent.name, seed, (x, y, "classification", idx)


def bind_last_cpu():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = ctypes.c_void_p(-1)
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        cpu = process.value.bit_length()-1
        if not kernel.SetProcessAffinityMask(handle, ctypes.c_size_t(1 << cpu)):
            raise ctypes.WinError(ctypes.get_last_error())
        return cpu
    if hasattr(os, "sched_getaffinity"):
        cpu = max(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        return cpu
    return None


def main():
    out = b.EVIDENCE/"dictator"
    out.mkdir(parents=True, exist_ok=True)
    methods = {"coefficient":partial(rank_features, method="coefficient"), "original":rank_features,
               "cached":partial(rank_bounds, cross_coalition=False), "bound":rank_bounds,
               "closure":partial(rank_bounds, closure=True)}
    cpu = bind_last_cpu()
    oracle = pd.read_csv(b.EVIDENCE/"audit/exact-minima.csv").set_index(["source","seed"]).sort_index()
    rng = np.random.default_rng(117)
    rows, bounds = [], []
    with threadpool_limits(limits=1):
        for panel, name, seed, args in jobs():
            x, y, task, idx = args
            ordering = rng.permutation(list(methods))
            outputs = {}
            for rep in range(len(methods)):
                # Each method occupies every clock position once.
                for pos, method in enumerate(np.roll(ordering, rep)):
                    def run_rank(*rank_args, method_name=method, **kwargs):
                        kwargs.pop("method")
                        val = methods[method_name](*rank_args, **kwargs)
                        outputs[method_name] = val
                        return val
                    with patch.object(b.engine, "rank_features", run_rank):
                        rec, pred = b.select_one("minpi", x, y, task, idx, seed)
                    rank = outputs[method]
                    pred_hash = sha256(pred.tobytes()).hexdigest()
                    cols = sorted(rec["order"][:rec["k"]])
                    outcome = {}
                    if panel == "confirmation":
                        dev = np.r_[idx["fit"],idx["vote"],idx["width"]]
                        outcome = b.metrics(y[idx["test"]], pred, task, y[dev].std())
                    rows.append(dict(panel=panel, source=name, seed=seed, task=task, cpu=cpu, repeat=rep,
                                     position=pos, method=method, p=rec["p"], k=rec["k"],
                                     columns=" ".join(map(str, cols)),
                                     order_sha256=sha256(np.asarray(rec["order"],dtype="<i8").tobytes()).hexdigest(),
                                     forecast_sha256=pred_hash,
                                     **{k:rec[k] for k in ["rank_s","selection_s","deployment_s"]},
                                     **rank.work, **outcome))
                original, local, cross = (outputs[k] for k in ["original","cached","bound"])
                closed = outputs["closure"]
                np.testing.assert_array_equal(original.minimum, local.minimum)
                np.testing.assert_array_equal(original.order, local.order)
                assert original.witness == local.witness == cross.witness
                assert np.all(cross.minimum <= local.minimum)
                assert np.all(closed.minimum <= cross.minimum)
                assert all(a is None or a == b for a,b in zip(cross.witness, closed.witness))
                assert closed.work["distinct_votes"] == cross.work["distinct_votes"]
            exact = None
            if panel == "confirmation" and (name, seed) in oracle.index:
                exact = oracle.loc[name, seed].sort_values("feature").exact.to_numpy()
                assert np.all(closed.minimum >= exact)
            for j in range(x.shape[1]):
                bounds.append(dict(panel=panel, source=name, seed=seed, feature=j,
                                   original=original.minimum[j], refined=cross.minimum[j], closed=closed.minimum[j],
                                   exact=exact[j] if exact is not None else None,
                                   witness=json.dumps(original.witness[j]),
                                   closed_witness=json.dumps(closed.witness[j])))
            print(f"{panel}/{name}/{seed}: matched negative witnesses",flush=True)
    clocks = pd.DataFrame(rows)
    clocks.to_csv(out/"runs.csv",index=False)
    pd.DataFrame(bounds).to_csv(out/"bounds.csv",index=False)
    clocks["retained"] = clocks.k/clocks.p
    clocks.groupby(["panel","source","seed","method"]).median(numeric_only=True).drop(
        columns=["position","repeat","cpu"]).to_csv(out/"splits.csv")
    for panel, frame in clocks.groupby("panel"):
        summary = frame.groupby(["source","seed","method"]).median(numeric_only=True)
        for chosen in ("bound","closure"):
            for alternative in ("original","cached","coefficient"):
                a = summary.xs(chosen,level="method")
                c = summary.xs(alternative,level="method")
                ratios = np.exp(np.log(a[["rank_s","selection_s","k"]]/c[["rank_s","selection_s","k"]]).groupby(
                    level="source").mean().mean())
                print(panel,chosen,alternative,ratios.round(4).to_dict())
        outcome = frame[frame.repeat==0].pivot(index=["source","seed"],columns="method",values="forecast_sha256")
        print("Changed forecasts (bound, closure):",((outcome[["bound","closure"]].ne(outcome["original"],axis=0)).sum()))
        print("Computations:", summary.groupby("method")[["evaluated_marginals","new_child_solves","factorizations"]].sum())


if __name__ == "__main__":
    main()
