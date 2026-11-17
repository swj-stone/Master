"""Replay acceptance schedules against archived forest supports."""
import argparse
import hashlib
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark, legacy_gain
from experiments.minpi_rework import quota_coalition_rf as rf, validation_predictive as external
from experiments.minpi_rework.predictive_selection import prepare_paths, select_width


OUT = benchmark.ROOT / "thesis-paper/evidence/minpi_rework/predictive-width"
POLICIES = ("eager", "intact_first", "min_first", "mdi_min_first")


def evaluate(panel, name, seed, x, y, task, idx):
    start = perf_counter()
    model = rf.fit(x[idx["fit"]], y[idx["fit"]], task, seed)
    fit_s = perf_counter() - start
    paths, null, costs = prepare_paths(model, x[idx["vote"]], y[idx["vote"]],
                                     x[idx["width"]], y[idx["width"]], task, rank_backend="native")
    tag = "predictive-confirmation" if panel == "external" else "predictive-prepared"
    folder = benchmark.ROOT / "results/minpi_rework" / tag / f"{name}-{seed}"
    receipt = json.loads((folder / "selection.json").read_bytes())["records"]
    scale = max(float(np.std(y[idx["fit"]])), 1e-12)
    rows = []
    for rep in range(3):
        for pos in np.random.default_rng(seed * 10 + rep).permutation(len(POLICIES)):
            policy = POLICIES[pos]
            mdi = policy == "mdi_min_first"
            start = perf_counter()
            columns, trace = select_width({"mdi": paths["mdi"]} if mdi else paths, null,
                                          y[idx["width"]], task, scale, policy="min_first" if mdi else policy)
            seconds = perf_counter() - start
            expected = receipt["mdi_min" if mdi else "mdi_plus_min_min"]["support"]
            np.testing.assert_array_equal(columns, expected)
            setup = sum(costs.values()) - ((costs["min_rank"] + costs["min_curve"]) if mdi else 0.)
            rows.append(dict(panel=panel, dataset=name, seed=seed, rep=rep, policy=policy,
                             p=x.shape[1], k=len(columns), native_calls=trace["native_calls"],
                             validation_s=seconds, setup_s=setup, fit_s=fit_s,
                             fit_to_selection_s=fit_s + setup + seconds,
                             support_sha=hashlib.sha256(json.dumps(expected).encode()).hexdigest()))
    return rows


def summarize(raw):
    med = raw.groupby(["panel", "dataset", "seed", "policy"]).median(numeric_only=True)
    records = []
    for panel, group in med.groupby(level="panel"):
        new = group.xs("min_first", level="policy")
        for other in ("eager", "intact_first", "mdi_min_first"):
            base = group.xs(other, level="policy")
            for metric in ("native_calls", "validation_s", "fit_to_selection_s", "k"):
                per_source = np.log(new[metric] / base[metric]).groupby(level="dataset").mean()
                rng = np.random.default_rng(171166)
                vals = per_source.to_numpy()
                sims = vals[rng.integers(len(vals), size=(20000, len(vals)))].mean(axis=1)
                value, lo, hi = np.exp([vals.mean(), *np.quantile(sims, [.025, .975])])
                records.append(dict(panel=panel, baseline=other, metric=metric,
                                    sources=len(vals), estimate=value, lower=lo, upper=hi))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    records = OUT / "timings.csv"
    if args.summarize:
        raw = pd.read_csv(records)
        if len(raw) != 65 * 3 * len(POLICIES):
            raise ValueError("Incomplete timing panel")
        summarize(raw).to_csv(OUT / "contrasts.csv", index=False)
        return
    # Warm Numba and native scoring before measuring, on data outside the panel.
    rng = np.random.default_rng(771)
    x = rng.normal(size=(90, 4))
    y = (x[:, 0] > 0).astype(int)
    for task, truth in (("classification", y), ("regression", x[:, 0])):
        model = rf.fit(x[:30], truth[:30], task, 89, n_estimators=3)
        paths, null, _ = prepare_paths(model, x[30:60], truth[30:60], x[60:], truth[60:], task,
                                       rank_backend="native")
        for policy in POLICIES[:3]:
            select_width(paths, null, truth[60:], task, 1., policy=policy)
    prepared = legacy_gain.jobs()
    jobs = [("external", name, seed) for name in external.SOURCES for seed in external.SEEDS]
    jobs.extend(("prepared", name, seed) for name, seed in sorted(prepared))
    rows = []
    started = perf_counter()
    for i, (panel, name, seed) in enumerate(jobs, 1):
        tick = perf_counter()
        if panel == "external":
            x, y, task, idx = external.split(name, seed)
        else:
            directory, pin = prepared[name, seed]
            x, y, _, idx = legacy_gain.read_split(directory, pin, seed)
            task = "regression" if pin["task"] == "regression" else "classification"
        rows.extend(evaluate(panel, name, seed, x, y, task, idx))
        pd.DataFrame(rows).to_csv(records, index=False)
        elapsed = perf_counter() - started
        print(f"[{i}/{len(jobs)}] {name}/{seed} {perf_counter()-tick:.2f}s; "
              f"elapsed {elapsed:.0f}s ETA {elapsed*(len(jobs)-i)/i:.0f}s", flush=True)
    summarize(pd.DataFrame(rows)).to_csv(OUT / "contrasts.csv", index=False)


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        process = ctypes.c_void_p(-1)
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        processors = [1 << i for i in range(allowed.value.bit_length()) if allowed.value >> i & 1]
        if not kernel.SetProcessAffinityMask(process, ctypes.c_size_t(processors[min(6, len(processors)-1)])):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        main()
