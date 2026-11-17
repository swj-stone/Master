"""Development check for pruning widths on the original direct Gain path.

Calibration excludes every split of the evaluated source. It consumes only
the archived inner width-label results, without an outer-test comparison.
"""
import json
import math
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark, legacy_gain, legacy_gain_receiver
from experiments.minpi_rework.nonlinear_pilot import ROOT
from minppi_mofs.modeling import make_model
from minppi_mofs.nonlinear_vote import _stratified_indices
from experiments.minpi_rework.nonlinear_vote_tree import path_evidence


def proxy(x, y, idx, task, order, seed):
    fit, vote = idx["fit"], idx["vote"]
    y_fit, y_vote = y[fit], y[vote]
    if task == "regression":
        cuts = np.unique(np.quantile(y_fit, [.25, .5, .75]))
        y_fit, y_vote = np.searchsorted(cuts, y_fit), np.searchsorted(cuts, y_vote)
    rng = np.random.default_rng(seed)
    fi = _stratified_indices(y_fit, 2048, rng)
    vi = _stratified_indices(y_vote, 512, rng)
    prep = SimpleImputer(strategy="median", keep_empty_features=True)
    a = prep.fit_transform(x[fit[fi]])
    b = prep.transform(x[vote[vi]])
    forest = RandomForestClassifier(n_estimators=16, max_depth=7, min_samples_leaf=3,
                                    random_state=seed, n_jobs=1).fit(a, y_fit[fi])
    one_hot = (y_vote[vi, None] == forest.classes_).astype(float)
    one_hot /= np.maximum(one_hot.sum(axis=0), 1)
    rows = [path_evidence(t, b, one_hot) for t in forest.estimators_]
    path, votes, margins = (np.concatenate(a) for a in zip(*rows))
    p = x.shape[1]
    levels = np.unique(np.rint(np.array([.125, .5, .875, 1]) * p).astype(int))
    degree = path.sum(axis=1)
    q = np.ones((degree.max()+1, len(levels)))
    for i in range(1, len(q)):
        q[i] = q[i-1] * np.maximum(levels-i+1, 0) / (p-i+1)
    # Under the full model S equals a random level-size set, and under the
    # restriction it equals its intersection with A. Only paths outside A differ.
    q = q[degree] / (len(forest.estimators_) * len(forest.classes_))
    reach = np.max(path[:, order] * np.arange(1, p+1), axis=1)
    result = {}
    for k in sorted({1, *(math.ceil(p*f) for f in benchmark.DESIGN["width_fractions"])}):
        missing = reach > k
        a, b = votes[missing] @ q[missing], margins[missing] @ q[missing]
        result[k] = dict(vote_min=a[:-1].max(), vote_pooled=a[:-1].mean(), vote_full=a[-1],
                         continuous_min=b[:-1].max(), continuous_full=b[-1],
                         mdi=forest.feature_importances_[order[k:]].sum())
    return result


def collect():
    rows = []
    for (n,s), (directory, info) in sorted(legacy_gain.jobs().items()):
        x,y,task,idx = legacy_gain.read_split(directory, info, s)
        key = f"{n}-{s}"
        archive = json.loads((legacy_gain_receiver.OUT / key / "selection.json").read_bytes())
        guard = archive["records"]["guard_gain"]
        prior = json.loads((legacy_gain.RUN / key / "selection.json").read_bytes())
        rank = next(v for v in prior["rows"] if v["method"] == "gain" and v["repeat"] == 0)
        order = np.array(rank["order"])
        model = make_model(task=info["task"], seed=s, backend=legacy_gain.CONFIG["backend"],
                           model_params=legacy_gain.CONFIG["model_params"])
        model.fit(x[idx["fit"]], y[idx["fit"]])
        gains = model.get_booster().get_score(importance_type="gain")
        weights = np.array([gains.get(f"f{i}", 0.) for i in range(x.shape[1])])
        np.testing.assert_array_equal(order, np.lexsort((-np.arange(len(weights)), -weights)))
        t = perf_counter()
        proxy_scores = proxy(x, y, idx, task, order, s)
        proxy_s = perf_counter()-t
        p = len(order)
        for k, scores in proxy_scores.items():
            metrics = guard["trace"][str(k)]
            full = guard["trace"][str(p)]
            rows.append(dict(dataset=n, seed=s, k=k, p=p, original=guard["k"],
                eligible=benchmark.eligible(metrics, full, task), fraction=1-k/p,
                gain_mass=weights[order[k:]].sum()/weights.sum(), **scores, proxy_s=proxy_s))
        print(key, f"proxy {proxy_s:.3f}s", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "guard-inputs.csv", index=False)


def evaluate():
    inputs = pd.read_csv(ROOT / "guard-inputs.csv")
    outputs = []
    # A score above the largest eligible calibration value advises skipping.
    # Report both preserved-width rate and actual count of checks up to first pass.
    for n, test in inputs.groupby("dataset"):
        train = inputs[(inputs.dataset != n) & inputs.eligible & (inputs.k < inputs.p)]
        for col in ["fraction", "gain_mass", "mdi", "vote_min", "vote_pooled", "vote_full",
                    "continuous_min", "continuous_full"]:
            threshold = train[col].max()
            for s, trace in test.groupby("seed"):
                scan = trace.loc[(trace[col] <= threshold) | (trace.k == trace.p)].sort_values("k")
                selected = int(scan.loc[scan.eligible, "k"].iloc[0])
                original = int(trace.original.iloc[0])
                # Both methods require an initial full model for their guard.
                baseline_checks = 1 + ((trace.k <= original) & (trace.k < trace.p)).sum()
                checks = 1 + ((scan.k <= selected) & (scan.k < scan.p)).sum()
                outputs.append(dict(dataset=n, seed=s, method=col, threshold=threshold,
                                    k=selected, original=original, saved=baseline_checks-checks,
                                    checks=checks, baseline=baseline_checks))
    out = pd.DataFrame(outputs)
    out.to_csv(ROOT / "guard-screen.csv", index=False)
    print(out.assign(worse=out.k>out.original).groupby("method")[["saved","worse","checks","baseline"]].sum())


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        collect()
        evaluate()
