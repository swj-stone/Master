"""Test the anchored quadratic game on exact raw TreeSHAP contributions."""
import argparse
import json
import math
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from xgboost import DMatrix

from experiments.minpi_compression import benchmark, legacy_gain, legacy_gain_receiver
from minppi_mofs.additive_minpi import quadratic, quadratic_order
from minppi_mofs.modeling import make_model


OUT = benchmark.ROOT / "results/minpi_rework/boosted-additive"
EVIDENCE = benchmark.ROOT / "thesis-paper/evidence/minpi_rework/boosted-additive"


def probability_parts(raw, bias):
    """Secant on class exponentials followed by the exact normalization identity."""
    if raw.shape[-1] == 1:
        raw = np.concatenate([-raw / 2, raw / 2], axis=2)
        bias = np.concatenate([-bias / 2, bias / 2], axis=1)
    # A common logit shift does not change any class probability.
    raw = raw - raw.mean(axis=2, keepdims=True)
    bias = bias - bias.mean(axis=1, keepdims=True)
    delta = raw.sum(axis=1)
    final = bias + delta
    shift = np.maximum(bias.max(axis=1), final.max(axis=1))[:, None]
    exponent = np.exp(bias-shift)
    z = np.exp(final-shift).sum(axis=1, keepdims=True)
    # Continuous extension of expm1(delta)/delta at zero.
    slope = np.divide(np.expm1(delta), delta, out=np.ones_like(delta), where=delta != 0)
    factors = exponent * slope / z
    base = softmax(bias, axis=1)
    changes = raw * factors[:, None, :]
    changes -= base[:, None, :] * changes.sum(axis=2, keepdims=True)
    np.testing.assert_allclose(base + changes.sum(axis=1), softmax(final, axis=1), atol=2e-9)
    return changes, base


def pair_order(gram, single, rule):
    """Remove the least supported two-feature block, updating the active game."""
    p = len(single)
    a, b = np.triu_indices(p, k=1)
    coupling = gram[a] + gram[b]
    coupling[np.arange(len(a)),a] = 0.
    coupling[np.arange(len(b)),b] = 0.
    solo_coupling = gram.copy()
    np.fill_diagonal(solo_coupling, 0.)
    def penalty(g):
        if rule == "min":
            return 2*np.maximum(g, 0.)
        return g if rule == "pooled" else 2*g
    penalty_pair = penalty(coupling)
    penalty_solo = penalty(solo_coupling)
    joint = single[a]+single[b]-2*gram[a,b]-penalty_pair.sum(axis=1)
    separate = single-penalty_solo.sum(axis=1)
    removed = []
    while len(removed)+2<=p:
        ix = len(a)-1-int(np.argmin(joint[::-1]))
        ordered = sorted([int(a[ix]),int(b[ix])], key=lambda j:(separate[j],-j))
        removed.extend(ordered)
        for j in ordered:
            joint += penalty_pair[:,j]
            separate += penalty_solo[:,j]
            joint[(a==j)|(b==j)] = np.inf
    if len(removed)<p:
        removed.extend(set(range(p))-set(removed))
    return np.array(removed[::-1])


def rank_all(x, y, task, model_task, idx, seed, pairs):
    fit, vote = idx["fit"], idx["vote"]
    begin = perf_counter()
    model = make_model(task=model_task, seed=seed, backend=legacy_gain.CONFIG["backend"],
                       model_params=legacy_gain.CONFIG["model_params"])
    model.fit(x[fit], y[fit])
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    p = x.shape[1]
    gain_order = np.array(sorted(range(p), key=lambda j: (gain.get(f"f{j}", 0.), j), reverse=True))
    fit_s = perf_counter()-begin
    began_shap = perf_counter()
    source = DMatrix(x[vote])
    contrib = booster.predict(source, pred_contribs=True, strict_shape=True)
    actual = booster.predict(source, output_margin=True, strict_shape=True)
    np.testing.assert_allclose(contrib.sum(axis=2), actual, atol=4e-5, rtol=1e-5)
    raw = contrib[:, :, :-1].transpose(0,2,1).astype(float)
    bias = contrib[:, :, -1].astype(float)
    shap_s = perf_counter()-began_shap
    t = perf_counter()
    absolute = np.abs(raw).mean(axis=(0,2))
    orders = dict(gain=gain_order, shap_abs=np.lexsort((np.arange(p), -absolute)))
    overhead = dict(gain=0., shap_abs=shap_s+perf_counter()-t)
    for game in ("raw", "supervised"):
        t = perf_counter()
        if game == "raw":
            parts, residual = raw, raw.sum(axis=1)
        elif task == "classification":
            parts, base = probability_parts(raw, bias)
            expected = model.predict_proba(x[vote])
            np.testing.assert_allclose(base+parts.sum(axis=1), expected, atol=5e-6)
            residual = np.eye(parts.shape[2])[y[vote]]-base
        else:
            parts, residual = raw, y[vote, None]-bias
        gram, single = quadratic(parts, residual)
        common = shap_s + perf_counter()-t
        for r in ("min", "pooled", "point", "energy"):
            t = perf_counter()
            name = f"{game}_{r}"
            orders[name], _ = quadratic_order(gram,single,r)
            overhead[name] = common+perf_counter()-t
        if game=="raw" and pairs:
            for r in ("min","pooled","point"):
                t = perf_counter()
                name = f"raw_pair_{r}"
                orders[name] = pair_order(gram,single,r)
                overhead[name] = common+perf_counter()-t
            break
    if pairs:
        orders = {k:v for k,v in orders.items() if k=="gain" or "pair_" in k}
    return orders, {m: fit_s + budget for m, budget in overhead.items()}


def run(name, seed, job, pairs):
    directory, row = job
    x,y,task,idx = legacy_gain.read_split(directory, row, seed)
    start = perf_counter()
    orders, rank_s = rank_all(x,y,task,row["task"],idx,seed,pairs)
    key = f"{name}-{seed}"
    old = json.loads((legacy_gain.RUN/key/"selection.json").read_bytes())
    expected = next(r["order"] for r in old["rows"] if r["repeat"]==0 and r["method"]=="gain")
    np.testing.assert_array_equal(expected, orders["gain"])
    previous = json.loads((legacy_gain_receiver.OUT/key/"selection.json").read_bytes())["records"]["guard_gain"]
    pool = np.sort(np.r_[idx["fit"], idx["vote"]])
    width = idx["width"]
    p = x.shape[1]
    ks = sorted({1, *(math.ceil(p*f) for f in benchmark.DESIGN["width_fractions"])})
    cache = {}
    def evaluate(columns):
        support = tuple(int(j) for j in np.sort(columns))
        if support not in cache:
            t = perf_counter()
            pred = legacy_gain_receiver.predict(x, y, pool, width, np.array(support), row["task"], seed)
            score = benchmark.metrics(y[width], pred, task, max(np.std(y[pool]),1e-12))
            cache[support] = (score, perf_counter()-t)
        return cache[support]
    full, full_s = evaluate(np.arange(p))
    rec = {}
    for method in np.random.default_rng(seed).permutation(list(orders)).tolist():
        order = orders[method]
        np.testing.assert_array_equal(np.sort(order), np.arange(p))
        cost = full_s + rank_s[method]
        trace = {p:full}
        k = p
        for k in ks:
            if k==p:
                break
            values, seconds = evaluate(order[:k])
            trace[k] = values
            cost += seconds
            if benchmark.eligible(values,full,task):
                break
        if method=="gain":
            assert k==previous["k"]
            for j,v in trace.items():
                for metric, value in v.items():
                    np.testing.assert_allclose(value, previous["trace"][str(j)][metric], atol=1e-12)
        rec[method] = dict(k=k, p=p, order=order.tolist(), trace=trace, rank_s=rank_s[method],
                          selection_component_s=cost, guard_fits=len(trace))
    dest = OUT / (f"{key}-pairs.json" if pairs else f"{key}.json")
    dest.write_text(json.dumps(dict(dataset=name,seed=seed,arms=rec,elapsed_s=perf_counter()-start)), encoding="utf-8")
    print(name, seed, f"{perf_counter()-start:.1f}s", {m:r["k"] for m,r in sorted(rec.items())},flush=True)


def export():
    rows=[]
    for path in sorted(OUT.glob("*.json")):
        rec = json.loads(path.read_bytes())
        for m, arm in rec["arms"].items():
            k = arm["k"]
            rows.append(dict(dataset=rec["dataset"],seed=rec["seed"],method=m,
                experiment="pair" if path.stem.endswith("-pairs") else "singleton",
                **{i:arm[i] for i in ("k","p","rank_s","selection_component_s","guard_fits")},
                **arm["trace"][str(k)]))
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(EVIDENCE/"development.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=legacy_gain.CONFIG["datasets"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--export", action="store_true")
    p.add_argument("--pairs", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(1):
        if args.export:
            export()
        else:
            jobs = legacy_gain.jobs()
            for name in args.datasets:
                for seed in args.seeds:
                    run(name,seed,jobs[name,seed],args.pairs)
