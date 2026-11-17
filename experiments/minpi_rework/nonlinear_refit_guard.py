"""Screen strong Gain prefixes with actual weak-learner refits on three coalition sizes."""
import json
import math
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark, legacy_gain
from experiments.minpi_rework.nonlinear_pilot import ROOT
from minppi_mofs.nonlinear_vote import _stratified_indices


def utility(pred, y, task, scale):
    if task == "classification":
        auc = np.mean([roc_auc_score(y == c, pred[:, c]) for c in range(pred.shape[1])])
        return np.array([auc, (pred.argmax(axis=1) == y).mean()])
    return np.array([1-np.mean((y-pred)**2)/max(np.var(y), 1e-12), -np.mean(abs(y-pred))/scale])


def weak_trace(x, y, task, idx, order, seed, ks):
    """A panel votes on deleting the tail from a fixed random background.

    The full-size panel uses the original Gain prefix. Lower-size panels use
    the intersection of that prefix with a random nested background.
    """
    rng = np.random.default_rng(seed + 781)
    fit, vote = idx["fit"], idx["vote"]
    labels = y if task == "classification" else np.searchsorted(np.unique(np.quantile(y[fit], [.25,.5,.75])), y)
    fit = fit[_stratified_indices(labels[fit], 2048, rng)]
    vote = vote[_stratified_indices(labels[vote], 512, rng)]
    if task == "classification":
        model = DecisionTreeClassifier
        classes = int(np.max(y[fit])) + 1
        prior = np.broadcast_to(np.bincount(y[fit], minlength=classes)/len(fit), (len(vote), classes))
    else:
        model = DecisionTreeRegressor
        prior = np.full(len(vote), y[fit].mean())
    p = len(order)
    shuffle = rng.permutation(p)
    scores, clocks = {}, {}
    def evaluate(cols):
        key = tuple(sorted(int(j) for j in cols))
        if key not in scores:
            begin = perf_counter()
            if not key:
                pred = prior
            else:
                estimator = model(max_depth=6, min_samples_leaf=5, random_state=seed)
                estimator.fit(x[np.ix_(fit, key)], y[fit])
                matrix = x[np.ix_(vote, key)]
                pred = estimator.predict_proba(matrix) if task == "classification" else estimator.predict(matrix)
            scores[key] = utility(pred, y[vote], task, max(np.std(y[fit]),1e-12))
            clocks[key] = perf_counter()-begin
        return key
    masks = [set(shuffle[:max(1, math.ceil(p*frac))]) for frac in (.25, .5, 1.)]
    all_full = [evaluate(m) for m in masks]
    contrasts = []
    for k in ks:
        prefix = set(order[:k])
        queried = [evaluate(m & prefix) for m in masks]
        loss = np.array([scores[f]-scores[q] for f,q in zip(all_full,queried)])
        contrasts.append(dict(k=k, refit_min=loss.max(), refit_pooled=loss.mean(axis=0).max(),
                              refit_full=loss[-1].max(), keys=queried))
    return contrasts, clocks, all_full


def collect():
    base = pd.read_csv(ROOT / "guard-inputs.csv")
    full = []
    timing = {}
    for (n,s), (directory, info) in sorted(legacy_gain.jobs().items()):
        key = f"{n}-{s}"
        r = json.loads((legacy_gain.RUN/key/"selection.json").read_bytes())
        order = next(a["order"] for a in r["rows"] if a["method"]=="gain" and a["repeat"]==0)
        x,y,task,idx = legacy_gain.read_split(directory, info, s)
        split = base[(base.dataset==n) & (base.seed==s)].copy()
        begin = perf_counter()
        contrasts, clocks, references = weak_trace(x,y,task,idx,order,s,split.k.tolist())
        rows = {}
        for c in contrasts:
            rows[c["k"]] = [str(k) for k in c.pop("keys")]
        timing[key] = dict(reference=[str(k) for k in references], widths=rows,
                           clocks={str(k):t for k,t in clocks.items()}, all_s=perf_counter()-begin)
        full.append(split.merge(pd.DataFrame(contrasts), on="k", validate="one_to_one"))
        print(key, f"{timing[key]['all_s']:.3f}s", flush=True)
    pd.concat(full, ignore_index=True).to_csv(ROOT/"guard-refit-inputs.csv", index=False)
    (ROOT/"guard-refit-clock.json").write_text(json.dumps(timing), encoding="utf-8")


def evaluate():
    df = pd.read_csv(ROOT/"guard-refit-inputs.csv")
    clocks = json.loads((ROOT/"guard-refit-clock.json").read_bytes())
    outputs = []
    for n, group in df.groupby("dataset"):
        train = df[(df.dataset!=n) & (df.k < df.p) & df.eligible]
        for col in ["refit_min", "refit_pooled", "refit_full", "fraction", "gain_mass", "mdi"]:
            cutoff = train[col].max()
            indices = [2] if col == "refit_full" else [0,1,2]
            for seed, split in group.groupby("seed"):
                record = clocks[f"{n}-{seed}"]
                eligible = split[(split[col]<=cutoff) | (split.k==split.p)]
                selected = eligible[eligible.eligible].k.min()
                original = split.original.iloc[0]
                baseline = 1 + ((split.k<=original) & (split.k<split.p)).sum()
                used = 1 + ((eligible.k<=selected) & (eligible.k<eligible.p)).sum()
                keys = {record["reference"][j] for j in indices}
                for width in split.loc[(split.k<=selected)&(split.k<split.p), "k"]:
                    keys.update(record["widths"][str(width)][j] for j in indices)
                proxy_s = sum(record["clocks"][key] for key in keys) if col.startswith("refit_") else None
                outputs.append(dict(dataset=n, seed=seed, method=col, baseline=baseline, used=used,
                    saved=baseline-used, k=selected, original=original, proxy_s=proxy_s))
    frame = pd.DataFrame(outputs)
    frame.to_csv(ROOT/"guard-refit-screen.csv", index=False)
    print(frame.assign(worse=frame.k>frame.original).groupby("method")[["saved","worse","proxy_s"]].sum(min_count=1))


def two_proposals():
    """Two deployment-width queries plus full, with equal proposal budgets.

    Select two grid positions minimizing empirical log width among other
    sources. Scalar probe scores localize the five calibration neighbors.
    The fixed pair uses every training split with equal weight.
    """
    data = pd.read_csv(ROOT/"guard-refit-inputs.csv")
    methods = ["fixed", "fraction", "gain_mass", "mdi", "vote_min", "vote_pooled",
               "refit_min", "refit_pooled", "refit_full"]
    jobs = []
    for (n,s), split in data.groupby(["dataset","seed"]):
        p = int(split.p.iloc[0])
        ks = np.array([1, *(math.ceil(p*f) for f in [.125,.25,.5,.75])])
        aligned = split.set_index("k").loc[ks]
        jobs.append(dict(dataset=n, seed=s, p=p, ks=ks, original=int(split.original.iloc[0]),
                         ok=aligned.eligible.to_numpy(), scores={m:aligned[m].to_numpy()
                         for m in methods if m!="fixed"}))
    pairs = list(combinations(range(5), 2))
    receipts = []
    for job in jobs:
        train = [j for j in jobs if j["dataset"]!=job["dataset"]]
        for m in methods:
            neighborhood = train if m=="fixed" else sorted(train, key=lambda j:
                np.linalg.norm(j["scores"][m]-job["scores"][m]))[:5]
            def width(j, pair):
                return next((j["ks"][i] for i in pair if j["ok"][i]), j["p"])
            pair = min(pairs, key=lambda pair: np.mean([np.log(width(j, pair)/j["p"]) for j in neighborhood]))
            k = int(width(job,pair))
            checked = list(dict.fromkeys(int(job["ks"][i]) for i in pair))
            if k in checked:
                checked = checked[:checked.index(k)+1]
            receipts.append(dict(dataset=job["dataset"], seed=job["seed"], method=m, k=k,
                original=job["original"], ratio=k/job["original"], pair=str(pair), checks=len(checked)+1))
    out = pd.DataFrame(receipts)
    out.to_csv(ROOT/"guard-two-proposals.csv", index=False)
    summary = out.assign(log_ratio=np.log(out.ratio),worse=out.ratio>1).groupby("method").agg(
        log_ratio=("log_ratio","mean"), worse=("worse","sum"), checks=("checks","sum"))
    print(summary.assign(width_ratio=np.exp(summary.log_ratio)).drop(columns="log_ratio"))


if __name__ == "__main__":
    with threadpool_limits(1):
        collect()
        evaluate()
        two_proposals()
