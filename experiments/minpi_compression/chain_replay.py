"""Replay the registered nested-chain sampler and its query-matched diagnostic control."""
from functools import partial
from hashlib import sha256
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_compression.dictator_audit import bind_last_cpu, jobs
from minppi_mofs.chain_minpi import rank_chain
from minppi_mofs.sampled_linear_minpi import rank_features


def real_test_labels(panel, name, seed, y, idx):
    if panel == "confirmation":
        return y[idx["test"]]
    folder = b.ROOT/"results/context_review/wide-uci"/name/f"fold-{seed-42}"
    with np.load(folder/"targets.npz") as targets, np.load(folder/"inputs.npz") as data:
        assert b.digest(folder/"targets.npz") == json.loads((folder/"layout.json").read_bytes())["targets_sha256"]
        np.testing.assert_array_equal(targets["test_row"], data["test_row"])
        return targets["y"]


def check_witnesses(x, y, z, truth, task, results):
    if task == "classification":
        classes = np.unique(y)
        y, truth = (y[:,None] == classes).astype(float), (truth[:,None] == classes).astype(float)
    else:
        y, truth = y[:,None], truth[:,None]
    scale = np.square(truth-y.mean(axis=0)).sum()/len(z) or 1.
    values = {}
    def utility(cols):
        key = tuple(sorted(cols))
        if key not in values:
            pred = y.mean(axis=0)
            if key:
                pred = Ridge(alpha=1.,solver="svd").fit(x[:,key],y).predict(z[:,key]).reshape(truth.shape)
            values[key] = 1/(1+np.square(truth-pred).sum()/len(z)/scale)
        return values[key]
    quota = .95*utility(range(x.shape[1]))
    for result in results:
        for j,(bound,witness) in enumerate(zip(result.minimum,result.witness,strict=True)):
            assert (bound == -1) == (witness is not None)
            if witness is not None:
                assert j not in witness
                assert utility(witness) >= quota > utility((*witness,j))


def summarize(clocks, folder):
    med = clocks.groupby(["panel","source","seed","task","method"]).agg(
        {k:("median" if k in ["rank_s","selection_s","deployment_s"] else "first")
         for k in ["rank_s","selection_s","deployment_s","columns","rank_hash","forecast","k",
                   "auc","accuracy","r2","scaled_mae","half_auc","half_accuracy","half_r2","half_scaled_mae"]})
    pairs = []
    for other in ["original","uniform","coefficient"]:
        a, c = (med.xs(m,level="method") for m in ("chain",other))
        fx = pd.DataFrame(index=a.index)
        for k in ["rank_s","selection_s","deployment_s","k"]:
            fx["ratio_"+k] = np.log(a[k]/c[k])
        for k in ["auc","accuracy","r2","scaled_mae","half_auc","half_accuracy","half_r2","half_scaled_mae"]:
            fx["delta_"+k] = a[k] - c[k]
        for k in ["columns","rank_hash","forecast"]:
            fx["changed_"+k] = (a[k]!=c[k]).astype(int)
        fx["splits"] = 1
        fx["baseline"] = other
        pairs.append(fx.reset_index())
    paired = pd.concat(pairs,ignore_index=True)
    numeric = [k for k in paired if k not in ["panel","source","seed","task","baseline"]]
    op = {k:"sum" if k.startswith("changed") or k=="splits" else "mean" for k in numeric}
    source = paired.groupby(["baseline","panel","source","task"]).agg(op).reset_index()
    domain = source.groupby(["baseline","panel","task"]).agg(op).reset_index()
    for frame,name in [(source,"source-effects"),(domain,"domain-effects")]:
        ratios = [k for k in frame if k.startswith("ratio_")]
        frame[ratios] = np.exp(frame[ratios])
        frame.to_csv(folder/f"{name}.csv",index=False)
    print(domain[["baseline","panel","task","ratio_rank_s","ratio_selection_s","ratio_k","delta_auc",
                  "delta_accuracy","delta_r2","delta_scaled_mae","changed_rank_hash","changed_columns"]].to_string(index=False))


def main():
    folder = b.EVIDENCE/"chain-replay"
    folder.mkdir(parents=True, exist_ok=True)
    cpu = bind_last_cpu()
    oracle = pd.read_csv(b.EVIDENCE/"audit/exact-minima.csv").set_index(["source","seed"]).sort_index()
    clocks, labels = [], []
    rng = np.random.default_rng(711)
    with threadpool_limits(1):
        for panel,name,seed,(x,y,task,idx) in jobs():
            prep = b.transform(x,idx["fit"])
            arrays = (prep.transform(x[idx["fit"]]), y[idx["fit"]], prep.transform(x[idx["vote"]]), y[idx["vote"]])
            engine = dict(task=task,seed=seed,**b.DESIGN["engine"])
            budget = rank_chain(*arrays,**engine).work["refinement_queries"]
            arms = dict(chain=rank_chain, uniform=partial(rank_chain,random_queries=budget),
                        original=rank_features, coefficient=partial(rank_features,method="coefficient"))
            arm_order = rng.permutation(list(arms))
            tests = real_test_labels(panel,name,seed,y,idx)
            dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
            scale = y[dev].std()
            results, half_metrics = {}, {}
            for repeat in range(len(arms)):
                for pos,arm in enumerate(np.roll(arm_order,repeat)):
                    def ranked(*args, **kwargs):
                        kwargs.pop("method")
                        results[arm] = arms[arm](*args,**kwargs)
                        return results[arm]
                    with patch.object(b.engine,"rank_features",ranked):
                        rec, pred = b.select_one("minpi",x,y,task,idx,seed)
                    if repeat == 0:
                        half = int(np.ceil(x.shape[1]/2))
                        fixed = b.predict(x,y,task,dev,idx["test"],results[arm].order,[half])[half]
                        half_metrics[arm] = {"half_"+k:v for k,v in b.metrics(tests,fixed,task,scale).items()}
                    q = results[arm]
                    clocks.append(dict(panel=panel,source=name,seed=seed,task=task,cpu=cpu,repeat=repeat,
                                       position=pos,method=arm,p=x.shape[1],k=rec["k"],
                                       columns=" ".join(map(str,sorted(rec["order"][:rec["k"]]))),
                                       rank_hash=sha256(q.order.astype("<i8").tobytes()).hexdigest(),
                                       forecast=sha256(pred.tobytes()).hexdigest(),
                                       **{k:rec[k] for k in ["rank_s","selection_s","deployment_s"]},
                                       **b.metrics(tests,pred,task,scale),**half_metrics[arm],**q.work))
            c, r = results["chain"],results["uniform"]
            for k in ("refinement_queries","distinct_votes","anchors","descending_brackets"):
                assert c.work[k] == r.work[k]
            if not budget:
                np.testing.assert_array_equal(c.minimum,r.minimum)
            check_witnesses(*arrays,task,[results[a] for a in ("chain","uniform","original")])
            exact = None
            if panel=="confirmation" and (name,seed) in oracle.index:
                exact = oracle.loc[name,seed].sort_values("feature").exact.to_numpy()
            for arm in ("chain","uniform","original"):
                result = results[arm]
                if exact is not None:
                    assert np.all(result.minimum >= exact)
                for j, upper in enumerate(result.minimum):
                    labels.append(dict(panel=panel,source=name,seed=seed,method=arm,feature=j,bound=upper,
                                       exact=exact[j] if exact is not None else None,
                                       witness=json.dumps(result.witness[j])))
            print(f"{panel}/{name}/{seed}: {budget} extra queries, chain {(c.minimum<0).sum()} / uniform {(r.minimum<0).sum()} negatives",flush=True)
    result = pd.DataFrame(clocks)
    result.to_csv(folder/"runs.csv",index=False)
    pd.DataFrame(labels).to_csv(folder/"bounds.csv",index=False)
    summarize(result,folder)


if __name__ == "__main__":
    main()
