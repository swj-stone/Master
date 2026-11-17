"""Matched comparison with greedy leave-one-out regularized least squares."""
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, chain_validation as validation


def greedy_ranking(x, y, ridge=1.):
    """Greedy pooled squared LOO loss with a free intercept on fixed feature scales."""
    x = np.asarray(x, float) - x.mean(axis=0)
    y = np.asarray(y, float).reshape(len(x), -1)
    y = y - y.mean(axis=0)
    # Store M X, M Y and diag(M), where M = I - H for the current ridge smoother.
    residual_x, residual_y = x.copy(), y.copy()
    diagonal = np.full(len(x), 1 - 1/len(x))
    available = np.arange(x.shape[1])
    order = []
    while len(available) > 1:
        direction = residual_x[:, available]
        denom = ridge + np.einsum("ij,ij->j", x[:, available], direction)
        slope = (direction.T @ y) / denom[:, None]
        base_loss = np.einsum("ij,ij->i", residual_y, residual_y)
        scores = (base_loss[:, None] - 2 * direction * (residual_y @ slope.T)
                  + direction**2 * np.einsum("ij,ij->i", slope, slope))
        leave_one_out = scores / (diagonal[:, None] - direction**2 / denom)**2
        best = leave_one_out.mean(axis=0).argmin()
        feature = available[best]
        order.append(feature)
        q = direction[:, best]
        residual_x -= np.outer(q, q @ x) / denom[best]
        residual_y -= np.outer(q, slope[best])
        diagonal -= q*q / denom[best]
        available = np.delete(available, best)
    return np.r_[order, available].astype(int)


def _rank(method, x, y, task, idx, seed):
    if method == "chain":
        return validation.rank(method, x, y, task, idx, seed)
    assert method == "greedy"
    pool = np.sort(np.r_[idx["fit"], idx["vote"]])
    z = b.transform(x, pool).transform(x[pool])
    target = b.targets(y[pool], task, len(np.unique(y[pool])))
    return greedy_ranking(z, target), {}


def run():
    validation.bind_cpu()
    out = b.EVIDENCE / "greedy-rls"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in validation.data.SOURCES:
        for seed in b.DESIGN["seeds"]:
            x, y, task, idx = validation.data.split(name, seed)
            train_y = y.copy()
            train_y[idx["test"]] = -1
            clocks, forecasts = {}, {}
            for rep in range(3):
                methods = ("greedy","chain") if (rep+seed)%2 else ("chain","greedy")
                for method in methods:
                    with patch.object(b, "ranked", _rank):
                        rec, pred = b.select_one(method, x, train_y, task, idx, seed)
                    clocks.setdefault(method, []).append(rec)
                    if rep:
                        np.testing.assert_array_equal(forecasts[method], pred)
                    forecasts[method] = pred
            for method, pred in forecasts.items():
                rec = clocks[method][0]
                validation.verify_predictions(x, y, task, idx, rec, pred)
                dev = np.r_[idx["fit"],idx["vote"],idx["width"]]
                loss = b.metrics(y[idx["test"]], pred, task, np.std(y[dev]))
                rows.append(dict(dataset=name, seed=seed, task=task, method=method,
                    retained=rec["k"]/rec["p"], k=rec["k"],
                    order=" ".join(map(str,rec["order"])), **loss,
                    **{k: np.median([t[k] for t in clocks[method]])
                       for k in ("selection_s","rank_s","deployment_s")}))
            print(f"{name} {seed} complete", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "scores.csv", index=False)
    deltas = []
    for task, subset in frame.groupby("task"):
        for key in ("auc","accuracy","r2","scaled_mae","retained","selection_s","deployment_s"):
            pairs = subset.pivot(index=["dataset","seed"],columns="method",values=key).dropna()
            if not len(pairs):
                continue
            is_ratio = key in ("retained","selection_s","deployment_s")
            d = np.log(pairs.chain / pairs.greedy) if is_ratio else pairs.chain - pairs.greedy
            value = d.groupby(level=0).mean().mean()
            deltas.append(dict(task=task, metric=key, value=np.exp(value) if is_ratio else value))
    result = pd.DataFrame(deltas)
    result.to_csv(out / "effects.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
