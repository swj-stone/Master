"""Compare the batched MinPI ranker to a cached direct fit of precisely the same game."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.context_review.timing_process import bind_logical_cpu
from minppi_mofs.sampled_linear_minpi import Ranking, coalition_batches, rank_features

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "thesis-paper/evidence/minpi_compression"


def direct(x, y, z, v, *, task, seed, rounds=4, alpha=.95, ridge=1.):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    if task != "regression":
        classes = np.unique(y)
        y, v = (a[:, None] == classes for a in (y, v))
    else:
        y, v = y.reshape(-1, 1), v.reshape(-1, 1)
    xbar, ybar = x.mean(axis=0), y.mean(axis=0)
    x, z, y, v = x - xbar, z - xbar, y - ybar, v - ybar
    p = x.shape[1]
    gram, xy = x.T @ x, x.T @ y
    gram.flat[::p+1] += ridge
    scale = np.einsum("ij,ij->", v, v) / len(v) or 1.
    full = tuple(range(p))
    tie = np.empty(p)
    cache = {}

    def utility(cols):
        if cols not in cache:
            error = v
            if cols:
                indices = np.asarray(cols)
                factor = cho_factor(gram[np.ix_(indices, indices)], check_finite=False)
                beta = cho_solve(factor, xy[indices], check_finite=False)
                if cols == full:
                    tie[:] = np.linalg.norm(beta, axis=1)
                error = v - z[:, indices] @ beta
            loss = np.einsum("ij,ij->", error, error) / len(v)
            cache[cols] = 1 / (1 + loss / scale)
        return cache[cols]

    quota = alpha * utility(full)
    minimum = np.ones(p, dtype=np.int8)
    witness = [None] * p
    for base, targets in coalition_batches(p, rounds, seed):
        active = targets[minimum[targets] > -1]
        if not len(active):
            continue
        cols = tuple(map(int, base))
        before = utility(cols) >= quota
        if not before:
            active = active[minimum[active] > 0]
        for i in active:
            after = utility(tuple(sorted((*cols, int(i))))) >= quota
            change = int(after) - int(before)
            minimum[i] = min(change, minimum[i])
            if change == -1:
                witness[i] = cols
    order = np.lexsort((np.arange(p), -tie, -minimum))
    return Ranking(order, minimum, minimum, tie, {"factorizations":len(cache) - int(() in cache)}, witness)


def wide_jobs():
    root = ROOT / "results/context_review/wide-uci"
    for folder in sorted(root.glob("*/fold-0")):
        data = folder / "inputs.npz"
        receipt = json.loads((folder / "layout.json").read_bytes())
        assert sha256(data.read_bytes()).hexdigest() == receipt["inputs_sha256"]
        with np.load(data) as a:
            x = a["x"][:, :100]
            train, vote = a["train"], a["proposal"]
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            fit = imputer.fit_transform(x[train])
            scaler = StandardScaler().fit(fit)
            yield folder.parent.name, "classification", (scaler.transform(fit), a["y"][train],
                scaler.transform(imputer.transform(x[vote])), a["y"][vote])


def run(panel, jobs, seed):
    rows = []
    cpu = bind_logical_cpu()
    with threadpool_limits(limits=1):
        for ix, (name, task, arrays) in enumerate(jobs):
            outputs = {}
            for rep in range(3):
                methods = [("batched", rank_features), ("direct", direct)]
                if (rep+ix) % 2:
                    methods.reverse()
                for position, (method, fn) in enumerate(methods):
                    t = perf_counter()
                    result = fn(*arrays, task=task, seed=seed)
                    elapsed = perf_counter() - t
                    record = (result.order.tolist(), result.minimum.tolist(), result.witness)
                    outputs[method] = record
                    digest = sha256(json.dumps(record).encode()).hexdigest()
                    rows.append(dict(source=name, panel=panel, seed=seed, repeat=rep, position=position,
                                     cpu=cpu, method=method, fit_rows=len(arrays[0]), vote_rows=len(arrays[2]),
                                     features=arrays[0].shape[1], seconds=elapsed, sha256=digest, **result.work))
                if outputs["batched"] != outputs["direct"]:
                    raise ValueError(f"Output mismatch on {name}: {outputs}")
            print(f"{name} verified three paired rankings and witnesses", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 6 * (6 if panel == "wide" else 8):
        raise ValueError("Incomplete panel")
    DEST.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEST / f"paired-cost-{panel}.csv", index=False)
    print(frame.groupby(["source","method"]).seconds.median().unstack().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=["wide","confirmation"])
    args = parser.parse_args()
    if args.panel == "wide":
        run("wide", wide_jobs(), 42)
    else:
        from experiments.minpi_compression.benchmark import cost_jobs
        run("confirmation", cost_jobs(42), 42)
