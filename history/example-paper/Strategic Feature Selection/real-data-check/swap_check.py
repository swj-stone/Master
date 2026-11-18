"""A single training-objective swap at the archived width on 31 development configurations."""
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.conditional_dense.original_tasks.paper_splits import f_support
from research.feature_game_eval import panels as bench
from research.lowdim import formal_study as formal
from research.lowdim.ablate_peers import top
from research.lowdim.method import fast_support


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INDEX = ROOT / "research/lowdim/peer_evidence/cases.csv"
RUNS = ROOT / "results/minpi-study"
KEY = ["cohort", "dataset", "fold", "seed"]
EPS = 1e-12


def sampled_rows(n, seed):
    """Same consecutive strata and unsigned SplitMix64 stream as native_votes.c."""
    m = min(n, 768)
    if m == n:
        return slice(None)
    mask = (1 << 64) - 1
    state = int(seed)
    rows = []
    for index in range(m):
        left, right = index * n // m, (index + 1) * n // m
        span = right - left
        offset = 0
        if span > 1:
            threshold = ((1 << 64) - span) % span
            while True:
                state = (state + 0x9e3779b97f4a7c15) & mask
                z = ((state ^ (state >> 30)) * 0xbf58476d1ce4e5b9) & mask
                z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
                word = z ^ (z >> 31)
                if word >= threshold:
                    offset = word % span
                    break
        rows.append(left + offset)
    return rows


def cross_products(X, y, task, seed):
    rows = sampled_rows(len(y), seed)
    z = np.array(X[rows], dtype=np.float64)
    z -= z.mean(axis=0)
    std = z.std(axis=0)
    z /= np.where(std > 0, std, 1.0)
    y = y[rows]
    if task == "classification":
        _, labels = np.unique(y, return_inverse=True)
        targets = np.eye(labels.max() + 1)[labels]
    else:
        targets = y.astype(np.float64).reshape(-1, 1)
        targets = targets / targets.std() if targets.std() > 0 else targets
    targets -= targets.mean(axis=0)
    p = z.shape[1]
    gram = z.T @ z + np.sqrt(len(z)) * np.eye(p)
    rhs = z.T @ targets
    return gram, rhs, np.sum(targets**2)


def verify_tiers(gram, rhs, tiers):
    p = len(gram)
    empty = np.sum(rhs**2, axis=1) / gram.diagonal()
    singleton = np.zeros(p)
    for k in range(p):
        weights = gram[:, k] / gram[k, k]
        others = np.arange(p) != k
        denom = gram.diagonal()[others] - weights[others] * gram[others, k]
        residual = rhs[others] - weights[others, None] * rhs[k]
        singleton[others] += np.sum(residual**2, axis=1) / denom
    np.testing.assert_allclose([empty, singleton / (p - 1)], tiers, rtol=2e-8, atol=2e-8)


def one_swap(cols, gram, rhs, target_ss):
    def loss(indices):
        block = gram[np.ix_(indices, indices)]
        return target_ss - np.sum(rhs[indices] * np.linalg.solve(block, rhs[indices]))

    initial = loss(cols)
    outside = np.setdiff1d(np.arange(len(gram)), cols)
    candidates = sorted(tuple(sorted((set(cols) - {j}) | {k})) for j in cols for k in outside)
    best = initial
    chosen = cols
    tol = EPS * max(1., abs(initial))
    for candidate in candidates:
        indices = np.asarray(candidate)
        value = loss(indices)
        if value < best:
            best, chosen = value, indices
    if initial - best <= tol:
        return cols, 0.0
    return chosen, initial - best


def check_case(key, record, X, y, task, fit, measure, ref):
    cohort, dataset, fold, seed = key
    initial, info = fast_support(X, y, task=task, seed=seed)
    k = len(initial)
    assert k == record.k < X.shape[1]
    gram, rhs, target_ss = cross_products(X, y, task, seed)
    verify_tiers(gram, rhs, info["tier_scores"])
    starts = {
        "peer": initial,
        "empty": top(info["tier_scores"][0], k),
        "F": f_support(X, y, task, k),
    }
    peer_file = RUNS / "lowdim-peer-ablation-20261118a" / f"{cohort}-{dataset}-{fold}-{seed}.npz"
    with np.load(peer_file) as old, np.load(ref) as panel:
        np.testing.assert_array_equal(initial, old["new_cols"])
        np.testing.assert_array_equal(starts["empty"], old["empty_cols"])
        np.testing.assert_array_equal(starts["F"], panel["F_cols"])

    # Freeze all decisions before any holdout prediction or scoring within this case.
    refined = {arm: one_swap(cols, gram, rhs, target_ss) for arm, cols in starts.items()}
    scores = {}
    for cols in [*starts.values(), *(support for support, _ in refined.values())]:
        key_cols = tuple(cols)
        if key_cols not in scores:
            scores[key_cols] = measure(*fit(cols))
    rows = []
    for arm, cols in starts.items():
        changed, improvement = refined[arm]
        before, after = scores[tuple(cols)], scores[tuple(changed)]
        expected = {"peer": record.score_new, "empty": record.score_empty, "F": record.score_F}[arm]
        np.testing.assert_allclose(before, expected, rtol=0, atol=EPS)
        rows.append(dict(zip(KEY, key)) | {
            "metric": record.metric, "p": X.shape[1], "k": k, "start": arm,
            "n_swaps_checked": k * (X.shape[1] - k),
            "initial_cols": " ".join(map(str, cols)), "refined_cols": " ".join(map(str, changed)),
            "train_loss_decrease": improvement, "score_before": before,
            "score_after": after, "score_delta": after-before,
        })
    return rows


def run():
    index = pd.read_csv(INDEX).set_index(KEY)
    assert len(index) == 31 and index.index.is_unique
    rows = []
    for cohort, panel, truth, units, _ in bench.panels():
        key = (cohort, panel.name, panel.fold, panel.seed)
        if key not in index.index:
            continue
        ref = RUNS / "lowdim-core-20261118a-pass1" / ("-".join(map(str,key)) + ".npz")
        def fit(cols):
            return bench.fit(panel, cols)
        def measure(pred, decision):
            return bench.deployment.score_predictions(panel.name, truth, pred, units, panel.task, decision)[0]
        rows.extend(check_case(key, index.loc[key], panel.X, panel.y, panel.task, fit, measure, ref))
    for name in ("breast_cancer", "diabetes"):
        raw, target, task, _, order, _, _ = formal.load_dataset(name)
        for fold, seed in enumerate(formal.P["formal_seeds"], 1):
            key = ("formal", name, fold, seed)
            record = index.loc[key]
            X, y, Xt, truth, *_ = formal.inputs(name, seed, raw, target, order, task)
            ref = RUNS / "lowdim-formal-20261118a-pass1" / f"{name}-{seed}.npz"
            def fit(cols):
                return formal.train("ridge", X[:, cols], y, Xt[:, cols], task, seed)
            def measure(pred, decision):
                return formal.metrics(truth, pred, decision)[record.metric]
            rows.extend(check_case(key, record, X, y, task, fit, measure, ref))
    frame = pd.DataFrame(rows).sort_values(KEY + ["start"])
    assert set(frame.set_index(KEY).index) == set(index.index)
    assert len(frame) == len(index)*3 and not frame.duplicated(KEY+["start"]).any()
    frame.to_csv(HERE / "cases.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
