"""Cross-check native blocks, caps and stopping decisions with exact dense refits."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research.block_checks.reference import block_path
from research.lowdim.method import block_support, fast_support, pair_support
from research.pair_checks.reference import standardize


HERE = Path(__file__).resolve().parent


def check(X, y, task, seed):
    old, ranks = fast_support(X, y, task=task, seed=seed)
    pair, prior = pair_support(X, y, task=task, seed=seed)
    cols, info = block_support(X, y, task=task, seed=seed, audit=True)
    regular, _ = block_support(X, y, task=task, seed=seed)
    repeated, again = block_support(X, y, task=task, seed=seed, audit=True)
    np.testing.assert_array_equal(cols, regular)
    np.testing.assert_array_equal(cols, repeated)
    for field in ["rank", "tier_scores"]:
        np.testing.assert_array_equal(info[field], ranks[field])
        np.testing.assert_array_equal(info[field], prior[field])
    for field in ["selection_order", "block_sizes"]:
        np.testing.assert_array_equal(info[field], again[field])
    p = X.shape[1]
    cap = info["width_cap"]
    assert 1 <= len(cols) == info["width"] <= cap == len(old) == len(pair) <= p
    assert np.all(np.diff(cols) > 0)
    path = info["selection_order"]
    if not np.any(ranks["rank"] > 0):
        assert not len(path) and not len(info["block_sizes"])
        np.testing.assert_array_equal(cols, np.arange(p))
        return dict(cap=cap, blocks=[], decisions=[], candidate_refits=0)
    np.testing.assert_array_equal(cols, np.sort(path))
    z, target = standardize(X, y, task, seed)
    m = len(z)
    if task == "classification":
        # Equivalent to m - sum(n_class²)/m for the sampled, centered one-hot target.
        counts = np.sum(target > 0, axis=0)
        np.testing.assert_allclose(np.sum(target**2), m - (counts @ counts)/m)
    else:
        np.testing.assert_allclose(np.sum(target**2), m)
    blocks, decisions, refits = block_path(z, target, cap)
    expected = [j for block in blocks for j in block]
    np.testing.assert_array_equal(path, expected, err_msg=f"{task}, seed={seed}, cap={cap}")
    np.testing.assert_array_equal(info["block_sizes"], list(map(len, blocks)))
    return dict(cap=cap, blocks=blocks, decisions=decisions, candidate_refits=refits)


def run():
    rng = np.random.default_rng(97513)
    summary = []
    for seed in range(8):
        X = rng.normal(size=(83+seed, 11))
        X[:, 5] = X[:, 0] + 1e-10*X[:, 5]
        X[:, 8] = 1
        y = X[:, :5].sum(axis=1) + rng.normal(0, 2, len(X))
        if seed % 2:
            X = X.astype(np.float32)[::-1, ::-1]
            y = y[::-1]
        summary.extend([check(X, y, "regression", seed),
                        check(X, np.digitize(y, [-1, 1]), "classification", seed)])
    large = rng.normal(size=(991, 12))
    signal = large[:, :6].sum(axis=1)
    summary.extend([check(large, signal, "regression", 987),
                    check(large, np.where(signal > 1, "high", "low"), "classification", 923)])

    U = np.array(list(product([-1., 1.], repeat=8)))
    X = U[:, [0, 1, 2, 3, 0, 1, 0, 1]].copy()
    X[:, 3] = .96*U[:, 2] + .28*U[:, 3]
    suppressors = check(X, U[:, 0] + .9*U[:, 1] + .7*U[:, 3], "regression", 999)
    assert suppressors["blocks"][:2] == [[0, 1], [2, 3]]
    assert suppressors["decisions"][0]["score"] > suppressors["decisions"][0]["best_single"]
    noisy = check(X, U[:, 0] + .9*U[:, 1] + 1.4*U[:, 7], "regression", 999)
    assert noisy["blocks"] == [[0, 1]] and noisy["cap"] > 2
    assert noisy["decisions"][0]["score"] < 0
    positive_full = check(U, U.sum(axis=1), "regression", 999)
    assert positive_full["cap"] == 8 and positive_full["blocks"] == [[0,1],[2,3],[4,5],[6,7]]
    odd_cap = check(U[:, :5], U[:, :5].sum(axis=1), "regression", 999)
    assert odd_cap["cap"] == 5 and odd_cap["blocks"] == [[0,1],[2,3],[4]]
    summary.extend([suppressors, noisy, positive_full, odd_cap])
    # Choose the fourth coefficient so that, after the seed, the added
    # likelihood-ratio term for including it equals exactly one log(m) penalty.
    X_tie = U[:, [0, 1, 2, 3, 0, 1, 0, 1]]
    m = len(U)
    attenuation = m/(m+np.sqrt(m))
    threshold = -np.expm1(-np.log(m)/m)
    fourth = np.sqrt(threshold*((1-attenuation)*2.36+4)/(attenuation-threshold))
    outcomes = []
    for adjustment in (1-1e-10, 1., 1+1e-10):
        labels = U[:, 0]+U[:, 1]+.6*U[:, 2]+adjustment*fourth*U[:, 3]+2*U[:, 7]
        _, audit = block_support(X_tie, labels, task="regression", seed=999, audit=True)
        expected, _, _ = block_path(*standardize(X_tie, labels, "regression", 999),
                                    audit["width_cap"])
        size = int(audit["block_sizes"][1])
        assert size == len(expected[1])
        outcomes.append(size)
    assert outcomes == [1, 1, 2]
    one = check(U[:, :1], U[:, 0], "regression", 999)
    cap_one = check(U, U[:, 0], "regression", 999)
    assert one["blocks"] == [[0]] and cap_one["blocks"] == [[0]]
    summary.extend([one, cap_one, check(U, np.ones(len(U)), "regression", 3),
                    check(np.ones_like(U), U[:, 0], "regression", 3),
                    check(U[:1], U[:1, 0], "regression", 3)])
    wide = rng.normal(size=(64, 33))
    y = wide[:, :4].sum(axis=1) + rng.normal(size=len(wide))
    summary.append(check(wide[:, :32], y, "regression", 11))
    for task, response in [("classification", y > 0), ("regression", y)]:
        old_cols = fast_support(wide, response, task=task)[0]
        block_cols, info = block_support(wide, response, task=task, audit=True)
        np.testing.assert_array_equal(old_cols, block_cols)
        assert info["width_cap"] == len(old_cols)
    result = dict(inputs=len(summary), wide_fallback_inputs=2,
                  cross_size_tie_sensitivity=dict(fourth_coefficient=fourth,
                      factors=[1-1e-10, 1., 1+1e-10], first_post_seed_sizes=outcomes),
                  direct_augmented_lstsq_refits=sum(s["candidate_refits"] for s in summary),
                  post_seed_decisions=sum(len(s["decisions"]) for s in summary),
                  early_stops=sum(any(not d["accepted"] for d in s["decisions"]) for s in summary),
                  post_seed_suppressors=suppressors, noise_floor=noisy)
    (HERE / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
