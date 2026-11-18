"""Check native choices independently using augmented least squares in direct data space."""
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research.lowdim.method import fast_support, pair_support
from research.pair_checks.reference import Refits, mechanism_supports, standardize


HERE = Path(__file__).resolve().parent


def ridge_loss(z, response, support):
    """Direct refit on rows with a penalty block, independently of kernel updates."""
    if not support:
        return float(np.sum(response**2))
    matrix = z[:, list(support)]
    k = len(support)
    design = np.vstack((matrix, len(z)**0.25 * np.eye(k)))
    target = np.vstack((response, np.zeros((k, response.shape[1]))))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(np.sum((design @ coefficients - target)**2))


def verify_case(X, y, task, seed):
    prev, old = fast_support(X, y, task=task, seed=seed)
    cols, info = pair_support(X, y, task=task, seed=seed, audit=True)
    normal, _ = pair_support(X, y, task=task, seed=seed)
    repeated, repeated_info = pair_support(X, y, task=task, seed=seed, audit=True)
    np.testing.assert_array_equal(cols, normal)
    np.testing.assert_array_equal(cols, repeated)
    np.testing.assert_array_equal(info["selection_order"], repeated_info["selection_order"])
    for k in ["rank", "tier_scores"]:
        np.testing.assert_array_equal(info[k], old[k])
    assert len(cols) == info["width"] == len(prev) == len(np.unique(cols))
    assert np.all(np.diff(cols) > 0) and np.all((0 <= cols) & (cols < X.shape[1]))
    path = list(map(int, info["selection_order"]))
    if len(cols) == X.shape[1]:
        np.testing.assert_array_equal(cols, np.arange(X.shape[1]))
        assert path == []
        return 0, 0
    np.testing.assert_array_equal(cols, sorted(path))
    z, response = standardize(X, y, task, seed)
    k = len(cols)
    start_width = min(k, 2)
    first_refits = [ridge_loss(z, response, s) for s in combinations(range(X.shape[1]), start_width)]
    actual_loss = ridge_loss(z, response, path[:start_width])
    scale = max(1., np.sum(response**2))
    assert actual_loss <= min(first_refits) + 3e-12 * scale
    count = len(first_refits)
    for width in range(start_width+1, k+1):
        prefix = path[:width-1]
        refits = [ridge_loss(z, response, prefix + [j]) for j in range(X.shape[1]) if j not in prefix]
        current = ridge_loss(z, response, path[:width])
        assert current <= min(refits) + 3e-12 * scale
        assert current <= actual_loss + 3e-12 * scale
        actual_loss = current
        count += len(refits)
    return count, max(0, k-2)


def run():
    rng = np.random.default_rng(4251)
    cases = []
    for seed in range(12):
        X = rng.normal(size=(64 + seed, 10))
        X[:, 6] = X[:, 0] + 1e-12 * X[:, 6]
        X[:, 7] = X[:, 2]
        X[:, 8] = 4.
        signal = X @ np.asarray([1., -.7, .7, 1., .4, .8, .2, -.2, 0., .5])
        y = signal + rng.normal(size=len(X))
        if seed % 2:
            X = X.astype(np.float32)[:, ::-1]
        cases.extend([(X, y, "regression", seed),
                      (X, np.digitize(y, [-1., 0., 1.]), "classification", seed)])
    high = rng.normal(size=(1165, 12))
    y = high[:, :6].sum(axis=1) + rng.normal(size=len(high))
    for seed in [192, 781]:
        cases.extend([(high[::-1], y[::-1], "regression", seed),
                      (high, np.where(y > 0, "positive", "negative"), "classification", seed)])
    U = np.asarray(list(product([-1., 1.], repeat=8)))
    X = U[:, [0, 0, 1, 2, 3, 4, 5, 6]]
    y = U[:, 0] + .6*U[:, 1] + .32*U[:, 2]
    cases.extend([(X, y, "regression", 999), (U, U[:, 0], "regression", 17),
                  (X, np.ones(len(X)), "regression", 16),
                  (X[:, :1], y, "regression", 16), (X[:1], y[:1], "regression", 16),
                  (np.ones_like(X), y, "regression", 16)])
    count = 0
    forward_steps = 0
    for item in cases:
        checks, steps = verify_case(*item)
        count += checks
        forward_steps += steps

    old, info = fast_support(X, y, task="regression", seed=999)
    ref = Refits(*standardize(X, y, "regression", 999))
    paths = mechanism_supports(ref, old, info["rank"])
    new, audit = pair_support(X, y, task="regression", seed=999, audit=True)
    np.testing.assert_array_equal(paths["old"], [0, 1, 2])
    np.testing.assert_array_equal(paths["pair_old_fill"], [0, 2, 1])
    np.testing.assert_array_equal(audit["selection_order"], [0, 2, 3])
    np.testing.assert_array_equal(new, [0, 2, 3])
    np.testing.assert_array_equal(pair_support(U, U[:, 0], task="regression")[0], [0])

    # Illustration designed after the development ablation. These centered,
    # orthogonal factors give a suppressor whose marginal covariance is zero.
    rho = .96
    suppressor = U.copy()
    suppressor[:, 1] = rho*U[:, 0] + np.sqrt(1-rho**2)*U[:, 1]
    response = U[:, 1] + .34*U[:, 2] + .32*U[:, 3] + .30*U[:, 4]
    init, scores = fast_support(suppressor, response, task="regression", seed=999)
    suppressor_ref = Refits(*standardize(suppressor, response, "regression", 999))
    s_paths = mechanism_supports(suppressor_ref, init, scores["rank"])
    chosen, details = pair_support(suppressor, response, task="regression", seed=999, audit=True)
    np.testing.assert_array_equal(chosen, [0, 1, 2, 3])
    np.testing.assert_array_equal(details["selection_order"], s_paths["pair_forward"])
    np.testing.assert_array_equal(s_paths["anchor_forward"], [2, 3, 4, 1])
    assert suppressor_ref.utility(chosen) > suppressor_ref.utility(s_paths["anchor_forward"])
    suppressed_checks = verify_case(suppressor, response, "regression", 999)

    # Verify legacy dispatch on either side of the 32-column boundary.
    large = rng.normal(size=(90, 33))
    target = large[:, :6].sum(axis=1)
    cases_at_cutoff = verify_case(large[:, :32], target, "regression", 245)
    for task, labels in [("regression", target), ("classification", target>0)]:
        np.testing.assert_array_equal(fast_support(large, labels, task=task)[0],
                                      pair_support(large, labels, task=task)[0])
    failures = [(np.zeros((4, 0)), np.arange(4), "classification"),
                (U, np.ones(len(U)), "classification"),
                (U, np.full(len(U), np.nan), "regression")]
    for data, labels, task in failures:
        try:
            pair_support(data, labels, task=task)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input unexpectedly accepted")
    result = {
        "low_dimensional_inputs": len(cases)+2,
        "direct_augmented_lstsq_candidate_refits": count+cases_at_cutoff[0]+suppressed_checks[0],
        "post_pair_conditional_steps_checked": forward_steps+cases_at_cutoff[1]+suppressed_checks[1],
        "wide_fallback_cases": 2, "invalid_inputs": len(failures),
        "duplicate_diagnostic": {
            "width": len(new),
            "paths": {arm: support.tolist() for arm, support in paths.items()},
            "train_ridge_gain": {arm: ref.utility(support) for arm, support in paths.items()},
        },
        "posthoc_suppressor_diagnostic": {
            "width": len(chosen), "rho": rho,
            "paths": {arm: support.tolist() for arm, support in s_paths.items()},
            "train_ridge_gain": {arm: suppressor_ref.utility(support) for arm, support in s_paths.items()},
        },
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
