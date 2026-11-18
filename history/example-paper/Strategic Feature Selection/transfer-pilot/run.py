"""Local fixed-width study of conditional ridge gains on static Gaussian data."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research.lowdim.method import fast_support


OUT = Path(__file__).resolve().parent
P = 8
PAIRS = tuple(combinations(range(P), 2))
RHOS = (0.0, 0.8, 0.98)
METHODS = ("single_peer", "empty", "F", "mean_only", "forward", "best_train_pair")


def sample(rng, n, rho, mechanism):
    # X0 observes z directly; X1 shares z with X0 and also measures u.
    values = rng.standard_normal((n, P + 1))
    z, u, v = values[:, 0], values[:, 1], values[:, 2]
    X = values[:, :P].copy()
    X[:, 1] = rho * z + np.sqrt(1 - rho**2) * u
    signal = z if mechanism == "redundant" else u
    y = signal + 0.4 * v + 0.6 * values[:, -1]
    return X, y


def best_two(scores):
    return tuple(sorted(np.argsort(-scores, kind="stable")[:2]))


def select_pairs(X, y, Z, seed):
    _, info = fast_support(X, y, task="regression", seed=seed)
    univariate_f = f_regression(X, y)[0]
    centered_y = (y - y.mean()) / y.std()
    G = Z.T @ Z + np.sqrt(len(y)) * np.eye(P)
    b = Z.T @ centered_y

    # The native marginal and pooled gains must agree with independent 2x2 solves.
    isolated = b**2 / np.diag(G)
    pair_totals = {}
    conditional_sums = np.zeros(P)
    for pair in PAIRS:
        idx = list(pair)
        total = float(b[idx] @ np.linalg.solve(G[np.ix_(idx, idx)], b[idx]))
        pair_totals[pair] = total
        j, k = pair
        conditional_sums[j] += total - isolated[k]
        conditional_sums[k] += total - isolated[j]
    averaged = conditional_sums / (P - 1)
    np.testing.assert_allclose(info["tier_scores"], [isolated, averaged], atol=1e-8)
    np.testing.assert_allclose(info["rank"], np.minimum(isolated, averaged), atol=1e-8)

    # Fix the highest marginal feature, then maximize the conditional gain.
    anchor = int(np.argmax(isolated))
    anchored = (pair for pair in PAIRS if anchor in pair)
    picks = {
        "single_peer": best_two(info["rank"]),
        "empty": best_two(info["tier_scores"][0]),
        "F": best_two(univariate_f),
        "mean_only": best_two(info["tier_scores"][1]),
        "forward": max(anchored, key=pair_totals.get),
        "best_train_pair": max(PAIRS, key=pair_totals.get),
    }
    assert picks["empty"] == picks["F"]
    return picks, anchor


def run_case(mechanism, rho, seed):
    # Common latent draws across rho and mechanism; separate train/test streams.
    seeds = np.random.SeedSequence([20261118, seed]).spawn(2)
    X, y = sample(np.random.default_rng(seeds[0]), 256, rho, mechanism)
    X_test, y_test = sample(np.random.default_rng(seeds[1]), 2048, rho, mechanism)
    scaler = StandardScaler().fit(X)
    Z, Z_test = scaler.transform(X), scaler.transform(X_test)
    picks, anchor = select_pairs(X, y, Z, seed)

    def test_r2(columns):
        columns = list(columns)
        model = Ridge(alpha=1, solver="cholesky").fit(Z[:, columns], y)
        return model.score(Z_test[:, columns], y_test)

    # Enumerating the reference distribution avoids variation from random draws.
    scores = {pair: test_r2(pair) for pair in PAIRS}
    uniform = np.asarray(list(scores.values()))
    record = {"mechanism": mechanism, "rho": rho, "seed": seed, "anchor": anchor,
              "full_r2": test_r2(range(P)), "uniform_mean_r2": uniform.mean(),
              "uniform_q25_r2": np.quantile(uniform, .25),
              "uniform_median_r2": np.median(uniform),
              "uniform_q75_r2": np.quantile(uniform, .75)}
    for name, pair in picks.items():
        record[name + "_cols"] = " ".join(str(int(i)) for i in pair)
        record[name + "_r2"] = scores[pair]
    return record


def main():
    cases = pd.DataFrame(run_case(mechanism, rho, seed)
                         for mechanism in ("redundant", "suppressor")
                         for rho in RHOS for seed in range(24))
    cases.to_csv(OUT / "cases.csv", index=False, lineterminator="\n")
    cols = ["full_r2", "uniform_mean_r2", *(name + "_r2" for name in METHODS)]
    print(cases.groupby(["mechanism", "rho"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
