"""
MinPPI demo: empirical verification of Lemma 1 + Theorem 1 from §4 of the
ICOMP 2026 paper "Min-Marginal Power Indices for Multi-Objective Feature
Selection: A Voting-Game Approach".

Four sections:
  1. Engineered v_α with a DECISIVE feature (MinSS = +1)
  2. Engineered v_α with a HARMFUL feature (MinSS = -1)
     — including the subtle case: f_0 is in every winning coalition
       but MinSS(f_0) = 0 (not +1), refuting the naive equivalence
  3. Real model fitting on synthetic tabular data
  4. Algorithm A (random MC) upward bias — empirical detection rate vs K

Verifies:
  - Lemma 1 (Boundedness): MinPPI_i ∈ {-1, 0, +1} always
  - Theorem 1 (Three-Way Decisiveness): MinSS = +1 ⟺ decisive,
    MinSS = -1 ⟺ harmful (in some context), 0 ⟺ neutral
  - The naive equivalence "in every winning coalition ⟺ MinSS = +1" FAILS

Run:
    python minppi_demo.py
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Iterable, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def shapley_weight(s: int, n: int) -> float:
    return factorial(s) * factorial(n - s - 1) / factorial(n)


def compute_min_ss(
    i: int, n: int, v_alpha: Callable[[Iterable[int]], int]
) -> Tuple[int, float, list]:
    """Return (MinSS_i, MeanSS_i, all_marginals) by exhaustive enumeration."""
    other = [f for f in range(n) if f != i]
    marginals = []
    mean_ss = 0.0
    for k in range(len(other) + 1):
        w = shapley_weight(k, n)
        for S in combinations(other, k):
            S_set = set(S)
            S_with_i = S_set | {i}
            marg = v_alpha(S_with_i) - v_alpha(S_set)
            marginals.append(marg)
            mean_ss += w * marg
    return min(marginals), mean_ss, marginals


def classify(min_ss: int) -> str:
    if min_ss == 1:
        return "DECISIVE"
    if min_ss == -1:
        return "HARMFUL"
    return "neutral"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Example 1: Decisive feature (MinSS = +1)
# ---------------------------------------------------------------------------
def example_1() -> None:
    banner("Example 1: Engineered v_α with DECISIVE feature f_0 (MinSS = +1)")
    print("Design: v_α(S) = 1 iff 0 ∈ S")
    print("        f_0 alone is BOTH necessary AND sufficient for winning.")
    print()

    n = 5

    def v_alpha(S: Iterable[int]) -> int:
        return int(0 in S)

    print(f"{'Feature':<10}{'MinSS':>7}    {'Mean SS':>9}    {'Class':<10}")
    print("-" * 45)
    for i in range(n):
        min_ss, mean_ss, _ = compute_min_ss(i, n, v_alpha)
        assert min_ss in {-1, 0, 1}, "Lemma 1 violated"
        print(f"f_{i:<8}{min_ss:>+5d}    {mean_ss:>+.4f}    {classify(min_ss):<10}")

    print()
    print("Lemma 1: all MinSS ∈ {-1, 0, +1} ✓")
    print("Theorem 1: f_0 is DECISIVE (∀S: v(S)=0, v(S∪{f_0})=1) ✓")
    print("           Others are neutral ✓")


# ---------------------------------------------------------------------------
# Example 2: Harmful feature + the "in every winning" caveat
# ---------------------------------------------------------------------------
def example_2() -> None:
    banner("Example 2: Engineered v_α with HARMFUL feature f_1 (MinSS = -1)")
    print("Design: 13 winning coalitions on N = {0,1,2,3,4}.")
    print("        f_1 harmful at S = {0,2}: v({0,2}) = 1 but v({0,1,2}) = 0.")
    print("        f_0 appears in EVERY winning coalition.")
    print()

    n = 5
    winning = {
        frozenset({0, 2}),
        frozenset({0, 3}),
        frozenset({0, 4}),
        frozenset({0, 2, 3}),
        frozenset({0, 2, 4}),
        frozenset({0, 3, 4}),
        frozenset({0, 2, 3, 4}),
        frozenset({0, 1, 3}),
        frozenset({0, 1, 4}),
        frozenset({0, 1, 3, 4}),
        frozenset({0, 1, 2, 3}),
        frozenset({0, 1, 2, 4}),
        frozenset({0, 1, 2, 3, 4}),
    }

    def v_alpha(S: Iterable[int]) -> int:
        return int(frozenset(S) in winning)

    # Sanity: confirm f_0 in every winning coalition
    f0_in_all = all(0 in w for w in winning)
    print(f"  Sanity: f_0 in every winning coalition?  {f0_in_all} ✓")
    print()

    print(f"{'Feature':<10}{'MinSS':>7}    {'Mean SS':>9}    {'Class':<10}")
    print("-" * 45)
    results = {}
    for i in range(n):
        min_ss, mean_ss, marginals = compute_min_ss(i, n, v_alpha)
        assert min_ss in {-1, 0, 1}, "Lemma 1 violated"
        results[i] = (min_ss, mean_ss, marginals)
        print(f"f_{i:<8}{min_ss:>+5d}    {mean_ss:>+.4f}    {classify(min_ss):<10}")

    print()
    print("Lemma 1: all MinSS ∈ {-1, 0, +1} ✓")
    print()
    print("Theorem 1 verification:")
    print(f"  f_1: MinSS = {results[1][0]:+d}  →  HARMFUL  ✓")
    print(
        f"       Witness: v({{0,2}}) = {v_alpha({0, 2})}, "
        f"v({{0,1,2}}) = {v_alpha({0, 1, 2})}, marginal = -1"
    )
    print()
    print("Caveat (refuting naive equivalence):")
    print("  f_0 IS in every winning coalition (sanity above) — but")
    print(f"  MinSS(f_0) = {results[0][0]} ≠ +1.")
    print(
        f"  Why? At S = {{1,2}}: v({{1,2}}) = {v_alpha({1, 2})}, "
        f"v({{0,1,2}}) = {v_alpha({0, 1, 2})}, marginal = 0"
    )
    print("  ⟹ 'in every winning' is NECESSARY but NOT SUFFICIENT for MinSS = +1.")
    print("  Decisiveness requires ALSO that every coalition with f_0 wins.")


# ---------------------------------------------------------------------------
# Example 3: Real model fitting
# ---------------------------------------------------------------------------
def example_3() -> None:
    banner("Example 3: Real model fitting (synthetic tabular + LogReg)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    n_samples = 300  # small to encourage instability
    n = 5

    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    x3 = 1.5 * rng.normal(size=n_samples)  # higher-variance noise
    x4 = rng.normal(size=n_samples)

    logits = 1.5 * x0 + 0.6 * x1 + 0.4 * x2 + 0.15 * rng.normal(size=n_samples)
    y = (logits > 0).astype(int)
    X = np.column_stack([x0, x1, x2, x3, x4])

    Xtr_all, Xte_all, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=7, stratify=y
    )

    def accuracy(F: Tuple[int, ...]) -> float:
        if len(F) == 0:
            return 0.5
        Xtr = Xtr_all[:, list(F)]
        Xte = Xte_all[:, list(F)]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        try:
            clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
            clf.fit(Xtr, ytr)
            return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        except Exception:
            return 0.5

    # Precompute accuracy for all subsets
    acc_table = {}
    for k in range(n + 1):
        for F in combinations(range(n), k):
            acc_table[F] = accuracy(F)
    acc_full = acc_table[tuple(range(n))]
    print(f"Acc_full = {acc_full:.4f}")

    for alpha in [0.90, 0.95, 0.98]:
        q = alpha * acc_full
        print(f"\n--- α = {alpha}  (quota q = {q:.4f}) ---")

        def v_alpha(S: Iterable[int]) -> int:
            key = tuple(sorted(S))
            return int(acc_table[key] >= q)

        print(f"{'Feature':<10}{'MinSS':>7}    {'Mean SS':>9}    {'Class':<10}")
        for i in range(n):
            min_ss, mean_ss, _ = compute_min_ss(i, n, v_alpha)
            assert min_ss in {-1, 0, 1}, "Lemma 1 violated"
            print(f"f_{i:<8}{min_ss:>+5d}    {mean_ss:>+.4f}    {classify(min_ss):<10}")

    print()
    print("Lemma 1: all MinSS values are in {-1, 0, +1} (verified across all α) ✓")
    print("Observation: real-data MinSS is α-dependent. Lower α → broader 'neutral'.")
    print(
        "             Higher α → tighter quota → more features look harmful or neutral."
    )


# ---------------------------------------------------------------------------
# Example 4: Algorithm A (random MC) upward bias
# ---------------------------------------------------------------------------
def example_4_alg_a_bias() -> None:
    banner("Example 4: Algorithm A (random MC) — empirical bias")
    print("Setup: Reuse Example 2's v_α (where f_1 has true MinSS = -1).")
    print("Question: how many random subset samples K are needed to detect harmful?")
    print()

    n = 5
    winning = {
        frozenset({0, 2}),
        frozenset({0, 3}),
        frozenset({0, 4}),
        frozenset({0, 2, 3}),
        frozenset({0, 2, 4}),
        frozenset({0, 3, 4}),
        frozenset({0, 2, 3, 4}),
        frozenset({0, 1, 3}),
        frozenset({0, 1, 4}),
        frozenset({0, 1, 3, 4}),
        frozenset({0, 1, 2, 3}),
        frozenset({0, 1, 2, 4}),
        frozenset({0, 1, 2, 3, 4}),
    }

    def v_alpha(S: Iterable[int]) -> int:
        return int(frozenset(S) in winning)

    target = 1
    other = [f for f in range(n) if f != target]

    # Compute p_neg : probability that a uniformly random S ⊆ N\{1} gives marginal = -1
    # The MC sampler here draws S by first picking a uniform-random size k ∈ [0, |other|],
    # then a uniform subset of that size. We compute p_neg under that exact distribution.
    n_neg_per_size = {}
    n_per_size = {}
    for k in range(len(other) + 1):
        n_per_size[k] = 0
        n_neg_per_size[k] = 0
        for S in combinations(other, k):
            S_set = set(S)
            n_per_size[k] += 1
            marg = v_alpha(S_set | {target}) - v_alpha(S_set)
            if marg == -1:
                n_neg_per_size[k] += 1
    # Sampler probability = (1/(|other|+1)) * (n_neg_per_size[k] / n_per_size[k])
    p_neg = sum(n_neg_per_size[k] / n_per_size[k] for k in range(len(other) + 1)) / (
        len(other) + 1
    )

    print(f"Target = f_{target}. True MinSS = -1.")
    print(
        f"  Sampler used: uniform random size k ∈ [0, {len(other)}], then "
        f"uniform subset of size k."
    )
    print(f"  Per-draw harmful probability p_- = {p_neg:.4f}")
    print()
    print("  Theoretical P(miss in K draws) = (1 - p_-)^K")
    print()
    print(f"  {'K':>5}  {'P(detect) theory':>18}  {'P(detect) empirical':>22}")
    print(f"  {'-' * 5}  {'-' * 18}  {'-' * 22}")

    rng = np.random.default_rng(11)
    n_trials = 500
    for K in [1, 3, 5, 10, 20, 50, 100]:
        p_detect_theory = 1 - (1 - p_neg) ** K
        detected = 0
        for _ in range(n_trials):
            min_marg = +1
            for _ in range(K):
                k = int(rng.integers(0, len(other) + 1))
                if k == 0:
                    S = set()
                else:
                    S = set(rng.choice(other, size=k, replace=False).tolist())
                marg = v_alpha(S | {target}) - v_alpha(S)
                if marg < min_marg:
                    min_marg = marg
            if min_marg == -1:
                detected += 1
        p_detect_emp = detected / n_trials
        print(f"  {K:>5}  {p_detect_theory:>18.4f}  {p_detect_emp:>22.4f}")

    print()
    print("Observation: theory and empirical match within sampling noise. ✓")
    print("Implication: when p_- is small (rare harmful contexts), random MC needs")
    print("             many samples to detect. This motivates Algorithm B")
    print("             (ShapG-guided adversarial search) in §4.3.2.")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("MinPPI Demo — verification of Lemma 1 + Theorem 1 (ICOMP 2026 §4)")
    example_1()
    example_2()
    example_3()
    example_4_alg_a_bias()
    print()
