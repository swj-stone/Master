"""
Cross-Shapley demo: bipartite Shapley interaction value φ_{i,o}^{cross}
for a 5-feature × 3-architecture-operation toy problem.

This is the minimum verification of the bipartite (feature x architecture)
factored game framework that underpins the master thesis. The single-layer
specialization corresponds to the ICOMP 2026 conference paper (MinPPI-MOFS).

Setup:
    Layer 1 (Features): 5 features f_0..f_4, with designed roles:
        - f_0, f_1 : useful as-is (linear-friendly)
        - f_2, f_3 : useful only as interaction (cross-feature friendly)
        - f_4      : pure noise
    Layer 2 (Architecture ops):
        - O_identity : keep features as-is (linear baseline)
        - O_poly     : add 2nd-order interaction features
        - O_log      : log1p(|x|)  (helps skewed features)
    v(F, A) = test AUC of LogisticRegression on features F, after applying ops A.

What the cross-Shapley matrix should reveal:
    - f_2, f_3 × O_poly  should have HIGH positive cross-Shapley
      (these features need the polynomial op to be useful)
    - f_4 × any op       should have ≈ 0 cross-Shapley  (noise feature)
    - f_0, f_1 × O_identity should be close to 0
      (linear features don't gain from added ops)

Run:
    python cross_shapley_demo.py
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ---------------------------------------------------------------------------
# 1. Synthetic data with designed structure
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

N_SAMPLES = 2000
N_FEATURES = 5

# Build features with controlled roles.
# f_0, f_1: linear-informative
# f_2, f_3: useful only through their interaction
# f_4: pure noise
X = RNG.normal(size=(N_SAMPLES, N_FEATURES))
linear_signal = 1.2 * X[:, 0] - 0.8 * X[:, 1]
interaction_signal = 1.5 * X[:, 2] * X[:, 3]
logits = linear_signal + interaction_signal + 0.3 * RNG.normal(size=N_SAMPLES)
y = (logits > np.median(logits)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# ---------------------------------------------------------------------------
# 2. Three "architecture operations"
# ---------------------------------------------------------------------------
def op_identity(Xtr: np.ndarray, Xte: np.ndarray):
    """O_0: keep features as-is."""
    return Xtr.copy(), Xte.copy()


def op_poly(Xtr: np.ndarray, Xte: np.ndarray):
    """O_1: add 2nd-order interaction features."""
    if Xtr.shape[1] < 2:
        return Xtr.copy(), Xte.copy()
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    return poly.fit_transform(Xtr), poly.transform(Xte)


def op_log(Xtr: np.ndarray, Xte: np.ndarray):
    """O_2: log1p(|x|), helpful for skewed features."""
    return np.log1p(np.abs(Xtr)), np.log1p(np.abs(Xte))


OPS = [op_identity, op_poly, op_log]
OP_NAMES = ["identity", "poly_x2", "log1p_abs"]
N_OPS = len(OPS)


# ---------------------------------------------------------------------------
# 3. Value function v(F, A) = test AUC
# ---------------------------------------------------------------------------
def v(F: Tuple[int, ...], A: Tuple[int, ...]) -> float:
    """Empty F → 0.5 (random); train LogReg and report test AUC."""
    if len(F) == 0:
        return 0.5

    Xtr_sub = X_train[:, list(F)]
    Xte_sub = X_test[:, list(F)]

    if len(A) == 0:
        Xtr_final, Xte_final = Xtr_sub, Xte_sub
    else:
        parts_tr = [Xtr_sub]
        parts_te = [Xte_sub]
        for op_idx in A:
            tr_op, te_op = OPS[op_idx](Xtr_sub, Xte_sub)
            # If op produces same shape (identity), don't double-count
            if op_idx == 0 and len(A) > 1:
                # identity adds no new features when combined with other ops
                continue
            parts_tr.append(tr_op)
            parts_te.append(te_op)
        Xtr_final = np.concatenate(parts_tr, axis=1)
        Xte_final = np.concatenate(parts_te, axis=1)

    # Standardize for numerical stability, then train LR.
    scaler = StandardScaler()
    Xtr_final = scaler.fit_transform(Xtr_final)
    Xte_final = scaler.transform(Xte_final)

    try:
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        clf.fit(Xtr_final, y_train)
        proba = clf.predict_proba(Xte_final)[:, 1]
        return float(roc_auc_score(y_test, proba))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# 4. Precompute v(F, A) for all subsets
# ---------------------------------------------------------------------------
def all_subsets(n: int):
    items = list(range(n))
    for k in range(n + 1):
        for combo in combinations(items, k):
            yield combo


print(
    f"Precomputing v(F, A) for all 2^{N_FEATURES} × 2^{N_OPS} = "
    f"{2**N_FEATURES * 2**N_OPS} subsets ..."
)
v_table: dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
for F in all_subsets(N_FEATURES):
    for A in all_subsets(N_OPS):
        v_table[(F, A)] = v(F, A)
print(
    f"Done. v(∅,∅) = {v_table[((), ())]:.4f}, "
    f"v(F=all, A=all) = {v_table[(tuple(range(N_FEATURES)), tuple(range(N_OPS)))]:.4f}\n"
)


# ---------------------------------------------------------------------------
# 5. Shapley computations
# ---------------------------------------------------------------------------
def shapley_weight(s: int, n: int) -> float:
    """w(s, n) = s! (n-s-1)! / n!  (prob that a random ordering puts exactly s items before i)"""
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def feature_shapley_given_arch(i: int, A_fixed: Tuple[int, ...]) -> float:
    """Standard Shapley φ_i^feature with architecture A_fixed."""
    n = N_FEATURES
    other = [f for f in range(n) if f != i]
    total = 0.0
    for s in range(n):
        w = shapley_weight(s, n)
        for S in combinations(other, s):
            S_with_i = tuple(sorted(S + (i,)))
            total += w * (v_table[(S_with_i, A_fixed)] - v_table[(S, A_fixed)])
    return total


def arch_shapley_given_features(o: int, F_fixed: Tuple[int, ...]) -> float:
    """Standard Shapley φ_o^arch with features F_fixed."""
    m = N_OPS
    other = [oo for oo in range(m) if oo != o]
    total = 0.0
    for t in range(m):
        w = shapley_weight(t, m)
        for T in combinations(other, t):
            T_with_o = tuple(sorted(T + (o,)))
            total += w * (v_table[(F_fixed, T_with_o)] - v_table[(F_fixed, T)])
    return total


def cross_shapley(i: int, o: int) -> float:
    """
    Bipartite Shapley interaction value:

        φ_{i,o} = Σ_{S ⊆ N₁\{i}} Σ_{T ⊆ N₂\{o}}
                    w₁(|S|, n₁) · w₂(|T|, n₂) · δ_{i,o}(S, T)

    where δ_{i,o}(S, T) = v(S∪{i}, T∪{o}) - v(S∪{i}, T) - v(S, T∪{o}) + v(S, T)

    Connects to Zhao & Parilina (2024) two-layer centrality:
    i ∈ N₁ is a Layer-1 player; o ∈ N₂ is a Layer-2 player; φ_{i,o} measures
    their inter-layer coupling strength.
    """
    n, m = N_FEATURES, N_OPS
    other_F = [f for f in range(n) if f != i]
    other_A = [oo for oo in range(m) if oo != o]
    total = 0.0
    for s in range(n):
        w_S = shapley_weight(s, n)
        for S in combinations(other_F, s):
            S_with_i = tuple(sorted(S + (i,)))
            for t in range(m):
                w_T = shapley_weight(t, m)
                for T in combinations(other_A, t):
                    T_with_o = tuple(sorted(T + (o,)))
                    delta = (
                        v_table[(S_with_i, T_with_o)]
                        - v_table[(S_with_i, T)]
                        - v_table[(S, T_with_o)]
                        + v_table[(S, T)]
                    )
                    total += w_S * w_T * delta
    return total


# Marginal single-layer Shapleys (averaged over the other layer's coalition)
def feature_shapley_marginalized(i: int) -> float:
    """φ_i^feat = E_T [φ_i^feat | T] where T is a random subset of ops."""
    n, m = N_FEATURES, N_OPS
    other_F = [f for f in range(n) if f != i]
    total = 0.0
    for t in range(m + 1):
        for T in combinations(range(m), t):
            # Probability that a uniformly random ordering of ops produces T as
            # the prefix... we use uniform over the entire size distribution
            # for simplicity (mean over all 2^m subsets weighted equally).
            w_T = 1.0 / (2**m)
            for s in range(n):
                w_S = shapley_weight(s, n)
                for S in combinations(other_F, s):
                    S_with_i = tuple(sorted(S + (i,)))
                    total += w_T * w_S * (v_table[(S_with_i, T)] - v_table[(S, T)])
    return total


def arch_shapley_marginalized(o: int) -> float:
    n, m = N_FEATURES, N_OPS
    other_A = [oo for oo in range(m) if oo != o]
    total = 0.0
    for s in range(n + 1):
        for S in combinations(range(n), s):
            w_S = 1.0 / (2**n)
            for t in range(m):
                w_T = shapley_weight(t, m)
                for T in combinations(other_A, t):
                    T_with_o = tuple(sorted(T + (o,)))
                    total += w_S * w_T * (v_table[(S, T_with_o)] - v_table[(S, T)])
    return total


# ---------------------------------------------------------------------------
# 6. Compute everything
# ---------------------------------------------------------------------------
print("Computing single-layer Shapleys (marginalized over the other layer)...")
phi_feat = np.array([feature_shapley_marginalized(i) for i in range(N_FEATURES)])
phi_arch = np.array([arch_shapley_marginalized(o) for o in range(N_OPS)])

print("Computing cross-Shapley matrix Φ_cross (5 × 3)...")
Phi_cross = np.zeros((N_FEATURES, N_OPS))
for i in range(N_FEATURES):
    for o in range(N_OPS):
        Phi_cross[i, o] = cross_shapley(i, o)


# ---------------------------------------------------------------------------
# 7. Display
# ---------------------------------------------------------------------------
def banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


banner("Single-layer Shapley (marginalized over the other layer)")
print("\nFeature Shapley φ_i^feat:")
feat_roles = [
    "linear-friendly",
    "linear-friendly",
    "needs interaction",
    "needs interaction",
    "pure noise",
]
for i, val in enumerate(phi_feat):
    print(f"  f_{i}  ({feat_roles[i]:<20s}) : {val:+.4f}")

print("\nArchitecture Shapley φ_o^arch:")
for o, val in enumerate(phi_arch):
    print(f"  O_{OP_NAMES[o]:<10s} : {val:+.4f}")

banner("Cross-Shapley matrix Φ_{i,o}^cross (the key two-layer object)")
header = "          " + "   ".join([f"{name:>10s}" for name in OP_NAMES])
print(header)
print("          " + "   ".join(["-" * 10] * N_OPS))
for i in range(N_FEATURES):
    row = f"f_{i} ({feat_roles[i][:13]:<13s})" + "   ".join(
        [f"{Phi_cross[i, o]:>+10.4f}" for o in range(N_OPS)]
    )
    print(row)

banner("Interpretation checks (the demo's main claim)")
poly_idx = OP_NAMES.index("poly_x2")
identity_idx = OP_NAMES.index("identity")

# Check 1: f_2, f_3 × poly should dominate
poly_synergy_23 = Phi_cross[2, poly_idx] + Phi_cross[3, poly_idx]
poly_synergy_01 = Phi_cross[0, poly_idx] + Phi_cross[1, poly_idx]
print(f"\n[1] (f_2, f_3) × poly synergy = {poly_synergy_23:+.4f}")
print(f"    (f_0, f_1) × poly synergy = {poly_synergy_01:+.4f}")
if poly_synergy_23 > poly_synergy_01:
    print("    ✓ As designed: interaction features benefit MORE from poly op")
else:
    print("    ✗ Unexpected: check the random seed / dataset")

# Check 2: f_4 (noise) cross-Shapleys should be near zero
noise_max = np.abs(Phi_cross[4, :]).max()
signal_max = np.abs(Phi_cross[:4, :]).max()
print(f"\n[2] |f_4 cross-Shapley|_max = {noise_max:.4f}")
print(f"    |signal cross-Shapley|_max = {signal_max:.4f}")
if noise_max < 0.5 * signal_max:
    print("    ✓ As designed: noise feature has weaker cross-couplings")
else:
    print("    ⚠ Noise feature has comparable cross-couplings — check setup")

# Check 3: Total decomposition
banner("Total decomposition (sanity check)")
v_empty = v_table[((), ())]
v_full = v_table[(tuple(range(N_FEATURES)), tuple(range(N_OPS)))]
total_gain = v_full - v_empty
sum_first_order = phi_feat.sum() + phi_arch.sum()
sum_cross = Phi_cross.sum()
residual = total_gain - sum_first_order - sum_cross
print(f"  v(F=all, A=all) - v(∅, ∅) = {total_gain:+.4f}")
print(f"  Σ φ_i^feat + Σ φ_o^arch    = {sum_first_order:+.4f}")
print(f"  Σ Σ φ_{{i,o}}^cross          = {sum_cross:+.4f}")
print(f"  Residual (higher-order)    = {residual:+.4f}")
print()
print("  Note: residual = 3-way and higher Shapley interactions.")
print("  For Topic 4 thesis, we'll show |residual| << |cross-Shapley| under")
print("  mild conditions — justifying the 2-layer truncation.")
