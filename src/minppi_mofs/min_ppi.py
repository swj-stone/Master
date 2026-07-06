"""
MinPPI estimation — MinSS / MinBz computation (§4.2–4.3).

Provides three estimation strategies for the Min-Marginal Power Index:

1. **Exact enumeration** (n ≤ 20): enumerate all 2^{n-1} coalitions per feature.
2. **Algorithm A — Random MC** (§4.3.1): sample K random coalitions, take min.
   Upward biased (sample min ≥ population min).  Proposition 1 gives the
   concentration bound for harmful detection.
3. **Algorithm B — ShapG-Guided Adversarial Search** (§4.3.2): use the ShapG
   correlation graph to find the coalitions where feature *i* is most likely
   to be harmful, then adversarially search them.

Key theoretical results implemented here:
- **Lemma 1** (Boundedness): MinPPI_i ∈ {-1, 0, +1}
- **Lemma 2** (Coincidence): MinSS_i = MinBz_i
- **Theorem 1** (Three-Way Classification): decisive / harmful / neutral

Usage::

    from minppi_mofs.min_ppi import estimate_minppi
    minppi = estimate_minppi(X_train, X_val, y_train, y_val, alpha=0.95,
                              method="shapg_guided", seed=42)
    # minppi: dict {feature_name: minppi_value}  (values ∈ {-1, 0, +1})
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pandas as pd

from minppi_mofs.voting_game import AccFullCache, make_acc_full_cache


# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------


def estimate_minppi_exact(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Exact MinPPI via full enumeration of 2^{n-1} coalitions per feature.

    Feasible only for n ≤ 20 (max ~1M evaluations for n=20).

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        task: One of ``"binary"``, ``"multiclass"``, ``"regression"``.
        acc_cache: Optional pre-built cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` where values are in {-1, 0, +1}.
    """
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train,
            X_val,
            y_train,
            y_val,
            task=task,
            seed=seed,
        )
    acc_cache.set_threshold(alpha)

    features = list(X_train.columns)
    n = len(features)
    if n > 20:
        raise ValueError(
            f"Exact enumeration requires n ≤ 20, got {n}. "
            f"Use estimate_minppi with method='random_mc' or 'shapg_guided'."
        )

    if verbose:
        print(f"  Exact MinPPI: {n} features, enumerating 2^{n - 1} per feature")

    minppi: dict[str, int] = {}
    all_features = set(features)

    for i, feat in enumerate(features):
        others = list(all_features - {feat})
        best_marg = 1  # init to max possible
        for r in range(len(others) + 1):
            for S_combo in itertools.combinations(others, r):
                S = set(S_combo)
                marg = acc_cache.eval_marginal(S, feat)
                if marg < best_marg:
                    best_marg = marg
                    if best_marg == -1:
                        break
            if best_marg == -1:
                break
        minppi[feat] = best_marg
        if verbose and (i + 1) % 5 == 0:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            n_pos = sum(1 for v in minppi.values() if v == +1)
            print(
                f"  Exact: {i + 1}/{n}  (harmful={n_neg}, decisive={n_pos})",
                flush=True,
            )

    return minppi


# ---------------------------------------------------------------------------
# Algorithm A — Random Monte Carlo (§4.3.1)
# ---------------------------------------------------------------------------


def estimate_minppi_random_mc(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K: int = 200,
    shared_coalitions: bool = False,
    per_coalition: int = 5,
    acc_cache: Optional[AccFullCache] = None,
    model_params: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Estimate MinPPI via random coalition sampling (Algorithm A).

    Two modes:

    **Per-feature mode** (``shared_coalitions=False``, default):
    For each feature *i*, samples *K* random coalitions S ⊆ N \\ {i} and
    computes  min_k [v_alpha(S_k ∪ {i}) - v_alpha(S_k)].

    **Shared-coalition mode** (``shared_coalitions=True``):
    Draws *K* coalitions once.  For each coalition S, up to *per_coalition*
    absent features are evaluated (same strategy as Banzhaf/MinShap).
    Total XGBoost evaluations: O(K · per_coalition) instead of O(p·K).
    Suitable for alpha sweeps and multiclass datasets.

    **Upward bias** (Lemma 3):  sample_min ≥ population_min almost surely.

    **Harmful detection** (Proposition 1): If MinPPI_i = -1 and the harmful
    event has probability p_-^i, then K ≥ log(δ)/log(1 - p_-^i) samples are
    needed to detect it with probability ≥ 1 − δ.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        K: Number of random coalitions (per-feature or shared).
        shared_coalitions: If True, use shared-coalition sampling.
        per_coalition: Max absent features to evaluate per coalition
            (shared-coalition mode only).
        acc_cache: Optional pre-built accuracy cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` (values ∈ {-1, 0, +1}).
    """
    rng = np.random.default_rng(seed)
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train,
            X_val,
            y_train,
            y_val,
            task=task,
            seed=seed,
            model_params=model_params,
        )
    acc_cache.set_threshold(alpha)

    features = list(X_train.columns)
    n = len(features)
    all_features = set(features)
    minppi: dict[str, int] = {}

    if shared_coalitions:
        # ── shared‑coalition mode (Banzhaf-style) ──
        # Draw K coalitions once.  For each S, evaluate v(S) and at most
        # *per_coalition* randomly chosen absent features.  Total evals:
        # K × (1 + per_coalition).
        if verbose:
            print(
                f"  Algorithm A (shared): K={K}, per_cc={per_coalition}, {n} features",
                flush=True,
            )

        minppi = {f: 1 for f in features}
        hits = {f: 0 for f in features}
        n_eval = 0

        for _ in range(K):
            s_size = rng.integers(0, n + 1)
            S_list = rng.choice(features, size=s_size, replace=False).tolist()
            S = set(S_list)
            vS = 1 if acc_cache.is_winning(S) else 0
            n_eval += 1

            absent = [f for f in features if f not in S]
            rng.shuffle(absent)
            for feat in absent[:per_coalition]:
                if minppi[feat] <= -0.99:
                    continue
                vSi = 1 if acc_cache.is_winning(S | {feat}) else 0
                n_eval += 1
                marg = vSi - vS
                if marg < minppi[feat]:
                    minppi[feat] = marg
                hits[feat] += 1

        # Second pass: any feature with < 2 evaluations?  Run a few
        # dedicated coalitions to get a baseline (coupon-collector edge).
        needs_more = [f for f in features if hits[f] < 2]
        for feat in needs_more:
            others = list(all_features - {feat})
            for _ in range(max(0, 5 - hits[feat])):
                s_size = rng.integers(0, len(others) + 1)
                S = set(rng.choice(others, size=s_size, replace=False).tolist())
                marg = acc_cache.eval_marginal(S, feat)
                n_eval += 1
                if marg < minppi[feat]:
                    minppi[feat] = marg
                hits[feat] += 1
                if marg == -1:
                    break

        minppi = {f: minppi[f] if hits[f] > 0 else 0 for f in features}
        if verbose:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            print(
                f"  Algorithm A (shared): {n_eval} evals, harmful={n_neg}/{n}",
                flush=True,
            )
        return minppi

    # ── per-feature mode ──
    if verbose:
        print(f"  Algorithm A (random MC): K={K}, {n} features", flush=True)

    for idx, feat in enumerate(features):
        others = list(all_features - {feat})
        best_marg = 1
        n_others = len(others)

        # Adaptive K: for tiny feature sets, exact enumeration is cheaper
        actual_K = min(K, 2**n_others)

        for _ in range(actual_K):
            s_size = rng.integers(0, n_others + 1)
            S = set(rng.choice(others, size=s_size, replace=False).tolist())
            marg = acc_cache.eval_marginal(S, feat)
            if marg < best_marg:
                best_marg = marg
                if best_marg == -1:
                    break

        minppi[feat] = best_marg
        if verbose and (idx + 1) % max(1, n // 5) == 0:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            print(
                f"  Algorithm A: {idx + 1}/{n}  (harmful found={n_neg})",
                flush=True,
            )

    return minppi


# ---------------------------------------------------------------------------
# Algorithm B — ShapG-Guided Adversarial Search (§4.3.2)
# ---------------------------------------------------------------------------


def estimate_minppi_shapg_guided(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K_adv: int = 30,
    shapg_neighbors: Optional[dict[str, list[str]]] = None,
    shapg_scores: Optional[dict[str, float]] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Estimate MinPPI via ShapG-guided adversarial search (Algorithm B).

    For each feature *i*:

    1. Identifies correlated neighbours C from the ShapG correlation graph.
    2. Constructs a seed coalition S_seed that includes C (the context where
       *i* is most likely to be redundant).
    3. Perturbs S_seed *K_adv* times, evaluating v_alpha margins.
    4. Exits early if a harmful marginal (-1) is found.

    **Sample efficiency**: When harmful events are sparse (small p_-^i),
    Algorithm B targets the vulnerable contexts directly rather than
    sampling uniformly.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        K_adv: Adversarial search budget per feature.
        shapg_neighbors: Pre-computed ShapG neighbor dict
            ``{feature: [neighbor_features, ...]}`` sorted by correlation.
            If None, falls back to random MC (Algorithm A).
        shapg_scores: ShapG importance scores ``{feature: value}``.
            Used to construct the seed coalition. If None, uses uniform.
        acc_cache: Optional pre-built accuracy cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` (values ∈ {-1, 0, +1}).
    """
    rng = np.random.default_rng(seed)
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train,
            X_val,
            y_train,
            y_val,
            task=task,
            seed=seed,
        )
    acc_cache.set_threshold(alpha)

    features = list(X_train.columns)
    n = len(features)
    all_features = set(features)
    minppi: dict[str, int] = {}

    # Build seed coalition: top features by ShapG score (or uniform)
    if shapg_scores is not None:
        top_order = sorted(shapg_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_half = [f for f, _ in top_order[: max(1, n // 2)]]
    else:
        top_half = list(features)

    if verbose:
        n_neighbors = (
            sum(len(v) for v in shapg_neighbors.values()) if shapg_neighbors else 0
        )
        print(
            f"  Algorithm B (ShapG-guided): K_adv={K_adv}, {n} features, "
            f"neighbors={n_neighbors}",
            flush=True,
        )

    for idx, feat in enumerate(features):
        best_marg = 1

        # Build seed context: include correlated neighbors when available
        if shapg_neighbors and feat in shapg_neighbors:
            correlated = shapg_neighbors.get(feat, [])
            # Build S_seed: features correlated with feat + top-scored features
            seed_core = set(correlated[: min(10, len(correlated))])
            # Pad to a reasonable coalition size with top features
            seed = seed_core.copy()
            for f in top_half:
                if len(seed) >= min(20, n // 3):
                    break
                if f not in seed and f != feat:
                    seed.add(f)
        else:
            # Fallback: random baseline coalition biased toward top features
            seed_size = min(20, n // 3)
            # Prefer top-ranked features
            top_pool = top_half[: max(seed_size * 2, len(top_half))]
            seed = set(
                rng.choice(
                    top_pool, size=min(seed_size, len(top_pool)), replace=False
                ).tolist()
            )

        # Adversarial search
        for _ in range(K_adv):
            # Perturb seed: randomly flip features, biased toward keeping
            # correlated neighbors
            S = seed.copy()
            for f in list(S):
                if rng.random() < 0.3:  # remove some features
                    S.discard(f)
            # Add some features from outside
            outside = list(all_features - S - {feat})
            if outside:
                n_add = min(rng.integers(0, min(5, len(outside)) + 1), len(outside))
                S.update(rng.choice(outside, size=n_add, replace=False).tolist())

            marg = acc_cache.eval_marginal(S, feat)
            if marg < best_marg:
                best_marg = marg
                if best_marg == -1:
                    break

        minppi[feat] = best_marg
        if verbose and (idx + 1) % max(1, n // 5) == 0:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            print(
                f"  Algorithm B: {idx + 1}/{n}  (harmful found={n_neg})",
                flush=True,
            )

    return minppi


# ---------------------------------------------------------------------------
# Hybrid dispatcher (§4.3.3)
# ---------------------------------------------------------------------------


def estimate_minppi_hybrid(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K: int = 200,
    K_adv: int = 30,
    shared_coalitions: bool = False,
    shapg_neighbors: Optional[dict[str, list[str]]] = None,
    shapg_scores: Optional[dict[str, float]] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Hybrid MinPPI estimation (§4.3.3).

    Dispatches based on feature count *n*:

    - n ≤ 20: exact enumeration
    - 20 < n ≤ 75: Algorithm B (ShapG-guided), fallback to Algorithm A
    - n > 75: Algorithm A + Algorithm B, take the min of both estimates

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        K: Random MC budget (Algorithm A).
        K_adv: Adversarial search budget (Algorithm B).
        shared_coalitions: Forwarded to Algorithm A.
        shapg_neighbors: ShapG neighbor dict for Algorithm B.
        shapg_scores: ShapG scores for seed construction.
        acc_cache: Optional pre-built accuracy cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` (values ∈ {-1, 0, +1}).
    """
    if acc_cache is None:
        acc_cache = make_acc_full_cache(X_train, X_val, y_train, y_val, seed=seed)

    n = len(X_train.columns)

    if n <= 20:
        if verbose:
            print(f"  Hybrid: n={n} ≤ 20 → exact enumeration", flush=True)
        return estimate_minppi_exact(
            X_train,
            X_val,
            y_train,
            y_val,
            alpha=alpha,
            acc_cache=acc_cache,
            seed=seed,
            verbose=verbose,
        )

    if n <= 75:
        if verbose:
            print(f"  Hybrid: 20 < n={n} ≤ 75 → Algorithm B", flush=True)
        if shapg_neighbors is None:
            if verbose:
                print(
                    "  No ShapG neighbors available → fallback to Algorithm A",
                    flush=True,
                )
            return estimate_minppi_random_mc(
                X_train,
                X_val,
                y_train,
                y_val,
                alpha=alpha,
                K=K,
                shared_coalitions=shared_coalitions,
                acc_cache=acc_cache,
                seed=seed,
                verbose=verbose,
            )
        return estimate_minppi_shapg_guided(
            X_train,
            X_val,
            y_train,
            y_val,
            alpha=alpha,
            K_adv=K_adv,
            shapg_neighbors=shapg_neighbors,
            shapg_scores=shapg_scores,
            acc_cache=acc_cache,
            seed=seed,
            verbose=verbose,
        )

    # n > 75: run both and take element-wise min
    if verbose:
        print(f"  Hybrid: n={n} > 75 → Algorithm A + B combined", flush=True)

    minppi_a = estimate_minppi_random_mc(
        X_train,
        X_val,
        y_train,
        y_val,
        alpha=alpha,
        K=K,
        shared_coalitions=shared_coalitions,
        acc_cache=acc_cache,
        seed=seed,
        verbose=False,
    )

    minppi_b: dict[str, int] = {}
    if shapg_neighbors is not None:
        minppi_b = estimate_minppi_shapg_guided(
            X_train,
            X_val,
            y_train,
            y_val,
            alpha=alpha,
            K_adv=K_adv,
            shapg_neighbors=shapg_neighbors,
            shapg_scores=shapg_scores,
            acc_cache=acc_cache,
            seed=seed,
            verbose=False,
        )

    combined: dict[str, int] = {}
    for feat in minppi_a:
        val_a = minppi_a[feat]
        val_b = minppi_b.get(feat, val_a)
        combined[feat] = min(val_a, val_b)

    if verbose:
        n_neg = sum(1 for v in combined.values() if v == -1)
        print(f"  Combined: {n_neg} harmful features found", flush=True)

    return combined


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def estimate_minppi(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    method: str = "hybrid",
    K: int = 200,
    K_adv: int = 30,
    shared_coalitions: bool = False,
    shapg_neighbors: Optional[dict[str, list[str]]] = None,
    shapg_scores: Optional[dict[str, float]] = None,
    model_params: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Estimate MinPPI for all features.

    This is the main entry point for MinPPI computation. Dispatches to one
    of four strategies based on *method*.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter α ∈ (0, 1].
        method: Estimation strategy:

            - ``"exact"`` — enumerate all coalitions (n ≤ 20 only)
            - ``"random_mc"`` — Algorithm A, random MC sampling
            - ``"shapg_guided"`` — Algorithm B, ShapG-guided adversarial search
            - ``"hybrid"`` — auto-select based on *n* (§4.3.3, default)
        K: Number of random samples per feature (Algorithm A).
        K_adv: Adversarial search budget per feature (Algorithm B).
        shared_coalitions: Forwarded to Algorithm A (shared-coalition mode).
        shapg_neighbors: ShapG neighbour dict for Algorithm B.
        shapg_scores: ShapG scores for seed coalition construction.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` where each value ∈ {-1, 0, +1},
        corresponding to the three-way classification of Theorem 1:
        ``+1`` = decisive, ``0`` = neutral, ``-1`` = harmful.

    Examples:
        >>> minppi = estimate_minppi(X_tr, X_val, y_tr, y_val, alpha=0.95)
        >>> harmful = {f for f, v in minppi.items() if v == -1}
        >>> decisive = {f for f, v in minppi.items() if v == +1}
    """
    methods = {
        "exact": estimate_minppi_exact,
        "random_mc": estimate_minppi_random_mc,
        "shapg_guided": estimate_minppi_shapg_guided,
        "hybrid": estimate_minppi_hybrid,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {sorted(methods.keys())}."
        )

    # Build acc_cache once and reuse
    acc_cache = make_acc_full_cache(
        X_train,
        X_val,
        y_train,
        y_val,
        task=task,
        seed=seed,
        model_params=model_params,
    )

    fn = methods[method]
    common: dict = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "alpha": alpha,
        "task": task,
        "acc_cache": acc_cache,
        "seed": seed,
        "verbose": verbose,
    }

    if method == "exact":
        return fn(**common)  # type: ignore[call-arg]
    elif method == "random_mc":
        return fn(K=K, shared_coalitions=shared_coalitions, **common)  # type: ignore[call-arg]
    elif method == "shapg_guided":
        return fn(
            K_adv=K_adv,
            shapg_neighbors=shapg_neighbors,
            shapg_scores=shapg_scores,
            **common,
        )  # type: ignore[call-arg]
    else:  # hybrid
        return fn(
            K=K,
            K_adv=K_adv,
            shared_coalitions=shared_coalitions,
            shapg_neighbors=shapg_neighbors,
            shapg_scores=shapg_scores,
            **common,
        )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Utility: MinPPI statistics
# ---------------------------------------------------------------------------


def minppi_stats(minppi: dict[str, int]) -> dict:
    """Compute summary statistics of a MinPPI result.

    Args:
        minppi: Dict ``{feature: value}`` with values in {-1, 0, +1}.

    Returns:
        Dict with ``n_total``, ``n_decisive``, ``n_neutral``, ``n_harmful``,
        ``pct_decisive``, ``pct_neutral``, ``pct_harmful``.
    """
    vals = list(minppi.values())
    n = len(vals)
    n_pos = sum(1 for v in vals if v == +1)
    n_zero = sum(1 for v in vals if v == 0)
    n_neg = sum(1 for v in vals if v == -1)
    return {
        "n_total": n,
        "n_decisive": n_pos,
        "n_neutral": n_zero,
        "n_harmful": n_neg,
        "pct_decisive": n_pos / n * 100 if n else 0,
        "pct_neutral": n_zero / n * 100 if n else 0,
        "pct_harmful": n_neg / n * 100 if n else 0,
    }


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("MinPPI estimation — synthetic demo")

    rng = np.random.default_rng(42)
    n_samp = 300
    X_demo = pd.DataFrame(
        {
            "f_signal": rng.normal(0, 1, n_samp),
            "f_harmful": rng.normal(0, 1, n_samp),
            "f_noise": rng.normal(0, 1, n_samp),
        }
    )
    # f_signal is predictive; f_harmful adds pure noise; f_noise is neutral
    y_demo = np.where(X_demo["f_signal"] > 0, 1, 0).astype(int)

    n_tr = int(0.7 * n_samp)
    X_tr, X_val = X_demo.iloc[:n_tr], X_demo.iloc[n_tr:]
    y_tr, y_val = y_demo[:n_tr], y_demo[n_tr:]

    # Exact enumeration (n=3 → only 2^{2}=4 coalitions per feature)
    minppi = estimate_minppi(
        X_tr, X_val, y_tr, y_val, alpha=0.90, method="exact", verbose=True
    )
    print("\nMinPPI results:")
    for feat, val in sorted(minppi.items(), key=lambda kv: kv[1]):
        label = {+1: "DECISIVE", 0: "neutral", -1: "HARMFUL"}[val]
        print(f"  {feat:20s} → {val:+d}  ({label})")

    stats = minppi_stats(minppi)
    print(
        f"\nSummary: {stats['n_total']} features — "
        f"{stats['n_decisive']} decisive, {stats['n_harmful']} harmful, "
        f"{stats['n_neutral']} neutral"
    )
    print(
        "\nObservation: Lemma 1 (boundedness) holds — all values ∈ {-1, 0, +1}. "
        "The harmful feature (pure noise added) should show MinPPI=-1 when "
        "α is high enough to detect the performance degradation."
    )

    # Compare with Algorithm A on the same data
    _demo_banner("Algorithm A comparison")
    minppi_mc = estimate_minppi(
        X_tr,
        X_val,
        y_tr,
        y_val,
        alpha=0.90,
        method="random_mc",
        K=50,
        verbose=True,
    )
    for feat in sorted(minppi_mc):
        ex = minppi[feat]
        mc = minppi_mc[feat]
        agree = "✓" if ex == mc else "✗ (upward bias)"
        print(f"  {feat:20s}  exact={ex:+d}  MC={mc:+d}  {agree}")
    print("\nObservation: Algorithm A shows upward bias (sample min >= true min).")
