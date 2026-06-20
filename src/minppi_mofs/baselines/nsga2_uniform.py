"""
NSGA-II-uniform baseline for feature selection.

NSGA-II-uniform is the **no-guidance** ablation (§4.4): the same
multi-objective evolutionary search as MOFS-MinPPI — minimizing
(|F|, 1-AUC, training-time) — but with **uniform** mutation and
crossover operators that carry no importance signal.

It serves as the lower-bound control:  any method whose selected
subsets outperform NSGA-II-uniform's Pareto front is benefiting
from its guidance mechanism (MinPPI, SHAP, or otherwise).

Pipeline
--------
1. Initialise a population of random binary masks.
2. For *G* generations:
   a. Binary tournament select parents from the ranked + crowded pool.
   b. Uniform crossover → two children.
   c. Uniform bit-flip mutation (p = 0.1, no bias).
   d. Evaluate children on the three objectives via actual XGBoost
      training.
   e. Merge parents + children, keep the best *N_pop* via
      non-dominated sorting + crowding distance.
3. From the final Pareto front, compute per-feature frequency
   as the importance score.

Usage::

    from minppi_mofs.baselines.nsga2_uniform import compute_nsga2_uniform_importance
    scores = compute_nsga2_uniform_importance(X_train, y_train)

Run (stand-alone)::

    python src/minppi_mofs/baselines/nsga2_uniform.py
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# NSGA‑II core (shared across multi-objective methods)
# ---------------------------------------------------------------------------


def _non_dominated_sort(fitness: np.ndarray) -> list[list[int]]:
    """Non-dominated sorting (Deb et al. 2002).

    *fitness* has shape (N, M).  Lower is better for all objectives.

    Returns a list of fronts (each front is a list of indices).
    """
    n = fitness.shape[0]
    dominated_count = np.zeros(n, dtype=int)
    dominates: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominates[i].append(j)
            elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominated_count[i] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if dominated_count[i] == 0]
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for i in current:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


def _crowding_distance(fitness: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for a single front."""
    dist = np.zeros(len(front), dtype=float)
    n_obj = fitness.shape[1]
    for obj in range(n_obj):
        order = sorted(front, key=lambda i: fitness[i, obj])
        f_min = fitness[order[0], obj]
        f_max = fitness[order[-1], obj]
        if f_max == f_min:
            continue
        dist[0] = dist[-1] = np.inf
        for k in range(1, len(front) - 1):
            dist[k] += (fitness[order[k + 1], obj] - fitness[order[k - 1], obj]) / (
                f_max - f_min
            )
    return dist


def _tournament_select(
    pop_size: int,
    fronts: list[list[int]],
    crowding: list[np.ndarray],
    rank: dict[int, int],
    pos: dict[int, int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Binary tournament: select two parents.

    *rank* maps individual index → front number (precomputed).  *pos* maps
    individual index → position within its front (precomputed).
    """
    cand = rng.choice(pop_size, size=4, replace=False)

    def _better(a: int, b: int) -> int:
        r_a, r_b = rank[a], rank[b]
        if r_a < r_b:
            return a
        if r_b < r_a:
            return b
        return a if crowding[r_a][pos[a]] > crowding[r_b][pos[b]] else b

    return _better(cand[0], cand[1]), _better(cand[2], cand[3])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_nsga2_uniform_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    pop_size: int = 50,
    generations: int = 30,
    mutation_rate: float = 0.1,
    seed: int = 42,
    verbose: bool = False,
    _result_box: Optional[list] = None,
) -> dict[str, float]:
    """Compute NSGA-II-uniform feature importance.

    Three objectives:  (|F|, 1−AUC, training_time).

    Args:
        X: Feature DataFrame.
        y: Target array.
        pop_size: Population size (must be even).
        generations: Number of NSGA-II generations.
        mutation_rate: Per-bit mutation probability.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: score}`` sorted descending.  The score is
        the frequency of each feature in the final Pareto front.
    """
    rng = np.random.default_rng(seed)
    features = list(X.columns)
    n_features = len(features)
    n_classes = len(np.unique(y))
    pop_size = (pop_size // 2) * 2  # ensure even

    # ------------------------------------------------------------------
    # 1.  Initialise population
    # ------------------------------------------------------------------
    pop = rng.choice([0, 1], size=(pop_size, n_features)).astype(np.int8)
    fitness = np.full((pop_size, 3), np.inf, dtype=float)

    if verbose:
        print(f"  NSGA-II-uniform: pop={pop_size}, gen={generations}", flush=True)

    for i in range(pop_size):
        sel = np.where(pop[i])[0]
        if len(sel) == 0:
            # Force at least one feature — empty mask is trivially non-dominated
            # on |F| but yields no useful AUC signal for ranking.
            j = rng.integers(0, n_features)
            pop[i, j] = 1
            sel = [j]
        cols = [features[j] for j in sel]
        X_sub = X[cols].values
        t0 = time.perf_counter()
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub,
                y,
                test_size=0.2,
                random_state=seed,
                stratify=y,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub,
                y,
                test_size=0.2,
                random_state=seed,
            )
        model = XGBClassifier(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.3,
            verbosity=0,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)
        if n_classes == 2:
            perf = 1.0 - float(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            perf = 1.0 - float(accuracy_score(y_te, model.predict(X_te)))
        elapsed = time.perf_counter() - t0
        fitness[i] = [float(len(sel)), float(perf), float(elapsed)]

    # ------------------------------------------------------------------
    # 2.  Evolve
    # ------------------------------------------------------------------
    for gen in range(generations):
        fronts = _non_dominated_sort(fitness)
        crowding = [_crowding_distance(fitness, f) for f in fronts]
        rank: dict[int, int] = {}
        pos: dict[int, int] = {}
        for ri, front in enumerate(fronts):
            for pi, idx in enumerate(front):
                rank[idx] = ri
                pos[idx] = pi

        offspring = np.empty_like(pop)
        for i in range(0, pop_size, 2):
            p1, p2 = _tournament_select(pop_size, fronts, crowding, rank, pos, rng)

            # Uniform crossover
            mask = rng.choice([0, 1], size=n_features, p=[0.5, 0.5])
            c1 = np.where(mask, pop[p1], pop[p2])
            c2 = np.where(mask, pop[p2], pop[p1])

            # Uniform bit-flip mutation (no guidance)
            for c in (c1, c2):
                flip = rng.random(n_features) < mutation_rate
                c[flip] = 1 - c[flip]

            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        # Evaluate offspring
        off_fitness = np.full((pop_size, 3), np.inf, dtype=float)
        for i in range(pop_size):
            sel = np.where(offspring[i])[0]
            if len(sel) == 0:
                offspring[i, rng.integers(0, n_features)] = 1
                sel = np.where(offspring[i])[0]
                continue
            cols = [features[j] for j in sel]
            X_sub = X[cols].values
            t0 = time.perf_counter()
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub,
                    y,
                    test_size=0.2,
                    random_state=seed,
                    stratify=y,
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub,
                    y,
                    test_size=0.2,
                    random_state=seed,
                )
            model = XGBClassifier(
                n_estimators=10,
                max_depth=2,
                learning_rate=0.3,
                verbosity=0,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)
            if n_classes == 2:
                perf = 1.0 - float(roc_auc_score(y_te, y_prob[:, 1]))
            else:
                perf = 1.0 - float(accuracy_score(y_te, model.predict(X_te)))
            off_fitness[i] = [float(len(sel)), float(perf), time.perf_counter() - t0]

        # Merge and select
        combined = np.vstack([pop, offspring])
        combined_fit = np.vstack([fitness, off_fitness])
        comb_fronts = _non_dominated_sort(combined_fit)
        comb_crowd = [_crowding_distance(combined_fit, f) for f in comb_fronts]

        selected: list[int] = []
        for front in comb_fronts:
            remaining = pop_size - len(selected)
            if len(front) <= remaining:
                selected.extend(front)
            else:
                cd = comb_crowd[comb_fronts.index(front)]
                order = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                selected.extend([front[k] for k in order[:remaining]])
                break

        pop = combined[selected].astype(np.int8)
        fitness = combined_fit[selected]

        if verbose and (gen + 1) % 5 == 0:
            n_one = int(pop.sum())
            avg_feats = n_one / pop_size
            print(
                f"  Gen {gen + 1:>3d}/{generations}: "
                f"front0 size={len(comb_fronts[0])}, "
                f"avg feats={avg_feats:.1f}, "
                f"best AUC={1 - fitness[0, 1]:.4f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 3.  Feature importance from final Pareto front
    # ------------------------------------------------------------------
    final_fronts = _non_dominated_sort(fitness)
    elite = final_fronts[0]

    if verbose:
        elite_feat_counts = [int(pop[i].sum()) for i in elite]
        print(
            f"  Final elite front: {len(elite)} solutions, "
            f"feat counts: {sorted(elite_feat_counts)[:10]}...",
            flush=True,
        )

    # Side-channel for HV / compression
    if _result_box is not None and len(elite) > 0:
        _result_box.append({
            "fitness": fitness[elite],
            "masks": pop[elite],
        })

    freq = pop[elite].mean(axis=0)  # type: ignore[call-overload]
    result = dict(zip(features, freq.astype(float)))
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Convert a score dict to a ranked list (highest first)."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("NSGA-II-uniform baseline — synthetic demo")
    rng = np.random.default_rng(42)
    n = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        },
    )
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    scores = compute_nsga2_uniform_importance(
        X_demo,
        y_demo,
        pop_size=20,
        generations=10,
        verbose=True,
    )
    print("Feature importance ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.4f}")
    print("\nObservation: f0 should appear most often in elite solutions.")
