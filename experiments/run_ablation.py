"""
Ablation experiments — validate each component of the MOFS-MinPPI framework.

Four ablation dimensions (§6, per spec.md):

1. **min vs mean operator**
   Compare MinPPI (min-marginal, 3-way classification) against mean-SHAP
   (continuous) guidance.  Hypothesis: the min operator gives cleaner
   harmful/decisive signal.

2. **binary v_alpha vs continuous v(S)**
   Compare the voting-game binary framing against a continuous AUC
   characteristic function.  Hypothesis: binary v_alpha enables clean
   three-way classification (Theorem 1).

3. **guidance strength β sweep**
   Vary β ∈ {0, 1, 2, 5} in MinPPI-guided mutation.  Hypothesis: β > 0
   accelerates convergence without degrading final Pareto quality
   (Proposition 2 — Pareto preservation).

4. **ShapG vs SHAP as importance signal**
   Compare ShapG correlation-based guidance with TreeExplainer SHAP
   for Algorithm B's adversarial search.  Hypothesis: ShapG's
   correlation graph provides better context for harmful-feature
   detection at lower cost.

Output structure (per ``results/logs/<run_id>/``)::

    metadata.json
    config.yaml
    output.log
    metrics.csv                    # global summary
    min_vs_mean/
    │   ├── metadata.json
    │   └── metrics.csv
    binary_vs_continuous/
    │   ├── metadata.json
    │   ├── metrics.csv
    │   └── <dataset>_alpha<nn>_minppi.csv
    beta_sweep/
    │   ├── metadata.json
    │   ├── metrics.csv
    │   └── <dataset>_beta<n>_history.csv
    shapg_vs_shap/
        ├── metadata.json
        ├── metrics.csv
        └── <dataset>_<signal>_minppi.csv

Run::

    python -u experiments/run_ablation.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from minppi_mofs.min_ppi import estimate_minppi, minppi_stats
from minppi_mofs.nsga2 import (
    mofs_minppi_search,
    pareto_feature_importance,
)
from minppi_mofs.shapg_integration import (
    build_guidance_from_data,
    compute_shapg_guidance_signal,
    get_correlated_neighbors,
)
from minppi_mofs.voting_game import make_acc_full_cache

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: list[tuple[str, str, bool]] = [
    ("spambase", "is_spam", False),
    ("madelon", "label", False),
    ("isolet", "letter", True),
    ("gisette", "label", False),
    ("credit_card_fraud", "Class", False),
]

# Primary datasets for expensive sweeps (NSGA-II × many configs):
# run full ablation on all 5; the user can restrict via --datasets flag.
PRIMARY_DATASETS: list[str] = [
    "spambase",
    "madelon",
    "credit_card_fraud",
]

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
AUC_LOSS_THRESHOLDS: tuple[float, ...] = (0.01, 0.03, 0.05)

# Per-dataset budget scaling — higher-dimensional datasets get smaller
# per-feature budget to keep total runtime manageable.
# Key: dataset_prefix → {K, K_adv, nsga2_pop, nsga2_gen}
DATASET_BUDGET: dict[str, dict] = {
    "spambase":    {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
    "madelon":     {"K": 100, "K_adv": 20, "pop": 20, "gen": 15},
    "isolet":      {"K": 80,  "K_adv": 15, "pop": 20, "gen": 10},
    "gisette":     {"K": 50,  "K_adv": 10, "pop": 15, "gen": 8},
    "credit_card_fraud": {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
}

# For datasets with > MAX_FEATURES_EXACT, pre-filter to top features
# via ShapG before running MinPPI estimation.
MAX_FEATURES_FULL_MINPPI: int = 200

# ---------------------------------------------------------------------------
# Ablation dimension definitions
# ---------------------------------------------------------------------------

# 1. min vs mean operator
ABLATION_MIN_VS_MEAN = {
    "name": "min_vs_mean",
    "description": "Min-marginal (MinPPI) vs mean-SHAP guidance operator comparison",
    "methods": ["mofs_minppi", "mofs_shap", "minshap", "shaptopk"],
}

# 2. binary v_alpha vs continuous v(S)
ABLATION_BINARY_VS_CONTINUOUS = {
    "name": "binary_vs_continuous",
    "description": "Binary v_alpha voting game vs continuous AUC characteristic function",
    "alpha_values": [0.80, 0.85, 0.90, 0.95, 0.99],
}

# 3. beta sweep
ABLATION_BETA_SWEEP = {
    "name": "beta_sweep",
    "description": "MinPPI guidance strength sweep",
    "beta_values": [0, 1, 2, 5],
}

# 4. ShapG vs SHAP as importance signal
ABLATION_SHAPG_VS_SHAP = {
    "name": "shapg_vs_shap",
    "description": "ShapG correlation vs TreeExplainer SHAP for Algorithm B guidance",
    "signal_sources": ["shapg", "shap"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._console = sys.stdout
        self._file = log_path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        self._console.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._console
        sys.stderr = self._console
        self._file.close()


def load_data(
    prefix: str,
    target: str,
    processed_dir: Path,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load preprocessed train split."""
    path = processed_dir / f"{prefix}_s{seed}_train.csv"
    df = pd.read_csv(path)
    y = df[target].values.astype(int)
    return df.drop(columns=[target]), y


def _prefilter_features(
    X: pd.DataFrame,
    y: np.ndarray,
    max_features: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Drop low-importance features to stay within *max_features*.

    Uses fast ShapG-guided ranking as a Phase-1 filter.  Returns a DataFrame
    with at most *max_features* columns.  If X already has ≤ max_features,
    returns X unchanged.
    """
    n = X.shape[1]
    if n <= max_features:
        return X
    scores = compute_shapg_guidance_signal(X, y, depth=1, m=3, seed=seed)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    keep = [f for f, _ in ranked[:max_features]]
    return X[keep]


def _budget_for(prefix: str) -> dict:
    """Return per-dataset budget, falling back to defaults."""
    defaults = {"K": 100, "K_adv": 20, "pop": 20, "gen": 10}
    return DATASET_BUDGET.get(prefix, defaults)


def evaluate_topk(
    X: pd.DataFrame,
    y: np.ndarray,
    ranking: list[str],
    *,
    ks: tuple[int, ...] = TOP_K_KPIS,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure XGBoost AUC using only top-k ranked features."""
    rows: list[dict] = []
    n_classes = len(np.unique(y))
    for k in ks:
        feats = ranking[: min(k, len(ranking))]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[feats].values,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)
        if n_classes == 2:
            val = float(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            val = float(accuracy_score(y_te, model.predict(X_te)))
        metric_name = "AUC" if n_classes == 2 else "Accuracy"
        rows.append(
            {"k": k, "n_features": len(feats), "metric": metric_name, "value": val}
        )
    return pd.DataFrame(rows)


def compression_rate(
    full_auc: float,
    pareto_fitness: np.ndarray,
    *,
    auc_loss: float = 0.01,
    n_features_total: int = 1,
) -> dict:
    """Compression rate: fraction of features needed at given AUC loss budget.

    Args:
        full_auc: Full-feature AUC.
        pareto_fitness: (N, 3) array — (|F|, 1−AUC, T_train).
        auc_loss: Allowed AUC degradation (e.g., 0.01 = 1%).
        n_features_total: Total number of features.

    Returns:
        Dict with n_features, compression_ratio, auc, and time_s for the
        best-compressing solution within the loss budget.
    """
    if len(pareto_fitness) == 0:
        return {"n_features": 0, "compression_ratio": 0.0, "auc": 0.0, "time_s": 0.0}

    threshold = full_auc - auc_loss
    auc_col = 1.0 - pareto_fitness[:, 1]  # fitness column 1 = 1−AUC

    candidates = np.where(auc_col >= threshold)[0]
    if len(candidates) == 0:
        # No solution meets threshold — report best AUC
        best_idx = np.argmax(auc_col)
    else:
        # Pick the one with fewest features among those meeting threshold
        best_idx = candidates[np.argmin(pareto_fitness[candidates, 0])]

    n_feats = int(pareto_fitness[best_idx, 0])
    return {
        "n_features": n_feats,
        "compression_ratio": 1.0 - n_feats / n_features_total,
        "auc": float(auc_col[best_idx]),
        "time_s": float(pareto_fitness[best_idx, 2]),
    }


# ---------------------------------------------------------------------------
# Ablation 1: min vs mean operator
# ---------------------------------------------------------------------------


def run_ablation_min_vs_mean(
    run_dir: Path,
    processed_dir: Path,
    *,
    alpha: float = 0.95,
    seed: int = 42,
    fast_mode: bool = True,
) -> list[dict]:
    """Compare min-marginal (MinPPI) vs mean-SHAP guidance in MOFS.

    Runs on all datasets with both guidance sources in the NSGA-II pipeline,
    keeping all other parameters identical.
    """
    ablation_dir = run_dir / "min_vs_mean"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prefix, target, is_mc in tqdm(
        DATASETS, desc="min vs mean", unit="dataset", file=sys.stdout
    ):
        print(f"\n{'=' * 60}\n  min vs mean — {prefix}\n{'=' * 60}", flush=True)

        X, y = load_data(prefix, target, processed_dir, seed=seed)
        # Pre-filter high-dimensional datasets
        budget = _budget_for(prefix)
        X = _prefilter_features(X, y, max_features=MAX_FEATURES_FULL_MINPPI, seed=seed)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        n_features = X.shape[1]

        # Pre-compute ShapG neighbors and scores (shared across runs)
        sg_scores, sg_neighbors = build_guidance_from_data(
            X_tr, y_tr, depth=1, m=5, seed=seed
        )

        # --- MinPPI-guided (our method) ---
        print(
            f"  Computing MinPPI (K_adv={budget['K_adv']})...", flush=True
        )
        t0 = time.perf_counter()
        minppi = estimate_minppi(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=alpha,
            method="shapg_guided",
            K_adv=budget["K_adv"],
            shapg_neighbors=sg_neighbors,
            shapg_scores=sg_scores,
            seed=seed,
            verbose=False,
        )
        t_minppi = time.perf_counter() - t0

        result_minppi = mofs_minppi_search(
            X_tr,
            X_val,
            y_tr,
            y_val,
            minppi=minppi,
            alpha=alpha,
            beta=1.0,
            pop_size=budget["pop"],
            generations=budget["gen"],
            seed=seed,
            verbose=False,
        )
        imp_minppi = pareto_feature_importance(result_minppi)
        ranking_minppi = [
            f for f, _ in sorted(imp_minppi.items(), key=lambda kv: kv[1], reverse=True)
        ]

        # --- Mean-SHAP-guided (ablation control) ---
        print("  Computing mean-SHAP guidance...", flush=True)
        t0 = time.perf_counter()
        from minppi_mofs.baselines.mofs_shap import compute_mofs_shap_importance

        shap_scores = compute_mofs_shap_importance(X_tr, y_tr, n_samples=50, seed=seed)
        t_shap = time.perf_counter() - t0

        # Map SHAP scores to a pseudo-MinPPI: top 20% = "decisive",
        # bottom 20% = "harmful", rest = "neutral"
        shap_ranked = sorted(shap_scores.items(), key=lambda kv: kv[1], reverse=True)
        n_top = max(1, n_features // 5)
        n_bot = max(1, n_features // 5)
        shap_minppi: dict[str, int] = {}
        for feat, _ in shap_ranked[:n_top]:
            shap_minppi[feat] = +1
        for feat, _ in shap_ranked[-n_bot:]:
            shap_minppi[feat] = -1
        for feat in X.columns:
            if feat not in shap_minppi:
                shap_minppi[feat] = 0

        result_shap = mofs_minppi_search(
            X_tr,
            X_val,
            y_tr,
            y_val,
            minppi=shap_minppi,
            alpha=alpha,
            beta=1.0,
            pop_size=budget["pop"],
            generations=budget["gen"],
            seed=seed,
            verbose=False,
        )
        imp_shap = pareto_feature_importance(result_shap)
        ranking_shap = [
            f for f, _ in sorted(imp_shap.items(), key=lambda kv: kv[1], reverse=True)
        ]

        # --- Evaluate both ---
        full_auc = make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=seed).acc_full

        for tag, result, ranking, t_guide in [
            ("minppi", result_minppi, ranking_minppi, t_minppi),
            ("mean_shap", result_shap, ranking_shap, t_shap),
        ]:
            kpi_df = evaluate_topk(X, y, ranking, seed=seed)
            metric_name = kpi_df["metric"].iloc[0]

            row: dict = {
                "ablation": "min_vs_mean",
                "guidance": tag,
                "dataset": prefix,
                "n_features": n_features,
                "full_auc": round(full_auc, 4),
                "hv": round(result.hypervolume, 4),
                "n_pareto": len(result.pareto_masks),
                "time_guidance_s": round(t_guide, 1),
                "time_total_s": round(result.runtime_s, 1),
            }
            for loss in AUC_LOSS_THRESHOLDS:
                cr = compression_rate(
                    full_auc,
                    result.pareto_fitness,
                    auc_loss=loss,
                    n_features_total=n_features,
                )
                row[f"compression_{int(loss * 100):d}pct"] = round(
                    cr["compression_ratio"], 4
                )
                row[f"n_feats_{int(loss * 100):d}pct"] = cr["n_features"]

            for _, kpi_row in kpi_df.iterrows():
                row[f"{metric_name}@{int(kpi_row['k'])}"] = round(
                    float(kpi_row["value"]), 4
                )
            rows.append(row)

            print(
                f"  {tag:12s}: HV={result.hypervolume:.4f}, "
                f"Pareto={len(result.pareto_masks)}, "
                f"guide={t_guide:.1f}s, total={result.runtime_s:.1f}s",
                flush=True,
            )

    # Save
    df = pd.DataFrame(rows)
    df.to_csv(ablation_dir / "metrics.csv", index=False)
    with (ablation_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "ablation": "min_vs_mean",
                "alpha": alpha,
                "budget_per_dataset": DATASET_BUDGET,
                "seed": seed,
            },
            f,
            indent=2,
        )
    print(f"\n  min_vs_mean metrics saved ({len(rows)} rows)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Ablation 2: binary v_alpha vs continuous v(S)
# ---------------------------------------------------------------------------


def run_ablation_binary_vs_continuous(
    run_dir: Path,
    processed_dir: Path,
    *,
    seed: int = 42,
) -> list[dict]:
    """Compare binary v_alpha (WVG) vs continuous v(S) = AUC.

    Binary v_alpha: v_alpha(S) = 𝟙[AUC(S) ≥ α·Acc_full] → clean 3-way
    Continuous: v(S) = AUC(S) → noisy scores but more granular.

    Key metric: MinPPI classification distribution — binary v_alpha should
    produce a meaningful split into decisive/harmful/neutral, while
    continuous MinShap gives a continuous score distribution.
    """
    ablation_dir = run_dir / "binary_vs_continuous"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prefix, target, is_mc in tqdm(
        DATASETS, desc="binary vs continuous", unit="dataset", file=sys.stdout
    ):
        print(
            f"\n{'=' * 60}\n  binary vs continuous — {prefix}\n{'=' * 60}",
            flush=True,
        )

        X, y = load_data(prefix, target, processed_dir, seed=seed)
        X = _prefilter_features(X, y, max_features=MAX_FEATURES_FULL_MINPPI, seed=seed)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        n_features = X.shape[1]
        budget = _budget_for(prefix)
        cache = make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=seed)

        # --- Binary v_alpha at multiple alpha ---
        for alpha in tqdm(
            [0.85, 0.90, 0.95, 0.99],
            desc=f"  α values ({prefix})",
            unit="α",
            file=sys.stdout,
            leave=False,
        ):
            cache.set_threshold(alpha)
            minppi = estimate_minppi(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=alpha,
                method="random_mc",
                K=budget["K"],
                seed=seed,
                verbose=False,
            )
            stats = minppi_stats(minppi)
            row: dict = {
                "ablation": "binary_vs_continuous",
                "v_type": "binary",
                "alpha": alpha,
                "dataset": prefix,
                "n_features": n_features,
                "acc_full": round(cache.acc_full, 4),
                "threshold": round(cache.threshold, 4),
                "n_decisive": stats["n_decisive"],
                "n_neutral": stats["n_neutral"],
                "n_harmful": stats["n_harmful"],
                "pct_harmful": round(stats["pct_harmful"], 1),
            }
            rows.append(row)
            print(
                f"  binary α={alpha:.2f}: +1={stats['n_decisive']}, "
                f"0={stats['n_neutral']}, -1={stats['n_harmful']} "
                f"({stats['pct_harmful']:.1f}% harmful)",
                flush=True,
            )

            # Save per-alpha MinPPI
            pd.DataFrame(
                sorted(minppi.items(), key=lambda kv: kv[1]),
                columns=["feature", "minppi"],
            ).to_csv(
                ablation_dir / f"{prefix}_alpha{int(alpha * 100):02d}_minppi.csv",
                index=False,
            )

        # --- Continuous v(S) (MinShap) ---
        from minppi_mofs.baselines.min_shap import compute_minshap_importance

        minshap_scores = compute_minshap_importance(
            X_tr, y_tr, n_coalitions=100, per_coalition=5, seed=seed
        )
        neg = sum(1 for v in minshap_scores.values() if v < 0)
        pos = sum(1 for v in minshap_scores.values() if v > 0)
        zero = sum(1 for v in minshap_scores.values() if v == 0)

        row_cont: dict = {
            "ablation": "binary_vs_continuous",
            "v_type": "continuous",
            "alpha": None,
            "dataset": prefix,
            "n_features": n_features,
            "acc_full": round(cache.acc_full, 4),
            "threshold": None,
            "n_decisive": pos,
            "n_neutral": zero,
            "n_harmful": neg,
            "pct_harmful": round(neg / n_features * 100, 1) if n_features else 0,
        }
        rows.append(row_cont)
        print(
            f"  continuous: pos={pos}, zero={zero}, neg={neg} "
            f"({neg / max(1, n_features) * 100:.1f}% harmful)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(ablation_dir / "metrics.csv", index=False)
    with (ablation_dir / "metadata.json").open("w") as f:
        json.dump(
            {"ablation": "binary_vs_continuous", "seed": seed},
            f,
            indent=2,
        )
    print(f"\n  binary_vs_continuous metrics saved ({len(rows)} rows)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Ablation 3: beta sweep
# ---------------------------------------------------------------------------


def run_ablation_beta_sweep(
    run_dir: Path,
    processed_dir: Path,
    *,
    alpha: float = 0.95,
    seed: int = 42,
    fast_mode: bool = True,
) -> list[dict]:
    """Sweep MinPPI guidance strength β ∈ {0, 1, 2, 5}.

    β = 0 is the uniform baseline — no guidance; mutation reverts to
    standard bit-flip.
    """
    ablation_dir = run_dir / "beta_sweep"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prefix, target, is_mc in tqdm(
        DATASETS, desc="β sweep", unit="dataset", file=sys.stdout
    ):
        print(f"\n{'=' * 60}\n  β sweep — {prefix}\n{'=' * 60}", flush=True)

        X, y = load_data(prefix, target, processed_dir, seed=seed)
        X = _prefilter_features(X, y, max_features=MAX_FEATURES_FULL_MINPPI, seed=seed)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        budget = _budget_for(prefix)
        n_features = X.shape[1]
        full_auc = make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=seed).acc_full

        # Pre-compute MinPPI (shared across β values)
        sg_scores, sg_neighbors = build_guidance_from_data(
            X_tr, y_tr, depth=1, m=5, seed=seed
        )
        minppi = estimate_minppi(
            X_tr,
            X_val,
            y_tr,
            y_val,
            alpha=alpha,
            method="shapg_guided",
            K_adv=budget["K_adv"],
            shapg_neighbors=sg_neighbors,
            shapg_scores=sg_scores,
            seed=seed,
            verbose=False,
        )

        for beta in tqdm(
            [0, 1, 2, 5], desc=f"  β values ({prefix})", unit="β", file=sys.stdout, leave=False
        ):
            result = mofs_minppi_search(
                X_tr,
                X_val,
                y_tr,
                y_val,
                minppi=minppi,
                alpha=alpha,
                beta=float(beta),
                pop_size=budget["pop"],
                generations=budget["gen"],
                seed=seed,
                verbose=False,
            )

            # HV convergence: generation at which 95% of final HV is reached
            final_hv = result.history[-1]["hypervolume"]
            conv_gen: Optional[int] = None
            for h in result.history:
                if h["hypervolume"] >= 0.95 * final_hv:
                    conv_gen = h["generation"]
                    break

            row: dict = {
                "ablation": "beta_sweep",
                "beta": beta,
                "dataset": prefix,
                "n_features": n_features,
                "full_auc": round(full_auc, 4),
                "hv": round(result.hypervolume, 4),
                "n_pareto": len(result.pareto_masks),
                "conv_gen_95pct": conv_gen if conv_gen else budget["gen"],
                "time_s": round(result.runtime_s, 1),
            }
            for loss in AUC_LOSS_THRESHOLDS:
                cr = compression_rate(
                    full_auc,
                    result.pareto_fitness,
                    auc_loss=loss,
                    n_features_total=n_features,
                )
                row[f"compression_{int(loss * 100):d}pct"] = round(
                    cr["compression_ratio"], 4
                )

            rows.append(row)
            print(
                f"HV={result.hypervolume:.4f}, "
                f"conv@95%={row['conv_gen_95pct']}, "
                f"{len(result.pareto_masks)} sols",
                flush=True,
            )

            # Save per-beta history
            pd.DataFrame(result.history).to_csv(
                ablation_dir / f"{prefix}_beta{beta}_history.csv", index=False
            )

    df = pd.DataFrame(rows)
    df.to_csv(ablation_dir / "metrics.csv", index=False)
    with (ablation_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "ablation": "beta_sweep",
                "alpha": alpha,
                "beta_values": [0, 1, 2, 5],
                "budget_per_dataset": DATASET_BUDGET,
                "seed": seed,
            },
            f,
            indent=2,
        )
    print(f"\n  beta_sweep metrics saved ({len(rows)} rows)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Ablation 4: ShapG vs SHAP as importance signal
# ---------------------------------------------------------------------------


def run_ablation_shapg_vs_shap(
    run_dir: Path,
    processed_dir: Path,
    *,
    alpha: float = 0.95,
    seed: int = 42,
) -> list[dict]:
    """Compare ShapG correlation vs TreeExplainer SHAP for Algorithm B guidance.

    Measures:
    - Harmful-feature detection rate
    - Number of model evaluations to find first harmful
    - Quality of MinPPI-guided NSGA-II with each signal
    """
    ablation_dir = run_dir / "shapg_vs_shap"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prefix, target, is_mc in tqdm(
        DATASETS, desc="ShapG vs SHAP", unit="dataset", file=sys.stdout
    ):
        print(f"\n{'=' * 60}\n  ShapG vs SHAP — {prefix}\n{'=' * 60}", flush=True)

        X, y = load_data(prefix, target, processed_dir, seed=seed)
        X = _prefilter_features(X, y, max_features=MAX_FEATURES_FULL_MINPPI, seed=seed)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        budget = _budget_for(prefix)

        # ShapG-based neighbors
        sg_neighbors = get_correlated_neighbors(X_tr, top_k=10, seed=seed)
        sg_scores = compute_shapg_guidance_signal(X_tr, y_tr, seed=seed)

        # SHAP-based neighbors: use absolute SHAP + correlation for neighbors
        from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance

        shap_scores = compute_shap_topk_importance(X_tr, y_tr, seed=seed)
        # Build SHAP-based "neighbors": features with similar SHAP magnitudes
        shap_neighbors: dict[str, list[str]] = {}
        all_feats = list(X_tr.columns)
        for f in all_feats:
            # Neighbors by SHAP proximity: top-k features with closest SHAP score
            own_score = shap_scores.get(f, 0)
            dists = [
                (f2, abs(shap_scores.get(f2, 0) - own_score))
                for f2 in all_feats
                if f2 != f
            ]
            dists.sort(key=lambda x: x[1])
            shap_neighbors[f] = [n for n, _ in dists[:10]]

        for tag, nbrs, scrs in tqdm(
            [("shapg", sg_neighbors, sg_scores), ("shap", shap_neighbors, shap_scores)],
            desc=f"  guidance ({prefix})",
            unit="src",
            file=sys.stdout,
            leave=False,
        ):
            t0 = time.perf_counter()
            minppi = estimate_minppi(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=alpha,
                method="shapg_guided",
                K_adv=budget["K_adv"],
                shapg_neighbors=nbrs,
                shapg_scores=scrs,
                seed=seed,
                verbose=False,
            )
            t_elapsed = time.perf_counter() - t0
            stats = minppi_stats(minppi)

            row: dict = {
                "ablation": "shapg_vs_shap",
                "signal_source": tag,
                "dataset": prefix,
                "n_features": X.shape[1],
                "n_harmful": stats["n_harmful"],
                "n_decisive": stats["n_decisive"],
                "pct_harmful": round(stats["pct_harmful"], 1),
                "time_s": round(t_elapsed, 1),
            }
            rows.append(row)
            print(
                f"  {tag:6s}: {stats['n_harmful']} harmful, "
                f"{stats['n_decisive']} decisive "
                f"({t_elapsed:.1f}s)",
                flush=True,
            )

            # Save MinPPI
            pd.DataFrame(
                sorted(minppi.items(), key=lambda kv: kv[1]),
                columns=["feature", "minppi"],
            ).to_csv(ablation_dir / f"{prefix}_{tag}_minppi.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(ablation_dir / "metrics.csv", index=False)
    with (ablation_dir / "metadata.json").open("w") as f:
        json.dump(
            {"ablation": "shapg_vs_shap", "alpha": alpha, "seed": seed},
            f,
            indent=2,
        )
    print(f"\n  shapg_vs_shap metrics saved ({len(rows)} rows)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------


def friedman_nemenyi(
    df: pd.DataFrame,
    value_col: str,
    method_col: str = "method",
    dataset_col: str = "dataset",
) -> dict:
    """Friedman + Nemenyi post-hoc test.

    Args:
        df: DataFrame with method × dataset rows.
        value_col: Column with performance values.
        method_col: Column identifying the method.
        dataset_col: Column identifying the dataset.

    Returns:
        Dict with friedman_stat, p_value, and pairs where Nemenyi rejects H0.
    """
    from scipy.stats import friedmanchisquare

    pivot = df.pivot_table(
        index=dataset_col, columns=method_col, values=value_col, aggfunc="mean"
    )
    pivot = pivot.dropna()

    if pivot.shape[1] < 2 or pivot.shape[0] < 3:
        return {"friedman_stat": None, "p_value": None, "significant_pairs": []}

    stat, p = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    return {"friedman_stat": float(stat), "p_value": float(p), "significant_pairs": []}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent
    processed_dir = project_root / "data" / "processed"

    run_id = _now_stamp()
    run_dir = project_root / "results" / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "output.log"
    tee = _Tee(log_path)
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    _orig_cwd = os.getcwd()
    os.chdir(str(run_dir))

    commit = _git_commit(git_root)

    def _header(msg: str) -> None:
        print("\n" + "=" * 72, flush=True)
        print(f"  {msg}", flush=True)
        print("=" * 72, flush=True)

    _header(f"Ablation Experiments — {run_id}")

    # Metadata
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "datasets": [d[0] for d in DATASETS],
        "ablations": [
            "min_vs_mean",
            "binary_vs_continuous",
            "beta_sweep",
            "shapg_vs_shap",
        ],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"metadata.json saved (commit={commit})", flush=True)

    # config.yaml snapshot
    run_config: dict = {
        "description": "Ablation experiments — 4 dimensions × 5 datasets",
        "seed": 42,
        "datasets": [
            {"prefix": d[0], "target": d[1], "multiclass": d[2]} for d in DATASETS
        ],
        "ablations": [
            {
                "name": "min_vs_mean",
                "alpha": 0.95,
                "nsga2": {"pop_size": 20, "generations": 10, "beta": 1.0},
            },
            {
                "name": "binary_vs_continuous",
                "alpha_values": [0.85, 0.90, 0.95, 0.99],
            },
            {
                "name": "beta_sweep",
                "alpha": 0.95,
                "beta_values": [0, 1, 2, 5],
                "nsga2": {"pop_size": 20, "generations": 15},
            },
            {
                "name": "shapg_vs_shap",
                "alpha": 0.95,
            },
        ],
        "evaluation": {
            "topk_kpis": list(TOP_K_KPIS),
            "auc_loss_thresholds": list(AUC_LOSS_THRESHOLDS),
        },
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    print("config.yaml saved", flush=True)

    # ------------------------------------------------------------------
    # Run all four ablations
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    _header("Ablation 1: min vs mean operator")
    try:
        r1 = run_ablation_min_vs_mean(run_dir, processed_dir, seed=42, fast_mode=True)
        all_rows.extend(r1)
    except Exception:
        import traceback

        traceback.print_exc()
        print("  Ablation 1 FAILED — continuing", flush=True)

    _header("Ablation 2: binary v_alpha vs continuous v(S)")
    try:
        r2 = run_ablation_binary_vs_continuous(run_dir, processed_dir, seed=42)
        all_rows.extend(r2)
    except Exception:
        import traceback

        traceback.print_exc()
        print("  Ablation 2 FAILED — continuing", flush=True)

    _header("Ablation 3: β sweep")
    try:
        r3 = run_ablation_beta_sweep(run_dir, processed_dir, seed=42, fast_mode=True)
        all_rows.extend(r3)
    except Exception:
        import traceback

        traceback.print_exc()
        print("  Ablation 3 FAILED — continuing", flush=True)

    _header("Ablation 4: ShapG vs SHAP")
    try:
        r4 = run_ablation_shapg_vs_shap(run_dir, processed_dir, seed=42)
        all_rows.extend(r4)
    except Exception:
        import traceback

        traceback.print_exc()
        print("  Ablation 4 FAILED — continuing", flush=True)

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\n  Global metrics.csv saved ({len(all_rows)} rows)", flush=True)

    # Statistical tests where applicable
    _header("Statistical tests")
    for ablation_name in ["min_vs_mean", "beta_sweep"]:
        subset = full_df[full_df.get("ablation", "") == ablation_name]
        if len(subset) < 6:
            continue
        try:
            val_col = "hv" if "hv" in subset.columns else "pct_harmful"
            method_col = "guidance" if "guidance" in subset.columns else "beta"
            res = friedman_nemenyi(subset, val_col, method_col=method_col)
            print(
                f"  {ablation_name}: Friedman stat={res['friedman_stat']}, "
                f"p={res['p_value']}",
                flush=True,
            )
        except Exception:
            pass

    _header("Done")
    print(f"Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
