"""
Alpha sensitivity experiment — quota parameter α ∈ {0.80, 0.85, 0.90, 0.95, 0.99}.

The quota α governs the voting-game threshold:

    v_alpha(S) = 𝟙[AUC(ℳ_S) ≥ α · Acc_full]

Higher α → fewer winning coalitions → more features classified as harmful
by MinPPI.  This experiment measures how α affects:

1. **MinPPI class distribution**: |decisive|, |neutral|, |harmful| vs α
2. **MOFS-MinPPI downstream performance**: hypervolume, compression rate
3. **Feature ranking stability**: Jaccard similarity between adjacent α levels
4. **Harmful-detection sensitivity**: at what α does a feature flip to -1?

Key findings (§5 / §6):
- α = 0.95 is the recommended main-table value (balances selectivity and
  stability)
- α ≥ 0.99 produces many harmful classifications but may be too strict
- α ≤ 0.85 produces almost no harmful classifications (too permissive)

Output structure (per ``results/logs/<run_id>/``)::

    metadata.json
    config.yaml
    output.log
    metrics.csv                  # per-(dataset, alpha) summary
    minppi_distribution.csv      # class counts per alpha per dataset
    table_harmful_pct_vs_alpha.csv
    table_hv_vs_alpha.csv
    table_compression_vs_alpha.csv
    <dataset>_alpha<nn>_minppi.csv       # per-alpha MinPPI values
    <dataset>_stability.csv              # Jaccard between alpha levels

Run::

    python -u experiments/run_alpha_sweep.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from minppi_mofs.min_ppi import estimate_minppi, minppi_stats
from minppi_mofs.nsga2 import mofs_minppi_search, pareto_feature_importance
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.voting_game import make_acc_full_cache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_VALUES: list[float] = [0.80, 0.85, 0.90, 0.95, 0.99]

DATASETS: list[tuple[str, str, bool]] = [
    ("spambase", "is_spam", False),
    ("madelon", "label", False),
    ("isolet", "letter", True),
    ("gisette", "label", False),
    ("credit_card_fraud", "Class", False),
]

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
AUC_LOSS_THRESHOLDS: tuple[float, ...] = (0.01, 0.03, 0.05)

# Per-dataset budget — shrinks MinPPI + NSGA-II cost on high-dimensional datasets
DATASET_BUDGET: dict[str, dict] = {
    "spambase":    {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
    "madelon":     {"K": 100, "K_adv": 20, "pop": 20, "gen": 15},
    "isolet":      {"K": 80,  "K_adv": 15, "pop": 20, "gen": 10},
    "gisette":     {"K": 50,  "K_adv": 10, "pop": 15, "gen": 8},
    "credit_card_fraud": {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
}
MAX_FEATURES_FULL_MINPPI: int = 200


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
    """Phase-1 filter: keep top *max_features* by ShapG score."""
    n = X.shape[1]
    if n <= max_features:
        return X
    from minppi_mofs.shapg_integration import compute_shapg_guidance_signal

    scores = compute_shapg_guidance_signal(X, y, depth=1, m=3, seed=seed)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    keep = [f for f, _ in ranked[:max_features]]
    return X[keep]


def _budget_for(prefix: str) -> dict:
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


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard index between two feature sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compression_rate(
    full_auc: float,
    fitness: np.ndarray,
    auc_loss: float,
    n_total: int,
) -> float:
    """Compression ratio at given AUC loss budget."""
    threshold = full_auc - auc_loss
    auc_col = 1.0 - fitness[:, 1]
    candidates = np.where(auc_col >= threshold)[0]
    if len(candidates) == 0:
        return 0.0
    n_feats = int(fitness[candidates[np.argmin(fitness[candidates, 0])], 0])
    return 1.0 - n_feats / max(1, n_total)


# ---------------------------------------------------------------------------
# Main α sweep
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

    _header(f"Alpha Sensitivity Sweep — {run_id}")
    print(f"Alpha values: {ALPHA_VALUES}", flush=True)
    print(f"Commit: {commit}", flush=True)

    # Metadata
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "alpha_values": ALPHA_VALUES,
        "datasets": [d[0] for d in DATASETS],
        "nsga2": {"beta": 1.0, "budget_per_dataset": DATASET_BUDGET},
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("metadata.json saved", flush=True)

    # config.yaml snapshot
    run_config: dict = {
        "description": "Alpha sensitivity sweep — quota parameter α ∈ {0.80, 0.85, 0.90, 0.95, 0.99}",
        "seed": 42,
        "alpha_values": ALPHA_VALUES,
        "datasets": [
            {"prefix": d[0], "target": d[1], "multiclass": d[2]} for d in DATASETS
        ],
        "nsga2": {"beta": 1.0, "budget_per_dataset": DATASET_BUDGET},
        "evaluation": {
            "topk_kpis": list(TOP_K_KPIS),
            "auc_loss_thresholds": list(AUC_LOSS_THRESHOLDS),
        },
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    print("config.yaml saved", flush=True)

    # ------------------------------------------------------------------
    # Per-dataset sweep
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []
    all_distributions: list[dict] = []

    for prefix, target, is_mc in tqdm(
        DATASETS, desc="α sweep", unit="dataset", file=sys.stdout
    ):
        _header(f"Alpha sweep — {prefix}")

        X, y = load_data(prefix, target, processed_dir)
        X = _prefilter_features(X, y, max_features=MAX_FEATURES_FULL_MINPPI, seed=42)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        budget = _budget_for(prefix)
        n_features = X.shape[1]
        features = list(X.columns)

        # Acc_full (independent of α)
        cache = make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=42)
        acc_full = cache.acc_full
        print(f"  Full-feature AUC: {acc_full:.4f}, n_features: {n_features}")

        # ShapG guidance (shared across α)
        sg_scores, sg_neighbors = build_guidance_from_data(
            X_tr, y_tr, depth=1, m=5, seed=42
        )

        # Storage for stability analysis
        minppi_per_alpha: dict[float, dict[str, int]] = {}
        ranking_per_alpha: dict[float, list[str]] = {}
        pareto_per_alpha: dict[float, dict] = {}

        for alpha in tqdm(
            ALPHA_VALUES,
            desc=f"  α values ({prefix})",
            unit="α",
            file=sys.stdout,
            leave=False,
        ):
            print(f"\n  --- α = {alpha:.2f} ---", flush=True)
            print(
                f"  Threshold: {alpha:.2f} × {acc_full:.4f} = {acc_full * alpha:.4f}",
                flush=True,
            )

            # --- MinPPI estimation ---
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
                seed=42,
                verbose=False,
            )
            t_minppi = time.perf_counter() - t0
            stats = minppi_stats(minppi)
            minppi_per_alpha[alpha] = minppi

            print(
                f"  MinPPI: +1={stats['n_decisive']}, 0={stats['n_neutral']}, "
                f"-1={stats['n_harmful']}  ({t_minppi:.1f}s)",
                flush=True,
            )

            all_distributions.append(
                {
                    "dataset": prefix,
                    "alpha": alpha,
                    "n_features": n_features,
                    "threshold": round(acc_full * alpha, 4),
                    **stats,
                }
            )

            # Save per-alpha MinPPI
            pd.DataFrame(
                [(f, minppi.get(f, 0)) for f in features],
                columns=["feature", "minppi"],
            ).to_csv(
                run_dir / f"{prefix}_alpha{int(alpha * 100):02d}_minppi.csv",
                index=False,
            )

            # --- MOFS-MinPPI search ---
            result = mofs_minppi_search(
                X_tr,
                X_val,
                y_tr,
                y_val,
                minppi=minppi,
                alpha=alpha,
                beta=1.0,
                pop_size=budget["pop"],
                generations=budget["gen"],
                seed=42,
                verbose=False,
            )
            imp = pareto_feature_importance(result)
            ranking = [
                f for f, _ in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
            ]
            ranking_per_alpha[alpha] = ranking

            # Evaluate top-k
            kpi_df = evaluate_topk(X, y, ranking, seed=42)
            metric_name = kpi_df["metric"].iloc[0]

            row: dict = {
                "dataset": prefix,
                "alpha": alpha,
                "n_features": n_features,
                "acc_full": round(acc_full, 4),
                "threshold": round(acc_full * alpha, 4),
                "n_harmful": stats["n_harmful"],
                "n_decisive": stats["n_decisive"],
                "n_neutral": stats["n_neutral"],
                "pct_harmful": round(stats["pct_harmful"], 1),
                "hv": round(result.hypervolume, 4),
                "n_pareto": len(result.pareto_masks),
                "time_minppi_s": round(t_minppi, 1),
                "time_total_s": round(result.runtime_s, 1),
            }

            # Compression rates
            for loss in AUC_LOSS_THRESHOLDS:
                cr = compression_rate(acc_full, result.pareto_fitness, loss, n_features)
                row[f"compression_{int(loss * 100):d}pct"] = round(cr, 4)

            # Top-k AUC
            for _, kpi_row in kpi_df.iterrows():
                row[f"{metric_name}@{int(kpi_row['k'])}"] = round(
                    float(kpi_row["value"]), 4
                )

            all_metrics.append(row)
            pareto_per_alpha[alpha] = {
                "fitness": result.pareto_fitness.tolist(),
                "hv": result.hypervolume,
            }

            print(
                f"  MOFS-MinPPI: HV={result.hypervolume:.4f}, "
                f"Pareto={len(result.pareto_masks)} sols, "
                f"total={result.runtime_s:.1f}s",
                flush=True,
            )

        # --- Stability analysis between adjacent α levels ---
        print("\n  --- Stability (Jaccard) ---", flush=True)
        stability_rows: list[dict] = []
        sorted_alphas = sorted(ALPHA_VALUES)

        for i in range(len(sorted_alphas)):
            for j in range(i + 1, len(sorted_alphas)):
                a1, a2 = sorted_alphas[i], sorted_alphas[j]

                # Harmful-feature set Jaccard
                h1 = {f for f, v in minppi_per_alpha[a1].items() if v == -1}
                h2 = {f for f, v in minppi_per_alpha[a2].items() if v == -1}
                jac_harmful = jaccard_similarity(h1, h2)

                # Decisive-feature set Jaccard
                d1 = {f for f, v in minppi_per_alpha[a1].items() if v == +1}
                d2 = {f for f, v in minppi_per_alpha[a2].items() if v == +1}
                jac_decisive = jaccard_similarity(d1, d2)

                # Top-10 ranking Jaccard
                r1 = set(ranking_per_alpha[a1][:10])
                r2 = set(ranking_per_alpha[a2][:10])
                jac_top10 = jaccard_similarity(r1, r2)

                stability_rows.append(
                    {
                        "dataset": prefix,
                        "alpha1": a1,
                        "alpha2": a2,
                        "jaccard_harmful": round(jac_harmful, 4),
                        "jaccard_decisive": round(jac_decisive, 4),
                        "jaccard_top10": round(jac_top10, 4),
                        "n_harmful_a1": len(h1),
                        "n_harmful_a2": len(h2),
                    }
                )
                print(
                    f"  α={a1:.2f} vs α={a2:.2f}: "
                    f"J_harmful={jac_harmful:.3f}, "
                    f"J_decisive={jac_decisive:.3f}, "
                    f"J_top10={jac_top10:.3f}",
                    flush=True,
                )

        # Save stability
        pd.DataFrame(stability_rows).to_csv(
            run_dir / f"{prefix}_stability.csv", index=False
        )

    # ------------------------------------------------------------------
    # Save aggregated results
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nmetrics.csv saved ({len(metrics_df)} rows)", flush=True)

    dist_df = pd.DataFrame(all_distributions)
    dist_df.to_csv(run_dir / "minppi_distribution.csv", index=False)
    print(f"minppi_distribution.csv saved ({len(dist_df)} rows)", flush=True)

    # Build pivot tables
    _header("Summary tables")

    # Harmful % vs α per dataset
    harmful_pivot = dist_df.pivot_table(
        index="dataset", columns="alpha", values="pct_harmful", aggfunc="first"
    )
    harmful_pivot = harmful_pivot.round(1)
    harmful_pivot.to_csv(run_dir / "table_harmful_pct_vs_alpha.csv")
    print("table_harmful_pct_vs_alpha.csv", flush=True)

    # HV vs α per dataset
    hv_pivot = metrics_df.pivot_table(
        index="dataset", columns="alpha", values="hv", aggfunc="first"
    )
    hv_pivot = hv_pivot.round(4)
    hv_pivot.to_csv(run_dir / "table_hv_vs_alpha.csv")
    print("table_hv_vs_alpha.csv", flush=True)

    # Compression at 3% loss vs α
    cr_pivot = metrics_df.pivot_table(
        index="dataset",
        columns="alpha",
        values="compression_3pct",
        aggfunc="first",
    )
    cr_pivot = cr_pivot.round(4)
    cr_pivot.to_csv(run_dir / "table_compression_vs_alpha.csv")
    print("table_compression_vs_alpha.csv", flush=True)

    # Recommendation
    _header("Alpha recommendation")
    # Find α that maximises HV while keeping > 5% harmful for signal
    for prefix in [d[0] for d in DATASETS]:
        ds_rows = metrics_df[metrics_df["dataset"] == prefix]
        if len(ds_rows) == 0:
            continue
        best_hv = ds_rows.loc[ds_rows["hv"].idxmax()]
        best_cr = ds_rows.loc[ds_rows["compression_3pct"].idxmax()]
        print(
            f"  {prefix:25s}: best HV at α={best_hv['alpha']:.2f} "
            f"(HV={best_hv['hv']:.4f}), "
            f"best compression at α={best_cr['alpha']:.2f} "
            f"(CR={best_cr['compression_3pct']:.4f})",
            flush=True,
        )

    _header("Done")
    print(f"Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
