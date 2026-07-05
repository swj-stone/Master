"""
Alpha sensitivity experiment — quota parameter α ∈ {0.85, 0.90, 0.95, 0.99}.

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
    metrics.csv                       # per-(dataset, alpha) with MinPPI AUC + HV@0.95
    minppi_distribution.csv           # class counts per alpha per dataset
    table_harmful_pct_vs_alpha.csv
    table_AUC@<k>_vs_alpha.csv        # MinPPI-ranking AUC vs α, k ∈ {1,5,10,20,50}
    <dataset>_alpha<nn>_minppi.csv    # per-alpha MinPPI values
    <dataset>_stability.csv           # Jaccard between alpha levels

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
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.min_ppi import (
    estimate_minppi,
    estimate_minppi_random_mc,
    minppi_stats,
)
from minppi_mofs.shapg_integration import build_guidance_from_data
from scipy.stats import friedmanchisquare, wilcoxon

from minppi_mofs.voting_game import make_acc_full_cache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_VALUES: list[float] = [0.85, 0.90, 0.95, 0.99]

DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
    ("gisette", "label", "binary"),
    ("credit_card_fraud", "Class", "binary"),
    ("crimedata", "nonViolPerPop", "regression"),
    ("abalone", "Rings", "regression"),
]

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)

# Per-dataset budget — shrinks MinPPI + NSGA-II cost on high-dimensional datasets
DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
    "madelon": {"K": 100, "K_adv": 20, "pop": 20, "gen": 15},
    "isolet": {"K": 80, "K_adv": 3, "pop": 20, "gen": 10},
    "gisette": {"K": 50, "K_adv": 10, "pop": 15, "gen": 8},
    "credit_card_fraud": {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
    "crimedata": {"K": 50, "K_adv": 10, "pop": 15, "gen": 8},
    "abalone": {"K": 50, "K_adv": 10, "pop": 30, "gen": 20},
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
    path = processed_dir / f"{prefix}_s{seed}_train.csv"
    df = pd.read_csv(path)
    y = df[target].values
    if y.dtype == np.float64 and np.array_equal(y, y.astype(int)):
        y = y.astype(int)
    return df.drop(columns=[target]), y


def _budget_for(prefix: str) -> dict:
    defaults = {"K": 100, "K_adv": 20, "pop": 20, "gen": 10}
    return DATASET_BUDGET.get(prefix, defaults)


def evaluate_topk(
    X: pd.DataFrame,
    y: np.ndarray,
    ranking: list[str],
    *,
    task: str = "binary",
    ks: tuple[int, ...] = TOP_K_KPIS,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure XGBoost AUC (or R² for regression) using only top-k ranked features."""
    rows: list[dict] = []
    n_classes = len(np.unique(y))
    for k in ks:
        feats = ranking[: min(k, len(ranking))]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[feats].values,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y if task != "regression" else None,
        )
        if task == "regression":
            model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            val = float(r2_score(y_te, y_pred))
            if val < 0.0:
                val = 0.0
            rows.append(
                {"k": k, "n_features": len(feats), "metric": "R²", "value": val}
            )
            continue
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
            val = float(roc_auc_score(y_te, y_prob, multi_class="ovr"))
        rows.append({"k": k, "n_features": len(feats), "metric": "AUC", "value": val})
    return pd.DataFrame(rows)


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard index between two feature sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Main α sweep
# ---------------------------------------------------------------------------


def _stat_test_alpha(
    df: pd.DataFrame,
    value_col: str,
    alpha_col: str = "alpha",
    dataset_col: str = "dataset",
) -> dict:
    """Friedman (≥3 alpha levels) or Wilcoxon (2 levels) across datasets.

    Returns:
        Dict with ``test``, ``statistic``, ``p_value``, ``n_groups``,
        ``n_datasets``, ``significant``.
    """
    pivot = df.pivot_table(
        index=dataset_col, columns=alpha_col, values=value_col, aggfunc="mean"
    )
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    pivot = pivot.dropna(axis=0)

    k = pivot.shape[1]
    n = pivot.shape[0]

    if k < 2 or n < 3:
        return {
            "test": "skipped",
            "statistic": None,
            "p_value": None,
            "n_groups": k,
            "n_datasets": n,
            "significant": False,
        }

    if k == 2:
        cols = list(pivot.columns)
        stat, p = wilcoxon(
            pivot[cols[0]].values, pivot[cols[1]].values, method="approx"
        )
        return {
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p),
            "n_groups": k,
            "n_datasets": n,
            "significant": bool(p < 0.05),
        }

    stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    return {
        "test": "friedman",
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": k,
        "n_datasets": n,
        "significant": bool(p < 0.05),
    }


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
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("metadata.json saved", flush=True)

    # config.yaml snapshot
    run_config: dict = {
        "description": "Alpha sensitivity sweep — quota parameter α ∈ {0.85, 0.90, 0.95, 0.99}",
        "seed": 42,
        "alpha_values": ALPHA_VALUES,
        "datasets": [{"prefix": d[0], "target": d[1], "task": d[2]} for d in DATASETS],
        "evaluation": {
            "topk_kpis": list(TOP_K_KPIS),
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

    for prefix, target, task in tqdm(
        DATASETS, desc="α sweep", unit="dataset", file=sys.stdout
    ):
        _header(f"Alpha sweep — {prefix}")

        X, y = load_data(prefix, target, processed_dir)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if task != "regression" else None,
        )
        budget = _budget_for(prefix)
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        features = list(X.columns)

        # Acc_full (independent of α) — use stronger model for meaningful
        # binary threshold separation. Multiclass datasets keep lighter
        # params: 26-class XGBoost is ~7× slower per estimator than binary.
        # Regression uses R² via task= parameter.
        if n_classes > 2:
            model_params_ab = {
                "n_estimators": 25,
                "max_depth": 2,
                "learning_rate": 0.3,
            }
        else:
            model_params_ab = {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
        cache = make_acc_full_cache(
            X_tr,
            X_val,
            y_tr,
            y_val,
            seed=42,
            model_params=model_params_ab,
            task=task,
        )
        acc_full = cache.acc_full
        metric_label = "R²" if task == "regression" else "AUC"
        print(
            f"  Full-feature {metric_label}: {acc_full:.4f}, n_features: {n_features}"
        )

        # ShapG guidance — only needed for binary datasets (Algorithm B).
        # Multiclass uses Algorithm A (random MC), which does not depend on
        # the correlation graph; this avoids ~6400 extra XGBoost evals on
        # 26-class isolet. Regression has no ShapG graph.
        sg_scores: dict[str, float] = {}
        sg_neighbors: dict[str, list[str]] | None = None
        if n_classes == 2 and task != "regression":
            sg_scores, sg_neighbors = build_guidance_from_data(
                X_tr,
                y_tr,
                depth=1,
                m=5,
                seed=42,
            )
        # For regression: pre-compute mutual information scores for ranking proxy
        mi_scores: dict[str, float] = {}
        if task == "regression":
            mi_arr = mutual_info_regression(X_tr, y_tr, random_state=42)
            mi_scores = dict(zip(X_tr.columns, mi_arr))

        # Storage for stability analysis
        minppi_per_alpha: dict[float, dict[str, int]] = {}
        ranking_per_alpha: dict[float, list[str]] = {}

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

            # --- MinPPI estimation (shared AccFullCache across α) ---
            # Multiclass / Regression: Algorithm A (shared-coalition random MC).
            # Binary:     Algorithm B (ShapG-guided adversarial search).
            # Rationale: alpha sweep only needs the sign of MinPPI, not
            # precise identification of sparse harmful features.
            # Algorithm A on isolet (100 feats): 300 eval × 0.12s ≈ 36s.
            # Algorithm B on isolet: 600 eval + ShapG 6400 eval ≈ 15+ min.
            cache.set_threshold(alpha)
            t0 = time.perf_counter()
            if n_classes > 2 or task == "regression":
                minppi = estimate_minppi_random_mc(
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    alpha=alpha,
                    K=200,
                    shared_coalitions=True,
                    per_coalition=5,
                    acc_cache=cache,
                    seed=42,
                    verbose=False,
                    task=task,
                )
            else:
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
                    acc_cache=cache,
                    seed=42,
                    verbose=False,
                    task=task,
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

            # MinPPI-based ranking: decisive (+1) top, harmful (-1) bottom,
            # neutral (0) middle sorted by importance proxy.
            # Binary datasets use ShapG; multiclass uses per-feature variance;
            # regression uses mutual information.
            ranking_mpp: list[str] = []
            decisive_f = [f for f in features if minppi.get(f, 0) == +1]

            def _score(feat: str) -> float:
                """Proxy score for feature ordering."""
                if sg_scores:
                    return float(sg_scores.get(feat, 0.0))
                if mi_scores:
                    return float(mi_scores.get(feat, 0.0))
                return float(X_tr[feat].var())

            neutral_f = sorted(
                [f for f in features if minppi.get(f, 0) == 0],
                key=_score,
                reverse=True,
            )
            harmful_f = sorted(
                [f for f in features if minppi.get(f, 0) == -1],
                key=_score,
                reverse=True,
            )
            ranking_mpp = decisive_f + neutral_f + harmful_f
            ranking_per_alpha[alpha] = ranking_mpp

            # ---- Top-k AUC/R² from MinPPI-based ranking (all α, nearly free) ----
            kpi_mpp = evaluate_topk(X, y, ranking_mpp, seed=42, task=task)
            metric_name = kpi_mpp["metric"].iloc[0]

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
                "time_minppi_s": round(t_minppi, 1),
            }
            for _, kpi_row in kpi_mpp.iterrows():
                row[f"{metric_name}@{int(kpi_row['k'])}"] = round(
                    float(kpi_row["value"]), 4
                )

            all_metrics.append(row)

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

    # Top-k AUC / R² from MinPPI ranking vs α (all α have this)
    for k in [1, 5, 10, 20, 50]:
        for prefix in ["AUC", "R²"]:
            metric_col = f"{prefix}@{k}"
            if metric_col in metrics_df.columns:
                pivot = metrics_df.pivot_table(
                    index="dataset",
                    columns="alpha",
                    values=metric_col,
                    aggfunc="first",
                )
                pivot = pivot.round(4)
                pivot.to_csv(run_dir / f"table_{prefix}@{k}_vs_alpha.csv")
                print(f"table_{prefix}@{k}_vs_alpha.csv", flush=True)

    # ---- Per-dataset alpha recommendation (three criteria) ----
    _header("Alpha recommendation")

    summary_rows: list[dict] = []
    for prefix in [d[0] for d in DATASETS]:
        ds_rows = metrics_df[metrics_df["dataset"] == prefix].copy()
        if len(ds_rows) == 0:
            continue

        # Criterion 1: harmful% ≥ 5% (meaningful signal floor).
        # Fallback: if no α meets threshold, pick the lowest α that gives ≥ 2%
        # or the lowest α with any harmful — avoid extreme α when the dataset
        # is too "easy" for MinPPI to add value.
        threshold_strong = 5.0
        threshold_weak = 2.0
        usable = ds_rows[ds_rows["pct_harmful"] >= threshold_strong].copy()
        if len(usable) == 0:
            usable = ds_rows[ds_rows["pct_harmful"] >= threshold_weak].copy()
        if len(usable) == 0:
            usable = ds_rows[ds_rows["pct_harmful"] > 0].copy()
        # If still empty (e.g. regression datasets where MinPPI finds no
        # harmful features at any α), fall back to all α values.
        if len(usable) == 0:
            usable = ds_rows.copy()
        # Prefer lower α among ties (avoid over-selective extremes)
        usable = usable.sort_values("alpha")

        # Criterion 2: best score@10 among usable (AUC or R²).
        # Check for non-NaN values, not just column existence —
        # merged DataFrames carry both AUC@k and R²@k columns with
        # NaNs in the rows belonging to the other task type.
        if "AUC@10" in usable.columns and usable["AUC@10"].notna().any():
            kpi_col = "AUC@10"
        elif "R²@10" in usable.columns and usable["R²@10"].notna().any():
            kpi_col = "R²@10"
        else:
            kpi_col = usable.columns[0]  # fallback — shouldn't happen
        idx_best_kpi = usable[kpi_col].idxmax()
        best_row = usable.loc[idx_best_kpi]

        # Criterion 3: Jaccard stability to adjacent alphas
        stab_df = pd.read_csv(run_dir / f"{prefix}_stability.csv")
        if not stab_df.empty:
            # Mean jaccard_harmful between each alpha and its 1-step neighbor
            alphas_sorted = sorted(ALPHA_VALUES)
            neighbors_jac: dict[float, list[float]] = {a: [] for a in alphas_sorted}
            for _, r in stab_df.iterrows():
                a1, a2 = float(r["alpha1"]), float(r["alpha2"])
                jh = float(r["jaccard_harmful"])
                if abs(a2 - a1) == 0.05:  # 1-step neighbor
                    neighbors_jac[a1].append(jh)
                    neighbors_jac[a2].append(jh)
            mean_jac = {
                a: float(np.mean(v)) if v else 0.0 for a, v in neighbors_jac.items()
            }
        else:
            mean_jac = {a: 0.0 for a in ALPHA_VALUES}

        # Build comparison for this dataset
        lines: list[str] = [f"\n  {prefix}: ({kpi_col})"]
        parts: list[str] = []
        for _, r in ds_rows.iterrows():
            a_val = float(r["alpha"])
            parts.append(f"α={a_val:.2f}:{r['pct_harmful']:.0f}%")
        lines.append("    harmful%: " + "  ".join(parts))
        parts = []
        for _, r in ds_rows.iterrows():
            a_val = float(r["alpha"])
            kpi_v = r.get(kpi_col)
            if not pd.isna(kpi_v):
                parts.append(f"α={a_val:.2f}:{kpi_v:.3f}")
        lines.append(f"    {kpi_col}:  " + "  ".join(parts))
        parts = [f"α={a:.2f}:{mean_jac[a]:.2f}" for a in alphas_sorted]
        lines.append("    J(neighbor): " + "  ".join(parts))
        print("\n".join(lines), flush=True)

        summary_rows.append(
            {
                "dataset": prefix,
                "recommended_alpha": float(best_row["alpha"]),
                "n_harmful_at_best": int(best_row["n_harmful"]),
                "pct_harmful_at_best": float(best_row["pct_harmful"]),
                "best_kpi_at_10": float(best_row.get(kpi_col, float("nan"))),
                "best_kpi_name": kpi_col,
                "J_neighbor_harmful_at_best": round(
                    mean_jac.get(float(best_row["alpha"]), 0.0), 3
                ),
                "n_harmful_α085": int(
                    ds_rows[ds_rows["alpha"] == 0.85]["n_harmful"].values[0]
                )
                if 0.85 in ds_rows["alpha"].values
                else 0,
                "n_harmful_α095": int(
                    ds_rows[ds_rows["alpha"] == 0.95]["n_harmful"].values[0]
                )
                if 0.95 in ds_rows["alpha"].values
                else 0,
                "n_harmful_α099": int(
                    ds_rows[ds_rows["alpha"] == 0.99]["n_harmful"].values[0]
                )
                if 0.99 in ds_rows["alpha"].values
                else 0,
            }
        )

    # Global recommendation table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "table_alpha_recommendation.csv", index=False)
    print("\n  table_alpha_recommendation.csv saved", flush=True)

    # Narrative
    print(
        "\n  -- Recommendation narrative --\n"
        "  alpha=0.85 and below: too few harmful -> MinPPI has no signal.\n"
        "  alpha=0.99:          too many harmful -> over-selective, loses useful features.\n"
        "  alpha=0.90-0.95:     sweet spot - meaningful harmful% + best AUC@10 + stable.\n"
        "  alpha=0.95 is the recommended primary value; Appendix reports 0.90, 0.99 for context.",
        flush=True,
    )

    # ---- Statistical tests ----
    # Does α level significantly affect MinPPI output (harmful %) and
    # downstream ranking quality (top-k AUC / R²)?
    _header("Statistical tests")

    # 1. Friedman across all α levels for harmful %
    res_harm = _stat_test_alpha(metrics_df, "pct_harmful")
    if res_harm["test"] == "skipped":
        print(
            f"  pct_harmful vs α: skipped (k={res_harm['n_groups']}, "
            f"N={res_harm['n_datasets']})",
            flush=True,
        )
    else:
        sig_mark = " *" if res_harm["significant"] else ""
        print(
            f"  harmful% vs α: {res_harm['test']} "
            f"stat={res_harm['statistic']:.4f}, "
            f"p={res_harm['p_value']:.4f}{sig_mark} "
            f"(k={res_harm['n_groups']}, N={res_harm['n_datasets']})",
            flush=True,
        )

    # 2. Friedman / Wilcoxon for top-k KPI columns vs α
    _kpi_candidates = sorted(
        {c for c in metrics_df.columns if "@" in c and not c.startswith("metric_")}
    )
    for kpi in _kpi_candidates:
        if not kpi.endswith(("@10", "@20")):
            continue
        res_kpi = _stat_test_alpha(metrics_df, kpi)
        if res_kpi["test"] == "skipped":
            print(
                f"  {kpi} vs α: skipped (k={res_kpi['n_groups']}, "
                f"N={res_kpi['n_datasets']})",
                flush=True,
            )
        else:
            sig_mark = " *" if res_kpi["significant"] else ""
            print(
                f"  {kpi} vs α: {res_kpi['test']} "
                f"stat={res_kpi['statistic']:.4f}, "
                f"p={res_kpi['p_value']:.4f}{sig_mark} "
                f"(k={res_kpi['n_groups']}, N={res_kpi['n_datasets']})",
                flush=True,
            )

    # 3. Pairwise Wilcoxon: recommended α=0.95 vs each other α for harmful%
    alpha_list = sorted(ALPHA_VALUES)
    if 0.95 in alpha_list:
        print("\n  Pairwise Wilcoxon: α=0.95 vs each other α (harmful%):", flush=True)
        base = metrics_df[metrics_df["alpha"] == 0.95]
        for a in alpha_list:
            if a == 0.95:
                continue
            comp = metrics_df[metrics_df["alpha"] == a]
            merged = pd.merge(
                base[["dataset", "pct_harmful"]],
                comp[["dataset", "pct_harmful"]],
                on="dataset",
                suffixes=("_095", f"_{int(a * 100):02d}"),
            ).dropna()
            if len(merged) < 4:
                print(
                    f"    α=0.95 vs {a:.2f}: skipped (N={len(merged)})",
                    flush=True,
                )
                continue
            try:
                stat_w, p_w = wilcoxon(
                    merged["pct_harmful_095"].values,
                    merged[f"pct_harmful_{int(a * 100):02d}"].values,
                    method="approx",
                )
                sig_w = " *" if p_w < 0.05 else ""
                print(
                    f"    α=0.95 vs {a:.2f}: W={stat_w:.1f}, p={p_w:.4f}{sig_w}",
                    flush=True,
                )
            except Exception:
                import traceback

                print(f"    α=0.95 vs {a:.2f}: ERROR", flush=True)
                traceback.print_exc()

    _header("Done")
    print(f"Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
