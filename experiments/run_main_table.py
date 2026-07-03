"""
Main-table experiment: MOFS-MinPPI + 7 baselines × 7 datasets × 5 seeds.

Methods
-------
  - MOFS-MinPPI      (proposed: MinPPI-guided NSGA-II)
  - SHAP-Top-k       (classic TreeExplainer ranking)
  - MOFS-SHAP        (SHAP-guided NSGA-II ablation)
  - NSGA-II-uniform  (no-guidance multi-objective control)
  - Banzhaf FS       (Karavas et al. 2017)
  - MinShap          (Zheng & Raskutti 2026)
  - ShapG            (Zhao, Liu, Parilina 2025)
  - Powershap        (Verhaeghe et al. 2022)

Metrics
-------
All methods:
  - top-k AUC @ {1, 3, 5, 10, 20, 50, 100}
NSGA-II methods (MOFS-MinPPI, MOFS-SHAP, NSGA2-uniform):
  - Pareto-front hypervolume (HV)
  - Compression rate @ 1% / 3% / 5% AUC-loss
Statistical:
  - Friedman + Nemenyi (all methods × datasets)

Outputs
-------
``results/logs/<run_id>/``   — per-method per-seed details + global metrics.
``results/tables/<run_id>/``  — aggregated tables (mean ± std), pivots, HV, CR.

Run:
    python -u experiments/run_main_table.py
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
import yaml
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.baselines.banzhaf_fs import (
    compute_banzhaf_importance,
    rank_features as rank_banzhaf,
)
from minppi_mofs.baselines.min_shap import (
    compute_minshap_importance,
    rank_features as rank_minshap,
)
from minppi_mofs.baselines.powershap import (
    compute_powershap_importance,
    rank_features as rank_powershap,
)
from minppi_mofs.baselines.shap_top_k import (
    compute_shap_topk_importance,
    rank_features as rank_shaptopk,
)
from minppi_mofs.baselines.shapg import (
    compute_shapg_importance,
    rank_features as rank_shapg,
)
from scipy.stats import friedmanchisquare, wilcoxon


# ---------------------------------------------------------------------------
# Registry — 5 baselines (Plan C: no NSGA-II methods)
# ---------------------------------------------------------------------------

BASELINES: dict[str, dict] = {
    "shaptopk": {
        "compute": compute_shap_topk_importance,
        "rank": rank_shaptopk,
        "kwargs": {"seed": 42, "verbose": True},
        "score_col": "shap_value",
    },
    "banzhaf": {
        "compute": compute_banzhaf_importance,
        "rank": rank_banzhaf,
        "kwargs": {
            "n_coalitions": 300,
            "per_coalition": 10,
            "seed": 42,
            "verbose": True,
        },
        "score_col": "banzhaf_value",
    },
    "minshap": {
        "compute": compute_minshap_importance,
        "rank": rank_minshap,
        "kwargs": {
            "n_coalitions": 300,
            "per_coalition": 5,
            "seed": 42,
            "verbose": True,
        },
        "score_col": "minshap_value",
    },
    "shapg": {
        "compute": compute_shapg_importance,
        "rank": rank_shapg,
        "kwargs": {"depth": 1, "m": 3, "seed": 42, "verbose": True},
        "score_col": "shapg_value",
    },
    "powershap": {
        "compute": compute_powershap_importance,
        "rank": rank_powershap,
        "kwargs": {"power_iterations": 10, "alpha": 0.01, "seed": 42, "verbose": True},
        "score_col": "powershap_value",
    },
}

DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
    ("gisette", "label", "binary"),
    ("credit_card_fraud", "Class", "binary"),
    ("crimedata", "nonViolPerPop", "regression"),
    ("abalone", "Rings", "regression"),
]

DATASET_BANZHAF_BUDGET: dict[str, dict] = {
    "gisette": {"per_coalition": 50},
    "crimedata": {"per_coalition": 10},
    "abalone": {"per_coalition": 10},
}

DATASET_MINSHAP_BUDGET: dict[str, dict] = {
    "spambase": {"n_coalitions": 100, "per_coalition": 5},
    "madelon": {"n_coalitions": 100, "per_coalition": 10},
    "isolet": {"n_coalitions": 100, "per_coalition": 10},
    "gisette": {"n_coalitions": 100, "per_coalition": 15},
    "credit_card_fraud": {"n_coalitions": 100, "per_coalition": 5},
    "crimedata": {"n_coalitions": 100, "per_coalition": 10},
    "abalone": {"n_coalitions": 100, "per_coalition": 5},
}

DATASET_MOFSPPI_BUDGET: dict[str, dict] = {
    "spambase":    {"K_adv": 30, "pop_size": 30, "generations": 20},
    "madelon":     {"K_adv": 20, "pop_size": 20, "generations": 15},
    "isolet":      {"K_adv": 8, "pop_size": 20, "generations": 10},
    "gisette":     {"K_adv": 10, "pop_size": 15, "generations": 8},
    "credit_card_fraud": {"K_adv": 30, "pop_size": 30, "generations": 20},
    "crimedata": {"K_adv": 8, "pop_size": 15, "generations": 8},
    "abalone": {"K_adv": 8, "pop_size": 30, "generations": 20},
}

# NSGA2-uniform shares MOFS-MinPPI's pop/gen per dataset for fair HV comparison
DATASET_NSGA2U_BUDGET: dict[str, dict] = {
    "spambase":    {"pop_size": 30, "generations": 20},
    "madelon":     {"pop_size": 20, "generations": 15},
    "isolet":      {"pop_size": 20, "generations": 10},
    "gisette":     {"pop_size": 15, "generations": 8},
    "credit_card_fraud": {"pop_size": 30, "generations": 20},
    "crimedata": {"pop_size": 15, "generations": 8},
    "abalone": {"pop_size": 30, "generations": 20},
}

# MOFS-SHAP n_samples scaled per dataset
DATASET_MOFSSHAP_BUDGET: dict[str, dict] = {
    "spambase":    {"n_samples": 100},
    "madelon":     {"n_samples": 80},
    "isolet":      {"n_samples": 80},
    "gisette":     {"n_samples": 60},
    "credit_card_fraud": {"n_samples": 100},
    "crimedata": {"n_samples": 60},
    "abalone": {"n_samples": 80},
}

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100)
AUC_LOSS_THRESHOLDS: tuple[float, ...] = (0.01, 0.03, 0.05)

SEEDS: list[int] = [42, 123, 456, 789, 1024]


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
    """Load a preprocessed train split for a given seed.

    Target dtype is preserved from the CSV — ``int`` for classification,
    ``float`` for regression.
    """
    path = processed_dir / f"{prefix}_s{seed}_train.csv"
    df = pd.read_csv(path)
    y = df[target].values
    # Integer labels required by XGBClassifier and for stratify; regression
    # targets are already saved as float in the preprocessed CSVs.
    if y.dtype == np.float64 and np.array_equal(y, y.astype(int)):
        y = y.astype(int)
    return df.drop(columns=[target]), y


def evaluate_topk(
    X: pd.DataFrame,
    y: np.ndarray,
    ranking: list[str],
    *,
    ks: tuple[int, ...] = TOP_K_KPIS,
    seed: int = 42,
    task: str = "binary",
) -> pd.DataFrame:
    rows: list[dict] = []
    n_classes = len(np.unique(y))
    # XGBoost forbids [, ], < in feature names
    safe_cols = {c: f"f{i}" for i, c in enumerate(X.columns)}
    X_safe = X.rename(columns=safe_cols)
    ranking_safe = [safe_cols.get(f, f) for f in ranking]
    for k in ks:
        feats = ranking_safe[: min(k, len(ranking_safe))]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_safe[feats],
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y if task != "regression" else None,
        )
        if task == "regression":
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0, random_state=seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            val = float(r2_score(y_te, y_pred))
            if val < 0.0:
                val = 0.0
            rows.append({"k": k, "n_features": len(feats), "metric": "R²", "value": val})
        else:
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
            rows.append(
                {"k": k, "n_features": len(feats), "metric": "AUC", "value": val}
            )
    return pd.DataFrame(rows)


def refit_pareto_front(
    masks: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    *,
    n_classes: int = 2,
    seed: int = 42,
    task: str = "binary",
) -> np.ndarray:
    """Re-evaluate Pareto masks with strong XGBoost (100 estimators, depth 4).

    All NSGA-II methods share this refit so HV and compression are computed
    on equal footing regardless of the weak model used during search.
    """
    import warnings
    import time as _time

    safe_cols = {c: f"f{i}" for i, c in enumerate(X.columns)}
    X_s = X.rename(columns=safe_cols)
    features_s = [safe_cols.get(f, f) for f in features]
    n_masks = len(masks)
    fitness = np.zeros((n_masks, 3), dtype=float)
    for i in range(n_masks):
        cols = [features_s[j] for j in range(len(features_s)) if masks[i, j]]
        if len(cols) == 0:
            fitness[i] = (0.0, 1.0, 0.0)
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s[cols], y, test_size=0.2, random_state=seed, stratify=y if task != "regression" else None,
        )
        t0 = _time.perf_counter()
        if task == "regression":
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            perf = float(r2_score(y_te, y_pred))
            if perf < 0.0:
                perf = 0.0
        else:
            model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                verbosity=0, random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)
            perf = float(
                roc_auc_score(y_te, y_prob[:, 1])
                if n_classes == 2
                else accuracy_score(y_te, model.predict(X_te))
            )
        elapsed = _time.perf_counter() - t0
        fitness[i] = (float(len(cols)), 1.0 - perf, float(elapsed))
    return fitness


def compression_rate_from_fitness(
    full_auc: float,
    fitness: np.ndarray,
    auc_loss: float,
    n_total: int,
) -> float:
    """Compression ratio (1 − |F|/n) for best solution within AUC-loss budget.

    *fitness* has columns (|F|, 1−AUC, time).  Returns 0 if no solution
    meets the budget.
    """
    threshold = full_auc - auc_loss
    auc_col = 1.0 - fitness[:, 1]
    candidates = np.where(auc_col >= threshold)[0]
    if len(candidates) == 0:
        return 0.0
    best = candidates[np.argmin(fitness[candidates, 0])]
    return 1.0 - float(fitness[best, 0]) / max(1, n_total)


def _build_tables(raw_df: pd.DataFrame, tables_dir: Path) -> None:
    """Build per-seed raw table, aggregated (mean ± std), and per-KPI pivots."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(tables_dir / "main_metrics.csv", index=False)
    print(f"\n  main_metrics.csv  ({len(raw_df)} rows)", flush=True)

    # Separate top-k KPI columns from HV/compression columns
    kpi_cols = [
        c
        for c in raw_df.columns
        if "@" in c and c.split("@")[0] in ("AUC", "R²")
    ]
    hv_cr_cols = [
        c for c in raw_df.columns if c.startswith("hv") or c.startswith("cr_")
    ]
    num_cols = ["n_features", "time_s"] + kpi_cols + hv_cr_cols
    num_cols = [c for c in num_cols if c in raw_df.columns]
    group_cols = ["method", "dataset"]

    # Mean aggregation
    mean_df = raw_df.groupby(group_cols, as_index=False)[num_cols].mean()
    mean_df.to_csv(tables_dir / "mean_metrics.csv", index=False)
    print(f"  mean_metrics.csv  ({len(mean_df)} rows)", flush=True)

    # Mean ± std aggregation
    agg_mean = raw_df.groupby(group_cols)[num_cols].mean()
    agg_std = raw_df.groupby(group_cols)[num_cols].std().fillna(0)
    combined = pd.DataFrame(index=agg_mean.index)
    for col in num_cols:
        combined[col] = agg_mean[col].round(4)
        combined[f"{col}_std"] = agg_std[col].round(4)
    combined = combined.reset_index()
    combined.to_csv(tables_dir / "aggregated_metrics.csv", index=False)
    print(f"  aggregated_metrics.csv  ({len(combined)} rows)", flush=True)

    # Per-KPI pivot tables (dataset × method, mean values)
    for kpi_col in kpi_cols:
        pivot = mean_df.pivot_table(
            index="dataset",
            columns="method",
            values=kpi_col,
            aggfunc="first",
        )
        metric, k = kpi_col.split("@")
        pivot = pivot.round(4)
        pivot_sorted = pivot.reindex(columns=sorted(pivot.columns))
        pivot_sorted.to_csv(tables_dir / f"table_{metric}_at_{k}.csv")
        print(f"  table_{metric}_at_{k}.csv", flush=True)

    # HV / compression pivot tables (NSGA-II methods only)
    for cr_col in hv_cr_cols:
        if cr_col in mean_df.columns:
            pivot = mean_df.pivot_table(
                index="dataset",
                columns="method",
                values=cr_col,
                aggfunc="first",
            )
            pivot = pivot.round(4)
            pivot_sorted = pivot.reindex(columns=sorted(pivot.columns))
            short = cr_col.replace("compression", "cr").replace("_pct", "pct")
            pivot_sorted.to_csv(tables_dir / f"table_{short}.csv")
            print(f"  table_{short}.csv", flush=True)


def _stat_test(
    df: pd.DataFrame,
    value_col: str,
    method_col: str = "method",
    dataset_col: str = "dataset",
) -> dict:
    """Friedman (≥3 groups) or Wilcoxon (2 groups) with automatic dispatch.

    Mixed-task DataFrames may have NaN values in *value_col* for datasets
    where the metric is not applicable (e.g. AUC@k is NaN for regression
    datasets).  The function drops rows and columns that are entirely NaN
    before testing.

    Returns:
        Dict with ``test``, ``statistic``, ``p_value``, ``n_groups``,
        ``n_datasets``, ``significant``.
    """
    # Aggregate: mean over seeds per (method, dataset)
    agg = df.groupby([method_col, dataset_col])[value_col].mean().reset_index()
    pivot = agg.pivot(index=dataset_col, columns=method_col, values=value_col)
    # Drop datasets that have no data for this metric at all
    pivot = pivot.dropna(axis=0, how="all")
    # Drop methods that have no data for any dataset
    pivot = pivot.dropna(axis=1, how="all")
    # Now drop any rows with partial NaN (some methods missing)
    pivot = pivot.dropna(axis=0)

    k = pivot.shape[1]  # number of groups (methods)
    n = pivot.shape[0]  # number of datasets

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
        # Wilcoxon signed-rank (paired by dataset)
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

    # k ≥ 3 → Friedman
    stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    return {
        "test": "friedman",
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": k,
        "n_datasets": n,
        "significant": bool(p < 0.05),
    }


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
    tables_dir = project_root / "results" / "tables" / run_id

    log_path = run_dir / "output.log"
    tee = _Tee(log_path)
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    _orig_cwd = os.getcwd()
    os.chdir(str(run_dir))

    commit = _git_commit(git_root)

    def _header(msg: str) -> None:
        print("\n" + "=" * 72, flush=True)
        print(msg, flush=True)
        print("=" * 72, flush=True)

    _header(f"Main Table — {run_id}")

    # ---- Metadata ----
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS,
        "datasets": [d[0] for d in DATASETS],
        "methods": list(BASELINES.keys()),
        "nsga2_methods": [],
        "topk_kpis": list(TOP_K_KPIS),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("metadata.json saved")

    # ---- config.yaml ----
    run_config: dict = {
        "description": "Main table — MOFS-MinPPI + 7 baselines × 7 datasets × 5 seeds",
        "baselines": {},
        "seeds": SEEDS,
        "datasets": [
            {"prefix": d[0], "target": d[1], "task": d[2]} for d in DATASETS
        ],
        "topk_kpis": list(TOP_K_KPIS),
        "nsga2_metrics": ["hv", "cr_1pct", "cr_3pct", "cr_5pct"],
    }
    for name, c in BASELINES.items():
        entry: dict = {
            "module": c["compute"].__module__,
            "kwargs": c["kwargs"],
            "score_col": c["score_col"],
        }
        if name == "banzhaf":
            entry["dataset_budget"] = DATASET_BANZHAF_BUDGET
        if name == "minshap":
            entry["dataset_budget"] = DATASET_MINSHAP_BUDGET
        # Plan C: mofs_minppi / nsga2_uniform / mofs_shap removed
        run_config["baselines"][name] = entry
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    print("config.yaml saved")

    # ---- Run all methods × seeds ----
    all_rows: list[dict] = []

    for method, cfg in BASELINES.items():
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "kwargs": cfg["kwargs"],
                    "score_col": cfg["score_col"],
                    "nsga2": cfg.get("nsga2", False),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        compute_fn = cfg["compute"]
        rank_fn = cfg["rank"]
        base_kwargs = cfg["kwargs"]
        score_col = cfg["score_col"]
        is_nsga2 = cfg.get("nsga2", False)
        method_metrics: list[dict] = []

        for prefix, target, task in DATASETS:
            if prefix in cfg.get("skip_datasets", set()):
                print(f"\n  Skipping {prefix} (excluded for {method})", flush=True)
                continue

            for seed in SEEDS:
                _header(f"{method.upper()} — {prefix}  seed={seed}")

                X, y = load_data(prefix, target, processed_dir, seed=seed)
                n_total = X.shape[1]
                n_classes = len(np.unique(y))
                print(f"  Shape: {X.shape}", flush=True)
                print(f"  Classes: {len(np.unique(y))}", flush=True)

                call_kwargs = dict(base_kwargs, seed=seed)
                if method == "banzhaf" and prefix in DATASET_BANZHAF_BUDGET:
                    call_kwargs.update(DATASET_BANZHAF_BUDGET[prefix])
                if method == "minshap" and prefix in DATASET_MINSHAP_BUDGET:
                    call_kwargs.update(DATASET_MINSHAP_BUDGET[prefix])
                if method == "mofs_minppi" and prefix in DATASET_MOFSPPI_BUDGET:
                    call_kwargs.update(DATASET_MOFSPPI_BUDGET[prefix])
                if method == "nsga2_uniform" and prefix in DATASET_NSGA2U_BUDGET:
                    call_kwargs.update(DATASET_NSGA2U_BUDGET[prefix])
                if method == "mofs_shap" and prefix in DATASET_MOFSSHAP_BUDGET:
                    call_kwargs.update(DATASET_MOFSSHAP_BUDGET[prefix])

                call_kwargs["task"] = task

                if method == "mofs_minppi" and task == "regression":
                    call_kwargs["verbose"] = False  # suppress verbose for regression run

                # ---- NSGA-II methods: capture Pareto front via side-channel ----
                result_box: list = []
                if is_nsga2:
                    call_kwargs["_result_box"] = result_box

                t0 = time.perf_counter()
                try:
                    scores = compute_fn(X, y, **call_kwargs)
                except Exception:
                    import traceback

                    print(f"  FAILED: {method}/{prefix}/seed={seed}", flush=True)
                    traceback.print_exc()
                    failed: dict = {
                        "method": method,
                        "dataset": prefix,
                        "seed": seed,
                        "n_features": 0,
                        "time_s": 0,
                        "top1": "FAILED",
                        "top1_score": 0.0,
                    }
                    _metric = "R²" if task == "regression" else "AUC"
                    for k in TOP_K_KPIS:
                        failed[f"{_metric}@{k}"] = None
                    if is_nsga2:
                        failed["hv"] = None
                        for loss in AUC_LOSS_THRESHOLDS:
                            failed[f"cr_{int(loss * 100):d}pct"] = None
                    method_metrics.append(failed)
                    all_rows.append(failed)
                    print()
                    continue
                elapsed = time.perf_counter() - t0

                ranking = rank_fn(scores)
                n_feat = len(ranking)
                print(f"  Time: {elapsed:.1f} s  ({n_feat} features ranked)")
                print("  Top-5 features:")
                for feat, val in ranking[:5]:
                    print(f"    {feat:30s}  {val:.6f}")

                imp_path = method_dir / f"{prefix}_s{seed}_importance.csv"
                pd.DataFrame(ranking, columns=["feature", score_col]).to_csv(
                    imp_path, index=False
                )

                kpi_df = evaluate_topk(X, y, [f for f, _ in ranking], seed=seed, task=task)
                kpi_path = method_dir / f"{prefix}_s{seed}_topk_kpi.csv"
                kpi_df.to_csv(kpi_path, index=False)
                for _, kpi_row in kpi_df.iterrows():
                    print(
                        f"  top-{int(kpi_row['k']):>3d}: "
                        f"{kpi_row['metric']}={kpi_row['value']:.4f}"
                    )

                row: dict = {
                    "method": method,
                    "dataset": prefix,
                    "seed": seed,
                    "n_features": n_total,
                    "time_s": round(elapsed, 1),
                    "top1": ranking[0][0],
                    "top1_score": round(ranking[0][1], 6),
                }
                metric_name = kpi_df["metric"].iloc[0]
                for _, kpi_row in kpi_df.iterrows():
                    row[f"{metric_name}@{int(kpi_row['k'])}"] = round(
                        float(kpi_row["value"]), 6
                    )

                # ---- NSGA-II: HV + compression rate (strong-model refit) ----
                if is_nsga2 and result_box:
                    pf_data = result_box[0]
                    pf_fitness = pf_data["fitness"]

                    # MOFS-MinPPI refits inside its wrapper; the other two use
                    # weak models during search — refit here for fair comparison.
                    if method != "mofs_minppi":
                        pf_fitness = refit_pareto_front(
                            pf_data["masks"], X, y,
                            features=list(X.columns),
                            n_classes=n_classes,
                            seed=seed,
                            task=task,
                        )

                    # Save Pareto front (refit fitness) for post-hoc
                    # IGD/IGD+/Spacing via compute_pareto_metrics.py
                    np.savetxt(
                        method_dir / f"{prefix}_s{seed}_pareto.csv",
                        pf_fitness,
                        delimiter=",",
                        header="n_features,one_minus_quality,time_s",
                        comments="",
                    )

                    # HV — not computed (Plan C: no NSGA-II methods)

                    # Compression rate — full-model baseline
                    safe_cols = {c: f"f{i}" for i, c in enumerate(X.columns)}
                    X_safe = X.rename(columns=safe_cols)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_safe, y, test_size=0.2, random_state=seed, stratify=y if task != "regression" else None,
                    )
                    if task == "regression":
                        full_model = XGBRegressor(
                            n_estimators=100, max_depth=4, learning_rate=0.1,
                            verbosity=0, random_state=seed,
                        )
                        full_model.fit(X_tr, y_tr)
                        y_pred = full_model.predict(X_te)
                        full_metric = float(r2_score(y_te, y_pred))
                        if full_metric < 0.0:
                            full_metric = 0.0
                    else:
                        full_model = XGBClassifier(
                            n_estimators=100, max_depth=4, learning_rate=0.1,
                            verbosity=0, random_state=seed,
                        )
                        full_model.fit(X_tr, y_tr)
                        full_prob = full_model.predict_proba(X_te)
                        full_metric = float(
                            roc_auc_score(y_te, full_prob[:, 1])
                            if n_classes == 2
                            else accuracy_score(y_te, full_model.predict(X_te))
                        )
                    for loss in AUC_LOSS_THRESHOLDS:
                        cr = compression_rate_from_fitness(
                            full_metric, pf_fitness, loss, n_total,
                        )
                        row[f"cr_{int(loss * 100):d}pct"] = round(cr, 4)
                    print(
                        "  "  + "  ".join(
                            f"CR@{int(loss * 100)}%={row[f'cr_{int(loss * 100):d}pct']:.3f}"
                            for loss in AUC_LOSS_THRESHOLDS
                        ),
                        flush=True,
                    )

                method_metrics.append(row)
                all_rows.append(row)
                print()

        pd.DataFrame(method_metrics).to_csv(method_dir / "metrics.csv", index=False)
        print(f"metrics.csv saved for {method}")

    # ---- Global summary ----
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(run_dir / "metrics.csv", index=False)

    # ---- Tables ----
    _header("Building tables")
    _build_tables(raw_df, tables_dir)

    # ---- Statistical tests ----
    # Run Friedman (≥3 methods) or Wilcoxon (2 methods) on every available
    # top-10 and top-20 KPI column.  Classification datasets contribute
    # AUC@k; regression datasets contribute R²@k.  Mixed-task columns are
    # handled by dropna(how='all') inside _stat_test.
    _header("Statistical tests")
    fn_results: list[dict] = []
    _kpi_candidates = sorted({c for c in raw_df.columns if "@" in c})
    for kpi in _kpi_candidates:
        if not kpi.endswith(("@10", "@20")):
            continue
        res = _stat_test(raw_df, kpi)
        fn_results.append({"kpi": kpi, **res})
        if res["test"] == "skipped":
            print(
                f"  {kpi}: skipped — k={res['n_groups']}, N={res['n_datasets']}",
                flush=True,
            )
        else:
            p_val = res["p_value"]
            sig = "***" if p_val < 0.001 else (
                "**" if p_val < 0.01 else (
                    "*" if p_val < 0.05 else "n.s."
                )
            )
            sig_mark = f" {sig}" if res["significant"] else ""
            print(
                f"  {kpi}: {res['test']} "
                f"stat={res['statistic']:.4f}, "
                f"p={p_val:.4f}{sig_mark} "
                f"(k={res['n_groups']}, N={res['n_datasets']})",
                flush=True,
            )
    if fn_results:
        pd.DataFrame(fn_results).to_csv(tables_dir / "friedman_tests.csv", index=False)
        print("  friedman_tests.csv saved", flush=True)

    _header("Done")
    print(f"Logs:    {run_dir}")
    print(f"Tables:  {tables_dir}")

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
