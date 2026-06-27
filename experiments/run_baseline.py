"""
Run baselines on all five preprocessed datasets (seed=42).

Loads each dataset and computes feature importance with:
  - Banzhaf FS  (Karavas et al. 2017)
  - MinShap  (Zheng & Raskutti 2026)
  - ShapG  (Zhao, Liu, Parilina 2025)
  - Powershap (Verhaeghe et al. 2022)

Results saved under ``results/logs/<run_id>/<method>/`` with
per-run isolation:

    config.yaml       — run configuration snapshot
    metadata.json     — seed, git commit, timestamp
    metrics.csv       — per-dataset summary (method, top1, time, …)
    <dataset>_importance.csv  — ranked feature list
    <dataset>_topk_kpi.csv    — top-k performance sweep
    output.log        — full console output

The <run_id> is ``YYYY-MM-DD_HH-MM-SS`` so successive runs never collide.

Run:
    'python -u experiments/run_baseline.py' in terminal
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
from minppi_mofs.baselines.shapg import (
    compute_shapg_importance,
    rank_features as rank_shapg,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BASELINES: dict[str, dict] = {
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

# Per-dataset Banzhaf sampling budget.
#
#   evals_per_feat ≈ (n_coalitions × per_coalition) / p
DATASET_BANZHAF_BUDGET: dict[str, dict] = {
    "gisette": {"per_coalition": 50},
    # spambase(56), madelon(100), isolet(100), credit_card(29) —
    # all <=100 feats after reduction; default per_coalition=10 is fine
    "crimedata": {"per_coalition": 10},
    "abalone": {"per_coalition": 10},
}

# Per-dataset MinShap sampling budget.
#
# MinShap estimates  min_S [v(S∪{i}) - v(S)]  via shared-coalition MC.
# With continuous v(S)=AUC, single-feature marginals are tiny, so each
# feature needs enough evaluations for the min estimate to be meaningful.
# Target: ~15+ evaluations per feature.
#
#   evals_per_feat ≈ (n_coalitions × per_coalition) / p
DATASET_MINSHAP_BUDGET: dict[str, dict] = {
    "spambase": {"n_coalitions": 100, "per_coalition": 5},
    "madelon": {"n_coalitions": 100, "per_coalition": 10},
    "isolet": {"n_coalitions": 100, "per_coalition": 10},
    "gisette": {"n_coalitions": 100, "per_coalition": 15},
    "credit_card_fraud": {"n_coalitions": 100, "per_coalition": 5},
    "crimedata": {"n_coalitions": 100, "per_coalition": 10},
    "abalone": {"n_coalitions": 100, "per_coalition": 5},
}

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)


def _now_stamp() -> str:
    """Return a sortable datetime stamp, e.g. ``2026-05-30_17-05-23``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit(repo_root: Path) -> str:
    """Return the short git commit hash of *repo_root*, or ``"unknown"``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class _Tee:
    """Write to both stdout and a log file."""

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
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the seed-42 train split."""
    path = processed_dir / f"{prefix}_s42_train.csv"
    df = pd.read_csv(path)
    y = df[target].values
    if y.dtype == np.float64 and np.array_equal(y, y.astype(int)):
        y = y.astype(int)
    return df.drop(columns=[target]), y


def evaluate_topk(
    X: pd.DataFrame,
    y: np.ndarray,
    ranking: list[str],
    *,
    task: str,
    ks: tuple[int, ...] = TOP_K_KPIS,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure XGBoost performance using only the top-k ranked features."""
    rows: list[dict] = []
    n_classes = len(np.unique(y))
    for k in ks:
        feats = ranking[: min(k, len(ranking))]
        X_sub = X[feats].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub,
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
        elif n_classes == 2:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)
            val = float(roc_auc_score(y_te, y_prob[:, 1]))
            rows.append(
                {"k": k, "n_features": len(feats), "metric": "AUC", "value": val}
            )
        else:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            val = float(accuracy_score(y_te, y_pred))
            rows.append(
                {"k": k, "n_features": len(feats), "metric": "Accuracy", "value": val}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent  # to_student/ git repo root
    processed_dir = project_root / "data" / "processed"

    run_id = _now_stamp()
    run_dir = project_root / "results" / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout+stderr to output.log
    log_path = run_dir / "output.log"
    tee = _Tee(log_path)
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    # cd to run_dir so CatBoost info lands inside the log directory
    _orig_cwd = os.getcwd()
    os.chdir(str(run_dir))

    commit = _git_commit(git_root)

    def _header(msg: str) -> None:
        print("\n" + "=" * 72, flush=True)
        print(msg, flush=True)
        print("=" * 72, flush=True)

    _header(f"Baseline run — {run_id}")

    # ------------------------------------------------------------------
    # Save top-level metadata
    # ------------------------------------------------------------------
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "datasets": [d[0] for d in DATASETS],
        "baselines": list(BASELINES.keys()),
        "topk_kpis": list(TOP_K_KPIS),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"metadata.json saved (commit={commit})", flush=True)

    # Snapshot the effective run configuration
    run_config: dict = {
        "description": "Baseline comparison run",
        "baselines": {},
        "datasets": [{"prefix": d[0], "target": d[1], "task": d[2]} for d in DATASETS],
        "topk_kpis": list(TOP_K_KPIS),
        "seed": 42,
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
        run_config["baselines"][name] = entry
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    print("config.yaml saved", flush=True)

    # ------------------------------------------------------------------
    # Run each baseline
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    for method, cfg in BASELINES.items():
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Per-method metadata  (can diverge from top-level if needed)
        m_meta = {
            "method": method,
            "kwargs": cfg["kwargs"],
            "score_col": cfg["score_col"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (method_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(m_meta, f, indent=2)

        compute_fn = cfg["compute"]
        rank_fn = cfg["rank"]
        kwargs = cfg["kwargs"]
        score_col = cfg["score_col"]

        method_metrics: list[dict] = []

        for prefix, target, task in DATASETS:
            if prefix in cfg.get("skip_datasets", set()):
                print(f"\n  Skipping {prefix} (excluded for {method})", flush=True)
                continue

            _header(f"{method.upper()} — {prefix}")

            X, y = load_data(prefix, target, processed_dir)
            print(f"  Shape: {X.shape}", flush=True)
            print(f"  Classes: {len(np.unique(y))}", flush=True)

            # Merge dataset-specific params
            call_kwargs = dict(kwargs)
            if method == "banzhaf" and prefix in DATASET_BANZHAF_BUDGET:
                call_kwargs.update(DATASET_BANZHAF_BUDGET[prefix])
            if method == "minshap" and prefix in DATASET_MINSHAP_BUDGET:
                call_kwargs.update(DATASET_MINSHAP_BUDGET[prefix])
                print(
                    f"  MinShap budget: n_coalitions={call_kwargs['n_coalitions']}, "
                    f"per_coalition={call_kwargs['per_coalition']}",
                    flush=True,
                )

            t0 = time.perf_counter()
            try:
                scores = compute_fn(X, y, task=task, **call_kwargs)
            except Exception:
                import traceback

                print(
                    f"  FAILED: {method} on {prefix} crashed:",
                    flush=True,
                )
                traceback.print_exc()
                failed_row: dict = {
                    "method": method,
                    "dataset": prefix,
                    "n_features": 0,
                    "time_s": 0,
                    "top1": "FAILED",
                    "top1_score": 0.0,
                }
                for k in TOP_K_KPIS:
                    failed_row[f"AUC@{k}"] = None
                    failed_row[f"Accuracy@{k}"] = None
                    failed_row[f"R²@{k}"] = None
                method_metrics.append(failed_row)
                all_rows.append(method_metrics[-1])
                print()
                continue
            elapsed = time.perf_counter() - t0

            ranking = rank_fn(scores)
            n_feat = len(ranking)

            print(f"  Time: {elapsed:.1f} s  ({n_feat} features ranked)")
            print("  Top-5 features:")
            for feat, val in ranking[:5]:
                print(f"    {feat:30s}  {val:.6f}")
            print("  Bottom-3 features:")
            for feat, val in ranking[-3:]:
                print(f"    {feat:30s}  {val:.6f}")

            # importance CSV
            imp_path = method_dir / f"{prefix}_importance.csv"
            pd.DataFrame(ranking, columns=["feature", score_col]).to_csv(
                imp_path, index=False
            )
            print(f"  Saved: {imp_path}")

            # top-k KPI CSV
            kpi_df = evaluate_topk(X, y, [f for f, _ in ranking], task=task, seed=42)
            kpi_path = method_dir / f"{prefix}_topk_kpi.csv"
            kpi_df.to_csv(kpi_path, index=False)
            for _, kpi_row in kpi_df.iterrows():
                print(
                    f"  top-{int(kpi_row['k']):>3d}: {kpi_row['metric']}={kpi_row['value']:.4f}"
                )

            row: dict = {
                "method": method,
                "dataset": prefix,
                "n_features": n_feat,
                "time_s": round(elapsed, 1),
                "top1": ranking[0][0],
                "top1_score": round(ranking[0][1], 6),
            }
            # Flatten KPI sweep into the metrics row
            metric_name = kpi_df["metric"].iloc[0]
            for _, kpi_row in kpi_df.iterrows():
                k = int(kpi_row["k"])
                row[f"{metric_name}@{k}"] = round(float(kpi_row["value"]), 6)
            method_metrics.append(row)
            all_rows.append(row)
            print()

        # per-method metrics.csv
        pd.DataFrame(method_metrics).to_csv(method_dir / "metrics.csv", index=False)
        print(f"metrics.csv saved for {method}", flush=True)

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    pd.DataFrame(all_rows).to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nAll done.  Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
