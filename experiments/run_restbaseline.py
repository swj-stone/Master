"""
Run remaining baselines on all five preprocessed datasets (seed=42).

Loads each dataset and computes feature importance with:
  - SHAP-Top-k       (classic TreeExplainer ranking)
  - MOFS-SHAP        (SHAP-guided NSGA-II ablation)
  - NSGA-II-uniform  (no-guidance multi-objective control)

Results saved under ``results/logs/<run_id>/<method>/`` with
the same per-run isolation as ``run_baseline.py``.

Run:
    python -u experiments/run_restbaseline.py
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

from minppi_mofs.baselines.mofs_shap import (
    compute_mofs_shap_importance,
    rank_features as rank_mofs_shap,
)
from minppi_mofs.baselines.nsga2_uniform import (
    compute_nsga2_uniform_importance,
    rank_features as rank_nsga2u,
)
from minppi_mofs.baselines.shap_top_k import (
    compute_shap_topk_importance,
    rank_features as rank_shaptopk,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BASELINES: dict[str, dict] = {
    "shaptopk": {
        "compute": compute_shap_topk_importance,
        "rank": rank_shaptopk,
        "kwargs": {"seed": 42, "verbose": True},
        "score_col": "shap_value",
    },
    "mofs_shap": {
        "compute": compute_mofs_shap_importance,
        "rank": rank_mofs_shap,
        "kwargs": {"n_samples": 100, "seed": 42, "verbose": True},
        "score_col": "mofs_shap_value",
    },
    "nsga2_uniform": {
        "compute": compute_nsga2_uniform_importance,
        "rank": rank_nsga2u,
        "kwargs": {"pop_size": 30, "generations": 15, "seed": 42, "verbose": True},
        "score_col": "nsga2u_freq",
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

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)


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
) -> tuple[pd.DataFrame, np.ndarray]:
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
        print(msg, flush=True)
        print("=" * 72, flush=True)

    _header(f"Remaining baselines — {run_id}")

    # ---- Metadata ----
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "datasets": [d[0] for d in DATASETS],
        "baselines": list(BASELINES.keys()),
        "topk_kpis": list(TOP_K_KPIS),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"metadata.json saved (commit={commit})")

    # ---- config.yaml ----
    run_config: dict = {
        "description": "Remaining baseline comparison run",
        "baselines": {},
        "datasets": [{"prefix": d[0], "target": d[1], "task": d[2]} for d in DATASETS],
        "topk_kpis": list(TOP_K_KPIS),
        "seed": 42,
    }
    for name, c in BASELINES.items():
        run_config["baselines"][name] = {
            "module": c["compute"].__module__,
            "kwargs": c["kwargs"],
            "score_col": c["score_col"],
        }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    print("config.yaml saved")

    # ---- Run ----
    all_rows: list[dict] = []

    for method, cfg in BASELINES.items():
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        m_meta = {
            "method": method,
            "kwargs": cfg["kwargs"],
            "score_col": cfg["score_col"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (method_dir / "metadata.json").write_text(
            json.dumps(m_meta, indent=2), encoding="utf-8"
        )

        compute_fn = cfg["compute"]
        rank_fn = cfg["rank"]
        kwargs = cfg["kwargs"]
        score_col = cfg["score_col"]
        method_metrics: list[dict] = []

        for prefix, target, task in DATASETS:
            _header(f"{method.upper()} — {prefix}")

            X, y = load_data(prefix, target, processed_dir)
            print(f"  Shape: {X.shape}", flush=True)
            print(f"  Classes: {len(np.unique(y))}", flush=True)

            call_kwargs = dict(kwargs)
            t0 = time.perf_counter()
            try:
                scores = compute_fn(X, y, task=task, **call_kwargs)
            except Exception:
                import traceback

                print(f"  FAILED: {method} on {prefix} crashed:", flush=True)
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
                all_rows.append(failed_row)
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

            # CSVs
            imp_path = method_dir / f"{prefix}_importance.csv"
            pd.DataFrame(ranking, columns=["feature", score_col]).to_csv(
                imp_path, index=False
            )
            print(f"  Saved: {imp_path}")

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
            metric_name = kpi_df["metric"].iloc[0]
            for _, kpi_row in kpi_df.iterrows():
                row[f"{metric_name}@{int(kpi_row['k'])}"] = round(
                    float(kpi_row["value"]), 6
                )
            method_metrics.append(row)
            all_rows.append(row)
            print()

        pd.DataFrame(method_metrics).to_csv(method_dir / "metrics.csv", index=False)
        print(f"metrics.csv saved for {method}")

    # ---- Global summary ----
    pd.DataFrame(all_rows).to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nAll done.  Run directory: {run_dir}")

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
