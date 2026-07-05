"""
MinPPI as a safe preprocessing filter — Stage 1+2 pipeline evaluation (P0.1).

Tests the Plan C hypothesis: MinPPI's core value is as a *safe deletion
preprocessor*.  Any downstream FS method benefits from having harmful
features removed first — faster training (fewer features) at no AUC cost.

Pipeline
--------
Stage 1 (MinPPI filter):
  - Estimate MinPPI at α=0.95 (Algorithm B for binary, A for multi/reg).
  - Identify harmful features (MinPPI = −1) → delete from train/val/test.
Stage 2 (downstream FS):
  - Run each baseline on the *filtered* feature set.
  - Measure top-k AUC (or R² for regression).
  - Compare against the same baseline on the original full feature set.

Output (results/tables/<run_id>/)::

    minppi_filter_pipeline.csv     # per-(dataset, seed, method)
    minppi_filter_pipeline_agg.csv # mean over seeds

Run::

    python -u experiments/minppi_filter_eval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi, estimate_minppi_random_mc
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
from minppi_mofs.baselines.shapg import compute_shapg_importance
from minppi_mofs.baselines.min_shap import compute_minshap_importance
from minppi_mofs.baselines.banzhaf_fs import compute_banzhaf_importance
from minppi_mofs.baselines.powershap import compute_powershap_importance


ALPHA = 0.95
SEEDS: list[int] = [42, 123, 456, 789, 1024]
TOP_K: tuple[int, ...] = (1, 3, 5, 10, 20, 50)

DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
    ("gisette", "label", "binary"),
    ("credit_card_fraud", "Class", "binary"),
    ("crimedata", "nonViolPerPop", "regression"),
    ("abalone", "Rings", "regression"),
]

DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K_adv": 30},
    "madelon": {"K_adv": 20},
    "isolet": {"K_adv": 8},
    "gisette": {"K_adv": 10},
    "credit_card_fraud": {"K_adv": 30},
    "crimedata": {"K_adv": 10},
    "abalone": {"K_adv": 10},
}


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
        self._console, self._file = sys.stdout, log_path.open("w", encoding="utf-8")

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


def load_split(prefix, target, processed_dir, *, seed):
    train = pd.read_csv(processed_dir / f"{prefix}_s{seed}_train.csv")
    val = pd.read_csv(processed_dir / f"{prefix}_s{seed}_val.csv")
    test = pd.read_csv(processed_dir / f"{prefix}_s{seed}_test.csv")
    return (
        train.drop(columns=[target]),
        train[target].values,
        val.drop(columns=[target]),
        val[target].values,
        test.drop(columns=[target]),
        test[target].values,
    )


def baseline_ranking(X, y, method, *, task="binary", seed=42):
    if method == "shaptopk":
        scores = compute_shap_topk_importance(X, y, seed=seed, task=task)
    elif method == "shapg":
        scores = compute_shapg_importance(X, y, seed=seed)
    elif method == "minshap":
        scores = compute_minshap_importance(
            X, y, n_coalitions=100, per_coalition=5, seed=seed, task=task
        )
    elif method == "banzhaf":
        scores = compute_banzhaf_importance(X, y, per_coalition=5, seed=seed, task=task)
    elif method == "powershap":
        scores = compute_powershap_importance(X, y, seed=seed, task=task)
    else:
        raise ValueError(f"Unknown: {method}")
    return [f for f, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def evaluate_topk(X_test, y_test, ranking, *, task="binary", seed=42):
    n_classes = len(np.unique(y_test))
    rows = []
    for k in TOP_K:
        feats = ranking[: min(k, len(ranking))]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_test[feats].values,
            y_test,
            test_size=0.2,
            random_state=seed,
            stratify=y_test if task != "regression" else None,
        )
        if task == "regression":
            m = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            m.fit(X_tr, y_tr)
            val = float(r2_score(y_te, m.predict(X_te)))
            if val < 0.0:
                val = 0.0
            rows.append({"k": k, "metric": "R2", "value": round(val, 4)})
        else:
            m = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)
            if n_classes == 2:
                val = float(roc_auc_score(y_te, proba[:, 1]))
            else:
                val = float(roc_auc_score(y_te, proba, multi_class="ovr"))
            rows.append({"k": k, "metric": "AUC", "value": round(val, 4)})
    return pd.DataFrame(rows)


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    run_id = _now_stamp()
    out_dir = project_root / "results" / "tables" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit(project_root.parent)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {"run_id": run_id, "commit": commit, "alpha": ALPHA, "seeds": SEEDS},
            indent=2,
        ),
        encoding="utf-8",
    )

    all_rows = []
    for prefix, target, task in tqdm(
        DATASETS, desc="filter-pipe", unit="ds", file=sys.stdout
    ):
        for seed in tqdm(
            SEEDS, desc=f"  seeds ({prefix})", unit="sd", file=sys.stdout, leave=False
        ):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te = load_split(
                    prefix, target, processed_dir, seed=seed
                )
            except FileNotFoundError:
                print(f"\n  SKIP {prefix} s{seed}: missing split")
                continue

            n_full = X_tr.shape[1]
            n_cls = len(np.unique(y_val))
            bud = DATASET_BUDGET.get(prefix, {"K_adv": 20})

            # Stage 1 — MinPPI filter
            t0 = time.perf_counter()
            if n_cls > 2 or task == "regression":
                minppi = estimate_minppi_random_mc(
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    alpha=ALPHA,
                    K=200,
                    shared_coalitions=True,
                    per_coalition=5,
                    seed=seed,
                    verbose=False,
                    task=task,
                )
            else:
                sg_s, sg_n = build_guidance_from_data(
                    X_tr, y_tr, depth=1, m=5, seed=seed
                )
                minppi = estimate_minppi(
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    alpha=ALPHA,
                    method="shapg_guided",
                    K_adv=bud["K_adv"],
                    shapg_neighbors=sg_n,
                    shapg_scores=sg_s,
                    seed=seed,
                    verbose=False,
                    task=task,
                )
            harmful = {f for f, v in minppi.items() if v == -1}
            t_ppi = time.perf_counter() - t0
            n_h = len(harmful)
            cr = n_h / n_full if n_full else 0
            keep = [c for c in X_tr.columns if c not in harmful]

            all_rows.append(
                {
                    "dataset": prefix,
                    "seed": seed,
                    "stage": "filter",
                    "method": "minppi",
                    "n_features": n_h,
                    "n_full": n_full,
                    "compression": round(cr, 4),
                    "time_s": round(t_ppi, 1),
                }
            )
            print(
                f"\n  {prefix} s{seed}: {n_h} harmful removed → {len(keep)} remain ({cr * 100:.1f}%)"
            )

            if not keep:
                print("    Stage 2 SKIP (0 features left)")
                continue

            X_tr_f = X_tr[keep] if n_h > 0 else X_tr
            X_te_f = X_te[keep] if n_h > 0 else X_te

            # Stage 2 — baselines on filtered features
            for mname in ["shaptopk", "shapg", "minshap", "banzhaf", "powershap"]:
                try:
                    rank = baseline_ranking(X_tr_f, y_tr, mname, task=task, seed=seed)
                except Exception as e:
                    print(f"      {mname}: FAILED ({e})")
                    continue
                kpi = evaluate_topk(X_te_f, y_te, rank, task=task, seed=seed)
                for _, r in kpi.iterrows():
                    all_rows.append(
                        {
                            "dataset": prefix,
                            "seed": seed,
                            "stage": "filtered",
                            "method": mname,
                            "n_features": len(keep),
                            "n_full": n_full,
                            "compression": round(cr, 4),
                            f"{r['metric']}@{int(r['k'])}": r["value"],
                        }
                    )
                k10 = kpi[kpi["k"] == 10]
                if len(k10):
                    print(
                        f"      {mname}: {kpi['metric'].iloc[0]}@10={k10['value'].iloc[0]:.4f}"
                    )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "minppi_filter_pipeline.csv", index=False)
    print(f"\n  minppi_filter_pipeline.csv saved ({len(df)} rows)")

    num_cols = [
        c for c in df.columns if "@" in c or c == "compression" or c == "time_s"
    ]
    agg = (
        df.groupby(["dataset", "stage", "method"])[num_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(out_dir / "minppi_filter_pipeline_agg.csv")
    print("  minppi_filter_pipeline_agg.csv saved")

    # ΔAUC@10 vs full baseline
    full_csv = (
        project_root / "results" / "tables" / "2026-06-28_07-44-10" / "main_metrics.csv"
    )
    if full_csv.exists():
        full = pd.read_csv(full_csv)
        fa = full.groupby(["dataset", "method"])["AUC@10"].mean()
        print("\n=== ΔAUC@10 (filtered − full) ===")
        for ds in sorted(df["dataset"].unique()):
            sub = df[(df["dataset"] == ds) & (df["stage"] == "filtered")]
            for mn in sub["method"].unique():
                fv = sub[sub["method"] == mn]["AUC@10"].mean()
                try:
                    bv = fa.loc[ds, mn]
                    print(f"  {ds:20s} {mn:12s}: {fv - bv:+.4f}")
                except KeyError:
                    pass

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
