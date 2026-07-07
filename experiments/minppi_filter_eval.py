"""
MinPPI filter verification — safety + one downstream baseline (§6 pipeline).

Fast redesign (2026-07-07): instead of re-running 5 expensive baselines on
filtered data (which ``run_main_table.py`` already covers on full data),
this script does a single XGBoost retrain on filtered features, extracts
feature-importance for free, and evaluates top-k AUC.  Total cost per
(dataset, seed) = MinPPI + 2 model fits.

Pipeline
--------
1. Compute MinPPI at α=0.95 → identify harmful features.
2. Delete harmful features from train/val/test.
3. Train XGBoost on filtered train set → AUC_filtered on test.
4. Extract ``feature_importances_`` (gain) from the model → rank features.
5. Evaluate top-k AUC of the ranking on a held-out split of the test set.

Output (results/tables/<run_id>/)::

    filter_verify.csv     # per-(dataset, seed): n_harmful, AUC_full, AUC_filtered, AUC@10, …
    filter_verify_agg.csv # mean ± std over seeds

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
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi, estimate_minppi_random_mc
from minppi_mofs.shapg_integration import build_guidance_from_data

# ============================================================
# Constants
# ============================================================

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
    "isolet": {"K": 30, "per_coalition": 2},
    "gisette": {"K_adv": 10},
    "credit_card_fraud": {"K_adv": 30},
    "crimedata": {"K": 40, "per_coalition": 2},
    "abalone": {"K": 40, "per_coalition": 2},
}

# ============================================================
# Helpers
# ============================================================


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


def train_eval_model(
    X_tr,
    y_tr,
    X_te,
    y_te,
    *,
    task="binary",
    seed=42,
    n_classes=None,
):
    """Train one XGBoost and return (utility, feature_importances_ dict)."""
    if n_classes is None:
        n_classes = len(np.unique(y_te))
    if task == "regression":
        m = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
        m.fit(X_tr.values, y_tr)
        y_pred = m.predict(X_te.values)
        val = float(r2_score(y_te, y_pred))
        if val < 0.0:
            val = 0.0
    else:
        m = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
        m.fit(X_tr.values, y_tr)
        y_prob = m.predict_proba(X_te.values)
        if n_classes == 2:
            val = float(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            val = float(accuracy_score(y_te, m.predict(X_te.values)))
    imp = dict(zip(X_tr.columns, m.feature_importances_.astype(float)))
    return val, imp


def eval_topk_from_ranking(X_test, y_test, ranking, *, task="binary", seed=42):
    """Evaluate top-k AUC / R² for a given feature ranking."""
    n_classes = len(np.unique(y_test))
    rows = []
    for k in TOP_K:
        feats = ranking[: min(k, len(ranking))]
        if not feats:
            continue
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
            v = float(r2_score(y_te, m.predict(X_te)))
            if v < 0.0:
                v = 0.0
            rows.append({"k": k, "metric": "R2", "value": round(v, 4)})
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
                v = float(roc_auc_score(y_te, proba[:, 1]))
            else:
                v = float(accuracy_score(y_te, m.predict(X_te)))
            rows.append({"k": k, "metric": "AUC", "value": round(v, 4)})
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================


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

    all_rows: list[dict] = []

    for prefix, target, task in tqdm(
        DATASETS,
        desc="filter-verify",
        unit="ds",
        file=sys.stdout,
    ):
        for seed in tqdm(
            SEEDS,
            desc=f"  seeds ({prefix})",
            unit="sd",
            file=sys.stdout,
            leave=False,
        ):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te = load_split(
                    prefix,
                    target,
                    processed_dir,
                    seed=seed,
                )
            except FileNotFoundError:
                print(f"\n  SKIP {prefix} s{seed}: missing split")
                continue

            n_full = X_tr.shape[1]
            n_cls = len(np.unique(y_val))
            bud = DATASET_BUDGET.get(prefix, {"K_adv": 20})
            metric_label = "R2" if task == "regression" else "AUC"

            # ── Stage 1: MinPPI ──────────────────────────────────────
            t0 = time.perf_counter()
            if n_cls > 2 or task == "regression":
                minppi = estimate_minppi_random_mc(
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    alpha=ALPHA,
                    K=bud.get("K", 200),
                    shared_coalitions=True,
                    per_coalition=bud.get("per_coalition", 5),
                    seed=seed,
                    verbose=False,
                    task=task,
                )
            else:
                sg_s, sg_n = build_guidance_from_data(
                    X_tr,
                    y_tr,
                    depth=1,
                    m=5,
                    seed=seed,
                )
                minppi = estimate_minppi(
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    alpha=ALPHA,
                    method="shapg_guided",
                    K_adv=bud.get("K_adv", bud.get("K", 20)),
                    shapg_neighbors=sg_n,
                    shapg_scores=sg_s,
                    seed=seed,
                    verbose=False,
                    task=task,
                )
            harmful = {f for f, v in minppi.items() if v == -1}
            t_ppi = time.perf_counter() - t0
            n_h = len(harmful)
            cr = n_h / n_full if n_full else 0.0
            keep = [c for c in X_tr.columns if c not in harmful]

            print(
                f"\n  {prefix} s{seed}: {n_h} harmful → {len(keep)} remain "
                f"({cr * 100:.1f}%)  [{t_ppi:.0f}s]",
            )

            # ── Stage 2: train on full & filtered ────────────────────
            # Full-feature model
            auc_full, _ = train_eval_model(
                X_tr,
                y_tr,
                X_te,
                y_te,
                task=task,
                seed=seed,
            )

            all_rows.append(
                {
                    "dataset": prefix,
                    "seed": seed,
                    "stage": "full",
                    "method": "xgb",
                    "n_features": n_full,
                    "n_full": n_full,
                    "compression": 0.0,
                    "time_s": round(t_ppi, 1),
                    metric_label: round(auc_full, 4),
                }
            )

            if n_h == 0:
                print("    (0 harmful — filtered = full, skipping retrain)")
                continue

            # Filtered-feature model
            X_tr_f, X_te_f = X_tr[keep], X_te[keep]
            auc_filt, imp_filt = train_eval_model(
                X_tr_f,
                y_tr,
                X_te_f,
                y_te,
                task=task,
                seed=seed,
            )
            ranking = sorted(imp_filt, key=imp_filt.get, reverse=True)  # type: ignore[arg-type]

            all_rows.append(
                {
                    "dataset": prefix,
                    "seed": seed,
                    "stage": "filtered",
                    "method": "xgb",
                    "n_features": len(keep),
                    "n_full": n_full,
                    "compression": round(cr, 4),
                    "time_s": round(t_ppi, 1),
                    metric_label: round(auc_filt, 4),
                }
            )

            # ── Stage 3: top-k from feature_importances_ (free) ──────
            kpi = eval_topk_from_ranking(
                X_te_f,
                y_te,
                ranking,
                task=task,
                seed=seed,
            )
            for _, r in kpi.iterrows():
                all_rows.append(
                    {
                        "dataset": prefix,
                        "seed": seed,
                        "stage": "filtered",
                        "method": "xgb-gain",
                        "n_features": len(keep),
                        "n_full": n_full,
                        "compression": round(cr, 4),
                        f"{r['metric']}@{int(r['k'])}": r["value"],
                    }
                )

            k10 = kpi[kpi["k"] == 10]
            if len(k10):
                print(
                    f"    filtered {metric_label}={auc_filt:.4f}  "
                    f"Δ={auc_filt - auc_full:+.4f}  "
                    f"{kpi['metric'].iloc[0]}@10={k10['value'].iloc[0]:.4f}",
                )

    # ── Save ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "filter_verify.csv", index=False)
    print(f"\n  filter_verify.csv saved ({len(df)} rows)")

    num_cols = [c for c in df.columns if "@" in c or c in ("compression", "AUC", "R2")]
    if num_cols:
        agg = (
            df.groupby(["dataset", "stage", "method"])[num_cols]
            .agg(["mean", "std"])
            .round(4)
        )
        agg.to_csv(out_dir / "filter_verify_agg.csv")
        print("  filter_verify_agg.csv saved")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
