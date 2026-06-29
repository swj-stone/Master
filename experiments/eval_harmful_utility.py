"""
Harmful-feature utility evaluation — c1 removal test, c2 ground-truth recovery,
and #8 MinShap-vs-MinPPI comparison for the ICOMP 2026 §6.

Three evaluation blocks
-----------------------
c1  — **Removal test** (real-data ΔAUC / ΔR² on held-out test):
      baseline → remove MinPPI −1 features → ΔAUC and |F| compression.
      Controls: mean-SHAP bottom-*k*, MinShap negative, random (10 trials).
c2  — **Ground-truth recovery** on Madelon (NIPS 2003 probe features):
      MinPPI / MinShap / SHAP vs known noise-labels → precision, recall, F₁.
#8  — **MinShap-vs-MinPPI agreement**: per-dataset comparison of the
      harmful-feature sets identified by each method.

Datasets
--------
c1 classification: spambase, madelon, isolet
c1 regression:     crimedata, abalone
c2:                madelon only (唯一有 ground-truth 的数据集)
#8:                spambase, madelon, isolet (α = 0.95)

Output (``results/logs/<run_id>/``)::

    metadata.json
    config.yaml
    output.log
    c1_removal_{cls,reg}.csv          # per-seed ΔAUC / ΔR²
    c2_madelon_ground_truth.csv       # precision, recall, F₁
    c8_minshap_vs_minppi.csv          # agreement stats

Run::

    python -u experiments/eval_harmful_utility.py
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
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm

from minppi_mofs.min_ppi import (
    estimate_minppi,
    estimate_minppi_random_mc,
)
from minppi_mofs.shapg_integration import build_guidance_from_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA = 0.95
SEEDS: list[int] = [42, 123, 456, 789, 1024]

# c1 classification datasets
C1_CLS_DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
]

# c1 regression datasets
C1_REG_DATASETS: list[tuple[str, str, str]] = [
    ("crimedata", "nonViolPerPop", "regression"),
    ("abalone", "Rings", "regression"),
]

# c2 ground-truth datasets
C2_DATASETS: list[tuple[str, str, str]] = [
    ("madelon", "label", "binary"),
]

# #8 comparison datasets
C8_DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
]

# Madelon ground-truth: original f1–f20 are real; f21–f500 are noise probes.
# After MI filtering to 100, we check which of the 100 are probes.
MADELON_REAL = {f"f{i}" for i in range(1, 21)}
MADELON_PROBES = {f"f{i}" for i in range(21, 501)}

# Random-control trials per seed
N_RANDOM_TRIALS = 10

# Per-dataset MinPPI budget (matches run_ablation.py)
DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K": 200, "K_adv": 30},
    "madelon": {"K": 100, "K_adv": 20},
    "isolet": {"K": 80, "K_adv": 8},
    "credit_card_fraud": {"K": 200, "K_adv": 30},
    "gisette": {"K": 50, "K_adv": 10},
    "crimedata": {"K": 50, "K_adv": 10},
    "abalone": {"K": 50, "K_adv": 10},
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


def load_split(
    prefix: str, target: str, processed_dir: Path, *, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train, val, and test splits.  Returns (X, y) tuples."""
    train = pd.read_csv(processed_dir / f"{prefix}_s{seed}_train.csv")
    val = pd.read_csv(processed_dir / f"{prefix}_s{seed}_val.csv")
    test = pd.read_csv(processed_dir / f"{prefix}_s{seed}_test.csv")
    return (
        train.drop(columns=[target]),
        train[target],
        val.drop(columns=[target]),
        val[target],
        test.drop(columns=[target]),
        test[target],
    )


def _budget_for(prefix: str) -> dict:
    defaults = {"K": 100, "K_adv": 20}
    return DATASET_BUDGET.get(prefix, defaults)


def train_and_eval(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
) -> tuple[float, float]:
    """Train XGBoost on *X_tr* / *y_tr*, evaluate on *X_te* / *y_te*.

    Returns ``(utility, elapsed_seconds)`` where utility is AUC (classification)
    or R² (regression).
    """
    t0 = time.perf_counter()
    n_classes = len(np.unique(y_te))

    if task == "regression":
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, random_state=seed,
        )
        model.fit(X_tr.values, y_tr)
        y_pred = model.predict(X_te.values)
        val = float(r2_score(y_te, y_pred))
        if val < 0.0:
            val = 0.0
    else:
        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, random_state=seed,
        )
        model.fit(X_tr.values, y_tr)
        y_prob = model.predict_proba(X_te.values)
        if n_classes == 2:
            val = float(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            val = float(roc_auc_score(y_te, y_prob, multi_class="ovr"))
    return val, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# c1 — Removal test
# ---------------------------------------------------------------------------


def run_c1_removal(
    run_dir: Path,
    processed_dir: Path,
    *,
    alpha: float = ALPHA,
) -> tuple[list[dict], list[dict]]:
    """Deletion-based harmful utility test.

    For each (dataset, seed):
    1. Compute MinPPI on (train, val) → identify harmful features.
    2. Compute mean-SHAP and MinShap scores on train → identify bottom-k.
    3. Baseline: full model on train → eval on held-out test.
    4. Delete MinPPI −1 features → retrain → eval → ΔAUC.
    5. Delete SHAP bottom-k (k = n_harmful) → retrain → eval (control).
    6. Delete MinShap-negative → retrain → eval (control).
    7. Delete k random features × N_RANDOM_TRIALS → avg ΔAUC (control).
    """
    removal_dir = run_dir / "c1_removal"
    removal_dir.mkdir(parents=True, exist_ok=True)

    cls_rows: list[dict] = []
    reg_rows: list[dict] = []

    all_configs: list[tuple[str, str, str, bool, int]] = []
    for prefix, target, task in C1_CLS_DATASETS:
        for seed in SEEDS:
            all_configs.append((prefix, target, task, True, seed))
    for prefix, target, task in C1_REG_DATASETS:
        for seed in SEEDS:
            all_configs.append((prefix, target, task, False, seed))

    for prefix, target, task, is_cls, seed in tqdm(
        all_configs, desc="c1 removal", unit="config", file=sys.stdout
    ):
        try:
            X_tr, y_tr, X_val, y_val, X_te, y_te = load_split(
                prefix, target, processed_dir, seed=seed,
            )
        except FileNotFoundError:
            print(f"\n  SKIP {prefix} s{seed}: missing split files", flush=True)
            continue

        n_features = X_tr.shape[1]
        budget = _budget_for(prefix)
        metric_label = "AUC" if is_cls else "R²"

        # ---- baseline ----
        base_util, base_time = train_and_eval(
            X_tr, y_tr.values, X_te, y_te.values, task=task, seed=seed,
        )
        print(
            f"\n  {prefix} s{seed}: base {metric_label}={base_util:.4f} "
            f"({n_features} feats)",
            flush=True,
        )

        # ---- MinPPI ----
        t0 = time.perf_counter()
        if task == "multiclass" or task == "regression":
            minppi = estimate_minppi_random_mc(
                X_tr, X_val, y_tr.values, y_val.values,
                alpha=alpha, K=200, shared_coalitions=True,
                per_coalition=5, seed=seed, verbose=False, task=task,
            )
        else:
            sg_scores, sg_neighbors = build_guidance_from_data(
                X_tr, y_tr.values, depth=1, m=5, seed=seed,
            )
            minppi = estimate_minppi(
                X_tr, X_val, y_tr.values, y_val.values,
                alpha=alpha, method="shapg_guided",
                K_adv=budget["K_adv"],
                shapg_neighbors=sg_neighbors,
                shapg_scores=sg_scores,
                seed=seed, verbose=False, task=task,
            )
        t_minppi = time.perf_counter() - t0
        harmful = {f for f, v in minppi.items() if v == -1}
        n_harmful = len(harmful)

        # ---- mean-SHAP ranking ----
        from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
        shap_scores = compute_shap_topk_importance(X_tr, y_tr.values, seed=seed, task=task)
        shap_ranked = sorted(shap_scores.items(), key=lambda kv: kv[1])
        shap_bottom = {f for f, _ in shap_ranked[:max(1, n_harmful)]}

        # ---- MinShap negative ----
        from minppi_mofs.baselines.min_shap import compute_minshap_importance
        minshap_scores = compute_minshap_importance(
            X_tr, y_tr.values, n_coalitions=100, per_coalition=5, seed=seed, task=task,
        )
        minshap_neg = {f for f, v in minshap_scores.items() if v < 0}
        # If MinShap finds no negative features, use bottom-n_harmful by score
        if not minshap_neg and n_harmful > 0:
            ranked = sorted(minshap_scores.items(), key=lambda kv: kv[1])
            minshap_neg = {f for f, _ in ranked[:n_harmful]}

        # ---- evaluate removal ----
        def _remove_and_eval(to_remove: set[str]) -> tuple[float, float, float]:
            if not to_remove:
                return base_util, 0.0, base_time
            keep = [f for f in X_tr.columns if f not in to_remove]
            if not keep:
                return base_util, 1.0, 0.0  # can't remove all features
            post_u, post_t = train_and_eval(
                X_tr[keep], y_tr.values, X_te[keep], y_te.values, task=task, seed=seed,
            )
            cr = len(to_remove) / n_features
            return post_u, cr, post_t

        # MinPPI removal
        ppi_u, ppi_cr, ppi_t = _remove_and_eval(harmful)
        ppi_delta = ppi_u - base_util

        # mean-SHAP control
        shap_u, shap_cr, shap_t = _remove_and_eval(shap_bottom)
        shap_delta = shap_u - base_util

        # MinShap control
        ms_u, ms_cr, ms_t = _remove_and_eval(minshap_neg)
        ms_delta = ms_u - base_util

        # Random control (n_trials)
        all_feats = list(X_tr.columns)
        rng = np.random.default_rng(seed)
        rand_deltas: list[float] = []
        for _ in range(N_RANDOM_TRIALS):
            rand_remove = set(rng.choice(all_feats, size=max(1, n_harmful), replace=False))
            ru, _, _ = _remove_and_eval(rand_remove)
            rand_deltas.append(ru - base_util)
        rand_delta_mean = float(np.mean(rand_deltas))
        rand_delta_std = float(np.std(rand_deltas, ddof=1)) if len(rand_deltas) > 1 else 0.0

        row: dict = {
            "dataset": prefix,
            "seed": seed,
            "task": task,
            "n_features": n_features,
            "base_util": round(base_util, 4),
            "n_harmful": n_harmful,
            "n_shap_bottom": len(shap_bottom),
            "n_minshap_neg": len(minshap_neg),
            # MinPPI
            "minppi_util": round(ppi_u, 4),
            "minppi_delta": round(ppi_delta, 4),
            "minppi_cr": round(ppi_cr, 4),
            "minppi_time_s": round(t_minppi, 1),
            # mean-SHAP control
            "shap_util": round(shap_u, 4),
            "shap_delta": round(shap_delta, 4),
            "shap_cr": round(shap_cr, 4),
            # MinShap control
            "minshap_util": round(ms_u, 4),
            "minshap_delta": round(ms_delta, 4),
            "minshap_cr": round(ms_cr, 4),
            # Random control
            "random_delta_mean": round(rand_delta_mean, 4),
            "random_delta_std": round(rand_delta_std, 4),
        }
        if is_cls:
            cls_rows.append(row)
        else:
            reg_rows.append(row)

        harmful_pct = n_harmful / n_features * 100
        print(
            f"    MinPPI: {n_harmful} harmful ({harmful_pct:.1f}%) → "
            f"Δ{metric_label}={ppi_delta:+.4f}, cr={ppi_cr:.3f}",
            flush=True,
        )
        print(
            f"    SHAP:   Δ{metric_label}={shap_delta:+.4f}, "
            f"MinShap: Δ{metric_label}={ms_delta:+.4f}, "
            f"Random:  Δ{metric_label}={rand_delta_mean:+.4f} ± {rand_delta_std:.4f}",
            flush=True,
        )

    # Save
    if cls_rows:
        pd.DataFrame(cls_rows).to_csv(removal_dir / "c1_removal_cls.csv", index=False)
        print(f"\n  c1_removal_cls.csv saved ({len(cls_rows)} rows)", flush=True)
    if reg_rows:
        pd.DataFrame(reg_rows).to_csv(removal_dir / "c1_removal_reg.csv", index=False)
        print(f"  c1_removal_reg.csv saved ({len(reg_rows)} rows)", flush=True)

    return cls_rows, reg_rows


# ---------------------------------------------------------------------------
# c2 — Madelon ground-truth recovery
# ---------------------------------------------------------------------------


def run_c2_ground_truth(
    run_dir: Path,
    processed_dir: Path,
    *,
    alpha: float = ALPHA,
) -> list[dict]:
    """Evaluate MinPPI / MinShap / SHAP against known Madelon probe labels.

    Madelon has 500 original features: f1–f20 are real (informative),
    f21–f500 are probes (pure noise).  After MI filtering to 100 features,
    we check which of the 100 are probes → ground-truth harmful labels.

    Metrics: precision, recall, F₁ for each method's harmful-feature set
    against the ground-truth probe set.
    """
    removal_dir = run_dir / "c2_ground_truth"
    removal_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prefix, target, task in tqdm(
        C2_DATASETS, desc="c2 ground truth", unit="dataset", file=sys.stdout
    ):
        for seed in SEEDS:
            X_tr, y_tr, X_val, y_val, X_te, y_te = load_split(
                prefix, target, processed_dir, seed=seed,
            )
            n_features = X_tr.shape[1]
            budget = _budget_for(prefix)

            # Ground-truth: which of the 100 MI-selected features are probes?
            selected_feats = set(X_tr.columns)
            gt_probes = selected_feats & MADELON_PROBES
            gt_real = selected_feats & MADELON_REAL
            n_gt_probes = len(gt_probes)
            if n_gt_probes == 0:
                print(
                    f"\n  {prefix} s{seed}: WARNING — 0 probes in MI top-100, "
                    f"all {n_features} are real features. Skipping.",
                    flush=True,
                )
                continue

            print(
                f"\n  {prefix} s{seed}: {n_features} features, "
                f"{n_gt_probes} probes (ground-truth), {len(gt_real)} real",
                flush=True,
            )

            # ---- MinPPI ----
            sg_scores, sg_neighbors = build_guidance_from_data(
                X_tr, y_tr.values, depth=1, m=5, seed=seed,
            )
            minppi = estimate_minppi(
                X_tr, X_val, y_tr.values, y_val.values,
                alpha=alpha, method="shapg_guided",
                K_adv=budget["K_adv"],
                shapg_neighbors=sg_neighbors,
                shapg_scores=sg_scores,
                seed=seed, verbose=False, task=task,
            )
            minppi_harmful = {f for f, v in minppi.items() if v == -1}

            # ---- MinShap ----
            from minppi_mofs.baselines.min_shap import compute_minshap_importance
            minshap_scores = compute_minshap_importance(
                X_tr, y_tr.values, n_coalitions=100, per_coalition=5,
                seed=seed, task=task,
            )
            # Use all negative-scored features
            minshap_neg = {f for f, v in minshap_scores.items() if v < 0}
            if not minshap_neg:
                ranked = sorted(minshap_scores.items(), key=lambda kv: kv[1])
                k = max(1, len(minppi_harmful))
                minshap_neg = {f for f, _ in ranked[:k]}

            # ---- SHAP ----
            from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
            shap_sc = compute_shap_topk_importance(
                X_tr, y_tr.values, seed=seed, task=task,
            )
            ranked = sorted(shap_sc.items(), key=lambda kv: kv[1])
            k = max(1, len(minppi_harmful))
            shap_bottom = {f for f, _ in ranked[:k]}

            # ---- Compute precision / recall / F1 ----
            def _prf(predicted: set[str]) -> tuple[float, float, float]:
                if not predicted:
                    return 0.0, 0.0, 0.0
                tp = len(predicted & gt_probes)
                prec = tp / len(predicted)
                rec = tp / len(gt_probes) if gt_probes else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                return round(prec, 4), round(rec, 4), round(f1, 4)

            for method, pred_set in [
                ("minppi", minppi_harmful),
                ("minshap", minshap_neg),
                ("shap", shap_bottom),
            ]:
                prec, rec, f1 = _prf(pred_set)
                rows.append({
                    "dataset": prefix,
                    "seed": seed,
                    "method": method,
                    "n_features": n_features,
                    "n_gt_probes": n_gt_probes,
                    "n_predicted": len(pred_set),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                })
                print(
                    f"    {method:8s}: predicted={len(pred_set):3d}, "
                    f"P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_csv(removal_dir / "c2_madelon_ground_truth.csv", index=False)
        print(f"\n  c2_madelon_ground_truth.csv saved ({len(df)} rows)", flush=True)

        # Aggregate: mean over seeds per method
        agg = df.groupby("method")[["precision", "recall", "f1"]].agg(
            ["mean", "std"]
        ).round(4)
        agg.to_csv(removal_dir / "c2_madelon_ground_truth_agg.csv")
        print("  c2_madelon_ground_truth_agg.csv saved", flush=True)
        print(agg.to_string(), flush=True)

    return rows


# ---------------------------------------------------------------------------
# #8 — MinShap vs MinPPI agreement
# ---------------------------------------------------------------------------


def run_c8_minshap_vs_minppi(
    run_dir: Path,
    processed_dir: Path,
    *,
    alpha: float = ALPHA,
) -> list[dict]:
    """Compare MinShap and MinPPI harmful-feature detection.

    For each (dataset, seed):
    - MinPPI harmful set (binary ±1 classification)
    - MinShap negative set (continuous, < 0)
    - Jaccard similarity between the two
    - Per-feature agreement: how often does MinShap < 0 correspond to MinPPI = −1?
    """
    removal_dir = run_dir / "c8_minshap_vs_minppi"
    removal_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prefix, target, task in tqdm(
        C8_DATASETS, desc="c8 MinShap vs MinPPI", unit="dataset", file=sys.stdout
    ):
        for seed in SEEDS:
            X_tr, y_tr, X_val, y_val, _, _ = load_split(
                prefix, target, processed_dir, seed=seed,
            )
            n_features = X_tr.shape[1]
            budget = _budget_for(prefix)

            # ---- MinPPI ----
            if task == "multiclass":
                minppi = estimate_minppi_random_mc(
                    X_tr, X_val, y_tr.values, y_val.values,
                    alpha=alpha, K=200, shared_coalitions=True,
                    per_coalition=5, seed=seed, verbose=False, task=task,
                )
            else:
                sg_scores, sg_neighbors = build_guidance_from_data(
                    X_tr, y_tr.values, depth=1, m=5, seed=seed,
                )
                minppi = estimate_minppi(
                    X_tr, X_val, y_tr.values, y_val.values,
                    alpha=alpha, method="shapg_guided",
                    K_adv=budget["K_adv"],
                    shapg_neighbors=sg_neighbors,
                    shapg_scores=sg_scores,
                    seed=seed, verbose=False, task=task,
                )
            minppi_harmful = {f for f, v in minppi.items() if v == -1}

            # ---- MinShap ----
            from minppi_mofs.baselines.min_shap import compute_minshap_importance
            minshap_scores = compute_minshap_importance(
                X_tr, y_tr.values, n_coalitions=100, per_coalition=5,
                seed=seed, task=task,
            )
            minshap_neg = {f for f, v in minshap_scores.items() if v < 0}

            # ---- Agreement ----
            inter_mp_ms = minppi_harmful & minshap_neg
            union_mp_ms = minppi_harmful | minshap_neg
            jac_mp_ms = (
                len(inter_mp_ms) / len(union_mp_ms) if union_mp_ms else 0.0
            )

            # Per-feature: how many MinShap < 0 are also MinPPI = −1?
            n_ms_neg = len(minshap_neg)
            n_ms_neg_and_mp_harm = len(inter_mp_ms)

            rows.append({
                "dataset": prefix,
                "seed": seed,
                "n_features": n_features,
                "n_minppi_harmful": len(minppi_harmful),
                "n_minshap_neg": n_ms_neg,
                "n_both_harmful": n_ms_neg_and_mp_harm,
                "jaccard_minppi_minshap": round(jac_mp_ms, 4),
                "minshap_neg_as_minppi_rate": (
                    round(n_ms_neg_and_mp_harm / n_ms_neg, 4)
                    if n_ms_neg > 0 else 0.0
                ),
            })
            print(
                f"\n  {prefix} s{seed}: MinPPI−1={len(minppi_harmful)}, "
                f"MinShap<0={n_ms_neg}, overlap={n_ms_neg_and_mp_harm}, "
                f"J={jac_mp_ms:.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_csv(removal_dir / "c8_minshap_vs_minppi.csv", index=False)
        print(f"\n  c8_minshap_vs_minppi.csv saved ({len(df)} rows)", flush=True)

        # Aggregate
        agg = df.groupby("dataset")[
            ["n_minppi_harmful", "n_minshap_neg", "n_both_harmful",
             "jaccard_minppi_minshap", "minshap_neg_as_minppi_rate"]
        ].agg(["mean", "std"]).round(4)
        agg.to_csv(removal_dir / "c8_minshap_vs_minppi_agg.csv")
        print("  c8_minshap_vs_minppi_agg.csv saved", flush=True)
        print(agg.to_string(), flush=True)

    return rows


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

    _header(f"Harmful-Feature Utility Evaluation — {run_id}")
    print(f"Commit: {commit}", flush=True)
    print(f"Alpha: {ALPHA}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)

    # Metadata
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": ALPHA,
        "seeds": SEEDS,
        "experiments": ["c1_removal", "c2_ground_truth", "c8_minshap_vs_minppi"],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # config.yaml
    run_config: dict = {
        "description": "Harmful-feature utility evaluation — c1/c2/#8",
        "alpha": ALPHA,
        "seeds": SEEDS,
        "random_trials": N_RANDOM_TRIALS,
        "c1_datasets": {
            "classification": [d[0] for d in C1_CLS_DATASETS],
            "regression": [d[0] for d in C1_REG_DATASETS],
        },
        "c2_datasets": [d[0] for d in C2_DATASETS],
        "c8_datasets": [d[0] for d in C8_DATASETS],
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    _header("c1 — Removal test (classification)")
    cls_rows, reg_rows = run_c1_removal(run_dir, processed_dir, alpha=ALPHA)

    if reg_rows:
        _header("c1 — Removal test (regression)")
        print(f"  {len(reg_rows)} regression rows (see c1_removal_reg.csv)", flush=True)

    _header("c2 — Madelon ground-truth recovery")
    run_c2_ground_truth(run_dir, processed_dir, alpha=ALPHA)

    _header("#8 — MinShap vs MinPPI")
    run_c8_minshap_vs_minppi(run_dir, processed_dir, alpha=ALPHA)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _header("Summary")

    if cls_rows:
        df_cls = pd.DataFrame(cls_rows)
        agg = df_cls.groupby("dataset")[
            ["minppi_delta", "shap_delta", "minshap_delta",
             "random_delta_mean", "minppi_cr"]
        ].mean().round(4)
        print("\n  c1 classification — mean ΔAUC by dataset:", flush=True)
        print(agg.to_string(), flush=True)

    _header("Done")
    print(f"Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
