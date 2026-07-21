"""
ISOLET utility-function ablation: accuracy-based vs OVR-AUC-based v_α.

**Structural guarantee (2026-07-21).**  A single ``AccFullCache`` trains
each coalition's XGBoost **exactly once** and caches the raw probability
matrix.  Two ``UtilityView`` instances read the identical predictions under
different metrics:

    coalition
       ↓
    train XGBoost once  →  cache.predict_proba(X_val)  ← single source of truth
       ↓                        ↓
    UtilityView("accuracy")   UtilityView("ovr_auc")
       ↓                        ↓
    harmful_acc               harmful_ovr

Base budget: K=8 shared coalitions, per_coalition=2, plus deterministic
supplemental coverage steps generated once and reused by both utilities.
Model: XGBoost (n_estimators=15, max_depth=2).

Run:
    python -u experiments/e4_isolet_utility_ablation.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from minppi_mofs.min_ppi import (
    estimate_minppi_random_mc,
    make_shared_coalition_plan,
)
from minppi_mofs.voting_game import AccFullCache, UtilityView

K_ABLATION = 8
PER_COALITION = 2
ALPHA = 0.95
SEED = 42


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _assert_same_prediction_keys(
    before: set[tuple[str, ...]],
    after: set[tuple[str, ...]],
) -> None:
    """Refuse results unless both utility passes used identical coalitions."""
    added = after - before
    missing = before - after
    if added or missing:
        raise RuntimeError(
            "Utility passes used different prediction-cache keys: "
            f"added={len(added)}, missing={len(missing)}"
        )


def _eval_removal(
    harmful_set: set[str],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
) -> float:
    keep = (
        [c for c in X_tr.columns if c not in harmful_set]
        if harmful_set
        else list(X_tr.columns)
    )
    m = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        verbosity=0, random_state=SEED,
    )
    m.fit(X_tr[keep].values, y_tr)
    proba = m.predict_proba(X_te[keep].values)
    return float(roc_auc_score(y_te, proba, multi_class="ovr"))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed_v2"
    run_id = _now_stamp()
    out_dir = project_root / "results" / "tables" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(data_dir / f"isolet_s{SEED}_train.csv")
    va = pd.read_csv(data_dir / f"isolet_s{SEED}_val.csv")
    te = pd.read_csv(data_dir / f"isolet_s{SEED}_test.csv")

    X_tr = tr.drop(columns=["letter"])
    y_tr = tr["letter"].values
    X_va = va.drop(columns=["letter"])
    y_va = va["letter"].values
    X_te = te.drop(columns=["letter"])
    y_te = te["letter"].values
    n_cls = len(np.unique(y_tr))

    print(f"ISOLET seed={SEED}: {X_tr.shape[1]} features, {n_cls} classes")
    print(f"  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")
    print(f"Budget: K={K_ABLATION}, per_coalition={PER_COALITION}")
    shared_plan = make_shared_coalition_plan(
        X_tr.columns,
        K=K_ABLATION,
        per_coalition=PER_COALITION,
        seed=SEED,
    )
    print(f"Fixed shared plan: {len(shared_plan)} steps")

    # ── Build ONE prediction cache (each coalition trained once) ──────
    print("\nBuilding shared prediction cache ...", flush=True)
    t0 = time.perf_counter()
    cache = AccFullCache(
        X_tr, X_va, y_tr, y_va,
        task="multiclass", seed=SEED,
        # default utility irrelevant — we use views for everything
    )
    print(f"  Cache ready [{time.perf_counter() - t0:.0f}s]  "
          f"(default utility: {cache._utility})")

    # ── Two views, SAME predictions, DIFFERENT metrics ────────────────
    view_acc = UtilityView(cache, utility="accuracy", alpha=ALPHA)
    view_auc = UtilityView(cache, utility="ovr_auc", alpha=ALPHA)
    print(f"  U_full(accuracy) = {view_acc.acc_full:.4f}")
    print(f"  U_full(ovr_auc)  = {view_auc.acc_full:.4f}")
    print(f"  Threshold accuracy = {view_acc.threshold:.4f}")
    print(f"  Threshold ovr_auc  = {view_auc.threshold:.4f}")

    # ── MinPPI under accuracy view ────────────────────────────────────
    print("\n[1/2] Accuracy-based MinPPI ...", flush=True)
    t_acc = time.perf_counter()
    minppi_acc = estimate_minppi_random_mc(
        X_tr, X_va, y_tr, y_va,
        alpha=ALPHA, K=K_ABLATION,
        shared_coalitions=True, per_coalition=PER_COALITION,
        shared_plan=shared_plan,
        seed=SEED, verbose=True, task="multiclass", utility="accuracy",
        acc_cache=view_acc,          # ← reads from shared predictions
    )
    t_acc = time.perf_counter() - t_acc
    harmful_acc = {f for f, v in minppi_acc.items() if v == -1}
    print(f"  Done: {len(harmful_acc)} harmful [{t_acc:.0f}s]")

    # ── MinPPI under OVR-AUC view (SAME predictions, DIFFERENT metric) ─
    cached_keys_before = set(cache._preds)
    n_cached = len(cached_keys_before)  # models already trained from first pass
    print(f"\n[2/2] OVR-AUC-based MinPPI  (cache has {n_cached} coalitions) ...",
          flush=True)
    t_ovr = time.perf_counter()
    minppi_ovr = estimate_minppi_random_mc(
        X_tr, X_va, y_tr, y_va,
        alpha=ALPHA, K=K_ABLATION,
        shared_coalitions=True, per_coalition=PER_COALITION,
        shared_plan=shared_plan,
        seed=SEED, verbose=True, task="multiclass", utility="ovr_auc",
        acc_cache=view_auc,         # ← same predictions, different metric
    )
    t_ovr = time.perf_counter() - t_ovr
    harmful_ovr = {f for f, v in minppi_ovr.items() if v == -1}

    # Verify: second pass should be entirely cache hits (all coalitions
    # were already trained in the first pass with identical inputs).
    cached_keys_after = set(cache._preds)
    _assert_same_prediction_keys(cached_keys_before, cached_keys_after)
    n_cached_after = len(cached_keys_after)
    print(f"  Done: {len(harmful_ovr)} harmful [{t_ovr:.0f}s]")
    print(f"  Cache: {n_cached} → {n_cached_after} coalitions stored  "
          f"({'+' if n_cached_after > n_cached else ''}"
          f"{n_cached_after - n_cached} new)")

    # ── Comparison ────────────────────────────────────────────────────
    union = harmful_acc | harmful_ovr
    inter = harmful_acc & harmful_ovr
    jaccard = len(inter) / len(union) if union else 1.0

    print(f"\n{'='*60}")
    print(f"  Accuracy utility: {len(harmful_acc)} harmful [{t_acc:.0f}s]")
    print(f"  OVR-AUC utility:  {len(harmful_ovr)} harmful [{t_ovr:.0f}s]")
    print(f"  Jaccard:          {jaccard:.3f}")
    print(f"  Intersection:     {len(inter)} features")
    print(f"  Acc-only:         {len(harmful_acc - harmful_ovr)}")
    print(f"  OVR-only:         {len(harmful_ovr - harmful_acc)}")

    # ── One-time test evaluation ──────────────────────────────────────
    base_auc = _eval_removal(set(), X_tr, y_tr, X_te, y_te)
    auc_acc = _eval_removal(harmful_acc, X_tr, y_tr, X_te, y_te)
    auc_ovr = _eval_removal(harmful_ovr, X_tr, y_tr, X_te, y_te)
    d_acc = auc_acc - base_auc
    d_ovr = auc_ovr - base_auc

    print(f"\n  Baseline OVR AUC (test):  {base_auc:.4f}")
    print(f"  Delete acc-H (test):      {auc_acc:.4f}  (Δ={d_acc:+.4f})")
    print(f"  Delete ovr-H (test):      {auc_ovr:.4f}  (Δ={d_ovr:+.4f})")

    if jaccard >= 0.70:
        print("\n  Jaccard >= 0.7 — accuracy is a valid proxy for OVR AUC.")
    else:
        print("\n  Jaccard < 0.7 — the two utilities produce different sets.")

    # ── Save ──────────────────────────────────────────────────────────
    rows = [
        {"dataset": "isolet", "seed": SEED, "n_features": X_tr.shape[1],
         "n_classes": n_cls, "alpha": ALPHA,
         "utility": "accuracy", "n_harmful": len(harmful_acc),
         "time_s": round(t_acc, 1)},
        {"dataset": "isolet", "seed": SEED, "n_features": X_tr.shape[1],
         "n_classes": n_cls, "alpha": ALPHA,
         "utility": "ovr_auc", "n_harmful": len(harmful_ovr),
         "time_s": round(t_ovr, 1)},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "e4_isolet_utility_ablation.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "run_id": run_id, "jaccard": round(jaccard, 4),
        "delta_auc_acc": round(d_acc, 4), "delta_auc_ovr": round(d_ovr, 4),
        "K": K_ABLATION, "per_coalition": PER_COALITION,
        "shared_plan_steps": len(shared_plan),
        "data": "processed_v2, train/val/test split",
        "model": "XGBClassifier(15,2) — trained ONCE per coalition",
        "protocol": "shared AccFullCache → 2 UtilityViews read SAME predictions",
        "cache_coalitions": n_cached_after,
    }, indent=2))
    print(f"\nSaved: {out_dir / 'e4_isolet_utility_ablation.csv'}")


if __name__ == "__main__":
    main()
