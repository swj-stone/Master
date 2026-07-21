"""
ISOLET utility-function ablation: accuracy-based vs OVR-AUC-based v_α.

**Fixed in Task 3**: Both conditions now use the *same* XGBoost model
producing the *same* probability predictions.  The only difference is
which utility metric is computed from those probabilities to decide
whether a coalition wins.  This isolates the utility function as the
sole independent variable.

Budget: K=8 shared coalitions, per_coalition=2 (lightweight for rapid
turnaround).  Model: XGBoost with 15 estimators, max_depth=2.

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

from minppi_mofs.min_ppi import estimate_minppi_random_mc

K_ABLATION = 8
PER_COALITION = 2
ALPHA = 0.95
SEED = 42


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _eval_removal(harmful_set: set[str], X_tr: pd.DataFrame, y_tr: np.ndarray,
                  X_te: pd.DataFrame, y_te: np.ndarray) -> float:
    """Retrain on X_tr without harmful features, evaluate OVR-AUC on X_te."""
    keep = [c for c in X_tr.columns if c not in harmful_set] if harmful_set \
           else list(X_tr.columns)
    m = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        verbosity=0, random_state=SEED,
    )
    m.fit(X_tr[keep].values, y_tr)
    proba = m.predict_proba(X_te[keep].values)
    return float(roc_auc_score(y_te, proba, multi_class="ovr"))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    run_id = _now_stamp()
    out_dir = project_root / "results" / "tables" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(processed_dir / "isolet_s42_train.csv")
    te = pd.read_csv(processed_dir / "isolet_s42_test.csv")
    X_tr = tr.drop(columns=["letter"])
    y_tr = tr["letter"].values
    X_te = te.drop(columns=["letter"])
    y_te = te["letter"].values
    n_cls = len(np.unique(y_tr))

    print(f"ISOLET seed=42: {X_tr.shape[1]} features, {n_cls} classes")
    print(f"Budget: K={K_ABLATION}, per_coalition={PER_COALITION}")

    # ── Condition 1: Accuracy utility ─────────────────────────────────
    print("\n[1/2] Accuracy-based MinPPI ...", flush=True)
    t0 = time.perf_counter()
    minppi_acc = estimate_minppi_random_mc(
        X_tr, X_tr, y_tr, y_tr,
        alpha=ALPHA, K=K_ABLATION,
        shared_coalitions=True, per_coalition=PER_COALITION,
        seed=SEED, verbose=True, task="multiclass", utility="accuracy",
    )
    t_acc = time.perf_counter() - t0
    harmful_acc = {f for f, v in minppi_acc.items() if v == -1}
    print(f"  Done: {len(harmful_acc)} harmful [{t_acc:.0f}s]")

    # ── Condition 2: OVR-AUC utility (SAME model, different metric) ───
    print("\n[2/2] OVR-AUC-based MinPPI ...", flush=True)
    t0 = time.perf_counter()
    minppi_ovr = estimate_minppi_random_mc(
        X_tr, X_tr, y_tr, y_tr,
        alpha=ALPHA, K=K_ABLATION,
        shared_coalitions=True, per_coalition=PER_COALITION,
        seed=SEED, verbose=True, task="multiclass", utility="ovr_auc",
    )
    t_ovr = time.perf_counter() - t0
    harmful_ovr = {f for f, v in minppi_ovr.items() if v == -1}
    print(f"  Done: {len(harmful_ovr)} harmful [{t_ovr:.0f}s]")

    # ── Comparison ────────────────────────────────────────────────────
    union = harmful_acc | harmful_ovr
    inter = harmful_acc & harmful_ovr
    jaccard = len(inter) / len(union) if union else 1.0

    print(f"\n{'='*60}")
    print(f"  Accuracy-based: {len(harmful_acc)} harmful [{t_acc:.0f}s]")
    print(f"  OVR-AUC-based:   {len(harmful_ovr)} harmful [{t_ovr:.0f}s]")
    print(f"  Jaccard:         {jaccard:.3f}")
    print(f"  Intersection:    {len(inter)} features")
    print(f"  Acc-only:        {len(harmful_acc - harmful_ovr)}")
    print(f"  OVR-only:        {len(harmful_ovr - harmful_acc)}")

    # ── ΔAUC comparison (both evaluated with OVR-AUC) ─────────────────
    base_auc = _eval_removal(set(), X_tr, y_tr, X_te, y_te)
    auc_acc = _eval_removal(harmful_acc, X_tr, y_tr, X_te, y_te)
    auc_ovr = _eval_removal(harmful_ovr, X_tr, y_tr, X_te, y_te)
    d_acc = auc_acc - base_auc
    d_ovr = auc_ovr - base_auc

    print(f"\n  Baseline OVR AUC:  {base_auc:.4f}")
    print(f"  Delete acc-H:      {auc_acc:.4f}  (Δ={d_acc:+.4f})")
    print(f"  Delete ovr-H:      {auc_ovr:.4f}  (Δ={d_ovr:+.4f})")

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
        "model": "XGBClassifier(15,2) — same for both conditions",
    }, indent=2))
    print(f"\nSaved: {out_dir / 'e4_isolet_utility_ablation.csv'}")


if __name__ == "__main__":
    main()
