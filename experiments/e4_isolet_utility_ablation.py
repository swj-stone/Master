"""
ISOLET utility-function ablation: accuracy-based vs OVR-AUC-based v_α.

Lightweight ablation — uses reduced sampling budget (K=8, per_coalition=2)
and lightweight XGBoost (n_estimators=5 for OVR path) to keep runtime under
15 minutes.  Reports Jaccard similarity and saves results to CSV.

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

from minppi_mofs.min_ppi import estimate_minppi_random_mc
from minppi_mofs.voting_game import AccFullCache

K_ABLATION = 8
PER_COALITION = 2
ALPHA = 0.95
SEED = 42


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def main():
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
    print(f"Evals per condition: {K_ABLATION} x (1+{PER_COALITION}) = "
          f"{K_ABLATION * (1 + PER_COALITION)} model fits")

    # ── Condition 1: Accuracy-based (current approach) ──────────────
    # Multiclass uses LogisticRegression internally → fast (~90s).
    print("\n[1/2] Accuracy-based MinPPI ...", flush=True)
    t0 = time.perf_counter()
    minppi_acc = estimate_minppi_random_mc(
        X_tr, X_tr, y_tr, y_tr,
        alpha=ALPHA, K=K_ABLATION,
        shared_coalitions=True, per_coalition=PER_COALITION,
        seed=SEED, verbose=True, task="multiclass",
    )
    t_acc = time.perf_counter() - t0
    harmful_acc = {f for f, v in minppi_acc.items() if v == -1}
    print(f"  Done: {len(harmful_acc)} harmful [{t_acc:.0f}s]")

    # ── Condition 2: OVR-AUC-based ──────────────────────────────────
    # Override _evaluate_subset to use OVR AUC instead of accuracy.
    # Use lightweight XGBoost (n_estimators=5) since we only need
    # binary threshold decisions, not precise AUC values.
    class _OVRAUCCache(AccFullCache):
        def _evaluate_subset(self, S, *, _seed_base=42):
            key = tuple(sorted(S))
            if key in self._cache:
                return self._cache[key]
            cols = list(S)
            if len(cols) == 0:
                val = 0.5
            else:
                import warnings
                from xgboost import XGBClassifier
                m = XGBClassifier(
                    n_estimators=5, max_depth=2, learning_rate=0.3,
                    verbosity=0, random_state=_seed_base,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(self._X_train[cols].values, self._y_train)
                proba = m.predict_proba(self._X_val[cols].values)
                val = float(roc_auc_score(self._y_val, proba,
                                          multi_class="ovr"))
            self._cache[key] = val
            return val

    print("\n[2/2] OVR-AUC-based MinPPI (n_estimators=5, ~10-15 min) ...",
          flush=True)
    t0 = time.perf_counter()
    ovr_cache = _OVRAUCCache(
        X_tr, X_tr, y_tr, y_tr, task="multiclass", seed=SEED,
    )
    minppi_ovr = estimate_minppi_random_mc(
        X_tr, X_tr, y_tr, y_tr,
        alpha=ALPHA, K=K_ABLATION,
        shared_coalitions=True, per_coalition=PER_COALITION,
        seed=SEED, verbose=True, task="multiclass",
        acc_cache=ovr_cache,
    )
    t_ovr = time.perf_counter() - t0
    harmful_ovr = {f for f, v in minppi_ovr.items() if v == -1}
    print(f"  Done: {len(harmful_ovr)} harmful [{t_ovr:.0f}s]")

    # ── Comparison ──────────────────────────────────────────────────
    union = harmful_acc | harmful_ovr
    inter = harmful_acc & harmful_ovr
    jaccard = len(inter) / len(union) if union else 1.0

    print(f"\n{'=' * 60}")
    print(f"  Accuracy-based: {len(harmful_acc)} harmful [{t_acc:.0f}s]")
    print(f"  OVR-AUC-based:   {len(harmful_ovr)} harmful [{t_ovr:.0f}s]")
    print(f"  Jaccard:         {jaccard:.3f}")
    print(f"  Intersection:    {len(inter)} features")
    print(f"  Acc-only:        {len(harmful_acc - harmful_ovr)}")
    print(f"  OVR-only:        {len(harmful_ovr - harmful_acc)}")

    # ── ΔAUC comparison ─────────────────────────────────────────────
    from xgboost import XGBClassifier

    def _eval_removal(harmful_set):
        keep = [c for c in X_tr.columns if c not in harmful_set] if harmful_set \
               else list(X_tr.columns)
        m = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, random_state=SEED,
        )
        m.fit(X_tr[keep].values, y_tr)
        proba = m.predict_proba(X_te[keep].values)
        return float(roc_auc_score(y_te, proba, multi_class="ovr"))

    base_auc = _eval_removal(set())
    auc_acc = _eval_removal(harmful_acc)
    auc_ovr = _eval_removal(harmful_ovr)
    d_acc = auc_acc - base_auc
    d_ovr = auc_ovr - base_auc

    print(f"\n  Baseline OVR AUC:  {base_auc:.4f}")
    print(f"  Delete acc-H:      {auc_acc:.4f}  (Delta={d_acc:+.4f})")
    print(f"  Delete ovr-H:      {auc_ovr:.4f}  (Delta={d_ovr:+.4f})")

    if jaccard >= 0.70:
        print("\n  Jaccard >= 0.7 — accuracy is a valid proxy for OVR AUC.")
    else:
        print("\n  Jaccard < 0.7 — the two utilities produce different sets.")

    # ── Save results ────────────────────────────────────────────────
    rows = [
        {
            "dataset": "isolet", "seed": SEED, "n_features": X_tr.shape[1],
            "n_classes": n_cls, "alpha": ALPHA,
            "utility": "accuracy", "n_harmful": len(harmful_acc),
            "time_s": round(t_acc, 1),
        },
        {
            "dataset": "isolet", "seed": SEED, "n_features": X_tr.shape[1],
            "n_classes": n_cls, "alpha": ALPHA,
            "utility": "ovr_auc", "n_harmful": len(harmful_ovr),
            "time_s": round(t_ovr, 1),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e4_isolet_utility_ablation.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(
        {"run_id": run_id, "jaccard": round(jaccard, 4),
         "delta_auc_acc": round(d_acc, 4), "delta_auc_ovr": round(d_ovr, 4),
         "K": K_ABLATION, "per_coalition": PER_COALITION},
        indent=2,
    ))
    print(f"\nSaved: {out_dir / 'e4_isolet_utility_ablation.csv'}")


if __name__ == "__main__":
    main()
