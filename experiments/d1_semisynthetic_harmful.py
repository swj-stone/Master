"""
Semi-synthetic harmful-feature injection experiment (§R.D1).

Creates three variants of Spambase and ISOLET, plus Madelon as
ground-truth baseline.  Each variant adds features with known
harmful labels to test whether MinPPI correctly identifies them.

Variants
--------
D1a — pure noise:  20 columns of N(0,1) added.  All are ground-truth
    harmful (zero predictive power).
D1b — permuted signal:  5 columns, each a random permutation of one
    of the top-5 MI-ranked existing features.  Harmful because they
    provide no information beyond what is already in the original
    feature, but are correlated with it.
D1c — injected harmful interaction:  5 columns of N(0,1).  When any
    of these columns is > 0 AND the top-2 MI features are both > 0,
    labels are flipped with p = 0.2 during training-set construction.
    This creates a *conditional* harmful witness — the column is only
    harmful when paired with specific existing features — exactly the
    scenario the binary v_α is designed to detect.

For each variant, three methods are evaluated:
  - MinPPI (α = 0.95)
  - MinShap (matched-budget: top-k most negative scores)
  - SHAP    (matched-budget: bottom-k ranked)

Output (results/tables/<run_id>/)::

    d1_semisynthetic.csv     # per-(dataset, variant, method)
    d1_semisynthetic_agg.csv # aggregated over seeds (if multi-seed)

Run::

    python -u experiments/d1_semisynthetic_harmful.py
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi, estimate_minppi_random_mc
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.baselines.min_shap import compute_minshap_importance
from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance

# ============================================================
# Constants
# ============================================================

SEED: int = 42
ALPHA: float = 0.95

# Variant parameters
N_NOISE: int = 20  # D1a: number of pure-noise columns
N_PERMUTED: int = 5  # D1b: number of permuted-signal columns
N_HARMFUL: int = 5  # D1c: number of harmful-interaction columns
HARMFUL_FLIP_PROB: float = 0.2

DATASETS: list[tuple[str, str, str]] = [
    # (prefix, target_col, task)
    ("spambase", "is_spam", "binary"),
    ("isolet", "letter", "multiclass"),
    ("madelon", "label", "binary"),  # ground-truth baseline
]

DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K_adv": 30},
    "isolet": {"K": 60, "per_coalition": 3},
    "madelon": {"K_adv": 20},
}

# Madelon probe-feature ground truth: after MI filtering to 100 features,
# f1—f20 are real, f21—f100 are probes (noise).  The MI-filter selects
# ~97 probes among the 100 retained features, but we use all 100 with
# the known labels f21–f100 = probe.
MADELON_PROBE_RANGE: tuple[int, int] = (21, 100)


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


class _Tee:
    """Duplicate stdout to a log file."""

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
    prefix: str,
    target: str,
    processed_dir: Path,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load train and test DataFrames for a dataset × seed."""
    train = pd.read_csv(processed_dir / f"{prefix}_s{seed}_train.csv")
    test = pd.read_csv(processed_dir / f"{prefix}_s{seed}_test.csv")
    return (
        train.drop(columns=[target]),
        train[target].values,
        test.drop(columns=[target]),
        test[target].values,
    )


def precision_recall(
    pred_harmful: set[str],
    true_harmful: set[str],
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 of predicted vs. true harmful set."""
    if not pred_harmful:
        return 1.0 if not true_harmful else 0.0, 0.0, 0.0
    inter = len(pred_harmful & true_harmful)
    prec = inter / len(pred_harmful)
    rec = inter / len(true_harmful) if true_harmful else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def train_eval(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    task: str = "binary",
) -> float:
    """Train XGBoost and evaluate utility."""
    n_cls = len(np.unique(y_te))
    if task == "regression":
        from xgboost import XGBRegressor
        from sklearn.metrics import r2_score

        m = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, random_state=SEED,
        )
        m.fit(X_tr.values, y_tr)
        val = float(r2_score(y_te, m.predict(X_te.values)))
        return max(val, 0.0)
    if n_cls <= 2:
        m = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, random_state=SEED,
        )
        m.fit(X_tr.values, y_tr)
        return float(roc_auc_score(y_te, m.predict_proba(X_te.values)[:, 1]))
    # multiclass — use accuracy
    from sklearn.metrics import accuracy_score

    m = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        verbosity=0, random_state=SEED,
    )
    m.fit(X_tr.values, y_tr)
    return float(accuracy_score(y_te, m.predict(X_te.values)))


# ============================================================
# Variant construction
# ============================================================


def _top_mi_features(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    n: int = 2,
) -> list[str]:
    """Return the top-n features by mutual information with the target."""
    mi = mutual_info_classif(X_tr, y_tr, random_state=SEED)
    ranked = sorted(zip(X_tr.columns, mi), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:n]]


def make_d1a_noise(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Add N_NOISE pure-noise columns.  All are ground-truth harmful."""
    X_tr_a = X_tr.copy()
    X_te_a = X_te.copy()
    harmful = set()
    for k in range(N_NOISE):
        col = f"__d1a_noise_{k}__"
        X_tr_a[col] = rng.normal(0, 1, len(X_tr_a))
        X_te_a[col] = rng.normal(0, 1, len(X_te_a))
        harmful.add(col)
    return X_tr_a, X_te_a, harmful


def make_d1b_permuted(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Add N_PERMUTED columns, each a permutation of a top signal feature."""
    top_feats = _top_mi_features(X_tr, y_tr, n=N_PERMUTED)
    X_tr_b = X_tr.copy()
    X_te_b = X_te.copy()
    harmful = set()
    for k, src in enumerate(top_feats):
        col = f"__d1b_perm_{k}__"
        idx_tr = rng.permutation(len(X_tr_b))
        idx_te = rng.permutation(len(X_te_b))
        X_tr_b[col] = X_tr_b[src].values[idx_tr]
        X_te_b[col] = X_te_b[src].values[idx_te]
        harmful.add(col)
    return X_tr_b, X_te_b, harmful


def make_d1c_injected(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, set[str]]:
    """Add N_HARMFUL noise columns with conditional label-flipping.

    When noise_col > 0 AND both top-2 MI features are > 0, labels
    are flipped with probability HARMFUL_FLIP_PROB.  This creates a
    conditional harmful witness that MinPPI's binary v_α can detect.
    """
    y_tr_c = y_tr.copy()
    pair = _top_mi_features(X_tr, y_tr_c, n=2)
    X_tr_c = X_tr.copy()
    X_te_c = X_te.copy()
    harmful = set()

    for k in range(N_HARMFUL):
        col = f"__d1c_harmful_{k}__"
        noise_tr = rng.normal(0, 1, len(X_tr_c))
        noise_te = rng.normal(0, 1, len(X_te_c))
        X_tr_c[col] = noise_tr
        X_te_c[col] = noise_te
        harmful.add(col)

        # Conditional harmful injection on training set
        mask = (
            (noise_tr > 0) & (X_tr_c[pair[0]].values > 0) & (X_tr_c[pair[1]].values > 0)
        )
        flip = mask & (rng.random(len(X_tr_c)) < HARMFUL_FLIP_PROB)
        y_tr_c[flip] = 1 - y_tr_c[flip]

    return X_tr_c, X_te_c, y_tr_c, harmful


# ============================================================
# Method evaluators
# ============================================================


def _run_minppi(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    task: str,
    budget: dict,
) -> set[str]:
    """Compute MinPPI (α = 0.95) and return the harmful-feature set."""
    n_cls = len(np.unique(y_tr))
    if n_cls > 2 or task == "regression":
        minppi = estimate_minppi_random_mc(
            X_tr,
            X_tr,
            y_tr,
            y_tr,
            alpha=ALPHA,
            K=budget.get("K", 200),
            shared_coalitions=True,
            per_coalition=budget.get("per_coalition", 5),
            seed=SEED,
            verbose=False,
            task=task,
        )
    else:
        sg_s, sg_n = build_guidance_from_data(X_tr, y_tr, seed=SEED)
        minppi = estimate_minppi(
            X_tr,
            X_tr,
            y_tr,
            y_tr,
            alpha=ALPHA,
            method="shapg_guided",
            K_adv=budget.get("K_adv", 30),
            shapg_neighbors=sg_n,
            shapg_scores=sg_s,
            seed=SEED,
            verbose=False,
            task=task,
        )
    return {f for f, v in minppi.items() if v == -1}


def _run_minshap_matched(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    k: int,
    task: str,
) -> set[str]:
    """MinShap top-k most-negative features."""
    scores = compute_minshap_importance(
        X_tr,
        y_tr,
        n_coalitions=100,
        per_coalition=5,
        seed=SEED,
        task=task,
    )
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    return {f for f, _ in ranked[:k]}


def _run_shap_matched(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    k: int,
    task: str,
) -> set[str]:
    """SHAP bottom-k features."""
    scores = compute_shap_topk_importance(
        X_tr,
        y_tr,
        seed=SEED,
        task=task,
    )
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    return {f for f, _ in ranked[:k]}


def _madelon_gt_probes(X_tr: pd.DataFrame) -> set[str]:
    """Return the set of Madelon probe features (f21–f100) present in X_tr."""
    cols = X_tr.columns.tolist()
    probes = set()
    for c in cols:
        if c.startswith("f"):
            try:
                fid = int(c[1:])
                if MADELON_PROBE_RANGE[0] <= fid <= MADELON_PROBE_RANGE[1]:
                    probes.add(c)
            except ValueError:
                pass
    return probes


# ============================================================
# Main
# ============================================================


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent
    processed_dir = project_root / "data" / "processed"

    run_id = _now_stamp()
    out_dir = project_root / "results" / "tables" / run_id
    run_dir = project_root / "results" / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(run_dir / "output.log")
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    commit = _git_commit(git_root)
    print("=" * 72)
    print(f"  D1 Semi-Synthetic Harmful Injection — {run_id}")
    print(f"  Commit: {commit}")
    print("=" * 72)

    metadata = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "alpha": ALPHA,
        "n_noise": N_NOISE,
        "n_permuted": N_PERMUTED,
        "n_harmful": N_HARMFUL,
        "harmful_flip_prob": HARMFUL_FLIP_PROB,
        "datasets": [d[0] for d in DATASETS],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rng = np.random.default_rng(SEED)
    all_rows: list[dict] = []

    for prefix, target, task in tqdm(
        DATASETS,
        desc="D1",
        unit="ds",
        file=sys.stdout,
    ):
        try:
            X_tr, y_tr, X_te, y_te = load_split(
                prefix,
                target,
                processed_dir,
                seed=SEED,
            )
        except FileNotFoundError:
            print(f"\n  SKIP {prefix}: no preprocessed data for seed {SEED}")
            continue

        budget = DATASET_BUDGET.get(prefix, {"K_adv": 30})
        t0 = time.perf_counter()
        print(f"\n{'=' * 60}")
        print(f"  {prefix} ({task}), n_features = {X_tr.shape[1]}")
        print(f"{'=' * 60}")

        # ── Ground-truth harmful sets ──────────────────────────────
        if prefix == "madelon":
            # Madelon: use known probe labels + run methods on original
            # data (no variant injection; probes are already present)
            gt_probes = _madelon_gt_probes(X_tr)
            variants = [
                ("original", X_tr, X_te, y_tr, gt_probes),
            ]
        else:
            gt_probes = set()
            variants = []

        # Build variants for non-Madelon datasets
        if prefix != "madelon":
            # D1a
            Xa_tr, Xa_te, gt_a = make_d1a_noise(X_tr, X_te, rng)
            variants.append(("D1a-noise", Xa_tr, Xa_te, y_tr, gt_a))

            # D1b
            Xb_tr, Xb_te, gt_b = make_d1b_permuted(X_tr, X_te, y_tr, rng)
            variants.append(("D1b-permuted", Xb_tr, Xb_te, y_tr, gt_b))

            # D1c (binary only — requires {0,1} labels for flipping)
            if task == "binary":
                Xc_tr, Xc_te, y_tr_c, gt_c = make_d1c_injected(X_tr, X_te, y_tr, rng)
                variants.append(("D1c-injected", Xc_tr, Xc_te, y_tr_c, gt_c))

        for vname, Xv_tr, Xv_te, yv_tr, gt_harmful in variants:
            n_total = Xv_tr.shape[1]
            print(f"\n  --- {vname} ---")
            print(f"  n_full = {n_total},  gt_harmful = {len(gt_harmful)}")

            if len(gt_harmful) == 0:
                print("  (no ground-truth harmful — skipping detection eval)")
                continue

            # Seed-consistent evaluation: train & eval on the variant
            baseline_auc = train_eval(Xv_tr, yv_tr, Xv_te, y_te, task=task)
            print(f"  Baseline utility = {baseline_auc:.4f}")

            # ── MinPPI ──────────────────────────────────────────────
            t_p = time.perf_counter()
            harmful_ppi = _run_minppi(Xv_tr, yv_tr, task, budget)
            t_ppi = time.perf_counter() - t_p
            prec_p, rec_p, f1_p = precision_recall(harmful_ppi, gt_harmful)

            # ΔAUC after deleting MinPPI-harmful features
            if harmful_ppi:
                keep = [c for c in Xv_tr.columns if c not in harmful_ppi]
                auc_filt = train_eval(Xv_tr[keep], yv_tr, Xv_te[keep], y_te)
            else:
                auc_filt = baseline_auc

            all_rows.append(
                {
                    "dataset": prefix,
                    "variant": vname,
                    "method": "minppi",
                    "n_full": n_total,
                    "n_gt_harmful": len(gt_harmful),
                    "n_predicted": len(harmful_ppi),
                    "precision": prec_p,
                    "recall": rec_p,
                    "f1": f1_p,
                    "baseline_auc": round(baseline_auc, 4),
                    "filtered_auc": round(auc_filt, 4),
                    "delta_auc": round(auc_filt - baseline_auc, 4),
                    "time_s": round(t_ppi, 1),
                }
            )
            print(
                f"  MinPPI:  flagged={len(harmful_ppi)}  "
                f"prec={prec_p:.3f}  rec={rec_p:.3f}  f1={f1_p:.3f}  "
                f"ΔAUC={auc_filt - baseline_auc:+.4f}  [{t_ppi:.1f}s]",
            )

            # ── MinShap (matched-budget) ────────────────────────────
            k = len(harmful_ppi)
            if k > 0:
                t_m = time.perf_counter()
                harmful_ms = _run_minshap_matched(Xv_tr, yv_tr, k, task)
                t_ms = time.perf_counter() - t_m
                prec_m, rec_m, f1_m = precision_recall(harmful_ms, gt_harmful)

                keep_ms = [c for c in Xv_tr.columns if c not in harmful_ms]
                auc_ms = train_eval(Xv_tr[keep_ms], yv_tr, Xv_te[keep_ms], y_te)

                all_rows.append(
                    {
                        "dataset": prefix,
                        "variant": vname,
                        "method": "minshap",
                        "n_full": n_total,
                        "n_gt_harmful": len(gt_harmful),
                        "n_predicted": len(harmful_ms),
                        "precision": prec_m,
                        "recall": rec_m,
                        "f1": f1_m,
                        "baseline_auc": round(baseline_auc, 4),
                        "filtered_auc": round(auc_ms, 4),
                        "delta_auc": round(auc_ms - baseline_auc, 4),
                        "time_s": round(t_ms, 1),
                    }
                )
                print(
                    f"  MinShap: flagged={len(harmful_ms)}  "
                    f"prec={prec_m:.3f}  rec={rec_m:.3f}  f1={f1_m:.3f}  "
                    f"ΔAUC={auc_ms - baseline_auc:+.4f}  [{t_ms:.1f}s]",
                )

                # ── SHAP (matched-budget) ───────────────────────────
                t_s = time.perf_counter()
                harmful_sh = _run_shap_matched(Xv_tr, yv_tr, k, task)
                t_sh = time.perf_counter() - t_s
                prec_s, rec_s, f1_s = precision_recall(harmful_sh, gt_harmful)

                keep_sh = [c for c in Xv_tr.columns if c not in harmful_sh]
                auc_sh = train_eval(Xv_tr[keep_sh], yv_tr, Xv_te[keep_sh], y_te)

                all_rows.append(
                    {
                        "dataset": prefix,
                        "variant": vname,
                        "method": "shap",
                        "n_full": n_total,
                        "n_gt_harmful": len(gt_harmful),
                        "n_predicted": len(harmful_sh),
                        "precision": prec_s,
                        "recall": rec_s,
                        "f1": f1_s,
                        "baseline_auc": round(baseline_auc, 4),
                        "filtered_auc": round(auc_sh, 4),
                        "delta_auc": round(auc_sh - baseline_auc, 4),
                        "time_s": round(t_sh, 1),
                    }
                )
                print(
                    f"  SHAP:    flagged={len(harmful_sh)}  "
                    f"prec={prec_s:.3f}  rec={rec_s:.3f}  f1={f1_s:.3f}  "
                    f"ΔAUC={auc_sh - baseline_auc:+.4f}  [{t_sh:.1f}s]",
                )
            else:
                print(
                    "  MinShap/SHAP: SKIP (MinPPI found 0 harmful → k=0, "
                    "matched-budget comparison not applicable)",
                )

        elapsed = time.perf_counter() - t0
        print(f"\n  {prefix} total: {elapsed:.0f}s")

    # ── Save ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "d1_semisynthetic.csv", index=False)
    print(f"\n  d1_semisynthetic.csv saved ({len(df)} rows)")

    # Aggregated view: mean over variants grouped by dataset-variant-method
    num_cols = [
        "precision",
        "recall",
        "f1",
        "n_predicted",
        "baseline_auc",
        "filtered_auc",
        "delta_auc",
        "time_s",
    ]
    avail = [c for c in num_cols if c in df.columns]
    agg = df.groupby(["dataset", "variant", "method"])[avail].mean().round(4)
    agg.to_csv(out_dir / "d1_semisynthetic_agg.csv")
    print("  d1_semisynthetic_agg.csv saved")

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Summary (MinPPI — D1c injected-harmful detection)")
    print("=" * 72)
    d1c = df[df["variant"] == "D1c-injected"]
    if len(d1c):
        for ds in sorted(d1c["dataset"].unique()):
            sub = d1c[(d1c["dataset"] == ds) & (d1c["method"] == "minppi")]
            if len(sub):
                row = sub.iloc[0]
                print(
                    f"  {ds:15s}: prec={row['precision']:.3f}  "
                    f"rec={row['recall']:.3f}  "
                    f"f1={row['f1']:.3f}  "
                    f"flagged={int(row['n_predicted'])}/{int(row['n_gt_harmful'])}",
                )

    print(f"\nDone.  Output: {out_dir}")
    tee.close()


if __name__ == "__main__":
    main()
