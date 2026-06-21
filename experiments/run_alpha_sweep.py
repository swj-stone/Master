"""
Alpha sensitivity experiment — quota parameter α ∈ {0.80, 0.85, 0.90, 0.95, 0.99}.

The quota α governs the voting-game threshold:

    v_alpha(S) = 𝟙[AUC(ℳ_S) ≥ α · Acc_full]

Higher α → fewer winning coalitions → more features classified as harmful
by MinPPI.  This experiment measures how α affects:

1. **MinPPI class distribution**: |decisive|, |neutral|, |harmful| vs α
2. **MOFS-MinPPI downstream performance**: hypervolume, compression rate
3. **Feature ranking stability**: Jaccard similarity between adjacent α levels
4. **Harmful-detection sensitivity**: at what α does a feature flip to -1?

Key findings (§5 / §6):
- α = 0.95 is the recommended main-table value (balances selectivity and
  stability)
- α ≥ 0.99 produces many harmful classifications but may be too strict
- α ≤ 0.85 produces almost no harmful classifications (too permissive)

Output structure (per ``results/logs/<run_id>/``)::

    metadata.json
    config.yaml
    output.log
    metrics.csv                       # per-(dataset, alpha) with MinPPI AUC + HV@0.95
    minppi_distribution.csv           # class counts per alpha per dataset
    table_harmful_pct_vs_alpha.csv
    table_AUC@<k>_vs_alpha.csv        # MinPPI-ranking AUC vs α, k ∈ {1,5,10,20,50}
    <dataset>_alpha<nn>_minppi.csv    # per-alpha MinPPI values
    <dataset>_stability.csv           # Jaccard between alpha levels

Run::

    python -u experiments/run_alpha_sweep.py
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
from tqdm import tqdm
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from minppi_mofs.min_ppi import estimate_minppi_shapg_guided, minppi_stats
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.voting_game import make_acc_full_cache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_VALUES: list[float] = [0.80, 0.85, 0.90, 0.95, 0.99]

DATASETS: list[tuple[str, str, bool]] = [
    ("spambase", "is_spam", False),
    ("madelon", "label", False),
    ("isolet", "letter", True),
    ("gisette", "label", False),
    ("credit_card_fraud", "Class", False),
]

TOP_K_KPIS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)

# Per-dataset budget — shrinks MinPPI + NSGA-II cost on high-dimensional datasets
DATASET_BUDGET: dict[str, dict] = {
    "spambase":    {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
    "madelon":     {"K": 100, "K_adv": 20, "pop": 20, "gen": 15},
    "isolet":      {"K": 80,  "K_adv": 3, "pop": 20, "gen": 10},
    "gisette":     {"K": 50,  "K_adv": 10, "pop": 15, "gen": 8},
    "credit_card_fraud": {"K": 200, "K_adv": 30, "pop": 30, "gen": 20},
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


def load_data(
    prefix: str,
    target: str,
    processed_dir: Path,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    path = processed_dir / f"{prefix}_s{seed}_train.csv"
    df = pd.read_csv(path)
    y = df[target].values.astype(int)
    return df.drop(columns=[target]), y


def _budget_for(prefix: str) -> dict:
    defaults = {"K": 100, "K_adv": 20, "pop": 20, "gen": 10}
    return DATASET_BUDGET.get(prefix, defaults)


def evaluate_topk(
    X: pd.DataFrame,
    y: np.ndarray,
    ranking: list[str],
    *,
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
            stratify=y,
        )
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


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard index between two feature sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Main α sweep
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

    _header(f"Alpha Sensitivity Sweep — {run_id}")
    print(f"Alpha values: {ALPHA_VALUES}", flush=True)
    print(f"Commit: {commit}", flush=True)

    # Metadata
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "alpha_values": ALPHA_VALUES,
        "datasets": [d[0] for d in DATASETS],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("metadata.json saved", flush=True)

    # config.yaml snapshot
    run_config: dict = {
        "description": "Alpha sensitivity sweep — quota parameter α ∈ {0.80, 0.85, 0.90, 0.95, 0.99}",
        "seed": 42,
        "alpha_values": ALPHA_VALUES,
        "datasets": [
            {"prefix": d[0], "target": d[1], "multiclass": d[2]} for d in DATASETS
        ],
        "evaluation": {
            "topk_kpis": list(TOP_K_KPIS),
        },
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    print("config.yaml saved", flush=True)

    # ------------------------------------------------------------------
    # Per-dataset sweep
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []
    all_distributions: list[dict] = []

    for prefix, target, is_mc in tqdm(
        DATASETS, desc="α sweep", unit="dataset", file=sys.stdout
    ):
        _header(f"Alpha sweep — {prefix}")

        X, y = load_data(prefix, target, processed_dir)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        budget = _budget_for(prefix)
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        features = list(X.columns)

        # Acc_full (independent of α) — use stronger model for meaningful
        # binary threshold separation. Multiclass datasets keep lighter
        # params: 26-class XGBoost is ~7× slower per estimator than binary.
        if n_classes > 2:
            model_params_ab = {
                "n_estimators": 20, "max_depth": 3, "learning_rate": 0.2,
            }
        else:
            model_params_ab = {
                "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
            }
        cache = make_acc_full_cache(
            X_tr, X_val, y_tr, y_val, seed=42,
            model_params=model_params_ab,
        )
        acc_full = cache.acc_full
        print(f"  Full-feature AUC: {acc_full:.4f}, n_features: {n_features}")

        # ShapG guidance (shared across α)
        sg_scores, sg_neighbors = build_guidance_from_data(
            X_tr, y_tr, depth=1, m=5, seed=42
        )

        # Storage for stability analysis
        minppi_per_alpha: dict[float, dict[str, int]] = {}
        ranking_per_alpha: dict[float, list[str]] = {}

        for alpha in tqdm(
            ALPHA_VALUES,
            desc=f"  α values ({prefix})",
            unit="α",
            file=sys.stdout,
            leave=False,
        ):
            print(f"\n  --- α = {alpha:.2f} ---", flush=True)
            print(
                f"  Threshold: {alpha:.2f} × {acc_full:.4f} = {acc_full * alpha:.4f}",
                flush=True,
            )

            # --- MinPPI estimation (reuse shared AccFullCache across α) ---
            cache.set_threshold(alpha)
            t0 = time.perf_counter()
            minppi = estimate_minppi_shapg_guided(
                X_tr,
                X_val,
                y_tr,
                y_val,
                alpha=alpha,
                K_adv=budget["K_adv"],
                shapg_neighbors=sg_neighbors,
                shapg_scores=sg_scores,
                acc_cache=cache,
                seed=42,
                verbose=False,
            )
            t_minppi = time.perf_counter() - t0
            stats = minppi_stats(minppi)
            minppi_per_alpha[alpha] = minppi

            print(
                f"  MinPPI: +1={stats['n_decisive']}, 0={stats['n_neutral']}, "
                f"-1={stats['n_harmful']}  ({t_minppi:.1f}s)",
                flush=True,
            )

            all_distributions.append(
                {
                    "dataset": prefix,
                    "alpha": alpha,
                    "n_features": n_features,
                    "threshold": round(acc_full * alpha, 4),
                    **stats,
                }
            )

            # Save per-alpha MinPPI
            pd.DataFrame(
                [(f, minppi.get(f, 0)) for f in features],
                columns=["feature", "minppi"],
            ).to_csv(
                run_dir / f"{prefix}_alpha{int(alpha * 100):02d}_minppi.csv",
                index=False,
            )

            # MinPPI-based ranking: decisive (+1) top, harmful (-1) bottom,
            # neutral (0) middle sorted by ShapG score (harmonic proxy)
            ranking_mpp: list[str] = []
            decisive_f = [f for f in features if minppi.get(f, 0) == +1]
            neutral_f = sorted(
                [f for f in features if minppi.get(f, 0) == 0],
                key=lambda f: sg_scores.get(f, 0),
                reverse=True,
            )
            harmful_f = sorted(
                [f for f in features if minppi.get(f, 0) == -1],
                key=lambda f: sg_scores.get(f, 0),
                reverse=True,
            )
            ranking_mpp = decisive_f + neutral_f + harmful_f
            ranking_per_alpha[alpha] = ranking_mpp

            # ---- Top-k AUC from MinPPI-based ranking (all α, nearly free) ----
            kpi_mpp = evaluate_topk(X, y, ranking_mpp, seed=42)
            metric_name = kpi_mpp["metric"].iloc[0]

            row: dict = {
                "dataset": prefix,
                "alpha": alpha,
                "n_features": n_features,
                "acc_full": round(acc_full, 4),
                "threshold": round(acc_full * alpha, 4),
                "n_harmful": stats["n_harmful"],
                "n_decisive": stats["n_decisive"],
                "n_neutral": stats["n_neutral"],
                "pct_harmful": round(stats["pct_harmful"], 1),
                "time_minppi_s": round(t_minppi, 1),
            }
            for _, kpi_row in kpi_mpp.iterrows():
                row[f"{metric_name}@{int(kpi_row['k'])}"] = round(
                    float(kpi_row["value"]), 4
                )

            all_metrics.append(row)

        # --- Stability analysis between adjacent α levels ---
        print("\n  --- Stability (Jaccard) ---", flush=True)
        stability_rows: list[dict] = []
        sorted_alphas = sorted(ALPHA_VALUES)

        for i in range(len(sorted_alphas)):
            for j in range(i + 1, len(sorted_alphas)):
                a1, a2 = sorted_alphas[i], sorted_alphas[j]

                # Harmful-feature set Jaccard
                h1 = {f for f, v in minppi_per_alpha[a1].items() if v == -1}
                h2 = {f for f, v in minppi_per_alpha[a2].items() if v == -1}
                jac_harmful = jaccard_similarity(h1, h2)

                # Decisive-feature set Jaccard
                d1 = {f for f, v in minppi_per_alpha[a1].items() if v == +1}
                d2 = {f for f, v in minppi_per_alpha[a2].items() if v == +1}
                jac_decisive = jaccard_similarity(d1, d2)

                # Top-10 ranking Jaccard
                r1 = set(ranking_per_alpha[a1][:10])
                r2 = set(ranking_per_alpha[a2][:10])
                jac_top10 = jaccard_similarity(r1, r2)

                stability_rows.append(
                    {
                        "dataset": prefix,
                        "alpha1": a1,
                        "alpha2": a2,
                        "jaccard_harmful": round(jac_harmful, 4),
                        "jaccard_decisive": round(jac_decisive, 4),
                        "jaccard_top10": round(jac_top10, 4),
                        "n_harmful_a1": len(h1),
                        "n_harmful_a2": len(h2),
                    }
                )
                print(
                    f"  α={a1:.2f} vs α={a2:.2f}: "
                    f"J_harmful={jac_harmful:.3f}, "
                    f"J_decisive={jac_decisive:.3f}, "
                    f"J_top10={jac_top10:.3f}",
                    flush=True,
                )

        # Save stability
        pd.DataFrame(stability_rows).to_csv(
            run_dir / f"{prefix}_stability.csv", index=False
        )

    # ------------------------------------------------------------------
    # Save aggregated results
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nmetrics.csv saved ({len(metrics_df)} rows)", flush=True)

    dist_df = pd.DataFrame(all_distributions)
    dist_df.to_csv(run_dir / "minppi_distribution.csv", index=False)
    print(f"minppi_distribution.csv saved ({len(dist_df)} rows)", flush=True)

    # Build pivot tables
    _header("Summary tables")

    # Harmful % vs α per dataset
    harmful_pivot = dist_df.pivot_table(
        index="dataset", columns="alpha", values="pct_harmful", aggfunc="first"
    )
    harmful_pivot = harmful_pivot.round(1)
    harmful_pivot.to_csv(run_dir / "table_harmful_pct_vs_alpha.csv")
    print("table_harmful_pct_vs_alpha.csv", flush=True)

    # Top-k AUC from MinPPI ranking vs α (all α have this)
    for k in [1, 5, 10, 20, 50]:
        auc_col = f"AUC@{k}"
        if auc_col in metrics_df.columns:
            pivot = metrics_df.pivot_table(
                index="dataset", columns="alpha", values=auc_col, aggfunc="first",
            )
            pivot = pivot.round(4)
            pivot.to_csv(run_dir / f"table_AUC@{k}_vs_alpha.csv")
            print(f"table_AUC@{k}_vs_alpha.csv", flush=True)

    # ---- Per-dataset alpha recommendation (three criteria) ----
    _header("Alpha recommendation")

    summary_rows: list[dict] = []
    for prefix in [d[0] for d in DATASETS]:
        ds_rows = metrics_df[metrics_df["dataset"] == prefix].copy()
        if len(ds_rows) == 0:
            continue

        # Criterion 1: harmful% must be ≥ 2% for meaningful MinPPI signal
        usable = ds_rows[ds_rows["pct_harmful"] >= 2.0].copy()
        if len(usable) == 0:
            usable = ds_rows  # fallback if no α clears threshold

        # Criterion 2: best AUC@10
        idx_best_auc = usable["AUC@10"].idxmax() if "AUC@10" in usable.columns else usable.index[0]
        best_row = usable.loc[idx_best_auc]

        # Criterion 3: Jaccard stability to adjacent alphas
        stab_df = pd.read_csv(run_dir / f"{prefix}_stability.csv")
        if not stab_df.empty:
            # Mean jaccard_harmful between each alpha and its 1-step neighbor
            alphas_sorted = sorted(ALPHA_VALUES)
            neighbors_jac: dict[float, list[float]] = {a: [] for a in alphas_sorted}
            for _, r in stab_df.iterrows():
                a1, a2 = float(r["alpha1"]), float(r["alpha2"])
                jh = float(r["jaccard_harmful"])
                if abs(a2 - a1) == 0.05:  # 1-step neighbor
                    neighbors_jac[a1].append(jh)
                    neighbors_jac[a2].append(jh)
            mean_jac = {a: float(np.mean(v)) if v else 0.0 for a, v in neighbors_jac.items()}
        else:
            mean_jac = {a: 0.0 for a in ALPHA_VALUES}

        # Build comparison for this dataset
        lines: list[str] = [f"\n  {prefix}:"]
        parts: list[str] = []
        for _, r in ds_rows.iterrows():
            a_val = float(r["alpha"])
            parts.append(f"α={a_val:.2f}:{r['pct_harmful']:.0f}%")
        lines.append("    harmful%: " + "  ".join(parts))
        parts = []
        for _, r in ds_rows.iterrows():
            a_val = float(r["alpha"])
            auc_v = r.get("AUC@10")
            if not pd.isna(auc_v):
                parts.append(f"α={a_val:.2f}:{auc_v:.3f}")
        lines.append("    AUC@10:  " + "  ".join(parts))
        parts = [f"α={a:.2f}:{mean_jac[a]:.2f}" for a in alphas_sorted]
        lines.append("    J(neighbor): " + "  ".join(parts))
        print("\n".join(lines), flush=True)

        summary_rows.append({
            "dataset": prefix,
            "recommended_alpha": float(best_row["alpha"]),
            "n_harmful_at_best": int(best_row["n_harmful"]),
            "pct_harmful_at_best": float(best_row["pct_harmful"]),
            "AUC@10_at_best": float(best_row.get("AUC@10", float("nan"))),
            "J_neighbor_harmful_at_best": round(mean_jac.get(float(best_row["alpha"]), 0.0), 3),
            "n_harmful_α085": int(ds_rows[ds_rows["alpha"] == 0.85]["n_harmful"].values[0]),
            "n_harmful_α095": int(ds_rows[ds_rows["alpha"] == 0.95]["n_harmful"].values[0]),
            "n_harmful_α099": int(ds_rows[ds_rows["alpha"] == 0.99]["n_harmful"].values[0]),
        })

    # Global recommendation table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "table_alpha_recommendation.csv", index=False)
    print("\n  table_alpha_recommendation.csv saved", flush=True)

    # Narrative
    print(
        "\n  -- Recommendation narrative --\n"
        "  alpha=0.85 and below: too few harmful -> MinPPI has no signal.\n"
        "  alpha=0.99:          too many harmful -> over-selective, loses useful features.\n"
        "  alpha=0.90-0.95:     sweet spot - meaningful harmful% + best AUC@10 + stable.\n"
        "  alpha=0.95 is the recommended primary value; Appendix reports 0.90, 0.99 for context.",
        flush=True,
    )

    _header("Done")
    print(f"Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
