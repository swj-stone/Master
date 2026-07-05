"""
Pipeline comparison: MinPPI vs. MinShap vs. Random preprocessing (P0.3).

Answers: "If I use MinShap's negative set as a pre-filter instead of MinPPI,
does downstream FS get better or worse?"

Three pipelines per dataset (seed 42 representative filter, tested on 5 seeds):
  A (MinPPI):   delete MinPPI−1 features   → SHAP-top-k on remaining
  B (MinShap):  delete MinShap<0 features   → SHAP-top-k on remaining
  C (Random):   delete N random features    → SHAP-top-k (10 trials)
  Baseline:     SHAP-top-k on full features (no deletion)

Output (results/tables/<run_id>/)::

    pipeline_comparison.csv     # per-(dataset, seed, pipeline)
    pipeline_comparison_agg.csv # mean over seeds
    pipeline_comparison.pdf     # ΔAUC bar chart

Run::

    python experiments/eval_pipeline_comparison.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tqdm import tqdm

from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
from minppi_mofs.baselines.min_shap import compute_minshap_importance

SEEDS: list[int] = [42, 123, 456, 789, 1024]
TOP_K: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
N_RAND = 10

CLS_DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
    ("gisette", "label", "binary"),
    ("credit_card_fraud", "Class", "binary"),
]

# Per-dataset Algorithm-B adversarial search budget (consistent with
# run_pareto_eval.py and run_main_table.py).
DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K_adv": 30},
    "madelon": {"K_adv": 20},
    "isolet": {"K_adv": 8},
    "gisette": {"K_adv": 10},
    "credit_card_fraud": {"K_adv": 30},
}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate_topk(X_test, y_test, ranking, *, task="binary", seed=42):
    n_classes = len(np.unique(y_test))
    result = {}
    for k in TOP_K:
        feats = ranking[: min(k, len(ranking))]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_test[feats].values,
            y_test,
            test_size=0.2,
            random_state=seed,
            stratify=y_test if task != "regression" else None,
        )
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
        result[f"AUC@{k}"] = round(val, 4)
    return result


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    run_id = _now_stamp()
    out_dir = project_root / "results" / "tables" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit(project_root.parent)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {"run_id": run_id, "commit": commit, "seeds": SEEDS, "n_random": N_RAND},
            indent=2,
        ),
        encoding="utf-8",
    )

    all_rows: list[dict] = []

    for prefix, target, task in tqdm(
        CLS_DATASETS, desc="pipe-comp", unit="ds", file=sys.stdout
    ):
        # Representative filter: seed 42
        tr42 = pd.read_csv(processed_dir / f"{prefix}_s42_train.csv")
        va42 = pd.read_csv(processed_dir / f"{prefix}_s42_val.csv")
        X_r = tr42.drop(columns=[target])
        y_r = tr42[target].values
        Xv_r = va42.drop(columns=[target])
        yv_r = va42[target].values
        n_full = X_r.shape[1]

        # MinPPI (seed 42)
        t0 = time.perf_counter()
        if task == "multiclass":
            from minppi_mofs.min_ppi import estimate_minppi_random_mc

            minppi_d = estimate_minppi_random_mc(
                X_r,
                Xv_r,
                y_r,
                yv_r,
                alpha=0.95,
                K=200,
                shared_coalitions=True,
                per_coalition=5,
                seed=42,
                verbose=False,
                task=task,
            )
        else:
            from minppi_mofs.min_ppi import estimate_minppi

            sg_s, sg_n = build_guidance_from_data(X_r, y_r, depth=1, m=5, seed=42)
            budget = DATASET_BUDGET.get(prefix, {"K_adv": 20})
            minppi_d = estimate_minppi(
                X_r,
                Xv_r,
                y_r,
                yv_r,
                alpha=0.95,
                method="shapg_guided",
                K_adv=budget["K_adv"],
                shapg_neighbors=sg_n,
                shapg_scores=sg_s,
                seed=42,
                verbose=False,
                task=task,
            )
        h_ppi = {f for f, v in minppi_d.items() if v == -1}
        n_h = len(h_ppi)
        print(f"\n  {prefix}: MinPPI={n_h} harmful ({time.perf_counter() - t0:.0f}s)")

        # MinShap (seed 42)
        ms = compute_minshap_importance(
            X_r, y_r, n_coalitions=100, per_coalition=5, seed=42, task=task
        )
        h_ms = {f for f, v in ms.items() if v < 0}
        if not h_ms:
            ranked = sorted(ms.items(), key=lambda kv: kv[1])
            h_ms = {f for f, _ in ranked[: max(1, n_h)]}
        print(f"    MinShap: {len(h_ms)} negative")

        # Test on all 5 seeds
        for seed in tqdm(
            SEEDS, desc=f"  seeds ({prefix})", unit="sd", file=sys.stdout, leave=False
        ):
            tr = pd.read_csv(processed_dir / f"{prefix}_s{seed}_train.csv")
            te = pd.read_csv(processed_dir / f"{prefix}_s{seed}_test.csv")
            Xt_full = tr.drop(columns=[target])
            yt = tr[target].values
            Xe_full = te.drop(columns=[target])
            ye = te[target].values

            def _pipe_row(pname, removed, aucs):
                return {
                    "dataset": prefix,
                    "seed": seed,
                    "pipeline": pname,
                    "n_removed": len(removed),
                    "n_full": n_full,
                    "cr": len(removed) / n_full if n_full else 0,
                    **aucs,
                }

            # Baseline — full
            sf = compute_shap_topk_importance(Xt_full, yt, seed=seed, task=task)
            rf = [f for f, _ in sorted(sf.items(), key=lambda kv: kv[1], reverse=True)]
            bl_auc = evaluate_topk(Xe_full, ye, rf, task=task, seed=seed)
            all_rows.append(_pipe_row("none", set(), bl_auc))

            # A — MinPPI filter
            if h_ppi:
                k_ppi = [c for c in Xt_full.columns if c not in h_ppi]
                sp = compute_shap_topk_importance(
                    Xt_full[k_ppi], yt, seed=seed, task=task
                )
                rp = [
                    f for f, _ in sorted(sp.items(), key=lambda kv: kv[1], reverse=True)
                ]
                au = evaluate_topk(Xe_full[k_ppi], ye, rp, task=task, seed=seed)
                all_rows.append(_pipe_row("minppi", h_ppi, au))
            else:
                all_rows.append(_pipe_row("minppi", set(), bl_auc))

            # B — MinShap filter
            if h_ms:
                k_ms = [c for c in Xt_full.columns if c not in h_ms]
                if k_ms:
                    sm = compute_shap_topk_importance(
                        Xt_full[k_ms], yt, seed=seed, task=task
                    )
                    rm = [
                        f
                        for f, _ in sorted(
                            sm.items(), key=lambda kv: kv[1], reverse=True
                        )
                    ]
                    au_m = evaluate_topk(Xe_full[k_ms], ye, rm, task=task, seed=seed)
                    all_rows.append(_pipe_row("minshap", h_ms, au_m))
                else:
                    all_rows.append(_pipe_row("minshap", h_ms, {}))
            else:
                all_rows.append(_pipe_row("minshap", set(), bl_auc))

            # C — Random filter (N = n_harmful)
            rng = np.random.default_rng(seed)
            all_f = list(Xt_full.columns)
            for _ in range(N_RAND):
                rr = set(rng.choice(all_f, size=max(1, n_h), replace=False))
                kr = [c for c in all_f if c not in rr]
                sr = compute_shap_topk_importance(Xt_full[kr], yt, seed=seed, task=task)
                rr_rank = [
                    f for f, _ in sorted(sr.items(), key=lambda kv: kv[1], reverse=True)
                ]
                ar = evaluate_topk(Xe_full[kr], ye, rr_rank, task=task, seed=seed)
                all_rows.append(_pipe_row("random", rr, ar))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "pipeline_comparison.csv", index=False)
    print(f"\n  pipeline_comparison.csv saved ({len(df)} rows)")

    kpis = [c for c in df.columns if "@" in c]
    agg = df.groupby(["dataset", "pipeline"])[kpis].mean().round(4)
    agg.to_csv(out_dir / "pipeline_comparison_agg.csv")
    print("  pipeline_comparison_agg.csv saved")

    # ΔAUC@10 summary
    if "AUC@10" in df.columns:
        print("\n=== ΔAUC@10 (pipeline − baseline) ===")
        base = df[df["pipeline"] == "none"].groupby("dataset")["AUC@10"].mean()
        for pn in ["minppi", "minshap", "random"]:
            sub = df[df["pipeline"] == pn]
            if not len(sub):
                continue
            pm = sub.groupby("dataset")["AUC@10"].mean()
            for ds in pm.index:
                d = pm[ds] - base.get(ds, 0)
                c = sub[sub["dataset"] == ds]["cr"].mean()
                print(f"  {ds:20s} {pn:8s}: ΔAUC@10={d:+.4f}  cr={c * 100:.1f}%")

    # Plot
    ds_names = sorted(df["dataset"].unique())
    pipe_names = ["minppi", "minshap", "random"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    fig, axes = plt.subplots(
        1, len(ds_names), figsize=(3.2 * len(ds_names), 3.5), squeeze=False
    )
    for ai, ds in enumerate(ds_names):
        ax = axes[0, ai]
        bv = df[(df["dataset"] == ds) & (df["pipeline"] == "none")]["AUC@10"].mean()
        vals = []
        for pi, pn in enumerate(pipe_names):
            sub = df[(df["dataset"] == ds) & (df["pipeline"] == pn)]
            vals.append(sub["AUC@10"].mean() - bv if len(sub) else 0)
        ax.bar(
            range(len(pipe_names)), vals, color=colors, edgecolor="black", linewidth=0.4
        )
        ax.set_xticks(range(len(pipe_names)))
        ax.set_xticklabels(["MinPPI", "MinShap", "Random"], fontsize=7, rotation=15)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(ds, fontsize=9)
        if ai == 0:
            ax.set_ylabel("ΔAUC@10")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fp = out_dir / "pipeline_comparison.pdf"
    fig.savefig(fp, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp.name}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
