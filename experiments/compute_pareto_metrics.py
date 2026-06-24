"""
Post-process a main-table run: compute IGD / IGD+ / Spacing for every NSGA-II
method against the per-dataset union reference front, and merge runtime.

Depends on the Pareto fronts saved by ``run_main_table.py`` as
``results/logs/<run_id>/<method>/<prefix>_s<seed>_pareto.csv`` (columns
``n_features,one_minus_quality,time_s``) and on ``metrics.csv`` for runtime.

Outputs (under the same run dir / its tables dir):
  - ``pareto_metrics.csv``        per (method, dataset, seed): igd, igd_plus,
                                  spacing, hv (if present), runtime_s
  - ``pareto_metrics_mean.csv``   mean +/- std over seeds
  - ``table_igd_plus.csv`` ...    dataset x method pivots (igd_plus, spacing)

Run (after run_main_table.py):
    python experiments/compute_pareto_metrics.py results/logs/<run_id>
If no run dir is given, the most recent results/logs/<run_id> is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto_metrics import build_reference_front, igd, normalizer, spacing  # noqa: E402

NSGA2_METHODS = ("mofs_minppi", "mofs_shap", "nsga2_uniform")


def _latest_run(project_root: Path) -> Path:
    logs = sorted((project_root / "results" / "logs").glob("*/"))
    if not logs:
        raise SystemExit("no results/logs/<run_id> found; run run_main_table.py first")
    return logs[-1]


def _load_fronts(run_dir: Path) -> dict[tuple[str, str, int], np.ndarray]:
    """Map (method, dataset, seed) -> fitness array from saved pareto csvs."""
    fronts: dict[tuple[str, str, int], np.ndarray] = {}
    for method in NSGA2_METHODS:
        for path in (run_dir / method).glob("*_pareto.csv"):
            # filename: <prefix>_s<seed>_pareto.csv  (prefix may contain '_')
            stem = path.name[: -len("_pareto.csv")]
            prefix, _, seed_tok = stem.rpartition("_s")
            try:
                seed = int(seed_tok)
            except ValueError:
                continue
            arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            if arr.size:
                fronts[(method, prefix, seed)] = arr
    return fronts


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _latest_run(project_root)
    print(f"run dir: {run_dir}")

    fronts = _load_fronts(run_dir)
    if not fronts:
        raise SystemExit(
            "no *_pareto.csv found. Re-run run_main_table.py with the front-saving "
            "patch applied (it writes <method>/<prefix>_s<seed>_pareto.csv)."
        )
    datasets = sorted({d for (_, d, _) in fronts})

    # Per-dataset reference front + normaliser (union of ALL methods & seeds)
    ref: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for d in datasets:
        union = [f for (m, dd, s), f in fronts.items() if dd == d]
        rf = build_reference_front(union)
        lo, span = normalizer(rf)
        ref[d] = (rf, lo, span)
        print(f"  {d}: reference front |Z|={len(rf)} from {len(union)} runs")

    # Runtime from metrics.csv (search time per method/dataset/seed)
    runtime: dict[tuple[str, str, int], float] = {}
    mpath = run_dir / "metrics.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        for _, r in m.iterrows():
            runtime[(r["method"], r["dataset"], int(r["seed"]))] = float(r.get("time_s", np.nan))

    rows: list[dict] = []
    for (method, d, seed), front in sorted(fronts.items()):
        rf, lo, span = ref[d]
        rows.append({
            "method": method,
            "dataset": d,
            "seed": seed,
            "igd": round(igd(front, rf, plus=False, lo=lo, span=span), 6),
            "igd_plus": round(igd(front, rf, plus=True, lo=lo, span=span), 6),
            "spacing": round(spacing(front, lo=lo, span=span), 6),
            "front_size": len(front),
            "runtime_s": runtime.get((method, d, seed), np.nan),
        })
    df = pd.DataFrame(rows)

    tables_dir = project_root / "results" / "tables" / run_dir.name
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "pareto_metrics.csv", index=False)
    print(f"\npareto_metrics.csv ({len(df)} rows) -> {tables_dir}")

    metrics = ["igd", "igd_plus", "spacing", "runtime_s"]
    g = df.groupby(["method", "dataset"])[metrics]
    mean = g.mean().round(6)
    std = g.std().fillna(0).round(6)
    out = mean.copy()
    for c in metrics:
        out[f"{c}_std"] = std[c]
    out.reset_index().to_csv(tables_dir / "pareto_metrics_mean.csv", index=False)
    print(f"pareto_metrics_mean.csv -> {tables_dir}")

    for c in ("igd_plus", "spacing"):
        piv = mean.reset_index().pivot(index="dataset", columns="method", values=c).round(6)
        piv.to_csv(tables_dir / f"table_{c}.csv")
        print(f"table_{c}.csv -> {tables_dir}")


if __name__ == "__main__":
    main()
