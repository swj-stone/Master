"""Ablate historical minima and conditional refits along the fixed dyadic schedule."""
import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import SEEDS, panels, quality
from experiments.minpi_study import ROOT, file_hash, train
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.piv2 import _folds
from minppi_mofs.refit_minpi import _refit_votes


def refit_halves(X, y, task, seed):
    row_cap = min(1024, max(256, 2 * X.shape[1]))
    halves = _folds(y, task, np.random.default_rng(seed), row_limit=row_cap)
    classes = np.unique(y) if task == "classification" else None
    result = []
    for fit, held in (halves, halves[::-1]):
        training, check = _standardize(X[fit], X[held])
        if classes is None:
            y_fit, y_check = y[fit, None], y[held, None]
        else:
            y_fit, y_check = y[fit, None] == classes, y[held, None] == classes
        intercept = y_fit.mean(axis=0)
        result.append((training, check, y_fit - intercept, y_check - intercept))
    return result


def paths_at_budget(gains, width, count):
    # gains caches the two-fold sums keyed by the ordered, realized support.
    stages = []
    supports = {}
    empty_gain = gains(())
    supports["empty_only"] = np.sort(np.argsort(-empty_gain, kind="stable")[:count])
    for rule in ("refit", "current_only"):
        selected = []
        minimum = np.full(width, np.inf)
        available = np.ones(width, dtype=bool)
        while len(selected) < count:
            gain = gains(tuple(selected))
            minimum = np.minimum(minimum, gain)
            size = len(selected)
            block_size = min(count, max(1, 2 * size)) - size
            pool = np.flatnonzero(available)
            current = pool[np.argsort(-gain[pool], kind="stable")[:block_size]]
            worst = pool[np.argsort(-minimum[pool], kind="stable")[:block_size]]
            if rule == "refit":
                stages.append(dict(size=size, block_size=block_size,
                                   overlap=len(np.intersect1d(worst, current)),
                                   worst=" ".join(map(str, worst)),
                                   current_on_same_support=" ".join(map(str, current))))
                block = worst
            else:
                block = current
            selected.extend(block)
            available[block] = False
        supports[rule] = np.sort(selected)
    return supports, stages


def run(archive, output):
    manifest = json.loads((archive / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in manifest["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Source changed since benchmark: {name}")
    expected_scores = pd.read_csv(archive / "cases.csv")
    expected_scores = expected_scores.query("method == 'refit'").set_index(
        ["phase", "source", "fold", "seed", "budget"])
    output.mkdir(parents=True, exist_ok=False)
    rows, staged = [], []
    for phase in ("development", "replay"):
        for panel in panels(phase, ("gas", "supercon")):
            name = f"{phase}-{panel.source}-{panel.fold}"
            with np.load(archive / f"{name}-split.npz") as split:
                np.testing.assert_array_equal(panel.tr, split["train"])
                np.testing.assert_array_equal(panel.te, split["test"])
                np.testing.assert_array_equal(panel.y[panel.te], split["y"])
            X, y, test = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
            for seed in SEEDS:
                folds = refit_halves(X, y, panel.task, seed)
                cache = {}

                def gains(active):
                    if active not in cache:
                        columns = np.asarray(active, dtype=int)
                        cache[active] = sum(_refit_votes(*fold, columns) for fold in folds)
                    return cache[active]

                with np.load(archive / f"{name}-s{seed}.npz") as saved:
                    for budget in manifest["budgets"]:
                        count = ceil(X.shape[1] * budget / 100)
                        supports, stages = paths_at_budget(gains, X.shape[1], count)
                        meta = dict(phase=phase, source=panel.source, fold=panel.fold,
                                    seed=seed, budget=budget, width=count)
                        staged.extend({**meta, **s} for s in stages)
                        default = supports["refit"]
                        np.testing.assert_array_equal(default, saved[f"b{budget}_refit_columns"])
                        score_default = expected_scores.loc[
                            (phase, panel.source, panel.fold, seed, budget), "score"]
                        predictions = {}
                        for method, columns in supports.items():
                            key = tuple(columns)
                            if key not in predictions:
                                predictions[key] = train("ridge", X[:, columns], y, test[:, columns],
                                                         panel.task, seed)[0]
                            pred = predictions[key]
                            score = quality(panel, pred)
                            if method == "refit":
                                np.testing.assert_array_equal(pred, saved[f"b{budget}_refit_pred"])
                                np.testing.assert_allclose(score, score_default, atol=1e-12, rtol=0)
                            rows.append({**meta, "method": method, "score": score,
                                         "delta_vs_refit": score - score_default,
                                         "same_support": np.array_equal(columns, default),
                                         "overlap_fraction": np.intersect1d(columns, default).size / count,
                                         "columns": " ".join(map(str, columns))})
            print(f"Verified {name}", flush=True)
    cases = pd.DataFrame(rows)
    cases.to_csv(output / "cases.csv", index=False)
    pd.DataFrame(staged).to_csv(output / "stages.csv", index=False)
    keys = ["phase", "source", "budget", "method"]
    by_fold = cases.groupby(keys + ["fold"])[
        ["score", "delta_vs_refit", "same_support", "overlap_fraction"]].mean()
    by_fold.groupby(keys).mean().to_csv(output / "summary.csv")
    sources = [Path(__file__), ROOT / "src/minppi_mofs/refit_minpi.py",
               ROOT / "src/minppi_mofs/masked_loss_minpi.py", ROOT / "src/minppi_mofs/piv2.py"]
    protocol = dict(benchmark=archive.name, benchmark_protocol_sha256=file_hash(archive / "protocol.json"),
                    hashes={str(f.relative_to(ROOT)): file_hash(f) for f in sources},
                    budgets=manifest["budgets"], seeds=SEEDS, phases=["development", "replay"],
                    threads=1, scoring="same outer train/test, feature count, ridge and metrics as archive",
                    refit="reconstruct full weakest-size rule and match archived columns and predictions",
                    current_only="same dyadic block schedule, own path, latest conditional refit gain only",
                    empty_only="only the empty coalition, rank by held-out singleton refit gain",
                    stages="alternative block at identical previously selected columns on full-rule path",
                    timing="not benchmarked")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.archive, args.output)
