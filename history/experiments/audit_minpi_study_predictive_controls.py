"""Verify the predictive controls, auxiliary materials surrogate and paired-support counts."""
import argparse
import json
from fractions import Fraction
from functools import cache
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score
from threadpoolctl import threadpool_limits

from experiments.minpi_study import P
from experiments.minpi_study_new_cohorts import LAYERS, SEEDS, load_cohort
from experiments.minpi_study_predictive_controls import RESULTS
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.piv2 import _folds


@cache
def signed_numerator(positive, negative, neutral, sign, k):
    """Count each peer composition with its literal signed change in majority."""
    positive -= sign == 1
    negative -= sign == -1
    total = 0
    for a in range(min(positive, k) + 1):
        for b in range(min(negative, k-a) + 1):
            c = k - a - b
            if c <= neutral:
                old, new = a - b, a - b + sign
                delta = int(new > 0) - int(new < 0) - int(old > 0) + int(old < 0)
                total += delta * comb(positive, a) * comb(negative, b) * comb(neutral, c)
    return total


def exact_layers(votes):
    w, p = votes.shape
    profiles = [(int((row == 1).sum()), int((row == -1).sum()), int((row == 0).sum()))
                for row in votes]
    layers = []
    for k in LAYERS:
        totals = [0] * p
        for profile, row in zip(profiles, votes.tolist(), strict=True):
            weights = {s: signed_numerator(*profile, s, k) for s in set(row) if s}
            for j, sign in enumerate(row):
                if sign:
                    totals[j] += weights[sign]
        layers.append([Fraction(total, w * comb(p-1, k)) for total in totals])
    return layers


def audit_materials():
    X, y, group, train, test, task = load_cohort("supercon")
    seed = 3469
    with np.load(RESULTS / f"new-cohorts/supercon-{seed}.npz") as archive:
        held = np.sign(witness_margins(X[test], y[test], task, seed + 10000)).astype(np.int8)
        np.testing.assert_array_equal(held, archive["held_votes"])
        assessment = exact_layers(held)
        weakest = list(map(min, zip(*assessment, strict=True)))
        np.testing.assert_allclose([float(x) for x in weakest], archive["held_min"], atol=1e-14,
                                   rtol=1e-13)
        rates_train = exact_layers(archive["train_votes"])
        train_min = list(map(min, zip(*rates_train, strict=True)))
        records = {}
        for method, rank in (("min", train_min), ("largest", rates_train[-1])):
            S = sorted(sorted(range(X.shape[1]), key=rank.__getitem__, reverse=True)[:9])
            np.testing.assert_array_equal(S, archive[f"columns_{method}"])
            a = sum(weakest[j] for j in S) / len(S)
            alt = min(sum(layer[j] for j in S) / len(S) for layer in assessment)
            quality = r2_score(y[test], archive[f"prediction_{method}"])
            records[method] = dict(columns=S, assessment_fraction=str(a), assessment=float(a),
                                   min_of_layer_means=float(alt), pooled_r2=quality)
        records["test_rows"] = len(test)
        records["test_groups"] = int(np.unique(group[test]).size)
        records["assessment_shape"] = list(held.shape)
        ref, query = _folds(y[test], task, np.random.default_rng(seed + 10000), row_limit=256)
        assert len(np.intersect1d(ref, query)) == 0
        records["assessment_reference_rows"] = test[ref].tolist()
        records["assessment_query_rows"] = test[query].tolist()
        records["assessment_definition"] = "mean over selected columns of their minimum held-out layer"
        records["assessment_population"] = ("The two supports share one assessment matrix built "
                                            "only from labelled prediction test rows.")
        records["assessment_seed"] = seed + 10000
        records["selection_seed"] = seed
        records["group_overlap"] = int(np.intersect1d(group[test], group[train]).size)
    return records


def support_agreement():
    panels = []
    for name in P["primary_sources"]:
        for seed in P["formal_seeds"]:
            with np.load(RESULTS / f"matched/formal-{name}-s{seed}.npz") as archive:
                agree = np.array_equal(archive["exact_support"], archive["k16_support"])
                panels.append((name, seed, agree))
    for name in ("gas", "supercon"):
        for seed in SEEDS:
            with np.load(RESULTS / f"new-cohorts/{name}-{seed}.npz") as archive:
                agree = np.array_equal(archive["columns_min"], archive["columns_largest"])
                panels.append((name, seed, agree))
    device = pd.read_csv(RESULTS / "device/supports.csv").set_index("method")
    panels.append(("mhealth", 0, device.at["minimum", "indices"] == device.at["largest", "indices"]))
    exact_records = pd.read_csv(RESULTS / "order-diagnostics/cases.csv").query("panel=='full'")
    references = {(r.dataset, r.seed): r.min_largest_same for r in exact_records.itertuples()}
    assert len(panels) == len(references)
    assert all(agree == references[(name, seed)] for name, seed, agree in panels)
    measured = [item for item in panels if item[0] != "madelon"]
    return {"measured_panels": len(measured), "agreed": sum(int(a) for _, _, a in measured),
            "different": [(name, seed) for name, seed, agree in measured if not agree],
            "additional_artificial_madelon_panels": len(panels) - len(measured)}


def audit_draws(stage):
    summary = pd.read_csv(stage / "summary.csv").set_index("source")
    for name in ("gas", "supercon", "device"):
        with np.load(stage / f"{name}-draws.npz") as archive:
            y, g = archive["test_y"], archive["test_groups"]
            predictions, supports = archive["predictions"], archive["columns"]
            scores = []
            for cols, pred in zip(supports, predictions, strict=True):
                assert np.all(np.diff(cols) > 0)
                if name == "supercon":
                    quality = r2_score(y, pred)
                else:
                    devices = [pred] if name == "gas" else pred
                    per_unit = []
                    for group in np.unique(g):
                        keep = group == g
                        per_unit.append(min(balanced_accuracy_score(y[keep], d[keep])
                                            for d in devices))
                    quality = np.mean(per_unit)
                scores.append(quality)
        row = summary.loc[name]
        np.testing.assert_allclose(np.quantile(scores, [.05, .5, .95]),
                                   [row.q05, row["median"], row.q95], atol=1e-13)
        np.testing.assert_allclose(np.mean(scores), row["mean"], atol=1e-13)
        assert np.count_nonzero(np.asarray(scores) > row.minpi) == row.above_minpi
    return {name: int(row.above_minpi) for name, row in summary.iterrows()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS / "predictive-controls")
    stage = parser.parse_args().input
    with threadpool_limits(1):
        raw = pd.read_csv(RESULTS / "arithmetic-consequence/gas-predictions.csv")
        paired = {}
        for batch, rows in raw.groupby("batch"):
            paired[int(batch)] = (balanced_accuracy_score(rows.target, rows["exact"])
                                  - balanced_accuracy_score(rows.target, rows["float"]))
        report = {"primary_random_control": {
                     "gas_and_materials": "first h of NumPy default_rng(1129).permutation(p)",
                     "device": "fourth child of SeedSequence(20261123), choice(40, 4, replace=False)",
                     "scope": "one uniform subset per source, reused for every held-out unit"},
                  "random_strict_exceedances": audit_draws(stage), "materials": audit_materials(),
                  "support_agreement": support_agreement(), "gas_paired_ba": paired,
                  "gas_equal_batch_mean": float(np.mean(list(paired.values())))}
    path = stage / "audit.json"
    serialized = json.dumps(report, indent=2)
    if path.is_file() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"Existing audit disagrees with recomputation: {path}")
    path.write_text(serialized, encoding="utf-8")
    print(f"Verified the archived predictions, materials scores, and support comparisons: {path.name}")
