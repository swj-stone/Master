"""Replay the single-exemption sequential voting rule from validation contribution archives."""

import argparse
import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import betaincc, betaln
from scipy.stats import rankdata

from minppi_mofs.experiment_protocol import atomic_write_json


ROOT = Path(__file__).resolve().parents[2]
PATHS = 96
AUC_TIE = .005
RISK = .05
TOL = 1e-12


def _auc_columns(margins, positive):
    """One AUC per column; columns represent evaluated prefix masks."""
    ranks = rankdata(margins, axis=0, method="average")
    n = int(positive.sum())
    return (ranks[positive].sum(axis=0) - n * (n + 1) / 2) / (n * (len(positive) - n))


def _schedule_counts(orders, tiers):
    counts = np.zeros((3, orders.shape[1]), dtype=int)
    for order in orders:
        counts[tiers, order] += 1
    if (counts == 0).any():
        raise ValueError("Each feature needs observations in all three size tiers.")
    return counts


def _calibrate(phi, positive, orders, tiers):
    deltas = np.empty(orders.shape, dtype=float)
    for row, order in enumerate(orders):
        aucs = _auc_columns(np.cumsum(phi[:, order], axis=1), positive)
        deltas[row] = np.diff(np.r_[.5, aucs])
    reference = deltas.mean(axis=0)
    excess = deltas - reference
    votes = (excess > AUC_TIE).astype(int) - (excess < -AUC_TIE).astype(int)
    sums = np.zeros((3, phi.shape[1]), dtype=int)
    for order, ballot in zip(orders, votes, strict=True):
        sums[tiers, order] += ballot
    min_vote = (sums / _schedule_counts(orders, tiers)).min(axis=0)
    eligible = (min_vote[:, None] > min_vote[None, :] + TOL).sum(axis=0) > phi.shape[1] / 2
    return reference, min_vote, eligible


def _validate(columns, phi, labels, orders, calibration_orders):
    if (phi.ndim != 2 or len(columns) != phi.shape[1] or phi.shape[1] < 3
            or len(set(columns)) != len(columns) or not np.isfinite(phi).all()):
        raise ValueError("Expected a finite matrix with at least three distinct feature names.")
    if labels.shape != (len(phi),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Expected aligned binary validation labels containing both classes.")
    indices = np.arange(phi.shape[1])
    for schedule in (orders, calibration_orders):
        if (schedule.shape != (PATHS, len(columns)) or not np.issubdtype(schedule.dtype, np.integer)
                or not (np.sort(schedule, axis=1) == indices).all()):
            raise ValueError(f"Expected {PATHS} full feature permutations per schedule.")
    if np.array_equal(orders, calibration_orders):
        raise ValueError("Calibration and monitoring require independently generated schedules.")


def replay(path):
    with np.load(path, allow_pickle=False) as archive:
        columns = archive["columns"].tolist()
        phi = archive["phi"].astype(float)
        labels = archive["y"]
        orders = archive["orders"]
        calibration_orders = archive["repeat_orders"]
    _validate(columns, phi, labels, orders, calibration_orders)
    positive = labels.astype(bool)
    width = len(columns)
    positions = np.arange(width)
    tiers = np.where(positions < width // 3, 0, np.where(positions < 2 * width // 3, 1, 2))
    start = time.monotonic()
    ref, cal_min, eligible = _calibrate(phi, positive, calibration_orders, tiers)
    calibration_s = time.monotonic() - start
    horizon = _schedule_counts(orders, tiers)
    remaining = horizon.copy()
    up = np.zeros_like(horizon)
    down = np.zeros_like(horizon)
    lock = np.zeros(width, dtype=int)
    excluded = np.zeros(width, dtype=int)
    crossed = np.zeros(width, dtype=int)
    full_auc = _auc_columns(phi.sum(axis=1, keepdims=True), positive)[0]

    # Uniform mixture over negative-vote probabilities q in [1/2, 1] among non-ties.
    neg, pos = np.indices((PATHS + 1, PATHS + 1))
    e_table = np.exp((neg + pos + 1) * np.log(2) + betaln(neg + 1, pos + 1)) * betaincc(neg + 1, pos + 1, .5)
    ranked_prefixes = counted_votes = 0
    for step, order in enumerate(orders, 1):
        active = positions[lock[order] == 0]
        endpoints = np.union1d(active, active + 1)
        interior = endpoints[(endpoints > 0) & (endpoints < width)]
        aucs = np.empty(width + 1)
        aucs[0], aucs[-1] = .5, full_auc
        if len(interior):
            # Absorbed features still contribute to the context of active features.
            margins = np.cumsum(phi[:, order], axis=1)
            aucs[interior] = _auc_columns(margins[:, interior - 1], positive)
        features, strata = order[active], tiers[active]
        excess = aucs[active + 1] - aucs[active] - ref[active]
        up[strata, features] += excess > AUC_TIE
        down[strata, features] += excess < -AUC_TIE
        remaining[strata, features] -= 1
        counted_votes += len(active)
        ranked_prefixes += len(interior)

        lower = ((up - down - remaining) / horizon).min(axis=0)
        upper = ((up - down + remaining) / horizon).min(axis=0)
        excluded[(excluded == 0) & (lower.max() > upper + TOL)] = step
        crossed[(crossed == 0) & (e_table[down, up].max(axis=0) >= 3 / RISK - 1e-10)] = step
        lock[(lock == 0) & eligible & (excluded > 0) & (crossed > 0)] = step

    champion = int(np.argmax(np.where(lock > 0, -np.inf, lower)))
    signs = np.where(lock > 0, -1, 0)
    signs[champion] = 1
    stats = dict(width=width, negative=int((signs < 0).sum()), zero=int((signs == 0).sum()),
                 positive=1, champion=columns[champion], champion_min=float(lower[champion]),
                 calibration_paths=PATHS, primary_paths=PATHS, primary_votes=counted_votes,
                 full_primary_votes=PATHS * width, primary_ranked_prefixes=ranked_prefixes,
                 full_primary_ranked_prefixes=PATHS * (width - 1), calibration_s=calibration_s,
                 total_voting_s=time.monotonic() - start)
    decisions = [dict(feature=name, sign=int(signs[i]), negative_lock_round=int(lock[i]) or None,
                      excluded_from_plus_round=int(excluded[i]) or None, evidence_crossing_round=int(crossed[i]) or None,
                      eligible_for_negative=bool(eligible[i]), calibration_min=float(cal_min[i]),
                      final_lower=float(lower[i]), final_upper=float(upper[i])) for i, name in enumerate(columns)]
    return stats, decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, nargs="+", metavar="game_evidence.npz",
                        help="Validation evidence produced by run_relative_voting.py; pass one or more archives.")
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.evidence]
    if len(set(paths)) != len(paths) or not all(p.is_file() for p in paths):
        parser.error("Supply distinct existing validation archives.")
    output = ROOT / "results/test/thesis_sampling" / datetime.now().strftime("online-relative-replay-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "protocol.json", dict(
        paths=PATHS, independent_calibration_paths=PATHS, auc_indifference=AUC_TIE, per_feature_risk=RISK,
        negative_e_threshold=3 / RISK, protected_seats=1, tie_break="archived input column order",
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows = []
    for i, path in enumerate(paths, 1):
        stats, decisions = replay(path)
        stats.update(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        atomic_write_json(output / f"arm-{i:02d}.json", dict(summary=stats, features=decisions))
        rows.append(stats)
        print(f"{path.parent.name}: {stats['negative']} negative, {stats['zero']} zero, 1 protected", flush=True)
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
