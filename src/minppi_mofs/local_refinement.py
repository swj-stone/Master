"""Fixed-budget validation of local exchanges around a frozen MinPI subset."""

from collections.abc import Mapping
import math
from numbers import Integral

import pandas as pd

from minppi_mofs.removal_selection import _feature_order, _gains, _scores, _supplemental_features


def _exchange_plans(features, bits, incumbent_mask, removed_before, eligible, witnesses, zeros,
                    gain, shap_importance, recovery_scores, retained_gain):
    gain = _gains(features, gain)
    inner_gain = _gains(features, retained_gain)
    if not isinstance(shap_importance, Mapping) or set(shap_importance) != set(features):
        raise ValueError("TreeSHAP importances must cover all features")
    shap = _gains(features, shap_importance)
    restorable = [f for f in features if f in removed_before and f in zeros]
    if not isinstance(recovery_scores, Mapping) or not set(restorable) <= recovery_scores.keys():
        raise ValueError("training recovery scores must cover every removed sampled zero")
    recovery = {f: float(recovery_scores[f]) for f in restorable}
    if not all(math.isfinite(v) for v in recovery.values()):
        raise ValueError("recovery scores must be finite")
    restore_order = sorted(restorable, key=lambda f: (-recovery[f], -shap[f], -gain[f]))
    droppable = [f for f in features if f in eligible and f not in removed_before]
    full_rank = pd.DataFrame({"gain": gain, "shap": shap}).rank(method="average").mean(axis=1)
    drop_orders = {
        "retained_gain": sorted(droppable, key=lambda f: (inner_gain[f], f not in witnesses)),
        "full_fused_rank": sorted(droppable, key=lambda f: (full_rank[f], f not in witnesses)),
    }

    # Interleave the two rankings for each operation, so a deadline does not
    # exhaust all operations from one ranking before the other starts.
    plans = {}
    operations = (("swap_one", 1, 0, 1), ("swap_two", 2, 0, 2),
                  ("swap_second", 1, 1, 1), ("drop_one", 1, 0, 0))
    for op, n_drop, start, n_restore in operations:
        restored = restore_order[start:start + n_restore]
        if len(restored) != n_restore:
            continue
        for path, order in drop_orders.items():
            if len(order) < n_drop:
                continue
            dropped = order[:n_drop]
            mask = incumbent_mask ^ sum(bits[f] for f in dropped + restored)
            if mask:
                plans.setdefault(mask, {"op": op, "sources": [], "newly_removed_features": dropped,
                                        "restored_features": restored})["sources"].append(f"{path}:{op}")

    return plans, drop_orders, restore_order, recovery


def select_local_refinement(features, sample_minima, incumbent_mask, gain, shap_importance,
                            recovery_scores, retained_gain, evaluate, incumbent_scores, full_scores,
                            margin, *, publish=None, allow_next=None, previous_trials=(), skip_ranking=False):
    """Compare two drop rankings and three exchanges or a single extra deletion per ranking.

    Rankings use training data only. A mask bit of one means retain the feature.
    evaluate(mask) returns scores on the whole validation partition and its two
    fixed halves. A candidate must pass the full-reference margin and improve the
    incumbent's mean half-score without a loss on any of the three coordinates.
    Only completed trials may replace the incumbent when allow_next() stops the search.
    The validation_frontier comparator may also choose a previously scored eligible
    mask. Only the retained mask and the three validation scores are consumed. If
    the training-ranking fit cannot start in budget, skip_ranking leaves the
    frontier comparison available and records an unknown planned local count.
    """
    features = _feature_order(features)
    full = (1 << len(features)) - 1
    if (isinstance(incumbent_mask, bool) or not isinstance(incumbent_mask, Integral)
            or not 0 < incumbent_mask <= full):
        raise ValueError("invalid incumbent retained mask")
    incumbent_mask = int(incumbent_mask)
    if not isinstance(sample_minima, Mapping):
        raise ValueError("explicit sample_minima are required")
    witnesses = {f for f, v in sample_minima.items() if v == -1}
    zeros = _supplemental_features(features, witnesses, sample_minima)
    eligible = witnesses | zeros
    bits = {f: 1 << i for i, f in enumerate(features)}
    removed_before = {f for f in features if not incumbent_mask & bits[f]}
    if not removed_before <= eligible:
        raise ValueError("incumbent removed protected features")

    reference = _scores(full_scores, 3)
    incumbent = _scores(incumbent_scores, 3)
    margin = float(margin)
    tol = 1e-12
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if any(value < ref - margin - tol for value, ref in zip(incumbent, reference, strict=True)):
        raise ValueError("incumbent fails its original validation gate")
    if incumbent_mask == full and incumbent != reference:
        raise ValueError("full-set incumbent scores disagree with full reference")

    if skip_ranking:
        plans, drop_orders, restore_order, recovery = {}, {}, [], {}
    else:
        plans, drop_orders, restore_order, recovery = _exchange_plans(
            features, bits, incumbent_mask, removed_before, eligible, witnesses, zeros,
            gain, shap_importance, recovery_scores, retained_gain)

    def trial(mask, scores, plan, origin):
        removed = [f for f in features if not mask & bits[f]]
        delta = [v - b for v, b in zip(scores, incumbent, strict=True)]
        full_deltas = [v - b for v, b in zip(scores, reference, strict=True)]
        full_gate = min(full_deltas) >= -margin - tol
        incumbent_gate = min(delta) >= -tol
        half_improved = (delta[1] + delta[2]) / 2 > tol
        return {**plan, "source": plan["sources"][0], "mask": mask, "scores": list(scores), "score_origin": origin,
                "removed_features": removed, "removed_count": len(removed),
                "removed_witness_count": len(witnesses.intersection(removed)),
                "validation_utility": scores[0], "incumbent_deltas": delta, "deltas": full_deltas,
                "worst_half_improvement": min(delta[1:]), "full_gate_passed": full_gate,
                "incumbent_gate_passed": incumbent_gate, "half_mean_improved": half_improved,
                "accepted": full_gate and incumbent_gate and (half_improved or mask == incumbent_mask)}

    baseline = trial(incumbent_mask, incumbent, {"op": "incumbent", "sources": ["guarded_incumbent"],
                                                "newly_removed_features": [], "restored_features": []}, "incumbent")
    trials = []
    for mask, plan in plans.items():
        if allow_next is not None and not allow_next():
            break
        record = trial(mask, _scores(evaluate(mask), 3), plan, "local_candidate")
        trials.append(record)
        if publish is not None:
            publish(record)

    # A qualifying change can have zero gain on its worst half and on the pooled
    # score. Keep the incumbent first if all policy tie-breakers also tie.
    accepted = [baseline, *(row for row in trials if row["accepted"])]
    def robust_key(row):
        return row["worst_half_improvement"], row["incumbent_deltas"][0], row["removed_count"]

    def pooled_key(row):
        return row["incumbent_deltas"][0], row["worst_half_improvement"], row["removed_count"]

    # Previously scored alternatives do not enlarge the fitted candidate budget.
    # Validate overlapping records before using them to reproduce a frozen score.
    scores_by_mask = {r["mask"]: tuple(r["scores"]) for r in [baseline, *trials]}
    scores_by_mask[full] = reference
    previous = {}
    for row in previous_trials:
        mask = row["mask"]
        if isinstance(mask, bool) or not isinstance(mask, Integral) or not 0 < mask <= full:
            raise ValueError("invalid previously scored mask")
        mask = int(mask)
        scores = _scores(row["scores"], 3)
        if scores != scores_by_mask.setdefault(mask, scores):
            raise ValueError("inconsistent validation scores for identical masks")
        if mask == incumbent_mask or mask in previous or mask.bit_count() > incumbent_mask.bit_count():
            continue
        removed = {f for f in features if not mask & bits[f]}
        if not removed <= eligible or not removed_before.intersection(witnesses) <= removed:
            continue
        previous[mask] = trial(mask, scores, {"op": "previous", "sources": [f"guarded_trial:{hex(mask)}"],
            "newly_removed_features": [f for f in features if f in removed - removed_before],
            "restored_features": [f for f in features if f in removed_before - removed]}, "previous_validation")

    drop_only = [row for row in accepted if row["op"] in {"incumbent", "drop_one"}]
    frontier = [*accepted, *(row for row in previous.values() if row["accepted"])]
    winners = {"local_robust": max(accepted, key=robust_key),
               "local_pooled": max(accepted, key=pooled_key),
               "local_drop": max(drop_only, key=robust_key),
               "validation_frontier": max(frontier, key=robust_key)}
    methods = {name: {**row, "selected_source": row["source"], "selected_mask": row["mask"]}
               for name, row in winners.items()}
    primary = methods["local_robust"]
    tried_removals = removed_before.union(*(row["removed_features"] for row in trials))
    frontier_removals = tried_removals.union(*(row["removed_features"] for row in previous.values()))
    candidates = [f for f in features if f in frontier_removals]
    return {"selected_method": "local_robust", "selected_source": primary["source"],
            "selected_mask": primary["mask"], "selected_scores": primary["scores"],
            "retained_features": [f for f in features if primary["mask"] & bits[f]],
            "removed_features": primary["removed_features"], "removed_witness_count": primary["removed_witness_count"],
            "validation_utility": primary["validation_utility"], "robust_removed_count": primary["removed_count"],
            "incumbent_mask": incumbent_mask, "incumbent_scores": list(incumbent), "full_scores": list(reference),
            "eligible_features": [f for f in features if f in eligible], "candidate_features": candidates,
            "local_candidate_features": [f for f in features if f in tried_removals],
            "all_candidate_features": candidates, "swap_out_candidates": drop_orders, "restore_candidates": restore_order,
            "recovery_scores": recovery, "planned_masks": list(plans), "planned_count": None if skip_ranking else len(plans),
            "evaluated_count": len(trials), "budget_stopped": skip_ranking or len(trials) < len(plans),
            "frontier_previous_count": len(previous), "frontier_previous_trials": list(previous.values()),
            "margin": margin, "baseline": baseline, "trials": trials, "methods": methods}
