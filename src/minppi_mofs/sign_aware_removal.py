"""Validation-budgeted sign-aware deletion for numerically sampled MinPI."""

from collections.abc import Mapping
import math

import pandas as pd

from minppi_mofs.removal_selection import _feature_order, _gains, _scores, _supplemental_features


def select_sign_aware_removals(features, witnesses, sample_minima, feature_gain, shap_importance,
                               evaluate, full_utility, margin, *, publish=None):
    """Compare sign-aware deletion and matched controls on at most 21 unique masks.

    Training-only gain and mean absolute TreeSHAP ranks order sampled zeros.
    Their midranks are averaged among zeros, with all witnessed features placed
    first. A second ranking orders all eligible features by gain alone, so it can
    retain a negative witness if deleting it alongside other features fails.
    Positive or unobserved minima are protected on both paths. Each validation
    callback receives a nonempty retained mask and returns (pooled, half A, half B).
    The primary policy charges the worst half's loss; the ablation charges pooled
    loss. Both must pass the same three noninferiority gates. Independent gain
    and TreeSHAP controls use the same size grid and worst-fold objective on
    their unrestricted feature rankings. Scores are shared across identical
    masks, while each method chooses only from its own admissible proposals.
    """
    features = _feature_order(features)
    if not isinstance(sample_minima, Mapping):
        raise ValueError("sample_minima must explicitly describe the sampling evidence")
    witnessed = set(witnesses)
    zeros = _supplemental_features(features, witnessed, sample_minima)
    gain = _gains(features, feature_gain)
    if not isinstance(shap_importance, Mapping) or set(shap_importance) != set(features):
        raise ValueError("TreeSHAP importances must cover all training features")
    shap = {f: float(shap_importance[f]) for f in features}
    if any(not math.isfinite(v) or v < 0 for v in shap.values()):
        raise ValueError("TreeSHAP importances must be finite and nonnegative")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    reference = _scores(full_utility, 3)
    n = len(features)
    full = (1 << n) - 1
    bits = {f: 1 << i for i, f in enumerate(features)}
    eligible = [f for f in features if f in witnessed or f in zeros]
    ordered_zeros = [f for f in features if f in zeros]
    train_scores = pd.DataFrame({"gain": gain, "shap": shap}).reindex(ordered_zeros)
    fused_rank = train_scores.rank(method="average").mean(axis=1).to_dict()
    ordered_witnesses = sorted((f for f in features if f in witnessed), key=gain.get)
    fused = ordered_witnesses + sorted(ordered_zeros, key=fused_rank.get)
    controls = {"matched_gain": sorted(features, key=gain.get),
                "matched_treeshap": sorted(features, key=shap.get)}
    rankings = {"witness_first_fused": fused, "eligible_gain": sorted(eligible, key=gain.get), **controls}
    sizes = (1, math.ceil(n / 4), math.ceil(n / 2), math.ceil(3 * n / 4), 7 * n // 8)
    proposals = {}
    candidate_pools, path_masks = {}, {}

    def register(removed, source):
        mask = full ^ sum(bits[f] for f in removed)
        if 0 < mask < full:
            proposals.setdefault(mask, []).append(source)
        return mask

    for path, ranking in rankings.items():
        counts = sorted({min(size, len(ranking), n - 1) for size in sizes if size})
        candidate_pools[path] = ranking[:max(counts, default=0)]
        path_masks[path] = [register(ranking[:count], f"{path}:{count}") for count in counts if count]
    witness_mask = register(ordered_witnesses, "witness_only")
    sign_masks = set(path_masks["witness_first_fused"] + path_masks["eligible_gain"]) | {witness_mask}

    def make_trial(mask, scores, sources):
        deltas = [value - ref for value, ref in zip(scores, reference, strict=True)]
        removed = [f for f in features if not mask & bits[f]]
        fraction = len(removed) / n
        worst_half = min(deltas[1:])
        threshold = -margin - 1e-12 if margin else 0.
        def objective(delta):
            return fraction - max(0., -delta) / margin if margin else fraction
        return {"mask": mask, "source": sources[0], "sources": sources, "scores": list(scores),
                "is_sign_candidate": mask != full and mask in sign_masks,
                "deltas": deltas, "delta": deltas[0], "validation_utility": scores[0],
                "feasible": True, "accepted": min(deltas) >= threshold, "pooled_accepted": deltas[0] >= threshold,
                "removed_count": len(removed), "removed_features": removed,
                "retained_features": [f for f in features if mask & bits[f]],
                "removed_witness_count": len(witnessed.intersection(removed)), "worst_slack": worst_half + margin,
                "objectives": {"sign_pooled": objective(deltas[0]), "sign_worst": objective(worst_half)}}

    baseline = make_trial(full, reference, ["full_set"])
    by_mask = {full: baseline}
    trials = []
    for mask, sources in proposals.items():
        record = make_trial(mask, _scores(evaluate(mask), 3), sources)
        by_mask[mask] = record
        trials.append(record)
        if publish is not None:
            publish(record)

    methods = {}
    accepted = [baseline, *(r for r in trials if r["accepted"] and r["is_sign_candidate"])]
    for method in ("sign_worst", "sign_pooled"):
        winner = max(accepted, key=lambda rec: (rec["objectives"][method], rec["worst_slack"],
                                                 rec["removed_witness_count"]))
        methods[method] = {**winner, "selected_mask": winner["mask"], "selected_source": winner["source"],
                           "objective": winner["objectives"][method]}
    for path in controls:
        rows = [baseline, *(by_mask[m] for m in path_masks[path] if by_mask[m]["accepted"])]
        winner = max(rows, key=lambda rec: (rec["objectives"]["sign_worst"], rec["worst_slack"]))
        source = next((s for s in winner["sources"] if s.startswith(path + ":")), "full_set")
        methods[path] = {**winner, "selected_mask": winner["mask"], "selected_source": source,
                         "objective": winner["objectives"]["sign_worst"]}
    primary = methods["sign_worst"]
    screened = {f for record in trials if record["is_sign_candidate"] for f in record["removed_features"]}
    candidates = [f for f in features if f in screened]
    external = set().union(*(candidate_pools[path] for path in controls))
    nomination = []
    for f in fused:
        if f in screened:
            entry = {"feature": f, "reason": "witness" if f in witnessed else "sampled_zero",
                     "gain": gain[f], "mean_abs_treeshap": shap[f], "zero_fused_midrank": fused_rank.get(f)}
            if f in witnessed and isinstance(witnesses, Mapping):
                entry["witness"] = witnesses[f]
            nomination.append(entry)
    largest_fused_mask = full ^ sum(bits[f] for f in candidate_pools["witness_first_fused"])
    witness_record = by_mask.get(witness_mask, {"mask": 0, "feasible": False, "accepted": False,
                                                "removed_features": list(features), "removed_count": n,
                                                "reason": "empty_retained_set"})
    return {"selected_method": "sign_worst", "selected_source": primary["source"],
            "selected_mask": primary["mask"], "selected_scores": primary["scores"],
            "removed_features": primary["removed_features"], "retained_features": primary["retained_features"],
            "validation_utility": primary["validation_utility"], "robust_removed_count": primary["removed_count"],
            "removed_witness_count": primary["removed_witness_count"], "full_scores": list(reference),
            "full_utility": reference[0], "margin": margin, "baseline": baseline, "witness_only": witness_record,
            "structured_pool": by_mask[largest_fused_mask], "candidate_pools": candidate_pools,
            "eligible_features": eligible, "candidate_features": candidates,
            "external_candidate_features": [f for f in features if f in external],
            "all_candidate_features": [f for f in features if f in screened or f in external],
            "nomination": nomination, "trials": trials, "methods": methods}
