"""Confirm two frozen MinPI subsets on one additional training-only holdout."""

from collections.abc import Mapping
import math
from numbers import Integral

from minppi_mofs.removal_selection import _feature_order, _scores


def select_confirmed_removal(features, candidate_masks, official_scores, margin, evaluate_confirm):
    """Select the largest subset deletion passing the original and confirmation gates.

    Mask bits identify retained columns in feature order. official_scores maps
    each mask, including full, to its locked (pooled, half A, half B) scores.
    evaluate_confirm returns one higher-is-better holdout score for the mask;
    all masks, including full, must use identical train-only rows and model settings.
    """
    features = _feature_order(features)
    full = (1 << len(features)) - 1
    methods = ("structured", "sign_worst")
    if not isinstance(candidate_masks, Mapping) or set(candidate_masks) != set(methods):
        raise ValueError("candidate_masks must contain structured and sign_worst")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    sources = {full: ["full_set"]}
    for name in methods:
        mask = candidate_masks[name]
        if isinstance(mask, bool) or not isinstance(mask, Integral) or not 0 < mask <= full:
            raise ValueError("each candidate must be a valid nonempty retained mask")
        sources.setdefault(int(mask), []).append(name)
    if not isinstance(official_scores, Mapping) or not sources.keys() <= official_scores.keys():
        raise ValueError("official scores are required for every distinct mask and full")
    scores = {mask: _scores(official_scores[mask], 3) for mask in sources}
    confirmation = {}
    for mask in sources:
        value = float(evaluate_confirm(mask))
        if not math.isfinite(value):
            raise ValueError("confirmation score must be finite")
        confirmation[mask] = value

    threshold = -margin - 1e-12 if margin else 0.
    trials = []
    for mask, names in sources.items():
        deltas = [val - ref for val, ref in zip(scores[mask], scores[full], strict=True)]
        delta = confirmation[mask] - confirmation[full]
        original_pass = min(deltas) >= threshold
        confirm_pass = delta >= threshold
        removed = [f for i, f in enumerate(features) if not mask & (1 << i)]
        trials.append({"mask": mask, "sources": names, "removed_count": len(removed),
                       "removed_features": removed, "scores": list(scores[mask]), "official_deltas": deltas,
                       "worst_slack": min(deltas[1:]) + margin, "confirmation_score": confirmation[mask],
                       "confirmation_delta": delta, "official_gate_passed": original_pass,
                       "confirmation_gate_passed": confirm_pass, "accepted": original_pass and confirm_pass})

    # The order of sources breaks exact ties: full first, then structured, then sign_worst.
    winner = max((row for row in trials if row["accepted"]),
                 key=lambda row: (row["removed_count"], row["confirmation_delta"], row["worst_slack"]))
    return {"selected_mask": winner["mask"], "selected_source": winner["sources"][0],
            "removed_count": winner["removed_count"], "removed_features": winner["removed_features"],
            "selected_scores": winner["scores"], "selected_confirmation_score": winner["confirmation_score"],
            "selected_confirmation_delta": winner["confirmation_delta"], "confirmation_full_score": confirmation[full],
            "full_scores": list(scores[full]), "margin": margin, "trials": trials}
