"""Choose between two locked feature sets by validation Pareto dominance."""

from collections.abc import Mapping
import math
from numbers import Integral

from minppi_mofs.removal_selection import _feature_order, _scores


def guarded_update(features, structured_mask, signed_mask, official_scores, full_scores, margin):
    """Adopt sign_worst only for a strict improvement over the feasible incumbent.

    Score triples refer to the same pooled validation set and its two fixed
    halves. Higher scores and more deleted columns are preferred. A difference
    up to 1e-12 is treated as a tie in each score coordinate.
    """
    features = _feature_order(features)
    full = (1 << len(features)) - 1
    masks = {"structured": structured_mask, "sign_worst": signed_mask}
    for mask in masks.values():
        if isinstance(mask, bool) or not isinstance(mask, Integral) or not 0 < mask <= full:
            raise ValueError("both candidates must have a valid nonempty retained mask")
    masks = {name: int(mask) for name, mask in masks.items()}
    if not isinstance(official_scores, Mapping) or not set(masks.values()) <= official_scores.keys():
        raise ValueError("official scores must cover both retained masks")
    reference = _scores(full_scores, 3)
    if full in official_scores and _scores(official_scores[full], 3) != reference:
        raise ValueError("official full scores disagree with the full reference")
    scores = {name: _scores(official_scores[mask], 3) for name, mask in masks.items()}
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    tol = 1e-12
    passes = {name: all(v >= ref - margin - tol for v, ref in zip(row, reference, strict=True))
              for name, row in scores.items()}
    if not passes["structured"]:
        raise ValueError("structured incumbent fails its original validation gate")

    deltas = [new - old for new, old in zip(scores["sign_worst"], scores["structured"], strict=True)]
    extra_removed = masks["structured"].bit_count() - masks["sign_worst"].bit_count()
    all_nonworse = min(deltas) >= -tol
    strict_improvement = extra_removed > 0 or max(deltas) > tol
    adopt = passes["sign_worst"] and extra_removed >= 0 and all_nonworse and strict_improvement
    source = "sign_worst" if adopt else "structured"
    selected = masks[source]
    removed = [f for i, f in enumerate(features) if not selected & (1 << i)]
    return {
        "selected_mask": selected, "selected_source": source, "selected_scores": list(scores[source]),
        "removed_count": len(removed), "removed_features": removed, "candidate_masks": masks,
        "candidate_scores": {name: list(row) for name, row in scores.items()}, "full_scores": list(reference),
        "margin": margin, "tolerance": tol, "challenger_minus_incumbent_scores": deltas,
        "challenger_minus_incumbent_removed_count": extra_removed,
        "predicates": {"incumbent_gate_passed": passes["structured"], "challenger_gate_passed": passes["sign_worst"],
                       "compression_nonworse": extra_removed >= 0, "all_scores_nonworse": all_nonworse,
                       "strict_improvement": strict_improvement, "adopt_challenger": adopt},
    }
