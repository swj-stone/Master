"""Cross-fitted training importance ranks for guarded MinPI feature selection."""

import hashlib
import math
from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance
from minppi_mofs.modeling import _validated_frame, _validated_target, make_model
from minppi_mofs.removal_selection import _feature_order, _gains, _scores, _supplemental_features


def _order_consensus(features, eligible, fold_ranks, *, descending=False):
    left, right = fold_ranks
    return sorted(
        (f for f in features if f in eligible),
        key=lambda f: (max(left[f], right[f]), (left[f] + right[f]) / 2),
        reverse=descending,
    )


def _fold_rankings(frame, labels, policy, allow_next, publish_ranking):
    features = frame.columns
    safe_names = [f"f{i}" for i in range(len(features))]
    safe = frame.set_axis(safe_names, axis=1)
    stratified = policy["task"] != "regression"
    halves = [
        np.sort(idx)
        for idx in train_test_split(
            np.arange(len(frame)), test_size=0.5, random_state=42, stratify=labels if stratified else None
        )
    ]

    def digest(rows):
        return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()

    split = {
        "seed": 42,
        "index_space": "zero-based positions in frozen training CSV",
        "stratified": stratified,
        "sizes": [len(idx) for idx in halves],
        "index_sha256": [digest(idx) for idx in halves],
        "shap_rows": [],
        "shap_index_sha256": [],
    }
    rankings = []
    for fit, other in (halves, halves[::-1]):
        if allow_next is not None and not allow_next():
            break
        hold = other
        if len(hold) > 512:
            positions = np.sort(np.random.default_rng(42).choice(len(hold), 512, replace=False))
            hold = hold[positions]
        model = make_model(**{k: policy[k] for k in ("task", "model_params", "backend", "seed")})
        model.fit(safe.iloc[fit], labels[fit])
        raw = _gains(safe_names, model.get_booster().get_score(importance_type="gain"))
        gain = {f: raw[s] for f, s in zip(features, safe_names, strict=True)}
        values = fitted_tree_shap_importance(model, safe.iloc[hold], seed=42, max_rows=512)
        shap = _gains(features, dict(zip(features, values, strict=True)))
        ranks = pd.DataFrame({"gain": gain, "shap": shap}).rank(method="average", pct=True).mean(axis=1)
        rankings.append({"gain": gain, "shap": shap, "normalized_mean_rank": ranks.to_dict()})
        split["shap_rows"].append(len(hold))
        split["shap_index_sha256"].append(digest(hold))
        if publish_ranking is not None:
            publish_ranking(len(rankings), 2)
    return split, rankings


def _fold_proposals(features, bits, incumbent, ordered, locked, to_restore):
    full = (1 << len(features)) - 1
    k = len(features) - incumbent.bit_count()
    block = math.ceil(len(features) / 16)
    plans = {}

    def propose(mask, source):
        if mask and mask != incumbent:
            plans.setdefault(mask, []).append(source)

    others = [f for f in ordered if f not in locked]
    locked_mask = sum(bits[f] for f in locked)
    for requested in (k, k + 1, k + block):
        count = min(requested, len(ordered), len(features) - 1)
        remove = others[: count - len(locked)]
        propose(full ^ (locked_mask | sum(bits[f] for f in remove)), f"global:{count}")

    to_drop = [f for f in ordered if incumbent & bits[f]]
    for num in sorted({1, min(block, k, len(to_drop), len(to_restore))}):
        if 0 < num <= min(len(to_drop), len(to_restore)):
            flips = to_drop[:num] + to_restore[:num]
            propose(incumbent ^ sum(bits[f] for f in flips), f"exchange:{num}")
    return plans


def select_fold_consensus(
    train,
    yt,
    sample_minima,
    incumbent_mask,
    evaluate,
    incumbent_scores,
    full_scores,
    margin,
    *,
    policy,
    publish=None,
    allow_next=None,
    publish_ranking=None,
):
    """Evaluate at most five distinct global or local candidates with no compression loss.

    Two training-half models supply Gain ranks. Each model supplies TreeSHAP ranks
    from the other training half. Per feature, the larger of its two mean ranks
    orders removal candidates; a tie uses their mean and then source column order.
    Each candidate is validated against both full and guarded incumbent scores.
    """
    frame = _validated_frame(train, name="training features")
    features = _feature_order(frame.columns)
    labels = _validated_target(yt, row_count=len(frame), task=policy["task"])
    full = (1 << len(features)) - 1
    if (
        isinstance(incumbent_mask, bool)
        or not isinstance(incumbent_mask, Integral)
        or not 0 < incumbent_mask <= full
    ):
        raise ValueError("invalid incumbent mask")
    incumbent_mask = int(incumbent_mask)
    if not isinstance(sample_minima, Mapping):
        raise ValueError("explicit sign evidence is required")
    witnesses = {f for f, v in sample_minima.items() if v == -1}
    zeros = _supplemental_features(features, witnesses, sample_minima)
    eligible = witnesses | zeros
    bits = {f: 1 << i for i, f in enumerate(features)}
    removed_before = {f for f in features if not incumbent_mask & bits[f]}
    if not removed_before <= eligible:
        raise ValueError("incumbent contains protected deletions")
    incumbent = _scores(incumbent_scores, 3)
    full_scores = _scores(full_scores, 3)
    tol = 1e-12
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid full-reference margin")
    if any(score < ref - margin - tol for score, ref in zip(incumbent, full_scores, strict=True)):
        raise ValueError("incumbent fails full validation gates")
    if incumbent_mask == full and incumbent != full_scores:
        raise ValueError("full-set incumbent disagrees with full scores")

    def trial(mask, scores, sources):
        diff = [a - b for a, b in zip(scores, incumbent, strict=True)]
        full_diff = [a - b for a, b in zip(scores, full_scores, strict=True)]
        removed = [f for f in features if not mask & bits[f]]
        full_pass = min(full_diff) >= -margin - tol
        guarded_pass = min(diff) >= -tol
        half_gain = (diff[1] + diff[2]) / 2 > tol
        return {
            "mask": mask,
            "scores": list(scores),
            "source": sources[0],
            "sources": sources,
            "removed_features": removed,
            "removed_count": len(removed),
            "removed_witness_count": len(witnesses.intersection(removed)),
            "incumbent_deltas": diff,
            "deltas": full_diff,
            "worst_half_improvement": min(diff[1:]),
            "full_gate_passed": full_pass,
            "incumbent_gate_passed": guarded_pass,
            "half_mean_improved": half_gain,
            "accepted": full_pass and guarded_pass and (half_gain or mask == incumbent_mask),
        }

    baseline = trial(incumbent_mask, incumbent, ["guarded_incumbent"])
    restorable = zeros & removed_before
    has_alternatives = bool(eligible - removed_before) and (incumbent_mask.bit_count() > 1 or bool(restorable))
    split, rankings, order, plans, trials = None, [], [], {}, []
    if has_alternatives:
        split, rankings = _fold_rankings(frame, labels, policy, allow_next, publish_ranking)
        if len(rankings) == 2:
            fold_ranks = [r["normalized_mean_rank"] for r in rankings]
            order = _order_consensus(features, eligible, fold_ranks)
            recovery = _order_consensus(features, restorable, fold_ranks, descending=True)
            plans = _fold_proposals(features, bits, incumbent_mask, order, witnesses & removed_before, recovery)
    for mask, sources in plans.items():
        if allow_next is not None and not allow_next():
            break
        record = trial(mask, _scores(evaluate(mask), 3), sources)
        trials.append(record)
        if publish is not None:
            publish(record)
    best = max(
        [baseline, *(r for r in trials if r["accepted"])],
        key=lambda r: (r["worst_half_improvement"], r["incumbent_deltas"][0], r["removed_count"]),
    )
    incomplete_ranks = has_alternatives and len(rankings) < 2
    candidates = removed_before.union(*(r["removed_features"] for r in trials))
    methods = {
        m: {**r, "selected_source": r["source"], "selected_mask": r["mask"]}
        for m, r in (("fold_consensus", best), ("guarded_incumbent", baseline))
    }
    return {
        "selected_method": "fold_consensus",
        "selected_source": best["source"],
        "selected_mask": best["mask"],
        "selected_scores": best["scores"],
        "validation_utility": best["scores"][0],
        "removed_features": best["removed_features"],
        "robust_removed_count": best["removed_count"],
        "removed_witness_count": best["removed_witness_count"],
        "full_scores": list(full_scores),
        "incumbent_scores": list(incumbent),
        "incumbent_mask": incumbent_mask,
        "training_split": split,
        "fold_rankings": rankings,
        "ranking_fits": len(rankings),
        "consensus_order": order,
        "candidate_features": [f for f in features if f in candidates],
        "trials": trials,
        "planned_masks": list(plans),
        "planned_count": None if incomplete_ranks else len(plans),
        "evaluated_count": len(trials),
        "budget_stopped": incomplete_ranks or len(trials) < len(plans),
        "methods": methods,
        "margin": margin,
        "baseline": baseline,
    }
