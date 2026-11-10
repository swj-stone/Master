"""Training-based removal nominations and bounded validation of joint deletions."""

from collections import deque
from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

_EXTERNAL_CONTROLS = {"gain_only", "treeshap_only"}


def _feature_order(features):
    order = tuple(features)
    if not order or any(not isinstance(f, str) or not f for f in order) or len(set(order)) != len(order):
        raise ValueError("features must be nonempty, unique column names")
    return order


def _gains(features, feature_gain):
    if not isinstance(feature_gain, Mapping) or set(feature_gain).difference(features):
        raise ValueError("feature_gain must map known column names to gains")
    # XGBoost omits features that were never used for a split.
    values = {f: float(feature_gain.get(f, 0.0)) for f in features}
    if any(not math.isfinite(g) or g < 0 for g in values.values()):
        raise ValueError("gains must be finite and nonnegative")
    return values


def _pool_size(n, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("pool_fraction must be in (0, 1]")
    return math.ceil(n * fraction)


def _supplemental_features(features, witnessed, sample_minima):
    if sample_minima is None:
        return set(features) - witnessed
    if (not isinstance(sample_minima, Mapping) or set(sample_minima).difference(features)
        or any(isinstance(v, (bool, np.bool_)) or v not in (-1, 0, 1, None) for v in sample_minima.values())):
        raise ValueError("sample_minima must map known feature names to -1, 0, 1 or None")
    if {f for f, score in sample_minima.items() if score == -1} != witnessed:
        raise ValueError("sampled -1 scores do not match the supplied witness identities")
    return {f for f, score in sample_minima.items() if score == 0}


def nominate_removals(train, feature_gain, witnesses, *, sample_minima=None,
                      pool_fraction=0.75, correlation_threshold=0.95):
    """Rank witnessed features, correlated proxies, then low-gain columns.

    Supply the training feature frame, full training-model gains keyed by column
    (unused columns may be omitted), and certified witness names or their records.
    Each returned dict has feature, reason and gain. A proxy also identifies its
    representative and absolute training Pearson correlation. The pool stops
    at ceil(pool_fraction*n) unless there are more witnesses. With sample_minima,
    only explicitly sampled zeros supplement witnesses. Positive and unseen
    features remain eligible as retained representatives.
    """
    if not isinstance(train, pd.DataFrame) or not len(train):
        raise ValueError("train must be a nonempty feature DataFrame")
    features = _feature_order(train.columns)
    gain = _gains(features, feature_gain)
    target = _pool_size(len(features), pool_fraction)
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1]")
    witnessed = set(witnesses)
    if witnessed.difference(features):
        raise ValueError("unknown feature in witnesses")
    eligible = _supplemental_features(features, witnessed, sample_minima)
    matrix = train.to_numpy(dtype=float, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("training features must be finite")
    # NaN correlations for constant columns do not provide proxy evidence.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.atleast_2d(np.corrcoef(matrix, rowvar=False))) if len(matrix) > 1 else None
    positions = {f: i for i, f in enumerate(features)}

    def low_first(feature):
        return gain[feature], positions[feature]

    nominations = []
    for f in sorted(witnessed, key=low_first):
        record = {"feature": f, "reason": "witness", "gain": gain[f]}
        if isinstance(witnesses, Mapping):
            record["witness"] = witnesses[f]
        nominations.append(record)

    reps = sorted((f for f in features if f not in witnessed and f not in eligible),
                  key=lambda f: (-gain[f], positions[f]))
    protected = set()
    proxies = []
    for f in sorted(eligible, key=lambda f: (-gain[f], positions[f])):
        candidates = []
        if corr is not None:
            for rep in reps:
                rho = float(corr[positions[f], positions[rep]])
                if math.isfinite(rho) and rho >= correlation_threshold - 4 * np.finfo(float).eps:
                    candidates.append((rep, rho))
        if not candidates:
            reps.append(f)
            continue
        # max retains the higher-priority representative for equal correlations.
        rep, rho = max(candidates, key=lambda item: item[1])
        protected.add(rep)
        proxies.append({"feature": f, "reason": "correlated_proxy", "gain": gain[f],
                        "representative": rep, "abs_corr": rho})
    proxies.sort(key=lambda e: (-e["abs_corr"], *low_first(e["feature"])))
    nominations.extend(proxies[:max(0, target - len(nominations))])
    for f in sorted((f for f in reps if f in eligible), key=lambda f: (f in protected, *low_first(f))):
        if len(nominations) >= target:
            break
        nominations.append({"feature": f, "reason": "low_gain", "gain": gain[f],
                            "represents_proxy": f in protected})
    return nominations


def _scores(values, count):
    values = tuple(float(x) for x in values)
    if len(values) != count or not all(math.isfinite(x) for x in values):
        raise ValueError(f"expected {count} finite validation scores")
    return values


def _nomination_order(features, nomination):
    order = [record["feature"] for record in nomination]
    if len(set(order)) != len(order) or not set(order).issubset(features):
        raise ValueError("nomination must contain distinct, known features")
    return order


def _select(features, nomination, paths, evaluate, baseline, margin, publish, *, rescue_on_veto=False):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    n = len(features)
    full = (1 << n) - 1
    bits = {f: 1 << i for i, f in enumerate(features)}
    witnessed = [r["feature"] for r in nomination if r["reason"] == "witness"]
    witness_mask = full ^ sum(bits[f] for f in witnessed)
    proposals = {}

    def register(removed, source):
        mask = full ^ sum(bits[f] for f in removed)
        if 0 < mask < full:
            proposals.setdefault(mask, []).append(source)
        return mask

    for method, ranking in paths.items():
        if method == "reserve_witness":
            continue  # This method proposes its next block after each joint validation.
        for size, label in ((len(ranking), "pool"), (n // 2, "half"), (n // 4, "quarter"),
                            (n // 8, "eighth"), (1, "one")):
            count = min(size, len(ranking), n - 1)
            if count:
                if label == "pool" and count != len(ranking):
                    label = "pool_keep_one"
                register(ranking[:count], f"{method}:{label}")
    register(witnessed, "witness_proxy_gain:witness_only")

    def trial(mask, scores, sources):
        delta = [score - reference for score, reference in zip(scores, baseline, strict=True)]
        removed = [f for f in features if not mask & bits[f]]
        return {"mask": mask, "removed_features": removed, "retained_features": [f for f in features if mask & bits[f]],
                "removed_count": len(removed), "source": sources[0], "sources": sources,
                "removed_witness_count": sum(not mask & bits[f] for f in witnessed),
                "scores": list(scores), "validation_utility": scores[0], "delta": delta[0], "deltas": delta,
                "accepted": all(d >= -margin - 1e-12 for d in delta), "pooled_accepted": delta[0] >= -margin - 1e-12,
                "worst_slack": min(delta[1:] or delta) + margin, "feasible": True}

    incumbent = trial(full, baseline, ["full_set"])
    tested = {}

    def evaluate_once(mask):
        if mask not in tested:
            tested[mask] = trial(mask, _scores(evaluate(mask), len(baseline)), proposals[mask])
            if publish is not None:
                publish(tested[mask])
        return tested[mask]

    for mask in sorted(proposals, key=int.bit_count):
        evaluate_once(mask)

    if rescue_on_veto and n > 1 and not any(row["accepted"] for row in tested.values()):
        zeros = [f for f in paths["witness_proxy_gain"] if f not in witnessed]
        if zeros:
            paths = {**paths, "reserve_witness": zeros}

    # Breadth-first bisection tests the sibling of a failed block even when
    # every strict prefix contains an indispensable feature. Accepted deletions
    # remain in the mask of every subsequent trial.
    initial = paths.get("reserve_witness", [])[:n - 1]
    blocks = deque([initial]) if initial else deque()
    accepted = []
    for attempt in range(1, 9):
        if not blocks:
            break
        block = blocks.popleft()
        combined = accepted + block
        mask = register(combined, f"reserve_witness:block_{attempt}")
        if evaluate_once(mask)["accepted"]:
            accepted = combined
        elif len(block) > 1:
            middle = (len(block) + 1) // 2
            blocks.extend((block[:middle], block[middle:]))

    def priority(r, robust, method):
        # Independent baselines rank by sparsity alone. MinPI first tries to
        # preserve as many witnessed deletions as validation permits.
        return (r["removed_witness_count"] if method not in _EXTERNAL_CONTROLS else 0,
                r["removed_count"], r["worst_slack"] if robust else r["delta"])

    def best(rows, robust, method):
        eligible = [incumbent, *(r for r in rows if r["accepted" if robust else "pooled_accepted"])]
        return max(eligible, key=lambda r: priority(r, robust, method))

    methods = {}
    selected = incumbent
    selected_method = "full_set"
    pooled_choice = incumbent
    for method in paths:
        rows = [r for r in tested.values() if any(s.startswith(method + ":") for s in r["sources"])]
        robust, pooled = best(rows, True, method), best(rows, False, method)
        source = next((s for s in robust["sources"] if s.startswith(method + ":")), "full_set")
        methods[method] = {**robust, "selected_mask": robust["mask"], "selected_source": source,
                           "pooled_only_removed_count": pooled["removed_count"], "pooled_only_mask": pooled["mask"]}
        if method in _EXTERNAL_CONTROLS:
            continue
        if priority(robust, True, method) > priority(selected, True, method):
            selected, selected_method = robust, method
        if priority(pooled, False, method) > priority(pooled_choice, False, method):
            pooled_choice = pooled

    def comparator(mask):
        if mask == full:
            return incumbent
        if mask == 0:
            return {"mask": 0, "removed_features": list(features), "removed_count": n, "feasible": False,
                    "accepted": False, "reason": "empty_retained_set"}
        return tested[mask]

    structured_mask = full ^ sum(bits[f] for f in paths["witness_proxy_gain"])
    candidates = {f for method, pool in paths.items() if method not in _EXTERNAL_CONTROLS for f in pool}
    external = {f for method, pool in paths.items() if method in _EXTERNAL_CONTROLS for f in pool}
    source = methods[selected_method]["selected_source"] if selected_method in methods else "full_set"
    return {"nomination": list(nomination), "trials": list(tested.values()), "methods": methods,
            "candidate_pools": paths, "candidate_features": [f for f in features if f in candidates],
            "external_candidate_features": [f for f in features if f in external],
            "all_candidate_features": [f for f in features if f in candidates or f in external],
            "full_scores": list(baseline), "full_utility": baseline[0], "baseline": incumbent, "margin": margin,
            "witness_only": comparator(witness_mask), "structured_pool": comparator(structured_mask),
            "selected_method": selected_method, "selected_source": source,
            "selected_mask": selected["mask"], "removed_features": selected["removed_features"],
            "retained_features": selected["retained_features"], "validation_utility": selected["validation_utility"],
            "selected_scores": selected["scores"], "robust_removed_count": selected["removed_count"],
            "removed_witness_count": selected["removed_witness_count"],
            "pooled_only_removed_count": pooled_choice["removed_count"], "pooled_only_mask": pooled_choice["mask"]}


def select_removals(features, nomination, evaluate, full_utility, margin, *, publish=None):
    """Gate structured removal prefixes and the witness set on one validation utility.

    evaluate(mask) scores the nonempty retained columns (bit i means retain
    features[i]). Return the winning selected_mask, retained/removed features,
    validation_utility, ordered nomination records, and one trial per distinct
    evaluated mask. publish(trial) is optional and receives each completed fit.
    """
    features = _feature_order(features)
    nomination = list(nomination)
    order = _nomination_order(features, nomination)
    return _select(features, nomination, {"witness_proxy_gain": order}, lambda mask: (evaluate(mask),),
                   _scores((full_utility,), 1), float(margin), publish)


def select_robust_removals(features, nomination, feature_gain, evaluate, full_utility, margin,
                           *, publish=None, pool_fraction=0.75, sample_minima=None, shap_importance=None,
                           structured_only=False):
    """Validate joint MinPI deletions and independent feature-importance controls.

    evaluate(retained_mask) and full_utility give (pooled, fold_0, fold_1) on
    the same fixed validation rows, using the same model for all three scores.
    Gate each score against its matching full-feature score minus margin.
    Among passing MinPI masks choose most witnesses removed, then most total
    removals, then largest worst-fold slack, then method insertion order. External
    gain and train-mean-absolute-TreeSHAP controls report independent choices.
    The witness-reserving path bisects a failed block and
    validates it cumulatively with accepted deletions, for at most eight attempts.
    methods records individual choices, trials shares fits across masks, and
    pooled_only_removed_count reports the pooled-gate comparator on tested masks.
    With structured_only=True, evaluate the witness/proxy/gain prefixes and
    the witness-only comparator. If every nonempty deletion fails, try the
    witness-reserving block rescue on supplemental candidates before the full-set
    fallback. Supply sample_minima to restrict supplemental candidates to sampled
    zeros. This mode uses the same three-score gate and needs no TreeSHAP scores.
    """
    features = _feature_order(features)
    nomination = list(nomination)
    structured = _nomination_order(features, nomination)
    gain = _gains(features, feature_gain)
    witnessed = {r["feature"] for r in nomination if r["reason"] == "witness"}
    eligible = _supplemental_features(features, witnessed, sample_minima)
    if not set(structured).issubset(eligible | witnessed):
        raise ValueError("structured MinPI nomination contains a positive or unseen feature")
    limit = max(_pool_size(len(features), pool_fraction), len(witnessed))
    if structured_only:
        paths = {"witness_proxy_gain": structured}
    else:
        paths = {"gain_only": sorted(features, key=gain.get)[:limit],
                 "witness_proxy_gain": structured,
                 "reserve_witness": [f for f in structured if f not in witnessed]}
    shap = None
    if not structured_only and shap_importance is not None:
        if not isinstance(shap_importance, Mapping) or set(shap_importance) != set(features):
            raise ValueError("shap_importance must give a train-mean-absolute TreeSHAP score for each feature")
        shap = {f: float(shap_importance[f]) for f in features}
        if any(not math.isfinite(v) or v < 0 for v in shap.values()):
            raise ValueError("TreeSHAP importances must be finite and nonnegative")
        shap_order = sorted(features, key=shap.get)
        paths["treeshap_only"] = shap_order[:limit]
        paths["witness_zero_shap"] = ([f for f in structured if f in witnessed]
                                      + [f for f in shap_order if f in eligible])[:limit]
    result = _select(features, nomination, paths, evaluate, _scores(full_utility, 3), float(margin), publish,
                     rescue_on_veto=structured_only)
    if structured_only:
        result["conditional_rescue_used"] = "reserve_witness" in result["methods"]
    result["feature_gain"] = gain
    if shap is not None:
        result["mean_absolute_treeshap"] = shap
    return result
