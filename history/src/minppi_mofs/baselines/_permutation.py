"""Budgeted global permutation marginals for the MinShap and MC-Shapley baselines.

The statistical selector follows Zheng and Raskutti, arXiv:2604.15107v2,
Section 4 Algorithm 1. Appendix B Algorithm 2 gives the permutation mean.
The experiment supplies train-only fits and an independent ranking-side oracle.
"""
from bisect import insort
from collections import Counter
from hashlib import sha256

import numpy as np


PATHS = 4
ALPHA = .05
BATCH = 16


def _schedule(p, active, paths, seed):
    """Draw full-space permutations before projecting out provable frozen-model dummies."""
    rng = np.random.default_rng(seed)
    enabled = np.zeros(p, dtype=bool)
    enabled[active] = True
    digest = sha256()
    projected = []
    for _ in range(paths):
        full_path = rng.permutation(p)
        digest.update(full_path.astype("<u4").tobytes())
        path = full_path[enabled[full_path]].tolist()
        projected.append(path)
    # Only prefixes of the same cardinality can coincide. Comparing all paths
    # one depth at a time avoids storing p full-width bitmaps per permutation.
    bits = [0]*paths
    repeated = {}
    for entrants in zip(*projected, strict=True):
        for i, j in enumerate(entrants):
            bits[i] |= 1 << j
        if len(set(bits)) < paths:
            repeated.update({key:n for key,n in Counter(bits).items() if n > 1})
    return projected, repeated, digest.hexdigest()


def _estimate(methods, mode, columns, evaluate, progress, *, active=None,
              seed=42, paths=PATHS, alpha=ALPHA, evaluate_batch=None):
    if mode not in ("native","frozen") or not methods or set(methods)-{"minshap","shapley"}:
        raise ValueError("Expected a MinShap or Shapley selector and a supported game")
    p = len(columns)
    if p == 0 or len(set(columns)) != p or paths < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid feature schema or permutation settings")
    active = list(range(p)) if active is None else list(active)
    if len(set(active)) != len(active) or any(not isinstance(i, (int, np.integer))
            or i < 0 or i >= p for i in active):
        raise ValueError("Invalid active feature coordinates")
    if mode == "native" and len(active) != p:
        raise ValueError("A refit game must cover every original coordinate")
    active = [int(i) for i in active]

    statistical = mode == "native" and "minshap" in methods
    scalar = mode == "frozen" or "shapley" in methods
    orders, repeated, digest = _schedule(p, active, paths, seed)
    utility_marginals = np.zeros((paths,p),dtype=float) if scalar else None
    loss_marginals = np.empty((paths,p),dtype=float) if statistical else None
    variances = np.empty((paths,p),dtype=float) if statistical else None

    def unpack(observation, expected_shape=None):
        if statistical and scalar:
            losses, utility = observation
        elif statistical:
            losses, utility = observation, None
        else:
            losses, utility = None, observation
        if statistical:
            losses = np.asarray(losses,dtype=float)
            if losses.ndim != 1 or len(losses) < 2 or not np.isfinite(losses).all():
                raise ValueError("Original MinShap needs finite per-row ranking losses")
            if expected_shape is not None and losses.shape != expected_shape:
                raise ValueError("Ranking-side losses must be aligned")
        if scalar:
            utility = float(utility)
            if not np.isfinite(utility):
                raise ValueError("Nonfinite coalition utility")
        return losses, utility

    empty = unpack(evaluate(()))
    loss_shape = empty[0].shape if statistical else None
    cache = {}
    calls = 1
    evaluations = paths * len(active)
    for path_id, order in enumerate(orders):
        prev = empty
        included = []
        bits = 0
        batch_size = (BATCH if mode=="frozen" else 4) if evaluate_batch is not None else 1
        for start in range(0, len(order), batch_size):
            current = order[start:start+batch_size]
            keys, fresh = [], []
            for j in current:
                insort(included, j)
                bits |= 1 << j
                keys.append(bits)
                if bits not in cache:
                    fresh.append((bits,tuple(included)))
            if batch_size == 1:
                vals = [evaluate(fresh[0][1])] if fresh else []
            else:
                vals = evaluate_batch([coord for _,coord in fresh]) if fresh else []
            all_vals = {key:unpack(val,loss_shape) for (key,_),val in zip(fresh,vals,strict=True)}
            calls += len(all_vals)
            for j,key in zip(current,keys):
                val = all_vals[key] if key in all_vals else cache[key]
                if statistical:
                    paired = prev[0] - val[0]
                    loss_marginals[path_id,j] = paired.mean()
                    variances[path_id,j] = paired.var(ddof=1) / len(paired)
                if scalar:
                    utility_marginals[path_id,j] = val[1] - prev[1]
                if key in repeated:
                    repeated[key] -= 1
                    if repeated[key]:
                        cache[key] = val
                    else:
                        cache.pop(key, None)
                prev = val
            completed = start+len(current)
            progress("permutation", checked=path_id*len(active)+completed,
                     total=evaluations, paths_completed=path_id+int(completed==len(order)),
                     path_position=completed, path_width=len(order),
                     columns_sampled=min(len(active),path_id*len(active)+completed),
                     width=p, proven_dummy_columns=p-len(active),
                     unique_prefixes=calls)

    common = dict(paths=paths, permutation_sha256=digest, feature_count=p,
                  active_indices=active, implied_zero_marginals=paths*(p-len(active)),
                  sampled_marginal_count=evaluations, all_covered=True, oracle_prefixes=calls)
    utility_evidence = utility_marginals.tolist() if scalar else None
    out = {}
    for method in methods:
        native_min = statistical and method == "minshap"
        marginals = loss_marginals if native_min else utility_marginals
        scores = marginals.min(axis=0) if method=="minshap" else marginals.mean(axis=0)
        details = dict(common, sampled_marginals=marginals.tolist() if native_min else utility_evidence)
        selected = None
        if native_min:
            # Algorithm 1 uses the variance accompanying the minimum observed
            # marginal. For ties use the first minimum on the prespecified paths.
            winners = marginals.argmin(axis=0)
            associated = variances[winners,np.arange(p)]
            thresholds = np.sqrt(-2*np.log(alpha)*associated)
            selected = [name for name,s,t in zip(columns,scores,thresholds,strict=True) if s >= t]
            details.update(alpha=alpha, thresholds=thresholds.tolist(), loss_rows=len(empty[0]),
                           variance_of_mean=variances.tolist(), worst_path=winners.tolist(),
                           variance_rule="Paired between-row sample variance/n on the first minimum-marginal path",
                           degenerate_zero_ties=int(np.count_nonzero((scores==0)&(thresholds==0))),
                           threshold_rule="Algorithm 1: minimum observed marginal >= threshold")
        out[method] = scores, selected, details
    return out


def permutation_scores(method, mode, columns, evaluate, progress, *, active=None,
                       seed=42, paths=PATHS, alpha=ALPHA, evaluate_batch=None):
    """Estimate one method from canonically sorted tuples of original column IDs.

    Native MinShap consumes a per-row loss vector. Scalar oracles supply
    higher-is-better utility for native Shapley and both frozen baselines.
    """
    return _estimate((method,),mode,columns,evaluate,progress,active=active,seed=seed,
                     paths=paths,alpha=alpha,evaluate_batch=evaluate_batch)[method]


def select_permutation_pair(context, *, joint_values=None):
    """Return both Selection objects after one traversal of the common permutations.

    In native mode joint_values(ids) must return the per-row loss vector and
    the scalar utility from the same coalition prediction. The empty coalition
    uses the same convention. Frozen games share the single utility oracle.
    """
    from minppi_mofs.baselines._selection_context import Selection

    frozen = context.mode == "frozen"
    if not frozen and not callable(joint_values):
        raise ValueError("A paired native oracle (row losses, utility) is required")
    estimates = _estimate(("minshap","shapley"),context.mode,context.columns,
                          context.value if frozen else joint_values,context.progress,
                          active=context.active if frozen else None,seed=context.seed,
                          evaluate_batch=_context_batch(context))
    return {method:Selection(scores=scores,selected=support,diagnostics=info)
            for method,(scores,support,info) in estimates.items()}


def _context_batch(context):
    if context.mode=="frozen":
        return context.values
    if getattr(context,"native_workers",1) > 1:
        if not callable(getattr(context,"native_batch",None)):
            raise ValueError("Parallel native evaluation needs an ordered native_batch oracle")
        return context.native_batch
    return None


def select_permutation(context, method):
    """Bridge the common cold-run context to either permutation estimator."""
    from minppi_mofs.baselines._selection_context import Selection

    statistical = method=="minshap" and context.mode=="native"
    scores, selected, diag = permutation_scores(
        method, context.mode, context.columns, context.losses if statistical else context.value,
        context.progress, active=context.active if context.mode=="frozen" else None,
        seed=context.seed, evaluate_batch=_context_batch(context))
    return Selection(scores=scores, selected=selected, diagnostics=diag)

