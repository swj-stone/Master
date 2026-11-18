"""First adaptive deletion pilot, extracted from its frozen validation-only probe."""

import numpy as np

from minppi_mofs.piv2 import prefilter


def search(train, labels, *, search_seed, tolerance, reference_scores, evaluate, budget=12):
    """Replay the original adaptive step and boundary randomization with fresh evaluations."""
    reference = np.asarray(reference_scores, dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference) & (reference > 0)):
        raise ValueError("Expected positive reference validation AUC and accuracy")
    if not 0 < tolerance < 1 or budget < 0:
        raise ValueError("Invalid loss tolerance or candidate budget")
    rng = np.random.default_rng(search_seed)
    keep = train.columns.tolist()
    best = reference.copy()
    accepted = True
    seen = set()
    steps = []
    step = 0
    for i in range(budget):
        if len(keep) <= 1:
            break
        if accepted:
            _, votes = prefilter(train[keep], labels, task="binary", seed=int(rng.integers(2**31)))
            order = np.lexsort((np.arange(len(keep)), votes["standalone"], votes["min_vote"]))
            if i == 0:
                step = max(1, len(votes["removed"]))
        step = min(step, len(keep)-1)
        band = min(step, len(keep)-step, max(1, int(np.sqrt(len(keep)))))
        first = order[:step-band]
        pool = order[step-band:step+band]
        for _ in range(8):
            drop = set(first.tolist() + rng.choice(pool, size=band, replace=False).tolist())
            proposal = tuple(name for j, name in enumerate(keep) if j not in drop)
            if proposal not in seen:
                break
        else:
            break

        seen.add(proposal)
        proposed = list(proposal)
        scores = np.asarray(evaluate(proposed), dtype=float)
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise ValueError("Expected finite candidate validation AUC and accuracy")
        ratio = float(max((reference-scores) / (reference*tolerance)))
        accepted = ratio <= 1 + 1e-12
        steps.append(dict(parent=keep.copy(), proposed=proposed, step_type="boundary_delete",
                          k=len(proposed), removed=step, accepted=accepted,
                          auc=float(scores[0]), accuracy=float(scores[1]), tightest_relative_budget_used=ratio))
        if accepted:
            keep = proposed
            best = scores
            step = max(1, round(step*float(np.clip(2-1.5*max(0, ratio), .5, 2))))
        else:
            step = max(1, int(np.floor(step*float(np.clip(.8/max(ratio, 1), .2, .75)))))

    return keep, dict(steps=steps, validation_scores=best.tolist(),
                      search_seed=search_seed, tolerance=tolerance, budget=budget)
