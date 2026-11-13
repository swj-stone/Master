"""Native PFI and ascending random-forest prefixes with a common validation floor."""

import numpy as np

from experiments.test.piv2_conventional_models import choose_model
from experiments.test.run_sampled_permutation_seed42 import REPEATS, sampled_drops


def select(backend, train, yt, val, yv, *, support, search_seed, policy, reference_scores,
           tolerance, evaluate, get_input_model, progress=None):
    """Select within an already certified support; evaluate always scores in original column order."""
    support = list(support)
    members = set(support)
    if (not support or not train.columns.is_unique or list(train.columns) != list(val.columns)
            or support != [col for col in train.columns if col in members]):
        raise ValueError("Expected an ordered, nonempty subset of aligned training and validation columns")
    floors = (1-tolerance)*np.asarray(reference_scores, dtype=float)
    if floors.shape != (2,) or not np.isfinite(floors).all() or not 0 < tolerance < 1:
        raise ValueError("Invalid reference or relative tolerance")
    if progress is None:
        def progress(phase, **counts):
            pass

    proposals = []

    def assess(columns):
        scores = np.asarray(evaluate(columns), dtype=float)
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise ValueError("Expected finite validation AUC and accuracy")
        accepted = bool(np.all(scores + 1e-12 >= floors))
        proposals.append(dict(proposed=columns, k=len(columns), auc=float(scores[0]),
                              accuracy=float(scores[1]), accepted=accepted))
        return accepted

    extra = {}
    if backend == "pfi":
        model, full_pred = get_input_model()
        safe_val = val.loc[:, support].set_axis([f"f{i}" for i in range(len(support))], axis=1)
        # The predictor is fixed; only permutation selection uses the varying search seed.
        rows, base, drops = sampled_drops(model, safe_val, np.asarray(yv), full_pred, "roc_auc",
                                          dict(policy, seed=search_seed), progress)
        scores = drops.mean(axis=1)
        order = np.argsort(-scores, kind="stable")
        ranking = [support[i] for i in order]
        native = [c for c,score in zip(support,scores,strict=True) if score > 0.] or ranking[:1]
        accepted = assess(native)
        selected = native if accepted else support
        fallback = not accepted
        extra.update(row_indices=rows.tolist(), repeats=REPEATS, sample_full_auc=float(base),
                     drops=drops.tolist(), native=native, ranking_seed=search_seed)
        auxiliary = 0
    elif backend == "rf_prefix":
        _, auxiliary, rf = choose_model("random_forest", train.loc[:, support], yt, policy,
                                         original_width=train.shape[1])
        scores = np.asarray(rf["scores"], dtype=float)
        order = np.argsort(-scores, kind="stable")
        ranking = [support[i] for i in order]
        included = set()
        for k,col in enumerate(ranking,1):
            included.add(col)
            selected = [c for c in support if c in included]
            accepted = assess(selected)
            progress("rf_prefix", checked=k, candidates=len(support))
            if accepted:
                break
        if not accepted:
            raise ValueError("Certified input unexpectedly failed the full-prefix validation check")
        fallback = len(selected) == len(support)
        extra.update(trees=rf["trees"], ranking_seed=policy["seed"],
                     ranking_sample_rows=rf["ranking_sample_rows"])
    else:
        raise ValueError(f"Unknown representative: {backend}")
    return selected, dict(backend=backend, support=support, ranking=ranking, importance=scores.tolist(),
                          proposals=proposals, selector_aux_fits=auxiliary, endpoint_fallback=fallback, **extra)
