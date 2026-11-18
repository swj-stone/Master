"""Experimental backward elimination with a checked single-column stopping rule."""
import numpy as np

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.piv3_compact_witness import _sample, _within


def _panel_weights(labels, task, seed):
    """One pooled vote and three stratified votes using a fixed split per search."""
    n = len(labels)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 659]))
    rows = np.arange(n)
    groups = [rows] if task == "regression" else [rows[labels == c] for c in np.unique(labels)]
    # A disjoint panel needs a representative of every class for a finite OvR AUC.
    disjoint = all(len(group) >= 3 for group in groups)
    weights = np.zeros((4, n), dtype=float)
    weights[0] = 1. / n
    for group in groups:
        if disjoint:
            for i, panel in enumerate(np.array_split(rng.permutation(group), 3), 1):
                weights[i, panel] = len(group) / (n * len(panel))
        else:
            for i in range(1, 4):
                draws = rng.integers(len(group), size=len(group))
                weights[i, group] = np.bincount(draws, minlength=len(group)) / n
    return rows, weights, "stratified_disjoint" if disjoint else "stratified_bootstrap_small_groups"


def select(anchor_info, model, train, ranking, yr, *, task, utility, search_seed, vote_mode,
           progress=None, background_seed=42):
    """Return a nonempty support locally irreducible for these fixed votes and contexts.

    Test singletons, weakest first. Every successful deletion strictly reduces
    the live support. Re-evaluate previous rejections after further deletions.
    A no-change pass or the nonempty-support boundary is the only normal exit.
    """
    order = tuple(anchor_info.get("anchor_order", anchor_info["order"]))
    k = anchor_info.get("anchor_k", anchor_info["k"])
    cols = list(train.columns)
    if (list(ranking.columns) != cols or not train.columns.is_unique or len(set(order)) != len(order)
            or not 1 <= k <= len(order) or not set(order).issubset(cols)):
        raise ValueError("Expected an aligned column schema and nonempty Gain anchor prefix")
    labels = np.asarray(yr)
    if labels.shape != (len(ranking),) or not len(labels):
        raise ValueError("Expected nonempty ranking rows and aligned labels")
    if vote_mode == "legacy":
        rows, weights = _sample(labels, task, np.random.SeedSequence([search_seed, 659]))
        panel_rule = "capped_paired_bootstrap"
    elif vote_mode == "full":
        rows, weights, panel_rule = _panel_weights(labels, task, search_seed)
    else:
        raise ValueError("vote_mode must be 'legacy' or 'full'")
    rare = task == "binary" and np.min(np.unique(labels, return_counts=True)[1]) <= .05 * len(labels)
    score = make_scorer(labels[rows], weights, task, utility, rare)
    x = np.ascontiguousarray(ranking.iloc[rows].to_numpy(), dtype=np.float32)
    rng = np.random.default_rng(background_seed)
    bg = rng.choice(len(train), min(512, len(train)), replace=False)
    fill = np.nanmedian(train.iloc[bg].to_numpy(dtype=np.float32), axis=0)
    masked = np.empty_like(x)
    cache = {}
    readout = model.predict if task == "regression" else model.predict_proba

    def assess(mask):
        if mask not in cache:
            masked[:] = fill
            indices = sorted(mask)
            masked[:, indices] = x[:, indices]
            cache[mask] = score(readout(masked))
        return cache[mask]

    indices = train.columns.get_indexer(order).tolist()
    anchor = frozenset(indices[:k])
    tails = (frozenset(), frozenset(indices[k:2*k]), frozenset(indices[k:]))
    # Identical contexts have identical votes and the same minimum.
    additions = tuple(dict.fromkeys(tails))
    original = assess(anchor)[0] if k > 1 else None

    def proposal(live, j):
        after = live - {j}
        if not _within(assess(after)[0], original):
            return False, "cumulative_anchor", None
        for tail in additions:
            before_votes, after_votes = assess(live | tail), assess(after | tail)
            yes = sum(_within(a, b) for a, b in zip(after_votes, before_votes, strict=True))
            if 2 * yes <= len(before_votes):
                return False, "context_" + str(tails.index(tail)), int(yes)
        return True, "accepted", None

    live = anchor
    sweeps = []
    proposals = 0
    while len(live) > 1:
        size = len(live)
        decisions = []
        removed = []
        for j in reversed(indices[:k]):
            if j not in live:
                continue
            accepted, rule, votes = proposal(live, j)
            proposals += 1
            decisions.append(dict(column=cols[j], accepted=accepted, decisive_rule=rule, votes_at_veto=votes))
            if accepted:
                live -= {j}
                removed.append(cols[j])
            if progress:
                progress(sweep=len(sweeps)+1, sweep_start_k=size, checked=len(decisions), k=len(live),
                         tried=proposals, voting_calls=len(cache), voting_rows=len(cache)*len(rows))
            if len(live) == 1:
                break
        sweeps.append(dict(start_k=size, end_k=len(live), covered=len(decisions), removed=removed,
                           decisions=decisions))
        if not removed:
            if len(decisions) != len(live):
                raise RuntimeError("Incomplete terminal sweep")
            break
    boundary = len(live) == 1
    selected = [name for j, name in enumerate(cols) if j in live]
    evidence = dict(stop_reason="minimum_size" if boundary else "no_accepted_singleton",
                    anchor_k=k, final_k=len(live), proposals=proposals, sweeps=sweeps,
                    accepted_per_sweep=[len(s["removed"]) for s in sweeps], sweeps_completed=len(sweeps),
                    final_sweep_covered=0 if boundary else sweeps[-1]["covered"],
                    voting_calls=len(cache), voting_rows=len(cache)*len(rows), ranking_rows_used=len(rows),
                    available_ranking_rows=len(labels), panels=panel_rule, search_seed=search_seed,
                    background_seed=background_seed, vote_mode=vote_mode, distinct_contexts=len(additions),
                    panel_sizes=np.count_nonzero(weights, axis=1).tolist(), voting_row_ids=rows.tolist())
    return selected, evidence
