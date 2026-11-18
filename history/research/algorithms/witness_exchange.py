"""One equal-width exchange evaluated on the original grouped minPI witness game."""
import numpy as np

from minppi_mofs.exact_minpi import _witness_margins_from_folds
from minppi_mofs.grouped_minpi import _representatives
from minppi_mofs.word_minpi import exact_top_from_votes


def _exchange_gains(votes, support, outgoing, incoming, margin):
    """Integer gain numerators for full context and uniform one-peer omission.

    The shared denominator is rows * max(1, len(support)-1). The fixed peer
    set for a proposed pair consists of the retained columns except outgoing.
    """
    old = votes[:, outgoing, None]
    new = votes[:, None, incoming]
    peers = len(support) - 1
    previous = margin[:, None, None]
    updated = previous - old + new
    full = (np.sign(updated) - np.sign(previous)).sum(axis=0) * max(1, peers)
    if peers == 0:
        return full, full
    omitted = np.zeros_like(full)
    selected = votes[:, support]
    for sign in (-1, 0, 1):
        multiplicity = np.count_nonzero(selected == sign, axis=1)[:, None, None] - (old == sign)
        omitted += ((np.sign(updated - sign) - np.sign(previous - sign)) * multiplicity).sum(axis=0)
    return full, omitted


def exchange_from_votes(votes, count, *, tiers=None):
    """Start at the exact minPI top k, then accept at most one strict gain.

    Propose up to eight low-cost removals and eight beneficial additions using
    the observed full-support majority. Exact paired gains can differ from
    these proposal scores. Ties in proposals and accepted pairs use column ID.
    """
    selected, info = exact_top_from_votes(votes, count, tiers=tiers)
    votes = np.asarray(votes, dtype=np.int8)
    remaining = np.setdiff1d(np.arange(votes.shape[1]), selected, assume_unique=True)
    info = dict(info, initial_support=selected.copy(), swapped=False)
    if not len(remaining):
        return selected, info
    margin = votes[:, selected].sum(axis=1, dtype=np.int64)
    original = np.sign(margin)
    # A one-column add/drop can change the sign only for |margin| <= 1.
    near = np.abs(margin) <= 1
    base = margin[near, None]
    rows = votes[near]
    losses = (original[near, None] - np.sign(base - rows[:, selected])).sum(axis=0)
    gains = (np.sign(base + rows[:, remaining]) - original[near, None]).sum(axis=0)
    outgoing = np.sort(selected[np.lexsort((selected, losses))[:8]])
    incoming = np.sort(remaining[np.lexsort((remaining, -gains))[:8]])
    full, omitted = _exchange_gains(votes, selected, outgoing, incoming, margin)
    worst = np.minimum(full, omitted)
    row, col = np.unravel_index(worst.argmax(), worst.shape)
    info.update(proposals=int(worst.size), denominator=len(votes) * max(1, count - 1),
                candidate_full=int(full[row, col]), candidate_omitted=int(omitted[row, col]))
    if worst[row, col] <= 0:
        return selected, info
    result = selected.copy()
    result[result == outgoing[row]] = incoming[col]
    info.update(swapped=True, dropped=int(outgoing[row]), added=int(incoming[col]))
    return np.sort(result), info


def exchange_top_from_training(X, y, count, *, random_state=42):
    """Fixed-budget dense classification selector. X and y are training rows."""
    X, y = np.asarray(X), np.asarray(y)
    reps, tiers, info, folds = _representatives(X, y, "classification", random_state, count, None)
    reps = np.sort(reps)
    if len(reps) == count:
        return reps, dict(info, initial_support=reps.copy(), swapped=False, proposals=0)
    margins = _witness_margins_from_folds(X, y, "classification", *folds, columns=reps)
    effective = sorted({min(k, len(reps) - 1) for k in tiers})
    columns, change = exchange_from_votes(np.sign(margins).astype(np.int8), count, tiers=effective)
    change["initial_support"] = reps[change["initial_support"]]
    if change["swapped"]:
        change.update(dropped=int(reps[change["dropped"]]), added=int(reps[change["added"]]))
    return reps[columns], dict(info, **change)

