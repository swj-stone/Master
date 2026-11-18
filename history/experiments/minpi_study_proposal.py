"""Propose a floating-point support and verify it against all columns."""
import numpy as np

from minppi_mofs.certified_minpi import _validate, top_from_votes
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _bounds, _choose_boundary, _dyadic_anchor, _layers
from minppi_mofs.word_minpi import native_scale


def verified_float_top(votes, count, *, tiers=None):
    """Return the exact support in column order and diagnostics of its proof.

    A floating-point proposal is accepted directly only if its weakest lower
    bound strictly exceeds every other upper bound. Otherwise resolve the
    overlapping boundary in column order. exact_columns counts columns scored
    with arbitrary-precision integers after the initial interval calculation.
    """
    votes = _validate(votes, count)
    count = int(count)
    w, p = votes.shape
    tiers = _selection_tiers(tiers, p)
    info = dict(arithmetic="all", refined=0, exact_columns=0,
                proposal_accepted=True, proposal_swaps=0)
    if count == p:
        return np.arange(p), info

    proposal = top_from_votes(votes, count, tiers=tiers)[0]
    common = native_scale(p, tiers, w)
    if common:
        info["arithmetic"] = "int64"
        layers = [(den, signed, np.asarray(signed, dtype=np.int64) * (common // den), 0)
                  for den, signed, _, _ in _layers(votes, tiers, None)]
        lower, upper = _bounds(votes, layers[0])
        silent = None  # Native endpoints are points for every column.
    else:
        info["arithmetic"] = "dyadic"
        scale = 1 << min(40, 62 - w.bit_length())
        layers = _layers(votes, tiers, scale)
        lower, upper, silent = _dyadic_anchor(votes, layers[0])

    def refine(ids):
        if len(layers) == 1 or not len(ids):
            return
        matrix = np.ascontiguousarray(votes[:, ids])
        for layer in layers[1:]:
            left, right = _bounds(matrix, layer)
            lower[ids] = np.minimum(lower[ids], left)
            upper[ids] = np.minimum(upper[ids], right)

    refine(proposal)
    threshold = lower[proposal].min()
    challengers = upper >= threshold
    challengers[proposal] = False
    outside = np.flatnonzero(challengers)
    refine(outside)
    info["refined"] = count + len(outside)
    if not len(outside) or np.all(upper[outside] < threshold):
        return proposal, info

    # The remaining outsiders are the only columns that could displace a
    # proposed member. Sorting indices first preserves the input-order tie rule.
    info["proposal_accepted"] = False
    eligible = np.sort(np.r_[proposal, outside[upper[outside] >= threshold]])
    low_cut = np.partition(lower[eligible], -count)[-count]
    high_cut = np.partition(upper[eligible], -count)[-count]
    inside = eligible[lower[eligible] > high_cut]
    boundary = eligible[(lower[eligible] <= high_cut) & (upper[eligible] >= low_cut)]
    remaining = count - len(inside)
    chosen, info["exact_columns"] = _choose_boundary(
        votes, layers, lower, upper, boundary, remaining, silent)
    result = np.sort(np.r_[inside, chosen])
    info["proposal_swaps"] = len(np.setdiff1d(result, proposal, assume_unique=True))
    return result, info
