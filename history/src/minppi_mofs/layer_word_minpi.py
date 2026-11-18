"""Exact pivotal support with unscaled, layer-specific signed-word sums."""
from functools import cmp_to_key
from math import comb

import numpy as np

from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _bounds, _layers, rational_top_from_votes


def layers_fit_word(p, tiers, w):
    """Sufficient bound for every partial sum on each separate layer."""
    return int(w) * max(comb(p - 1, k) for k in tiers) <= 2**63 - 1


def _compare_ratios(a, b):
    # The numerator pairs below contain Python ints, including for native sums.
    if a[1] == b[1]:
        return a[0] - b[0]
    return a[0] * b[1] - b[0] * a[1]


_ratio_key = cmp_to_key(_compare_ratios)


def layer_word_top_from_votes(votes, count, *, tiers=None):
    """Return the exact support in input order and the arithmetic path used.

    Accumulate each unscaled layer numerator in int64 when its worst-case
    magnitude fits. Compare rational scores by Python-integer cross products.
    If any layer cannot be accumulated safely, use the certified dyadic path.
    """
    votes = _validate(votes, count)
    count = int(count)
    w, p = votes.shape
    tiers = _selection_tiers(tiers, p)
    if count == p:
        return np.arange(p), dict(arithmetic="all", refined=0, exact_columns=0)
    if not layers_fit_word(p, tiers, w):
        selected, info = rational_top_from_votes(votes, count, tiers=tiers)
        return selected, dict(arithmetic="dyadic", **info)

    layers = [(d, signed, np.asarray(signed, dtype=np.int64), 0)
              for d, signed, _, _ in _layers(votes, tiers, None)]
    anchor_layer = layers[0]
    anchor_den = anchor_layer[0]
    anchor, _ = _bounds(votes, anchor_layer)
    head = np.sort(np.argsort(-anchor, kind="stable")[:count])
    if len(layers) == 1:
        return head, dict(arithmetic="layer_int64", refined=count, exact_columns=count)

    def complete(ids):
        matrix = np.ascontiguousarray(votes[:, ids])
        minima = [(n, anchor_den) for n in anchor[ids].tolist()]
        for layer in layers[1:]:
            denominator = layer[0]
            numerators, _ = _bounds(matrix, layer)
            # tolist() converts int64 to Python int before any multiplication.
            for i, numerator in enumerate(numerators.tolist()):
                score = (numerator, denominator)
                if _compare_ratios(score, minima[i]) < 0:
                    minima[i] = score
        return minima

    scores = complete(head)
    cut_num, cut_den = min(scores, key=_ratio_key)
    # Keep all equality cases: an outsider might win by its original index.
    anchor_cutoff = -((-cut_num * anchor_den) // cut_den)
    eligible = anchor >= anchor_cutoff
    eligible[head] = False
    outside = np.flatnonzero(eligible)
    if not len(outside):
        return head, dict(arithmetic="layer_int64", refined=count, exact_columns=count)

    scores.extend(complete(outside))
    ids = np.concatenate((head, outside))
    rows = zip(ids.tolist(), scores)
    ordered = sorted(rows, key=lambda row: (_ratio_key(row[1]), -row[0]), reverse=True)
    selected = np.sort([col for col, _ in ordered[:count]])
    return selected, dict(arithmetic="layer_int64", refined=len(ids), exact_columns=len(ids))
