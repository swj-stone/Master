"""Restoration masks from a minimum completely separating system (Spencer, 1970)."""

from itertools import combinations
from math import comb
from random import Random


def separating_contexts(tail_order):
    """Return empty, constant-weight-code, and full restorations in that order.

    Every ordered pair of distinct tail features occurs as present/absent in
    some mask. The caller supplies the rank order used to assign codewords.
    No row labels, predictions or observed votes enter the design.
    """
    tail = tuple(tail_order)
    full = frozenset(tail)
    if len(full) != len(tail):
        raise ValueError("Tail features must be unique")
    if not tail:
        return (full,)
    if len(tail) == 1:
        return (frozenset(), full)

    bits = 2
    while comb(bits, bits // 2) < len(tail):
        bits += 1
    masks = [set() for _ in range(bits)]
    signatures = combinations(range(bits), bits // 2)
    for feature, signature in zip(tail, signatures):
        for bit in signature:
            masks[bit].add(feature)
    return (frozenset(), *map(frozenset, masks), full)


def matched_random_contexts(contexts, tail_order, *, seed):
    """Replace each interior mask by a same-size uniform draw, rejecting duplicates."""
    tail = tuple(tail_order)
    full = frozenset(tail)
    if len(full) != len(tail):
        raise ValueError("Tail features must be unique")
    template = tuple(map(frozenset, contexts))
    if not template or template[0] or template[-1] != full or any(mask - full for mask in template):
        raise ValueError("Expected masks within the tail, with empty and full endpoints")
    if len(set(template)) != len(template):
        raise ValueError("The template must have distinct masks")
    generator = Random(seed)
    sampled = []
    used = set()
    for source in template:
        mask = source
        if source and source != full:
            mask = frozenset(generator.sample(tail, len(source)))
            while mask in used:
                mask = frozenset(generator.sample(tail, len(source)))
        used.add(mask)
        sampled.append(mask)
    return tuple(sampled)
