"""Backward block search with fixed-reference votes in distinct coalition contexts."""
from collections import deque


def search(anchor_order, extensions, ballot, *, rule="min", eager=False):
    """Find a nonempty single-deletion-irreducible subset for a deterministic ballot.

    anchor_order lists strongest first. ballot(S, E) returns four Boolean ballots
    comparing S union E with the initial anchor union E on the same fixed panels.
    The input extensions are disjoint from the anchor; repeated sets count once.
    """
    order = tuple(anchor_order)
    anchor = frozenset(order)
    tails = tuple(dict.fromkeys(map(frozenset, extensions)))
    if not anchor or len(anchor) != len(order) or not tails or tails[0]:
        raise ValueError("Expected unique anchor features and an empty first extension")
    if any(anchor & tail for tail in tails) or rule not in {"min", "mean", "point"}:
        raise ValueError("Invalid extension or vote rule")
    if rule == "point":
        tails = tails[:1]
    live = anchor
    trace = []
    def try_block(block, phase):
        nonlocal live
        candidate = live.difference(block)
        if not candidate:
            return False
        votes = []
        for tail in tails:
            panel = tuple(bool(x) for x in ballot(candidate, tail))
            if len(panel) != 4:
                raise ValueError("Expected the point panel and three replicated panels")
            votes.append(panel)
            if rule == "min" and not eager and sum(panel) <= 2:
                break
        if rule == "point":
            accepted = votes[0][0]
        elif rule == "mean":
            accepted = sum(map(sum, votes)) > 2 * len(votes)
        else:
            accepted = all(sum(panel) > 2 for panel in votes)
        trace.append(dict(phase=phase, before=len(live), block=list(block),
                          votes=votes, accepted=accepted))
        if accepted:
            live = candidate
        return accepted

    weak = order[::-1]
    pending = deque([weak])
    while pending and len(live) > 1:
        block = pending.popleft()
        if not try_block(block, "blocks") and len(block) > 1:
            mid = (len(block) + 1) // 2
            pending.extendleft((block[mid:], block[:mid]))
    sweep = 0
    while len(live) > 1:
        sweep += 1
        previous = live
        for column in weak:
            if column in live:
                try_block((column,), sweep)
            if len(live) == 1:
                break
        if previous == live:
            break
    return live, dict(trace=trace, singleton_sweeps=sweep, anchor_order=list(order),
                      context_extensions=[sorted(t) for t in tails],
                      stop="nonempty_boundary" if len(live) == 1 else "no_feasible_single_deletion")
