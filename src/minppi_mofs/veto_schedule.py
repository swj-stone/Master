"""Evaluate the minimum vote by visiting the last nonempty veto immediately after the empty context."""
from .context_vote import search


def search_with_recent_veto(anchor_order, extensions, ballot):
    """Run the standard block/singleton search and record the contexts actually visited.

    With deterministic ballots, the proposal path and selected support equal the
    fixed-order minimum rule. Reordering can raise or lower the number of queries.
    The first empty restoration is retained as a cheap common check.
    """
    fixed = tuple(dict.fromkeys(map(frozenset, extensions)))
    sequence = list(fixed)
    slots = {tail: index for index, tail in enumerate(fixed)}
    visited = []

    def scheduled_ballot(support, slot):
        index = slots[slot]
        if index == 0:
            visited.append([])
        tail = sequence[index]
        votes = tuple(bool(v) for v in ballot(support, tail))
        visited[-1].append(sorted(tail))
        if sum(votes) <= 2 and index > 1:
            sequence.insert(1, sequence.pop(index))
        return votes

    support, record = search(anchor_order, fixed, scheduled_ballot)
    for decision, contexts in zip(record["trace"], visited, strict=True):
        decision["evaluated_contexts"] = contexts
    record["context_schedule"] = "empty_then_most_recent_veto"
    return support, record
