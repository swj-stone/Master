"""Nominate one extra deletion by the largest minimum yes count over restorations."""


def nominate(order, contexts, ballot, *, prune=True):
    """Return the deletion with the best worst-context vote, ties weakest first.

    order lists the surviving features strongest first. Unseen contexts cannot
    raise a running minimum. Once it reaches the best earlier candidate's
    score, the remaining queries cannot change this tie-broken choice.
    """
    live = tuple(order)
    family = tuple(contexts)
    if not live or len(set(live)) != len(live) or not family:
        raise ValueError("Expected a nonempty, unique feature order and at least one context")
    support = frozenset(live)
    if len(live)==1:
        return support, dict(removed=None, min_yes=None, visits=0)
    best_count = -1
    visits = 0
    for feature in reversed(live):
        candidate = support-{feature}
        running = 4
        for context in family:
            panel = tuple(ballot(candidate,context))
            if len(panel)!=4:
                raise ValueError("A context must return four panel votes")
            running = min(running, sum(bool(v) for v in panel))
            visits += 1
            if prune and running<=best_count:
                break
        if running>best_count:
            best_count, removed = running, feature
    return support-{removed}, dict(removed=removed,min_yes=best_count,visits=visits)
