"""Generate separating constraints only when a provisional support violates one."""
from functools import lru_cache

from minppi_mofs.context_vote import search


def search_working_set(anchor_order, extensions, ballot):
    """Find a locally irreducible support for a deterministic, anchor-feasible full family."""
    order = tuple(anchor_order)
    family = tuple(dict.fromkeys(map(frozenset, extensions)))
    if not family or family[0]:
        raise ValueError("The first restoration must be empty")
    active = {family[0]}
    stages = []

    @lru_cache(maxsize=None)
    def fixed_ballot(support, extension):
        votes = tuple(bool(v) for v in ballot(support, extension))
        if len(votes) != 4:
            raise ValueError("Expected four paired ballots")
        return votes

    while True:
        masks = [extension for extension in family if extension in active]
        result, trace = search(order, masks, fixed_ballot, rule="min")
        # Every descending trajectory starts from the same feasible reference.
        blocker = next((extension for extension in family if sum(fixed_ballot(result, extension)) < 3), None)
        stages.append(dict(active=len(active), retained=len(result), proposals=len(trace["trace"]),
                           added=None if blocker is None else family.index(blocker)))
        if blocker is None:
            return result, stages
        if blocker in active or result == frozenset(order):
            raise ValueError("The anchor or active-family result violates its assumed feasibility")
        active.add(blocker)
