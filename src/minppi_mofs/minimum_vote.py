"""Select the first maximum of finite, integer-valued row minima."""
from heapq import heappop, heappush
from operator import index


def minimum_vote_winner(candidates, contexts, count_votes, *, panels=4):
    """Query the leading possible winner, using the supplied order for ties.

    Each callback returns the number of approving panels for one candidate
    and one context. Context order is fixed within each row. An empty slate
    returns None so the caller can fall back to its reference feature set.
    """
    candidates, contexts = tuple(candidates), tuple(contexts)
    panels = index(panels)
    if panels < 1 or not contexts:
        raise ValueError("Supply a positive panel count and at least one context")
    if not candidates:
        return None
    # Negative upper bounds give heap priority to the largest bound, then
    # to the earliest original candidate. The initial list is already a heap.
    leaders = [(-panels, row) for row in range(len(candidates))]
    depth = [0] * len(candidates)
    while True:
        neg_bound, row = heappop(leaders)
        upper = -neg_bound
        if depth[row] == len(contexts) or upper == 0:
            return candidates[row]
        if row == 0 and (not leaders or leaders[0][0] == 0):
            return candidates[row]
        votes = index(count_votes(candidates[row], contexts[depth[row]]))
        if not 0 <= votes <= panels:
            raise ValueError("A vote count must lie between zero and the panel count")
        depth[row] += 1
        heappush(leaders, (-min(upper, votes), row))
