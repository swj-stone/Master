from itertools import product

from minppi_mofs.context_vote import search
from minppi_mofs.veto_schedule import search_with_recent_veto


def test_recent_veto_equivalence_and_trace_on_arbitrary_predicates():
    order = (0, 1, 2)
    family = ((), (3,), (4,))
    for pattern in product(range(4), repeat=6):
        def ballot(support, tail):
            mask = sum(1 << column for column in support)
            fail = pattern[mask-1] == family.index(tuple(tail))
            return (True, not fail, not fail, False)

        direct, original = search(order, family, ballot)
        recent, scheduled = search_with_recent_veto(order, family, ballot)
        assert direct == recent
        assert [(p["block"], p["accepted"]) for p in original["trace"]] == [
            (p["block"], p["accepted"]) for p in scheduled["trace"]]
        live = frozenset(order)
        for decision in scheduled["trace"]:
            candidate = live.difference(decision["block"])
            actual = decision["evaluated_contexts"]
            assert actual[0] == []
            votes = [ballot(candidate, frozenset(t)) for t in actual]
            assert votes == decision["votes"]
            assert all(sum(v) > 2 for v in votes[:-1])
            if decision["accepted"]:
                assert len(actual) == len(family)
                live = candidate
