from itertools import combinations, product

import pytest

from minppi_mofs.context_working_set import search_working_set


def test_constraint_generation_on_every_small_feasibility_table():
    order = (1, 2, 3)
    anchor = frozenset(order)
    family = (frozenset(), frozenset({4}))
    proper = [frozenset(s) for k in (1, 2) for s in combinations(order, k)]
    keys = list(product(proper, family))
    observed_restart = False
    for assignment in range(1 << len(keys)):
        rules = {key: bool(assignment >> i & 1) for i, key in enumerate(keys)}
        def feasible(s, e):
            return s == anchor or rules[s, e]
        s, stages = search_working_set(order, family, lambda s,e: (feasible(s,e),)*4)
        observed_restart |= len(stages)>1
        assert len(stages)<=len(family)
        assert s and all(feasible(s,e) for e in family)
        assert len(s)==1 or all(any(not feasible(s-{i},e) for e in family) for i in s)
    assert observed_restart


def test_invalid_anchor_rejected():
    with pytest.raises(ValueError, match="feasibility"):
        search_working_set([1], [set(), {2}], lambda s,e: (False,)*4)
