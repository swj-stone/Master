from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.context_vote import search


@pytest.mark.parametrize("width", range(1, 9))
def test_lazy_and_eager_decide_the_same_path_and_certify_terminal_support(width):
    rng = np.random.default_rng(440+width)
    order = list(range(width))
    tails = (frozenset(), frozenset({9}), frozenset({9,10}))
    truth = {(frozenset(sub), tail): tuple(rng.random(4) > .17)
             for n in range(1,width+1) for sub in combinations(order,n) for tail in tails}
    def ballot(sub, tail):
        return truth[sub,tail]
    final, lazy = search(order,tails,ballot)
    eager_final, eager = search(order,tails,ballot,eager=True)
    assert final == eager_final
    def path(info):
        return [(step["block"],step["accepted"],step["phase"]) for step in info["trace"]]
    assert path(lazy) == path(eager)
    for j in final:
        assert len(final)==1 or any(sum(ballot(final-{j}, t))<=2 for t in tails)
    assert sum(len(e["votes"]) for e in lazy["trace"]) <= sum(len(e["votes"]) for e in eager["trace"])


def test_strict_majority_context_veto_and_scope_of_the_certificate():
    tails = (frozenset(), frozenset({"x"}), frozenset({"x"}))
    def ballot(mask, extension):
        if mask == {"a"} and extension:
            return (False,False,True,True)
        return (True,True,True,True) if mask == {"a"} else (False,False,False,False)
    min_set, info = search(("a","b"),tails,ballot)
    assert min_set == {"a","b"}
    assert len(info["context_extensions"]) == 2
    assert search(("a","b"),tails,ballot,rule="mean")[0] == {"a"}
    assert search(("a","b"),tails,ballot,rule="point")[0] == {"a"}


def test_previously_rejected_singleton_may_become_removable():
    # Nonmonotone support feasibility needs a complete terminal sweep.
    good = {frozenset(s) for s in ("abc","ac","a")}
    final, info = search(list("abc"),[()],lambda m,_: (m in good,)*4)
    assert final == {"a"} and info["stop"] == "nonempty_boundary"
