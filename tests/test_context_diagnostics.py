"""Regression checks for paired context controls and finite holdout coverage."""
import numpy as np

from experiments.context_diagnostics.design import config, decide, families, novel_masks


def test_held_out_contexts_exclude_all_seven_decisions():
    assert config()["archived_trees"]["candidates_per_split"] == 12
    tail = tuple(range(10))
    rules = families(tail, 5, np.random.default_rng(18))
    exposed = {e for group in rules.values() for e in group}
    masks = novel_masks(tail, 5, exposed, np.random.default_rng(3), 4)
    assert all(mask.issubset(tail) and mask not in exposed for mask in masks)
    assert len(masks) == len(set(masks)) == 12
    assert {len(mask) for mask in masks} == {2, 5, 8}
    for a,b in (("nested","random_middle"),("separating","random_matched")):
        assert list(map(len,rules[a])) == list(map(len,rules[b]))


def test_absent_interior_context_does_not_fabricate_holdout():
    for tail in ((),("a",),("a","b")):
        rules = families(tail, 3, np.random.default_rng(11))
        seen = {e for group in rules.values() for e in group}
        assert novel_masks(tail, 3, seen, np.random.default_rng(11), 4) == []


def test_rule_visits_and_rejection_on_ties():
    assert decide("pooled", [0,4,4]) == (True,3)
    assert decide("nested", [0,4,4]) == (False,1)
    assert decide("separating", [4,2,4]) == (False,2)
    assert decide("separating", [3,4,3]) == (True,3)
