from itertools import permutations
from math import comb

import pytest

from minppi_mofs.context_design import matched_random_contexts, separating_contexts


@pytest.mark.parametrize("size", [2, 3, 4, 5, 6, 7, 10, 11, 20, 21, 65])
def test_minimal_probes_cover_every_ordered_pair(size):
    probes = separating_contexts(range(size))
    assert probes[0] == frozenset() and probes[-1] == frozenset(range(size))
    informative = probes[1:-1]
    assert len(set(probes)) == len(probes)
    q = len(informative)
    assert comb(q - 1, (q - 1) // 2) < size <= comb(q, q // 2)
    for feature, other in permutations(range(size), 2):
        assert any(feature in probe and other not in probe for probe in informative)
    assert all(sum(feature in probe for probe in informative) == q // 2 for feature in range(size))


def test_lexicographic_signature_assignment_and_degenerate_inputs():
    assert separating_contexts([]) == (frozenset(),)
    assert separating_contexts(["z"]) == (frozenset(), frozenset("z"))
    expected = ["", "cba", "ced", "be", "ad", "cbaed"]
    assert separating_contexts(iter("cbaed")) == tuple(map(frozenset, expected))
    with pytest.raises(ValueError, match="unique"):
        separating_contexts([2, 2])


def test_random_comparator_preserves_ordered_sizes_without_duplicate_queries():
    tail = tuple(range(9))
    template = separating_contexts(tail)
    panels = matched_random_contexts(template, tail, seed=119)
    assert panels == matched_random_contexts(template, iter(tail), seed=119)
    assert len(set(panels)) == len(template)
    assert list(map(len, panels)) == list(map(len, template))
    assert panels[0] == frozenset() and panels[-1] == frozenset(tail)
    assert panels != matched_random_contexts(template, tail, seed=120)
    assert matched_random_contexts((frozenset(),), [], seed=13) == (frozenset(),)


def test_random_comparator_validates_source_family():
    with pytest.raises(ValueError, match="distinct"):
        matched_random_contexts([[], [0], [0], [0, 1]], [0, 1], seed=5)
    with pytest.raises(ValueError, match="tail"):
        matched_random_contexts([[], [3], [0, 1]], [0, 1], seed=5)
