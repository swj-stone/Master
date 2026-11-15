from collections import defaultdict
from itertools import product

import numpy as np
import pytest

from experiments.context_review.matched_stopping import choose
from experiments.context_review.vote_certificates import prefix_minimum
from minppi_mofs.minimum_vote import minimum_vote_winner


@pytest.mark.parametrize("rows,contexts,panels", [(2, 2, 2), (3, 2, 1), (2, 3, 1)])
def test_nomination_and_minimum_certificates_by_enumerating_completions(rows, contexts, panels):
    row_values = tuple(product(range(panels + 1), repeat=contexts))
    tables = tuple(product(row_values, repeat=rows))
    prefixes = tuple(product(range(contexts + 1), repeat=rows))
    completions = defaultdict(set)

    def observed(table, lengths):
        return tuple(row[:n] for row, n in zip(table, lengths, strict=True))

    for table in tables:
        winner = max(range(rows), key=lambda i: min(table[i]))
        for lengths in prefixes:
            completions[observed(table, lengths)].add(winner)

    for table in tables:
        seen = set()

        def read(i, j):
            assert (i, j) not in seen
            assert j == 0 or (i, j-1) in seen
            seen.add((i, j))
            return table[i][j]

        nominee = minimum_vote_winner(range(rows), range(contexts), read, panels=panels)
        depths = [sum(i == k for k, _ in seen) for i in range(rows)]
        assert completions[observed(table, depths)] == {nominee}
        # This reference examines all tables with the same prefixes, regardless of the search.
        shortest = min(sum(lens) for lens in prefixes if len(completions[observed(table, lens)]) == 1)
        assert prefix_minimum(np.asarray(table), panels) == shortest
        if nominee != 0 or any(min(row) > 0 for row in table[1:]):
            assert len(seen) == shortest
        visited = set()

        def prior_score(i, j):
            visited.add((i, j))
            # The legacy engine uses a fixed maximum count of four.
            return (4 * table[i][j] // panels,)

        assert choose(range(rows), range(contexts), prior_score, "vote", True) == nominee
        assert seen <= visited


def test_empty_slate_and_unopposed_first_candidate_need_no_predictions():
    def unexpected(*_):
        pytest.fail("The decision already follows from candidate order")

    assert minimum_vote_winner([], [0], unexpected) is None
    assert minimum_vote_winner(["only"], [0, 1], unexpected) == "only"


def test_first_winner_zero_rivals_can_have_a_much_shorter_certificate():
    table = np.array([[4] * 12, [0] * 12])
    queries = []

    def read(i, j):
        queries.append((i, j))
        return table[i, j]

    assert minimum_vote_winner([0, 1], range(12), read) == 0
    assert len(queries) == 12 and prefix_minimum(table, 4) == 1
