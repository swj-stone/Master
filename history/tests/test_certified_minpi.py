from itertools import product

import numpy as np
import pytest

from minppi_mofs.certified_minpi import select_minpi_support, top_from_votes
from minppi_mofs.exact_minpi import ExactMinPI, pivotal_rates


def test_every_two_by_four_vote_matrix_and_budget():
    # Exhaust all games at this width, including identical columns and tied boundaries.
    for entries in product((-1, 0, 1), repeat=8):
        votes = np.asarray(entries, dtype=np.int8).reshape(2, 4)
        reference = pivotal_rates(votes, (0, 3)).min(axis=0)
        for h in (1, 2, 3, 4):
            answer, diagnostics = top_from_votes(votes, h)
            np.testing.assert_array_equal(answer, np.sort(np.argsort(-reference, kind="stable")[:h]))
            assert diagnostics["evaluated"] <= 2*votes.shape[1]


def test_random_inputs_and_frontier_rescues():
    rng = np.random.default_rng(914)
    rescue_observed = False
    for width in (1, 2, 5, 16, 30, 217):
        for _ in range(6):
            votes = rng.integers(-1, 2, (23, width), dtype=np.int8)
            # Deliberate duplicate columns must tie under both evaluation paths.
            votes[:, -1] = votes[:, 0]
            for h in sorted({1, max(1, width//4), width}):
                expected, _ = top_from_votes(votes, h, prune=False)
                got, info = top_from_votes(np.asfortranarray(votes), h)
                np.testing.assert_array_equal(got, expected)
                rescue_observed |= info["refined"] > h and h < width
    assert rescue_observed


def test_public_api_matches_ranker_and_rejects_bad_budget():
    rng = np.random.default_rng(64)
    X = rng.normal(size=(64, 80))
    y = np.tile([0, 1], 32)
    expected = ExactMinPI(random_state=13, fraction=.17).fit(X, y).get_support(indices=True)
    np.testing.assert_array_equal(select_minpi_support(X, y, random_state=13, fraction=.17), expected)
    with pytest.raises(ValueError):
        select_minpi_support(X, y, fraction=0)
