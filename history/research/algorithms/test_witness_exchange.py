"""Compare the optimized two-context exchange rule with literal peer removal."""
import numpy as np
import pytest

from research.algorithms.witness_exchange import _exchange_gains, exchange_from_votes, exchange_top_from_training
from minppi_mofs.grouped_minpi import grouped_top_from_training


def test_strict_complementarity_and_plateau():
    votes = np.array([[0, 1, 1, 0], [1, 0, -1, 1], [1, 1, 1, 0]])
    final, info = exchange_from_votes(votes, 2, tiers=(0, 1, 2))
    np.testing.assert_array_equal(info["initial_support"], [0, 3])
    np.testing.assert_array_equal(final, [0, 1])
    assert info["swapped"] and info["candidate_full"] == info["candidate_omitted"] == 1
    unchanged, plateau = exchange_from_votes(np.ones((4, 6)), 2)
    assert not plateau["swapped"]
    np.testing.assert_array_equal(unchanged, [0, 1])


@pytest.mark.parametrize("k", [1, 2, 5, 9])
def test_literal_omissions(k):
    votes = np.random.default_rng(1129 + k).integers(-1, 2, size=(19, 12), dtype=np.int8)
    support = np.arange(k)
    outgoing = support
    incoming = np.arange(k, 12)
    margin = votes[:, support].sum(axis=1)
    full, omitted = _exchange_gains(votes, support, outgoing, incoming, margin)
    for oi, old in enumerate(outgoing):
        peers = support[support != old]
        for ni, new in enumerate(incoming):
            def gain(context):
                base = votes[:, context].sum(axis=1)
                return (np.sign(base + votes[:, new]) - np.sign(base + votes[:, old])).sum()
            expected = gain(peers)
            assert full[oi, ni] == expected * max(1, len(peers))
            literal = sum(gain(peers[peers != q]) for q in peers) if len(peers) else expected
            assert omitted[oi, ni] == literal


def test_training_starts_at_grouped_reference():
    rng = np.random.default_rng(678)
    X = rng.normal(size=(240, 41))
    y = np.repeat(np.arange(3), 80)
    for width in (1, 20, 41):
        old, _ = grouped_top_from_training(X, y, width, random_state=1129)
        selected, info = exchange_top_from_training(X, y, width, random_state=1129)
        np.testing.assert_array_equal(old, info["initial_support"])
        assert len(np.unique(selected)) == width and len(np.setdiff1d(old, selected)) <= 1
