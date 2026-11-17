import numpy as np
from itertools import product

from experiments.minpi_compression.oracle_audit import complete_votes
from minppi_mofs.dictator_minpi import rank_bounds, sharp_upper_bounds
from minppi_mofs.sampled_linear_minpi import rank_features


def test_partial_transcript_excludes_a_false_positive_before_its_zero_edge():
    g = np.random.default_rng(0)
    x = g.normal(size=(50, 6))
    y = x[:, 0] + x[:, 1] + g.normal(size=50)
    arrays = (x[:30], y[:30], x[30:], y[30:])
    config = dict(task="regression", seed=1, rounds=1)
    original = rank_features(*arrays, **config)
    local = rank_bounds(*arrays, **config, cross_coalition=False)
    revised = rank_bounds(*arrays, **config)
    np.testing.assert_array_equal(original.minimum, local.minimum)
    assert original.minimum[0] == 1 and revised.minimum[0] == 0
    assert revised.witness == local.witness == original.witness
    votes = complete_votes(*arrays, task="regression")
    for j, upper in enumerate(revised.minimum):
        bases = np.array([s for s in range(len(votes)) if not s & (1 << j)])
        assert upper >= np.min(votes[bases | (1 << j)] - votes[bases])


def test_a_true_dictator_survives_cross_coalition_inference():
    x = np.zeros((40, 6))
    x[:, 3] = np.arange(40)
    y = x[:, 3]
    result = rank_bounds(x[:20], y[:20], x[20:], y[20:], task="regression")
    assert result.minimum.tolist() == [0, 0, 0, 1, 0, 0]
    assert result.witness == [None] * x.shape[1]


def test_sharp_boolean_bounds_over_every_partial_two_feature_game():
    all_votes = list(product((False, True), repeat=4))
    for partial in product((None, False, True), repeat=4):
        votes = {s: k for s,k in enumerate(partial) if k is not None}
        complete = [v for v in all_votes if all(v[k] == w for k,w in votes.items())]
        bounds, edges = sharp_upper_bounds(votes, 2)
        for j in (0,1):
            bit = 1 << j
            best = max(min(int(v[s | bit])-int(v[s]) for s in range(4) if not s & bit)
                       for v in complete)
            assert bounds[j] == best
            if best == -1:
                base = sum(1 << i for i in edges[j])
                assert votes[base] and not votes[base | bit]
