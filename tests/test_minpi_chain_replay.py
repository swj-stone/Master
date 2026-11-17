"""Boolean-chain search and its limitations without relying on a fitted ridge model."""
import math
from itertools import combinations, product
from fractions import Fraction

import numpy as np

from minppi_mofs.chain_minpi import _brackets, _query, refine_brackets, rank_chain
from minppi_mofs.dictator_minpi import sharp_upper_bounds
from experiments.minpi_compression.chain_uniform_expectation import detection_probabilities


def test_every_binary_descending_bracket_of_length_up_to_thirteen():
    for length in range(1,14):
        chain = [(1 << j)-1 for j in range(length+1)]
        for inside in product((False,True),repeat=length-1):
            values = dict(zip(chain,(True,*inside,False),strict=True))
            observed = {chain[0]:True,chain[-1]:False}
            def read(mask):
                assert mask not in observed
                observed[mask] = values[mask]
                return values[mask]
            refine_brackets(read,[chain])
            assert len(observed)-2 <= math.ceil(math.log2(length))
            assert any(observed.get(chain[i]) is True and observed.get(chain[i+1]) is False
                       for i in range(length))


class BooleanBallot:
    def __init__(self,p,rule):
        self.bits = [1 << i for i in range(p)]
        self.rule = rule
        self.votes = {0:rule(0), (1 << p)-1:rule((1 << p)-1)}

    def base_vote(self,cols,mask):
        assert sum(1 << int(j) for j in cols) == mask
        self.votes[mask] = self.rule(mask)
        return self.votes[mask], None


def test_a_nonmonotone_game_has_genuine_witnesses_after_bisection():
    p=12
    # Every level-three coalition wins; every level-six coalition loses.
    ballot = BooleanBallot(p,lambda s: 1 <= s.bit_count() <= 4 or s.bit_count() >= 8)
    intervals = _brackets(ballot,4,np.random.default_rng(42))
    assert len(intervals) == 16
    refine_brackets(lambda mask:_query(ballot,mask),intervals)
    bounds, witnesses = sharp_upper_bounds(ballot.votes,p)
    assert len(witnesses) > 0
    for j, base in witnesses.items():
        mask = sum(1 << k for k in base)
        assert bounds[j] == -1 and j not in base and ballot.rule(mask) and not ballot.rule(mask | (1 << j))


def test_a_loss_between_two_winning_anchors_can_stay_hidden():
    p=12
    ballot = BooleanBallot(p,lambda s: s.bit_count() != 5)
    intervals = _brackets(ballot,4,np.random.default_rng(42))
    assert not intervals
    bounds, witnesses = sharp_upper_bounds(ballot.votes,p)
    assert np.all(bounds == 0) and not witnesses
    # In the complete game every column can take a winning size-4 set to a loss.
    assert all(any(ballot.rule(s) and not ballot.rule(s | (1 << i)) for s in range(1<<p) if not s & (1<<i))
               for i in range(p))


def test_no_refinement_control_is_identical_on_a_constant_target():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(30,9)), np.ones(30)
    args = x[:20],y[:20],x[20:],y[20:]
    adaptive = rank_chain(*args,task="regression")
    uniform = rank_chain(*args,task="regression",random_queries=0)
    assert adaptive.work["refinement_queries"] == uniform.work["refinement_queries"] == 0
    np.testing.assert_array_equal(adaptive.minimum,uniform.minimum)


def test_conditional_uniform_expectation_by_complete_small_pool_enumeration():
    for bits in product((False,True),repeat=8):
        votes = dict(enumerate(bits))
        anchors = {0,3,7}
        interior = sorted(votes.keys()-anchors)
        for q in range(len(interior)+1):
            expected = detection_probabilities(3,votes,anchors,q)
            draws = list(combinations(interior,q))
            sums = np.zeros(3,dtype=int)
            for draw in draws:
                sample = {m:votes[m] for m in anchors.union(draw)}
                bounds, _ = sharp_upper_bounds(sample,3)
                sums += bounds == -1
            assert expected == [Fraction(int(h),len(draws)) for h in sums]
