"""Regression test for the distinction between a minimum and a pooled vote."""
from unittest.mock import patch

import numpy as np

from experiments.refit_voting.race import Voter


def test_nonnegative_size_votes_are_never_vetoed_by_minimum():
    voter = Voter.__new__(Voter)
    voter.w = np.array([.5, .5])
    voter.y = np.array([0, 1])
    voter.null = 1.
    voter.task = "classification"
    # Both sizes have high within-size variability. Pooling cancels it.
    voter.cases = [(np.array([[1.], [-.8]]), np.zeros(2)),
                   (np.array([[-.8], [1.]]), np.zeros(2))]
    with patch("experiments.refit_voting.race.contribution", side_effect=lambda _,x,*args: x[:,0]):
        assert not voter.vote(None, [0], "minimum")
        assert voter.vote(None, [0], "pooled")

