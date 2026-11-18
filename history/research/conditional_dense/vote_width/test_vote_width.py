"""Budget boundary and ranking-prefix checks for the concentration rule."""
import numpy as np
from numpy.testing import assert_array_equal

from research.conditional_dense.vote_width import method


def test_width_boundaries_and_scale():
    assert method.effective_width([0., 0., 0.], 2) == 2
    assert method.effective_width([7., 7., 7.], 3) == 3
    assert method.effective_width([7., 7., 7.], 2) == 2
    assert method.effective_width([0., 4., 0.], 3) == 1
    for scale in (1e-160, 1., 1e150):
        assert method.effective_width(np.array([4., 1., 0.]) * scale, 3) == 2


def test_single_call_and_stable_ties(monkeypatch):
    calls = []
    def frozen(X, y, *, seed):
        calls.append(seed)
        return np.arange(5), {"width": 5, "rank": np.array([0., 1., 1., 1., 7.])}
    monkeypatch.setattr(method, "conditional_support", frozen)
    selected, info = method.vote_support(None, None, seed=3)
    assert calls == [3] and info["width"] == 5
    assert_array_equal(selected, [1, 4])
