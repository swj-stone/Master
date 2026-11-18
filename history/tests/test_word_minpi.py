from math import comb, lcm

import numpy as np

from minppi_mofs.integer_minpi import rational_top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale, word_support


def test_native_common_scale_and_dispatch():
    limit = 2**63 - 1
    p, ks = 30, (16, 4, 0)
    scale = lcm(*(comb(p - 1, k) for k in ks))
    assert native_scale(p, ks, limit // scale) == scale
    assert native_scale(p, ks, limit // scale + 1) == 0
    rng = np.random.default_rng(3315)
    for p, ks, branch in ((8, (7, 1, 0), "int64"), (31, (16, 4, 0), "int64"),
                          (180, (16, 4, 0), "dyadic")):
        votes = rng.integers(-1, 2, size=(13, p), dtype=np.int8)
        votes[:, 1::2] = votes[:, :p-1:2]  # tied pairs on the budget boundary
        for h in (np.uint64(1), p//2, p-1):
            want = rational_top_from_votes(votes, int(h), tiers=ks, prune=False)[0]
            selected, info = exact_top_from_votes(votes, h, tiers=ks)
            np.testing.assert_array_equal(selected, want)
            assert info["arithmetic"] == branch
            if branch == "int64":
                d = native_scale(p, ks, len(votes))
                for prune in (False, True):
                    got = word_support(votes, int(h), ks, d, prune)[0]
                    np.testing.assert_array_equal(got, want)

