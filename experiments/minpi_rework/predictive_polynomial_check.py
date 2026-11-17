"""Compare discrete-interval maxima and tolerance decisions to scalar exhaustive evaluation."""
import numpy as np
from numba import njit
from minppi_mofs.availability_minpi import probabilities
from minppi_mofs.size_extrema import difference_tree, maximum, select, _dot
from experiments.minpi_rework.predictive_veto_native import straight


@njit(cache=True)
def full(c, w):
    return np.array([_dot(c, w, s) for s in range(len(w))])


def check():
    rng = np.random.default_rng(302)
    stack = np.empty((64, 3), np.int64)
    ncases = interior = plateaus = 0
    for n in (1, 2, 3, 7, 8, 16, 43, 79, 128, 257, 520, 1024, 1558):
        degree = min(n, 8)
        w = np.ascontiguousarray(probabilities(n, degree)[1:]*n/np.arange(1,n+1)[:, None])
        low, high, absmax = difference_tree(w)
        for i in range(150):
            c = rng.normal(size=degree+1)
            c[0] = 0.
            if i % 9 == 0:
                c[2:] = 0.
            if i % 9 == 1:
                c[1:] = (-1.)**np.arange(degree)*rng.lognormal(size=degree)
            if i % 9 == 2 and degree >= 3:
                c[:] = 0.
                c[2], c[3] = 1., -1.14
            if i % 9 == 3:
                c *= 1e-307
            if i % 9 == 4:
                c *= 1e130
            if i % 9 == 5:
                c[2:] *= 1e-15
            if i % 9 == 6:
                c[:] = 0.
            if i % 9 == 7:
                c[0] = rng.normal()
            values = full(c, w)
            expected = values.max()
            last = int(rng.integers(n))
            peak, arg, _, _ = maximum(c,w,low,high,absmax,last,np.inf,stack)
            assert peak == expected == values[arg], (n, i, peak, expected)
            interior += int(max(values[0], values[-1]) < expected)
            plateaus += int(np.sum(values == expected) > 1)
            # The decision must also handle features just inside/outside the tie tolerance.
            tol = np.finfo(float).eps*max(float(np.abs(c).sum()), 1e-300)*128
            delta = np.tile(c,(6,1))
            delta[:,1] += tol * np.array([2., 1., 0., .5, -1., -2.])
            keep = rng.uniform(size=6) < .8
            keep[0] = True
            zero = np.zeros_like(c)
            got = select(zero, delta, keep, w, tol, last, low, high, absmax)[0]
            want = straight(zero, delta, keep, w, tol, last)[0]
            assert got == want, (n, i, got, want)
            ncases += 1
    print(f"{ncases} scalar extrema and tolerance decisions; {interior} strict interior and {plateaus} tied maxima")


if __name__ == "__main__":
    check()
