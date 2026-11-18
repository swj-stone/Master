"""Interior calibration for a non-monotone empirical native Gain path."""
from math import ceil, sqrt

import numpy as np


def search(profile, measure, tolerance=.005):
    p=len(profile)
    probe=ceil(sqrt(p))
    trace={p:measure(p)}
    if probe not in trace:
        trace[probe]=measure(probe)
    floor=trace[p]-tolerance
    for _ in range(3):
        hi=min(k for k,u in trace.items() if u>=floor)
        lo=max((k for k in trace if k<hi),default=0)
        if hi-lo<2:
            break
        candidates=np.arange(lo+1,hi)
        if lo:
            fraction=(floor-trace[lo])/(trace[hi]-trace[lo])
            target=profile[lo-1]+fraction*(profile[hi-1]-profile[lo-1])
        else:
            # With no rejected observation, bisect the unexplored vote mass.
            target=(1.+profile[hi-1])/2
        k=int(candidates[np.argmin(abs(profile[candidates-1]-target))])
        trace[k]=measure(k)
    return min(k for k,u in trace.items() if u>=floor),trace
