"""All peer-size minimum for additive, presence-and-absence LOO evidence."""
import numpy as np
from research.text_peer.vote import presences


def _signed_evidence(own, other, prior):
    # Smoothed balanced posterior transformed from [0,1] to [-1,1].
    same = np.maximum(0., own-1) / prior
    different = other / (1-prior)
    return (same - different) / (same + different + 2.)


def square_scores(X, y, *, absence=True):
    X=presences(X)
    y=np.asarray(y)
    if y.shape!=(X.shape[0],) or not X.shape[1]: raise ValueError("Invalid document labels")
    classes, labels, sizes=np.unique(y,return_inverse=True,return_counts=True)
    if len(classes)!=2 or sizes.min()<2: raise ValueError("Two classes with two rows each required")
    blocks=[X[labels==c] for c in (0,1)]
    counts=[np.bincount(block.indices,minlength=X.shape[1]) for block in blocks]
    first=np.zeros(X.shape[1]); second=np.zeros_like(first); cross=np.zeros_like(first)
    sum_z=0.;square_z=0.
    for c,block in enumerate(blocks):
        n=sizes[c];other=1-c;prior=(n-1)/(len(y)-1)
        present=_signed_evidence(counts[c],counts[other],prior)
        absent=(_signed_evidence(n-counts[c],sizes[other]-counts[other],prior)
                if absence else np.zeros_like(present))
        diff=present-absent
        prediction=absent.sum() + block @ diff
        factor=1/(2*n)
        first += factor * (n*absent+counts[c]*diff)
        second += factor * ((n-counts[c])*absent**2+counts[c]*present**2)
        cross += factor * (absent*prediction.sum()+diff*(block.T @ prediction))
        sum_z+=factor*prediction.sum();square_z+=factor*np.dot(prediction,prediction)
    slope=max(0.,sum_z/square_z) if square_z>0 else 0.
    zero=2*slope*first-slope**2*second
    full=2*slope*first-2*slope**2*cross+slope**2*second
    return np.stack([zero,full]),slope


def choose(X,y, *, count=None,rule="both",absence=True):
    gains,_=square_scores(X,y,absence=absence)
    if rule not in ("both","zero","all"):raise ValueError("Unknown rule")
    score=gains.min(axis=0) if rule=="both" else gains[0 if rule=="zero" else 1]
    if count is None:
        positive=np.count_nonzero(score>1e-15)
        count=positive or X.shape[1]
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or not 1<=count<=X.shape[1]:
        raise ValueError("Invalid width")
    return np.sort(np.lexsort((np.arange(len(score)),-np.round(gains[0],14),-np.round(score,14)))[:count])
