"""Sparse leave-one-document-out minimum payoff gain over two peer sizes."""
import numpy as np
from scipy.special import expit
from research.text_peer.vote import presences


def _softsign(x):
    return x / (1. + np.abs(x))


def scores(X, y, *, utility="softsign", scale="sqrt", rule="both"):
    X = presences(X)
    y = np.asarray(y)
    if y.shape != (X.shape[0],) or X.shape[1] < 1:
        raise ValueError("Expected aligned document features and labels")
    classes, labels, sizes = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) != 2 or min(sizes) < 2 or utility not in ("log","softsign") or scale not in ("sqrt","unit"):
        raise ValueError("Two nontrivial classes and known utility and scaling are required")
    positive = X[labels == 1]
    count = np.bincount(X.indices, minlength=X.shape[1])
    pos = np.bincount(positive.indices, minlength=X.shape[1])
    counts = (count - pos, pos)
    gain = np.zeros((2,X.shape[1]))
    for c, block in ((1,positive),(0,X[labels==0])):
        n = sizes[c]
        prior = (n-1) / (len(y)-1)
        weights = np.log1p(np.maximum(0,counts[c]-1)/prior) - np.log1p(counts[1-c]/(1-prior))
        lengths = np.diff(block.indptr)
        temperature = np.sqrt(np.maximum(lengths,1)) if scale=="sqrt" else np.ones(n)
        component = weights[block.indices] / np.repeat(temperature, lengths)
        if rule in ("both","zero"):
            single = (_softsign(component) if utility=="softsign" else
                      np.log(2.) - np.logaddexp(0., -component))
            gain[0] += np.bincount(block.indices,weights=single,minlength=block.shape[1]) / (2*n)
        if rule in ("both","all"):
            total = np.repeat((block @ weights)/temperature, lengths)
            joint = (_softsign(total)-_softsign(total-component) if utility=="softsign" else
                     np.log1p(expit(-total)*np.expm1(component)))
            gain[1] += np.bincount(block.indices,weights=joint,minlength=block.shape[1]) / (2*n)
    return gain


def select(X, y, *, count=None, rule="both", utility="softsign", scale="sqrt"):
    gain=scores(X,y,utility=utility,scale=scale,rule=rule)
    if rule not in ("zero","all","both"):
        raise ValueError("Unknown peer sizes")
    vote = gain.min(axis=0) if rule=="both" else gain[0 if rule=="zero" else 1]
    if count is None:
        count=max(1,np.count_nonzero(vote>1e-15))
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or not 1<=count<=len(vote):
        raise ValueError("Invalid width")
    order=np.lexsort((np.arange(len(vote)), -np.round(gain[0] if rule != "all" else gain[1],14), -np.round(vote,14)))
    return np.sort(order[:count])
