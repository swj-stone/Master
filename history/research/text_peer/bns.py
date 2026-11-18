"""Forman (JMLR 2003), Table 1 Bi-Normal Separation on binary term-presence."""
import numpy as np
from scipy.special import ndtri
from research.text_peer.vote import presences


def bns_scores(X, y):
    X=presences(X)
    y=np.asarray(y)
    if y.shape!=(X.shape[0],): raise ValueError("Labels must align with input rows")
    classes,labels,sizes=np.unique(y,return_inverse=True,return_counts=True)
    if len(classes)!=2: raise ValueError("BNS requires two classes")
    rate=np.stack([np.bincount(X[labels==k].indices,minlength=X.shape[1])/sizes[k] for k in (0,1)])
    # The paper specifies .0005 for zero; complement symmetry supplies .9995
    # for a rate of exactly one. Interior probabilities remain unchanged.
    rate=np.where(rate==0,.0005,np.where(rate==1.,.9995,rate))
    return np.abs(ndtri(rate[0])-ndtri(rate[1]))


def bns(X,y,width):
    if not 1 <= width <= X.shape[1]: raise ValueError("Invalid width")
    return np.sort(np.argsort(-bns_scores(X,y),kind="stable")[:width])
