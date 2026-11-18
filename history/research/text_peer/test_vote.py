from statistics import NormalDist
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from research.text_peer.sparse_game import scores, select
from research.text_peer.bns import bns_scores


@pytest.mark.parametrize("size", [1,5,17])
def test_exact_literal_leave_one_out_and_both_coalition_sizes(size):
    rng=np.random.default_rng(331+size)
    matrix=rng.integers(2,size=(17,size)); y=np.arange(17)%2
    expected=np.zeros((2,size))
    for i,row in enumerate(matrix):
        train=np.arange(len(y))!=i
        label=y[i]; prior=np.mean(y[train]==label)
        counts=matrix[train & (y==label)].sum(axis=0)
        other=matrix[train & (y!=label)].sum(axis=0)
        term=np.log((1+counts/prior)/(1+other/(1-prior))) / np.sqrt(max(1,row.sum()))
        def payoff(columns):
            return -np.logaddexp(0,-np.dot(row[columns],term[columns]))
        for j in range(size):
            all_cols=np.arange(size)
            for g, peers in ((0,[]),(1,all_cols[all_cols!=j].tolist())):
                delta=payoff(peers+[j])-payoff(np.asarray(peers,dtype=int))
                expected[g,j]+=delta/(2*np.count_nonzero(y==label))
    np.testing.assert_allclose(scores(csr_matrix(matrix),y,utility="log"),expected,atol=5e-16)
    cols=select(csr_matrix(matrix),y,utility="log");assert len(cols)==max(1,np.count_nonzero(expected.min(axis=0)>1e-15))


def test_bns_endpoints_and_configured_positive_label_invariance():
    X=csr_matrix([[1,1,0,0],[1,0,0,0],[1,0,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0]])
    y=np.repeat([0,1],3)
    z=NormalDist().inv_cdf
    expected=np.abs([z(.9995)-z(.0005),z(1/3)-z(.9995),z(1/3)-z(2/3),0])
    np.testing.assert_allclose(bns_scores(X,y),expected)
    np.testing.assert_array_equal(bns_scores(X,1-y),bns_scores(X,y))
