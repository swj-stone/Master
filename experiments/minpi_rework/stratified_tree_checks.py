"""Exhaustively verify size-conditioned votes and compare complete greedy paths."""
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from experiments.minpi_rework.quota_compress_tree import Game
from minppi_mofs.stratified_tree import greedy_path, marginal_anchors


def check():
    rng = np.random.default_rng(531)
    vectors = games = 0
    for p in range(1,9):
        groups = [cols for k in range(1,min(3,p)+1) for cols in combinations(range(p),k)]
        inc = csr_matrix([[int(j in cols) for j in range(p)] for cols in groups])
        for trial in range(8):
            a = csc_matrix(rng.integers(-2,3,size=(5,len(groups))).astype(float))
            if trial==1:
                a = a * 0.
            dense = (a @ inc).toarray().T
            vals = marginal_anchors(a,inc,range(p),dense)
            def value(mask):
                return np.asarray(a[:,[all(mask>>j&1 for j in cols) for cols in groups]].sum(axis=1)).ravel()
            utilities = [value(mask) for mask in range(1<<p)]
            for j in range(p):
                for s in range(p):
                    subset = [m for m in range(1<<p) if m.bit_count()==s and not (m>>j&1)]
                    direct = np.mean([utilities[m|(1<<j)]-utilities[m] for m in subset],axis=0)
                    np.testing.assert_allclose(vals[s,j],direct,atol=2e-14)
                    vectors += 1
            g = Game(a,inc)
            order = rng.permutation(p)
            for rho in (0.,.2,.9,1.0):
                ref, diag = g.search("exact_greedy",order,rho)
                for mode in ("cache","last","all"):
                    cols, out = greedy_path(a,inc,order,rho,mode=mode)
                    np.testing.assert_array_equal(cols,ref)
                    assert out["deletion_order"]==diag["deletion_order"]
                games += 1
    # Two cancelling interactions: only a small-coalition voter detects the last veto.
    a = csc_matrix([[1.,1.,-1.]])
    inc = csr_matrix([[1,0],[0,1],[1,1]])
    _,full = greedy_path(a,inc,[0,1],mode="last")
    _,multi = greedy_path(a,inc,[0,1])
    assert full["deletion_order"]==multi["deletion_order"]==[1]
    assert full["refreshed"]==1 and multi["refreshed"]==0 and multi["extra_stratum_vetoes"]==1
    inc = csr_matrix([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])
    for exponent in (-900,-270,0,270,900):
        a = csc_matrix(np.ldexp([[1.,-1.,15.,1.]],exponent))
        for mode in ("all","last","cache"):
            _, d = greedy_path(a,inc,[0,1,2],mode=mode)
            assert d["deletion_order"]==[1,0]
    for dtype in (np.int64,np.float32,np.float64):
        _,d=greedy_path(csc_matrix([[4000000000]],dtype=dtype),csr_matrix([[1]]),[0],rho=1.)
        assert d["deletion_order"]==[0]
    # After removing j=1, its +A and -A groups vanish together. The sole
    # remaining singleton has effect one, even when float64 rounds A + 1 to A.
    inc = csr_matrix([[1,1],[0,1],[1,0]])
    for exponent in (-500,0,500):
        a = csc_matrix(np.ldexp([[2.**60,-2.**60,1.]],exponent))
        for mode in ("cache","last","all"):
            keep,d = greedy_path(a,inc,[0,1],rho=0.,mode=mode)
            assert keep.tolist()==[0] and d["deletion_order"]==[1]
    print(f"{vectors} brute-force cardinality marginals; {games} games × three paths; strict veto witness checked")


if __name__=="__main__":
    check()
