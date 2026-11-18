import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from minppi_mofs.piv3_lite_peers import _comparison, _merge, _neighbors, screen


def test_peer_votes_match_direct_same_coalition_mse():
    rng = np.random.default_rng(42)
    s = rng.normal(size=(10,27))
    s[4] = s[1]
    target = rng.normal(size=27)
    a,b = np.triu_indices(10,1)
    edges = np.column_stack((a,b))
    ab = np.einsum("ij,ij->i",s[a],s[b])
    for _ in range(8):
        order = rng.permutation(10)
        tiers=[1,3,8]
        votes = _comparison(s,target,edges,order,tiers,ab)
        for h,k in enumerate(tiers):
            for i,(j,other) in enumerate(edges):
                coalition = [c for c in order if c!=j and c!=other][:k]
                total = s[coalition].sum(axis=0)
                gap = np.sum((target-(total+s[other])/(k+1))**2-(target-(total+s[j])/(k+1))**2)
                assert votes[h,i] == np.sign(gap)


def test_representatives_are_retained_and_min_protects_a_weak_scale():
    edges = np.array([[0,1],[1,2],[0,3]])
    correlations = np.full((3,2),.94)
    rates = np.zeros((2,3,3))
    rates[:,:,2] = [-.5,1.,1.]
    names = list("abcd")
    keep,links = _merge(names,edges,correlations,rates,range(4),"min")
    assert keep == ["a","c","d"] and len(links) == 1
    means,_ = _merge(names,edges,correlations,rates,range(4),"mean")
    assert means == ["a","c"]
    assert all(link["representative"] in keep for link in links)


def test_neighbor_verification_rejects_independent_and_inconsistent_signs():
    rng = np.random.default_rng(21)
    samples = [rng.normal(size=(220,18)) for _ in range(3)]
    for batch in samples:
        batch[:,1] = -batch[:,0]
        batch[:,3] = batch[:,2]
        batch[:,4] = 1.
    samples[2][:,3] *= -1
    pairs,corr,_ = _neighbors(samples[0],samples[1:],rng)
    np.testing.assert_array_equal(pairs,[[0,1]])
    np.testing.assert_allclose(corr,[[-1,-1]])


def test_null_signal_can_merge_only_verified_signed_duplicates_in_full_entry():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1400,82))
    x[:,80] = x[:,1]
    x[:,81] = -x[:,2]
    y = rng.integers(0,2,len(x))
    cols = [f"f{i}" for i in range(x.shape[1])]
    with threadpool_limits(limits=1):
        selected,diag = screen(pd.DataFrame(x,columns=cols),y,task="binary")
    assert diag["shadow_bypass"] == "weak global signal" and diag["bypass"] is None
    assert len(selected) == 80 and diag["peer_removed_count"] == 2 and diag["front_fits"] == 0
    assert diag["peer_proposal_rows"]+sum(diag["peer_check_rows"])<=1024
    pairs={frozenset((w["removed"],w["representative"])) for w in diag["witnesses"]}
    assert pairs == {frozenset(("f1","f80")),frozenset(("f2","f81"))}
    assert all(w["representative"] in selected for w in diag["witnesses"])
    assert all(np.allclose(w["tier_rates"],0) for w in diag["witnesses"])
    json.dumps(diag,allow_nan=False)
