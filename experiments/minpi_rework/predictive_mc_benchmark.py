"""Compare all-size analytic votes with matched nested-permutation votes on opened sources."""
import argparse
import ctypes
import os
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from threadpoolctl import threadpool_limits

from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework.availability_trial import DEST, SOURCES
from experiments.minpi_rework.predictive_mask_study import centered,curve
from experiments.minpi_rework.predictive_union_control import (
    MatchedSearch, Terms, antithetic_ranks, direct_mask_curve)
from minppi_mofs.predictive_quota import PredictiveGame


def fixture():
    rng = np.random.default_rng(7821)
    for p in (2,4,7):
        groups = rng.integers(0,2,(25,p)).astype(bool)
        groups = np.unique(groups[groups.any(axis=1)],axis=0)
        inc = csr_matrix(groups.astype(float))
        a = csc_matrix(rng.integers(-4,5,(5,len(groups))).astype(float))
        target = rng.integers(-8,9,5).astype(float)
        native = PredictiveGame.from_paths(a,inc,target)
        terms = Terms.build(a,inc,target)
        exact = MatchedSearch(terms)
        assert exact.run()[1] == exact.run(recount=True)[1] == native.search()[1]
        ranks = antithetic_ranks(p,128,rng)
        for b in (8,32,128):
            empirical = MatchedSearch(terms,ranks[:b])
            for mask in range(1<<p):
                keep = np.flatnonzero(mask >> np.arange(p) & 1)
                np.testing.assert_allclose(exact.curve(keep),curve(native,keep),atol=2e-12)
                if mask%max(1,2**p//5)==0:
                    direct = direct_mask_curve(a,inc,target,keep,ranks[:b])
                    np.testing.assert_allclose(empirical.curve(keep),direct,atol=2e-12)
            assert empirical.run()[1] == empirical.run(recount=True)[1]
    print("Direct prediction and both incremental/recount controls passed.",flush=True)


def one(source,name,seed,draws):
    t = perf_counter()
    x,y,task,idx = source.split(name,seed)
    data_s = perf_counter()-t
    pool = np.sort(np.r_[idx["fit"],idx["vote"]])
    t = perf_counter()
    forest = rf.fit(x[pool],y[pool],task,seed)
    fit_s = perf_counter()-t
    t = perf_counter()
    game = forest.game(x[idx["width"]])
    target = centered(forest,y[idx["width"]],task)
    paths_s = perf_counter()-t
    t = perf_counter()
    terms = Terms.build(game.a,game.incidence,target)
    dict_s = perf_counter()-t
    t = perf_counter()
    analytic = MatchedSearch(terms)
    analytic_setup = perf_counter()-t
    # The prediction-level reference is independent of risk selection and timing.
    t = perf_counter()
    production = PredictiveGame.from_paths(game.a,game.incidence,target)
    production_setup = perf_counter()-t
    t = perf_counter()
    ref,ref_path = production.search()
    production_search = perf_counter()-t
    actual,oracle_path = analytic.run()
    assert np.array_equal(actual,ref) and oracle_path==ref_path
    src_truth = source._partition(name,seed)[1][idx["test"]]
    outer = forest.game(x[idx["test"]])
    outer_game = PredictiveGame.from_paths(outer.a,outer.incidence,centered(forest,src_truth,task))
    ref_outer = curve(outer_game,ref)
    ref_curve = analytic.curve(ref)
    rows = []
    for draw in range(draws):
        rng = np.random.default_rng(16000+draw)
        # Reconstruct every prefix inside its own timed block to charge its actual cost.
        for method in rng.permutation(["exact","recount","mc8","mc32","mc128"]):
            t = perf_counter()
            b = int(method[2:]) if method.startswith("mc") else 0
            ranks = antithetic_ranks(terms.p,b,np.random.default_rng(420901+draw)) if b else None
            selector = MatchedSearch(terms,ranks)
            setup = perf_counter()-t
            t = perf_counter()
            cols,path = selector.run(recount=method=="recount")
            search = perf_counter()-t
            true_curve = analytic.curve(cols)
            sample_curve = selector.curve(cols)
            outer_curve = curve(outer_game,cols)
            union = len(np.union1d(cols,ref))
            agreement = len(np.intersect1d(cols,ref))/union if union else 1.
            shared = next((i for i,(a,b_) in enumerate(zip(path,ref_path)) if a!=b_),min(len(path),len(ref_path)))
            if not b:
                assert path==ref_path
            direct_s = np.nan
            if method=="mc8" and draw==0:
                t = perf_counter()
                direct = direct_mask_curve(game.a,game.incidence,target,cols,ranks)
                direct_s = perf_counter()-t
                np.testing.assert_allclose(sample_curve,direct,rtol=3e-8,atol=1e-9)
            common = fit_s+paths_s
            rows.append(dict(dataset=name,seed=seed,draw=draw,method=method,p=terms.p,k=len(cols),
                permutations=b,reference_k=len(ref),unions=len(terms.mass),groups=game.a.shape[1],
                data_s=data_s,fit_s=fit_s,path_cache_s=paths_s,union_dictionary_s=dict_s,
                vote_setup_s=setup,search_s=search,selector_s=dict_s+setup+search,
                pipeline_s=common+dict_s+setup+search,legacy_setup_s=production_setup,
                legacy_search_s=production_search,cal_worst=true_curve.max(),estimated_worst=sample_curve.max(),
                max_curve_error=np.max(np.abs(sample_curve-true_curve)),
                fixed_support_error=np.max(np.abs(selector.curve(ref)-ref_curve)),true_violation=true_curve.max()>.02+1e-9,
                independent_direct_s=direct_s,exact_support=agreement==1.,exact_path=path==ref_path,
                jaccard=agreement,prefix_steps=shared,reference_steps=len(ref_path),
                outer_worst=outer_curve.max(),outer_delta_worst=outer_curve.max()-ref_outer.max()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names",nargs="+")
    parser.add_argument("--draws",type=int,default=3)
    parser.add_argument("--tag",default="matched-mc-seed42")
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    fixture()
    if args.check:
        return
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    jobs = [(src,name) for src in SOURCES for name in src.SOURCES
            if args.names is None or name in args.names]
    rows=[]
    start=perf_counter()
    for i,(src,name) in enumerate(jobs,1):
        tick=perf_counter()
        rows.extend(one(src,name,42,args.draws))
        pd.DataFrame(rows).to_csv(DEST/f"{args.tag}.csv",index=False)
        elapsed=perf_counter()-start
        print(f"[{i}/{len(jobs)}] {name}: {perf_counter()-tick:.2f}s, elapsed {elapsed:.1f}s,"
              f" ETA {elapsed*(len(jobs)-i)/i:.1f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
