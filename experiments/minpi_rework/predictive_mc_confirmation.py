"""Collect the prespecified seven-source time/decision comparison without test labels."""
import ctypes
import json
import os
import zlib
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import validation_predictive as source,quota_coalition_rf as rf
from experiments.minpi_rework import predictive_union_control as union,predictive_direct_mc as direct
from experiments.minpi_rework.predictive_mask_study import centered
from experiments.minpi_rework.predictive_mc_report import summary
from minppi_mofs import predictive_quota,availability_minpi


STORE=b.ROOT/"results/minpi_rework/predictive-mc-confirmation"
OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-mc-confirmation"
FILES=("forecasts.npz","increments.npz","incidence.npz","selection.json")
METHODS=("exact","recount","production",*[f"{flavor}{n}" for n in (8,32,128)
                                          for flavor in ("mc","sparsemc")])


def build_hashes():
    paths=[Path(__file__),Path(__file__).with_suffix(".md")]
    paths.extend(Path(m.__file__) for m in (source,rf,union,direct,predictive_quota,availability_minpi))
    paths.extend(Path(source.__file__).parent.glob("validation_predictive*.json"))
    return {p.name:b.digest(p) for p in paths}


def allowed(p,n,g,count):
    memory=(8*n*p+4*g)*count
    work=count*n*p*p*(p+1)//2
    return (memory<=512*1024**2 and work<=6*10**9),memory,work


def collect_one(name,seed,sign):
    start=perf_counter()
    x,y,task,idx=source.split(name,seed)
    data_s=perf_counter()-start
    assert (y[idx["test"]]==-1).all() if task=="classification" else np.isnan(y[idx["test"]]).all()
    start=perf_counter()
    forest=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    fit_s=perf_counter()-start
    start=perf_counter()
    game=forest.game(x[idx["vote"]])
    target=centered(forest,y[idx["vote"]],task)
    trace_s=perf_counter()-start
    p=game.p
    draw=zlib.crc32(f"{name}/{seed}".encode("ascii")) & 0xfffffff
    correct=union.Terms.build(game.a,game.incidence,target)
    oracle=union.MatchedSearch(correct)
    ref,ref_path=oracle.run()
    reference_curve=oracle.curve(ref)
    paths={}
    keep={}
    rows=[]
    choices=list(METHODS)
    direct_limits={}
    for count in (8,32,128):
        ok,memory,work=allowed(p,len(target),game.a.shape[1],count)
        direct_limits[count]=dict(covered=bool(seed==42 and ok),memory=memory,work=work)
        if seed==42 and ok:
            choices.append(f"direct{count}")
    for repeat in range(3):
        # Build once for the matched union arms and charge it in full to each.
        start=perf_counter()
        terms=union.Terms.build(game.a,game.incidence,target)
        union_s=perf_counter()-start
        for m in np.random.default_rng(draw+repeat).permutation(choices):
            m=str(m)
            count=int("".join(filter(str.isdigit,m)) or 0)
            setup=perf_counter()
            if m=="production":
                other=predictive_quota.PredictiveGame.from_paths(game.a,game.incidence,target)
                setup_s=perf_counter()-setup
                start=perf_counter()
                cols,path=other.search()
                search_s=perf_counter()-start
                base_s=0.
                sampled=None
            elif m.startswith("direct"):
                path,setup_s,search_s,_=direct.run(game.a,game.incidence,target,count,draw)
                cols=np.setdiff1d(np.arange(p),path).astype(int)
                base_s=0.
                sampled=None
            else:
                ranks=union.antithetic_ranks(p,count,np.random.default_rng(420901+draw)) if count else None
                sampled=union.MatchedSearch(terms,ranks,sparse_mc=m.startswith("sparsemc"))
                setup_s=perf_counter()-setup
                start=perf_counter()
                cols,path=sampled.run(recount=m=="recount")
                search_s=perf_counter()-start
                base_s=union_s
            if m in paths:
                assert paths[m]==path
            paths[m]=path
            keep[m]=cols.tolist()
            if not count:
                assert path==ref_path
            norm=oracle.curve(cols)
            if sampled is not None:
                est=sampled.curve(cols)
                fixed=sampled.curve(ref)
            elif count and repeat==0:
                ranks=union.antithetic_ranks(p,count,np.random.default_rng(420901+draw))
                val=union.MatchedSearch(terms,ranks,sparse_mc=True)
                est,fixed=val.curve(cols),val.curve(ref)
            elif count:
                previous=next(r for r in rows if r["method"]==m)
                est=fixed=None
            else:
                est,fixed=norm,reference_curve
            disagreement=len(np.union1d(ref,cols))
            rows.append(dict(dataset=name,seed=seed,draw=draw,repeat=repeat,method=m,p=p,k=len(cols),
                reference_k=len(ref),groups=game.a.shape[1],unions=len(terms.mass),data_s=data_s,
                fit_s=fit_s,path_cache_s=trace_s,union_dictionary_s=base_s,vote_setup_s=setup_s,
                search_s=search_s,selector_s=base_s+setup_s+search_s,
                pipeline_s=fit_s+trace_s+base_s+setup_s+search_s,cal_worst=float(norm.max()),
                exact_path=path==ref_path,exact_support=np.array_equal(cols,ref),
                jaccard=len(np.intersect1d(ref,cols))/disagreement if disagreement else 1.,
                true_violation=bool(norm.max()>.02+1e-9),
                max_curve_error=np.max(np.abs(est-norm)) if est is not None else previous["max_curve_error"],
                fixed_support_error=np.max(np.abs(fixed-reference_curve)) if fixed is not None else previous["fixed_support_error"]))
    for count in (8,32,128):
        assert paths[f"mc{count}"]==paths[f"sparsemc{count}"]
        if direct_limits[count]["covered"]:
            assert paths[f"mc{count}"]==paths[f"direct{count}"]
    ranks=union.antithetic_ranks(p,8,np.random.default_rng(420901+draw))
    picked=np.asarray(keep["mc8"],int)
    direct_curve=union.direct_mask_curve(game.a,game.incidence,target,picked,ranks)
    np.testing.assert_allclose(direct_curve,union.MatchedSearch(correct,ranks).curve(picked),atol=1e-9,rtol=3e-8)
    folder=STORE/f"{name}-{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    held=forest.game(x[idx["test"]])
    sparse.save_npz(folder/"increments.npz",held.a)
    sparse.save_npz(folder/"incidence.npz",held.incidence)
    preds={m:forest.predict(x[np.ix_(idx["test"],np.asarray(cols,int))],np.asarray(cols,int)) for m,cols in keep.items()}
    np.savez_compressed(folder/"forecasts.npz",**preds)
    rec=dict(code=sign,dataset=name,seed=seed,empty=forest.empty.tolist(),task=task,
        splits={k:v.tolist() for k,v in idx.items()},paths=paths,supports=keep,limits=direct_limits,
        hashes={n:b.digest(folder/n) for n in FILES[:-1]},rows=rows)
    temp=folder/"selection.tmp"
    temp.write_text(json.dumps(rec),encoding="utf8")
    temp.replace(folder/"selection.json")


def main():
    if os.name=="nt":
        if not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise OSError("CPU affinity not set")
    OUT.mkdir(parents=True,exist_ok=True)
    STORE.mkdir(parents=True,exist_ok=True)
    a=sparse.csc_matrix(np.eye(3))
    inc=sparse.csr_matrix(np.eye(3))
    y=np.arange(3,dtype=float)
    terms=union.Terms.build(a,inc,y)
    ranks=union.antithetic_ranks(3,8,np.random.default_rng(0))
    for sparse_mc in (True,False):
        union.MatchedSearch(terms,ranks,sparse_mc=sparse_mc).run()
    union.MatchedSearch(terms).run()
    direct.run(a,inc,y,8,42)
    predictive_quota.PredictiveGame.from_paths(a,inc,y).search()
    sign=build_hashes()
    jobs=[(name,seed) for name in source.SOURCES for seed in range(42,47)]
    rows=[]
    lock={}
    start=perf_counter()
    for i,(name,seed) in enumerate(jobs,1):
        tick=perf_counter()
        folder=STORE/f"{name}-{seed}"
        receipt=folder/"selection.json"
        if not receipt.exists():
            collect_one(name,seed,sign)
        rec=json.loads(receipt.read_bytes())
        assert rec["code"]==sign and all(b.digest(folder/n)==h for n,h in rec["hashes"].items())
        lock[folder.name]=b.digest(receipt)
        rows.extend(rec["rows"])
        elapsed=perf_counter()-start
        print(f"[{i}/{len(jobs)}] {name}/{seed} {perf_counter()-tick:.1f}s, elapsed {elapsed:.1f}s,"
              f" ETA {elapsed*(len(jobs)-i)/i:.1f}s",flush=True)
    file=OUT/"lock.json"
    file.write_text(json.dumps(dict(code=sign,receipts=lock),indent=2),encoding="utf8")
    archive=b.ROOT/"packages/minpi-predictive-mc-seven-forecasts.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        z.write(file,"lock.json")
        for key in lock:
            for part in FILES:
                z.write(STORE/key/part,f"{key}/{part}")
    (OUT/"archive-sha256.txt").write_text(b.digest(archive)+"  "+archive.name+"\n",encoding="utf8")
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/"repeats.csv",index=False)
    jobs=frame.groupby(["dataset","seed","draw","method"],as_index=False).median(numeric_only=True)
    jobs.to_csv(OUT/"jobs.csv",index=False)
    summary(jobs[~jobs.method.str.startswith("direct")]).to_csv(OUT/"contrasts.csv",index=False)
    print("Sealed",len(lock),"jobs.",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
