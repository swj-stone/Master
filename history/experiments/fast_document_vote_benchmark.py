"""Direct same-support comparison of native binary voting on frozen text and mail panels."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import document_vote_benchmark as old
from experiments import mail_vote_transfer as mail
from experiments.audit_document_peers import pooled_gain
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training, library_path


ARMS = ("fast_vote", "pooled_match", "pooled_50", *old.ARMS)
ARCHIVE = old.ROOT/"results/minpi-study"
REFERENCES = {"uci": ARCHIVE/"document-vote-20261117b", "mail": ARCHIVE/"mail-vote-20261117a"}
ABLATION = ARCHIVE/"document-peer-audit-20261117a"
CASE = old.CASE


def panels():
    for case in old.panels():
        yield "uci", case
    X, y, Xt, truth, record = mail.prepare()
    for seed in mail.SEEDS:
        yield "mail", (X, y, Xt, truth, dict(record, seed=seed))


def pooled_binary_gain(X, y):
    """All-document marginal unit-ridge gain for binary term presence and binary labels."""
    classes, labels, counts = np.unique(y,return_inverse=True,return_counts=True)
    if len(classes) != 2:
        raise ValueError("Binary class labels required")
    term = np.bincount(X.indices,minlength=X.shape[1]).astype(float)
    minority = int(np.argmin(counts))
    observed = np.bincount(X[labels==minority].indices,minlength=X.shape[1])
    rhs = np.empty((X.shape[1],2))
    rhs[:,minority] = observed
    rhs[:,1-minority] = term-observed
    rhs -= term[:,None]*(counts/len(y))
    return np.einsum("ij,ij->i",rhs,rhs)/(term*(1-term/len(y))+1.)


def select(method, X, y, seed, width):
    if method == "fast_vote":
        return fast_document_vote_top_from_training(X,y,random_state=seed,backend="native")[0]
    if method in ("pooled_match","pooled_50"):
        count = width if method=="pooled_match" else (X.shape[1]+1)//2
        return np.sort(np.argsort(-pooled_binary_gain(X,y),kind="stable")[:count])
    return old.select(method,X,y,seed,width)


def run_case(suite, X, y, Xt, truth, meta, out, repeat_id, verify):
    key = {k:meta[k] for k in CASE}
    seed = key["seed"]
    filename = f"{key['dataset']}-{key['fold']}-{seed}.npz"
    reference = REFERENCES[suite]/filename
    cols_native, new = fast_document_vote_top_from_training(X, y, random_state=seed, backend="native")
    cols_ref, prior = old.document_vote_top_from_training(X,y,random_state=seed)
    np.testing.assert_array_equal(cols_native,cols_ref,err_msg=filename)
    exact = dict(**key, suite=suite, width=len(cols_native), p=X.shape[1])
    independent = pooled_gain(X,y)
    np.testing.assert_array_equal(pooled_binary_gain(X,y),independent,err_msg=filename+" pooled gain")
    pooled_order = np.lexsort((np.arange(X.shape[1]), -independent))
    for name in ("scores","zero_peer_scores","one_peer_scores"):
        a, b = np.asarray(new[name]), np.asarray(prior[name])
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12,err_msg=f"{filename} {name}")
        exact[name+"_bitwise"] = a.tobytes() == b.tobytes()
        exact[name+"_max_abs"] = float(np.max(np.abs(a-b)))
    arrays, scores, rows, audit = {}, {}, [], []
    with np.load(reference) as previous:
        for arm in ARMS:
            cols = select(arm,X,y,seed,len(cols_native))
            check_columns(cols,X.shape[1])
            pred = old.fit_predict(X,y,Xt,cols,"classification",seed,False)
            ref_arm = "document_vote" if arm == "fast_vote" else arm
            if arm.startswith("pooled_"):
                expected_cols = np.sort(pooled_order[:len(cols)])
                np.testing.assert_array_equal(cols,expected_cols)
                if arm=="pooled_match" and suite=="uci":
                    with np.load(ABLATION/filename) as baseline:
                        np.testing.assert_array_equal(cols,baseline["pooled_cols"])
                        np.testing.assert_array_equal(pred,baseline["pooled_pred"])
            for attr,value in (("cols",cols),("pred",pred)):
                if not arm.startswith("pooled_"):
                    np.testing.assert_array_equal(value,previous[ref_arm+"_"+attr],err_msg=filename+" "+arm)
                arrays[arm+"_"+attr]=value
            scores[arm]=old.balanced_accuracy_score(truth,pred)
            audit.append(dict(**key, method=arm, width=len(cols), score=scores[arm],
                              support_sha256=old.digest(cols), prediction_sha256=old.digest(pred)))
    if verify:
        with np.load(out/filename) as stored:
            assert set(arrays) == set(stored)
            for k,v in arrays.items():
                np.testing.assert_array_equal(v,stored[k],err_msg=filename+" "+k)
        return rows, audit, exact
    order = np.random.default_rng([643319,repeat_id,key["fold"],seed]).permutation(ARMS)
    for r in range(len(ARMS)):
        for pos, arm in enumerate(np.roll(order,r)):
            t0=perf_counter_ns()
            cols=select(arm,X,y,seed,len(cols_native))
            t1=perf_counter_ns()
            pred=old.fit_predict(X,y,Xt,cols,"classification",seed,False)
            t2=perf_counter_ns()
            np.testing.assert_array_equal(cols,arrays[arm+"_cols"])
            np.testing.assert_array_equal(pred,arrays[arm+"_pred"])
            rows.append(dict(**key,method=arm,repeat=r,position=pos,width=len(cols),p=X.shape[1],score=scores[arm],
                             select_ms=(t1-t0)/1e6,refit_ms=(t2-t1)/1e6,total_ms=(t2-t0)/1e6))
    np.savez_compressed(out/filename,**arrays)
    return rows, audit, exact


def summarise(out, rows, meta, exact):
    cases=summarize_timings(out,rows)
    meta.to_csv(out/"panels.csv",index=False)
    exact.to_csv(out/"numerical-equivalence.csv",index=False)
    vote=cases[cases.method=="fast_vote"].set_index(CASE).join(
        meta.set_index(CASE)[["suite","task","common_prepare_ms"]])
    vote["ratio_with_vocabulary"]=(vote.total_ms+vote.common_prepare_ms)/(vote.full_total_ms+vote.common_prepare_ms)
    paired=[]
    for method, group in cases.groupby("method"):
        controls=group.set_index(CASE)
        sub=vote[["suite","task","joint","select_ms"]].copy()
        sub["baseline"]=method
        sub["delta"]=vote.score-controls.score
        sub["width_ratio"]=vote.width/controls.width
        sub["time_ratio"]=vote.total_ms/controls.total_ms
        sub["select_ratio"]=vote.select_ms/controls.select_ms
        acceptable=vote.joint&(sub.delta>=-.02)&(sub.time_ratio<1)
        sub["relative_joint"]=acceptable&(sub.width_ratio<=1)
        sub["relative_joint_10pct"]=acceptable&(sub.width_ratio<=1.1)
        paired.append(sub)
    paired=pd.concat(paired).reset_index()
    paired.to_csv(out/"paired-baselines.csv",index=False)
    for field in ("relative_joint","relative_joint_10pct"):
        f=paired.pivot(index=CASE,columns="baseline",values=field)
        vote["F_random_chi50_"+field]=f.F_50&f.chi_50&f.random_50
        vote["F_random_chi_pooled50_"+field]=f.F_50&f.chi_50&f.random_50&f.pooled_50
    draws=pd.concat([pd.read_csv(folder/"random-draws.csv") for folder in REFERENCES.values()])
    vote=vote.join(draws.groupby(CASE).agg(wins=("win","sum"), width_ref=("width","first"), random_median=("score","median")))
    assert len(draws)==400 and vote.width.eq(vote.width_ref).all()
    vote["strong_joint"]=vote.joint&(vote.wins>=15)
    vote.to_csv(out/"vote-comparisons.csv")


def run(out, repeat_id, verify):
    files = [Path(__file__),Path(old.__file__),Path(mail.__file__),old.ROOT/"docs/minpi-study/fast-document-timing.md",
             old.ROOT/"experiments/audit_document_peers.py",ABLATION/"protocol.json",ABLATION/"cases.csv",
             old.ROOT/"src/minppi_mofs/fast_document_vote_minpi.py",old.ROOT/"src/minppi_mofs/document_vote_minpi.py",
             old.ROOT/"src/minppi_mofs/document_vote_kernel.c",old.ROOT/"experiments/build_native_document.py",
             old.ROOT/"src/minppi_mofs/signal_vote_minpi.py",old.ROOT/"experiments/minpi_constant_transfer.py",
             old.ROOT/"experiments/minpi_study.py",old.ROOT/"configs/minpi-study.json",
             old.ROOT/"experiments/compare_predictive_minpi.py",old.ROOT/"experiments/fast_minpi_five_source.py"]
    reference_sha = {source:{f.name:old.file_hash(f) for f in
                            [*folder.glob("*.npz"),folder/"protocol.json",folder/"random-draws.csv"]}
                     for source,folder in REFERENCES.items()}
    state = dict(files={p.relative_to(old.ROOT).as_posix():old.file_hash(p) for p in files}, methods=list(ARMS),
                 native_library_sha256=old.file_hash(library_path()), reference_sha256=reference_sha,
                 independent_pass=repeat_id, rounds=len(ARMS), old_sources=old.ASSETS, mail_sources=mail.ARCHIVES)
    if verify:
        assert state==json.loads((out/"protocol.json").read_text())
    else:
        out.mkdir(parents=True,exist_ok=False)
        old.write_json(out/"protocol.json",state)
    trials,audits,info,exact=[],[],[],[]
    for suite, (X,y,Xt,truth,meta) in panels():
        timed, replay, diagnostic=run_case(suite, X,y,Xt,truth,meta,out,repeat_id,verify)
        trials.extend(timed)
        audits.extend(replay)
        info.append(dict(meta,suite=suite))
        exact.append(diagnostic)
        print("Verified" if verify else "Measured",meta["dataset"],meta["fold"],meta["seed"],flush=True)
    assert len(info)==20 and len(audits)==len(info)*len(ARMS)
    exact=pd.DataFrame(exact)
    if verify:
        pd.testing.assert_frame_equal(exact,pd.read_csv(out/"numerical-equivalence.csv"),atol=1e-25,rtol=1e-12)
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv",index=False)
    else:
        assert len(trials)==20*len(ARMS)**2
        summarise(out,trials,pd.DataFrame(info),exact)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pass-id",type=int,choices=(1,2),required=True)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.output,args.pass_id,args.verify)
