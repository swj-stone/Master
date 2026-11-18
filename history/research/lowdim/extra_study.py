"""Cross-check spectroscopy, acoustic, four-stage Ridge and mechanism experiments."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments import allocation_nir_vote_benchmark as nir
from experiments.grouped_solver_cost_audit import fit as acoustic_fit
from experiments.minpi_backward_confirmation import evaluate
from experiments.stable_minpi_stress import prediction, digest, metric
from research.conditional_dense.confirmation import Panel
from research.conditional_dense.fsdd import features as speech
from research.conditional_dense.vote_width import study as dense
from research.feature_game_eval import confirmation as simulation, panels as bench
from research.four_stage import benchmark as four
from research.lowdim.panel_study import METHODS, f_support, fast_support, regression_support, vote_support

HERE = Path(__file__).resolve().parent
ARCH = bench.ARCHIVE
SIZES = {"dense":19, "four_stage_ridge":30, "spectra":36, "mechanism":64}
KEY = ["cohort","dataset","fold","seed"]


def cases(cohort):
    if cohort == "dense":
        for d in dense.panels():
            yield d, ARCH/"vote-width-timing1-20261118a"/f"{d.name}-{d.fold}-{d.seed}.npz", "full_pred"
        X,y,sp,_,_ = speech.read_corpus(bench.ROOT/"data/raw/fsdd-v1.0.10.zip")
        for d,_,_ in speech.panels(X,y,sp):
            yield d, ARCH/"fsdd-confirm-timing1-20261118b"/f"fold-{d.fold}.npz", "full_pred"
    elif cohort == "four_stage_ridge":
        for name in four.DATASETS:
            X,y,task,_ = four.load(name)
            raw_hash = [digest(X),digest(y)]
            for fold,seed in enumerate(four.SEEDS,1):
                archive = ARCH/"four-stage-crossed-20261118b"/f"{name}-{seed}.npz"
                with np.load(archive) as z:
                    assert raw_hash == [str(z[k]) for k in ("input_X_sha256","input_y_sha256")]
                    tr,te=z["train"],z["test"]
                yield Panel(name,fold,seed,X[tr],y[tr],X[te],y[te],None,task), archive, "ridge_full_pred"
    elif cohort == "spectra":
        for X,y,Xt,gold,info in nir.sources():
            for seed in nir.nir.SEEDS:
                name = info["dataset"]
                yield Panel(name,1,seed,X,y,Xt,gold,None,"regression"), ARCH/"allocation-nir-20261117a"/f"{name}-{seed}.npz", "full_pred"
    elif cohort == "mechanism":
        for family in ("proxy","diffuse"):
            for task in ("classification","regression"):
                for shape in (0,1):
                    for fold in range(1,9):
                        s = simulation.panel(family,task,shape,fold)
                        name = f"sim_{family}_{task}_p{s.X.shape[1]}"
                        yield Panel(name,fold,s.seed,s.X,s.y,s.Xt,s.truth,None,task), None, None
    else:
        raise ValueError(cohort)


def deploy(d, cohort, missing, arm, width):
    start=perf_counter_ns()
    X,y=d.X,d.y
    if missing and arm in ("F","previous","new"):
        valid=np.isfinite(X)
        mean=np.where(valid,X,0.).sum(axis=0)/np.maximum(1,valid.sum(axis=0))
        X=np.where(valid,X,mean)
    if arm=="new":
        cols=fast_support(X,y,task=d.task,seed=d.seed)[0]
    elif arm=="previous":
        fn=regression_support if d.task=="regression" else vote_support
        cols=fn(X,y,seed=d.seed)[0]
    elif arm=="F":
        cols=f_support(X,y,d.task,width)
    elif arm=="random":
        cols=np.sort(np.random.default_rng(d.seed).permutation(X.shape[1])[:width])
    elif arm=="full":
        cols=np.arange(X.shape[1])
    else:
        raise ValueError(arm)
    selected=perf_counter_ns()
    if cohort=="dense":
        pred=acoustic_fit(d,cols,"cholesky")[0]
    elif cohort=="four_stage_ridge":
        A,B=(d.X,d.Xt) if len(cols)==X.shape[1] else (d.X[:,cols],d.Xt[:,cols])
        A,B=four.encode(A,B,False)
        cls=Ridge if d.task=="regression" else RidgeClassifier
        pred=cls(alpha=1.,solver="cholesky").fit(A,y).predict(B)
    elif cohort=="spectra":
        pred=prediction(d.X,y,d.Xt,cols,d.task,d.seed)
    elif cohort=="mechanism":
        pred=bench.fit(d,cols)[0]
    else:
        raise ValueError(cohort)
    end=perf_counter_ns()
    return cols,pred,(selected-start)/1e6,(end-start)/1e6


def quality(d,pred):
    if d.name=="fsdd":
        return np.mean([balanced_accuracy_score(d.gold[d.units==u],pred[d.units==u]) for u in np.unique(d.units)])
    if d.units is not None:
        return evaluate(d.name,d.gold,pred,d.units,d.task)[0]
    return metric(d.gold,pred,d.task)


def run(base,cohort,run_id):
    out=base/(cohort+str(run_id))
    out.mkdir(parents=True,exist_ok=False)
    fields=KEY+["task","source_kind","metric","p","k","n_train","n_test","classes","method","repeat","position","score","select_ms","total_ms","train_sha","target_sha","test_sha","truth_sha"]
    i=0
    with (out/"times.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for d,archive,full_key in cases(cohort):
            i+=1
            hashes=[digest(a) for a in (d.X,d.y,d.Xt,d.gold)]
            kind="generated" if cohort=="mechanism" else "artificial_challenge" if d.name=="madelon" else "observational"
            metric_name={"har":"mean_person_BA","gas":"mean_batch_BA","fsdd":"mean_speaker_BA"}.get(d.name,"R2" if d.task=="regression" else "BA")
            meta=dict(zip(KEY,(cohort,d.name,d.fold,d.seed)),task=d.task,source_kind=kind,metric=metric_name,p=d.X.shape[1],n_train=len(d.y),n_test=len(d.gold),classes=len(np.unique(d.y)) if d.task=="classification" else 0)
            meta.update(zip(fields[-4:],hashes))
            missing=not np.isfinite(d.X).all()
            key=f"{d.name}-{d.fold}-{d.seed}.npz"
            arrays={"truth":d.gold,"hashes":np.array(hashes)}
            if d.units is not None:
                arrays["units"]=d.units
            width=None
            scores={}
            for arm in METHODS:
                cols,pred,_,_=deploy(d,cohort,missing,arm,width)
                if arm=="new":
                    width=len(cols)
                if arm in ("F","random"):
                    assert len(cols)==width
                np.testing.assert_array_equal(cols,np.unique(cols))
                assert len(cols)>0 and cols.min()>=0 and cols.max()<d.X.shape[1]
                if arm=="full" and archive is not None:
                    with np.load(archive) as z:
                        np.testing.assert_array_equal(z["truth"],d.gold)
                        np.testing.assert_array_equal(z[full_key],pred)
                if arm=="previous" and meta["p"]>32:
                    np.testing.assert_array_equal(cols,arrays["new_cols"])
                    np.testing.assert_array_equal(pred,arrays["new_pred"])
                arrays[arm+"_cols"]=cols
                arrays[arm+"_pred"]=pred
                scores[arm]=quality(d,pred)
            if run_id==2:
                with np.load(base/(cohort+"1")/key) as first:
                    assert set(first.files)==set(arrays)
                    for name,data in arrays.items():
                        np.testing.assert_array_equal(first[name],data)
            np.savez_compressed(out/key,**arrays)
            order=np.random.default_rng([d.fold,d.seed,run_id,9277]).permutation(METHODS)
            for repeat in range(5):
                for pos,arm in enumerate(np.roll(order,repeat)):
                    cols,pred,select_ms,total_ms=deploy(d,cohort,missing,arm,width)
                    np.testing.assert_array_equal(cols,arrays[arm+"_cols"])
                    np.testing.assert_array_equal(pred,arrays[arm+"_pred"])
                    writer.writerow(dict(meta,k=len(cols),method=arm,repeat=repeat,position=pos,score=scores[arm],select_ms=select_ms,total_ms=total_ms))
            stream.flush()
            if i%10==0:
                print(cohort,run_id,i,flush=True)
    assert i==SIZES[cohort],(cohort,i)
    paths=[Path(__file__),HERE/"method.py",HERE/"native_votes.c",Path(simulation.__file__),Path(bench.__file__),Path(four.__file__),Path(nir.__file__),Path(dense.__file__),Path(speech.__file__)]
    state=dict(cohort=cohort,process=run_id,n=i,sklearn=sklearn.__version__,numpy=np.__version__,sha256={q.relative_to(bench.ROOT).as_posix():bench.checksum(q) for q in paths})
    (out/"complete.json").write_text(json.dumps(state,indent=2),encoding="utf-8")


def summarize(base,out):
    records=[]
    for cohort,n in SIZES.items():
        frames=[]
        for idx in (1,2):
            folder=base/(cohort+str(idx))
            assert json.loads((folder/"complete.json").read_text())["n"]==n
            f=pd.read_csv(folder/"times.csv")
            assert len(f)==n*25 and f.groupby(KEY+["method"]).position.nunique().eq(5).all()
            f["process"]=idx
            frames.append(f)
        df=pd.concat(frames)
        groups=KEY+["method","process"]
        med=df.groupby(groups,as_index=False).agg({c:("median" if c in ("select_ms","total_ms") else "first") for c in df if c not in groups+["position","repeat"]})
        for case_id,group in med.groupby(KEY):
            a,b=(group[group.process==i].set_index("method") for i in (1,2))
            x=a.loc["new"]
            case=dict(zip(KEY,case_id))
            for c in ("task","source_kind","metric","p","k","classes","n_train","n_test","train_sha","target_sha","test_sha","truth_sha"):
                case[c]=x[c]
            for arm in METHODS:
                assert a.loc[arm,"score"]==b.loc[arm,"score"]
                case["score_"+arm]=a.loc[arm,"score"]
                case["width_"+arm]=a.loc[arm,"k"]
                for i,f in enumerate((a,b),1):
                    case[f"ms{i}_{arm}"]=f.loc[arm,"total_ms"]
                    case[f"select_ms{i}_{arm}"]=f.loc[arm,"select_ms"]
            records.append(case)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/"cases.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("--cohort",choices=SIZES)
    parser.add_argument("--process",type=int,choices=(1,2))
    parser.add_argument("--summary",type=Path)
    opts=parser.parse_args()
    if opts.summary:
        summarize(opts.base,opts.summary)
    else:
        with threadpool_limits(1):
            run(opts.base,opts.cohort,opts.process)

