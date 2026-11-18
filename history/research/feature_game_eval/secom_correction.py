"""Correct float32 cancellation for the three SECOM ANOVA screens and retime all arms."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.feature_game_eval import panels as bench
from research.feature_game_eval import summarize as reports
from research.feature_game_eval.audit_controls import effect_size


def stable_f(X,y,width):
    X,_=bench._impute_validation(X,X[:0])
    Z=X.astype(np.float64,copy=True)
    Z-=Z[0].copy()
    Z-=Z.mean(axis=0)
    return bench.choose("f",Z,y,"classification",None,width)


def measure(out, pass_id, base, reference):
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    scores=[]
    state=bench.protocol() | {"correction":"only SECOM F: float64 and center before official one-way F",
                             "correction_sha256":bench.checksum(Path(__file__))}
    for seed,X,y,groups,Xt,truth,units,task,extra in bench.deployment.configurations("secom"):
        case=f"original-secom-1-{seed}"
        with np.load(base/(case+".npz")) as old:
            arrays={k:old[k] for k in old}
        np.testing.assert_array_equal(arrays["input_fingerprints"],[bench.digest(a) for a in (X,y,Xt,truth)])
        width=len(arrays["minpi_columns"])
        panel=bench.original.previous.Panel("secom",1,seed,X,y,Xt,truth,task,groups,units,arrays["full_pred"])
        schedule=np.random.default_rng([5399,seed,1,pass_id]).permutation(bench.ARMS)
        for repeat in range(-1,bench.REPEATS):
            for pos,arm in enumerate(np.roll(schedule,repeat)):
                if arm=="F":
                    start=perf_counter_ns()
                    cols=stable_f(X,y,width)
                    selected=perf_counter_ns()
                    pred,decision=bench.fit(panel,cols)
                    end=perf_counter_ns()
                    select_ms,total_ms=(selected-start)/1e6,(end-start)/1e6
                else:
                    cols,pred,decision,select_ms,total_ms=bench.pipeline(panel,arm,width)
                if arm=="F" and repeat==-1:
                    arrays["F_columns"],arrays["F_pred"],arrays["F_decision"]=cols,pred,decision
                    imputed,_=bench._impute_validation(X,X[:0])
                    effects=effect_size(imputed,y,task)
                    np.testing.assert_array_equal(cols,np.sort(np.argsort(-effects,kind="stable")[:width]))
                for suffix,actual in [("_columns",cols),("_pred",pred),("_decision",decision)]:
                    np.testing.assert_array_equal(arrays[arm+suffix],actual)
                if repeat<0:
                    score,_,ap=bench.deployment.score_predictions("secom",truth,pred,units,task,decision)
                    scores.append(dict(cohort="original",dataset="secom",fold=1,seed=seed,family="original",
                                       task=task,train=len(y),test=len(truth),p=X.shape[1],method=arm,
                                       width=len(cols),score=score,secom_ap=ap))
                else:
                    rows.append(dict(cohort="original",dataset="secom",fold=1,seed=seed,method=arm,
                                     repeat=repeat,position=pos,select_ms=select_ms,total_ms=total_ms))
        if reference:
            with np.load(reference/(case+".npz")) as old:
                for k,v in arrays.items():
                    np.testing.assert_array_equal(old[k],v)
        np.savez_compressed(out/(case+".npz"),**arrays)
    pd.DataFrame(rows).to_csv(out/"times.csv",index=False)
    pd.DataFrame(scores).to_csv(out/"scores.csv",index=False)
    (out/"protocol.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    (out/"complete.json").write_text(json.dumps(dict(stage="cost",pass_id=pass_id)),encoding="utf-8")
    print("Verified all corrected F supports, non-F predictions and 75 fresh timed calls")


def integrate(out, base, original_times, corrections):
    out.mkdir(parents=True,exist_ok=False)
    quality=out/"quality"
    quality.mkdir()
    state=json.loads((corrections[0]/"protocol.json").read_text(encoding="utf-8"))
    assert state==json.loads((corrections[1]/"protocol.json").read_text(encoding="utf-8"))
    protocol=json.dumps(state,indent=2)
    (quality/"protocol.json").write_text(protocol,encoding="utf-8")
    scores=pd.read_csv(base/"scores.csv")
    revised=pd.read_csv(corrections[0]/"scores.csv")
    keys=["seed","method"]
    pd.testing.assert_frame_equal(revised.sort_values(keys).reset_index(drop=True),
                                  pd.read_csv(corrections[1]/"scores.csv").sort_values(keys).reset_index(drop=True))
    mix=scores[~((scores.dataset=="secom") & scores.method.isin(bench.ARMS))]
    pd.concat((mix,revised),ignore_index=True).to_csv(quality/"scores.csv",index=False)
    # Arrays remain in their immutable generating runs; explicit provenance per panel.
    paths=[]
    for source in sorted(base.glob("*.npz")):
        final=(corrections[0]/source.name) if source.name.startswith("original-secom-") else source
        paths.append(dict(panel=source.stem,source=final.as_posix(),sha256=bench.checksum(final)))
    pd.DataFrame(paths).to_csv(quality/"arrays.csv",index=False)
    merged_times=[]
    for index,(clock,correction) in enumerate(zip(original_times,corrections,strict=True),1):
        before=pd.read_csv(clock/"times.csv")
        after=pd.read_csv(correction/"times.csv")
        folder=out/f"cost-{index}"
        folder.mkdir()
        (folder/"protocol.json").write_text(protocol,encoding="utf-8")
        (folder/"complete.json").write_text(json.dumps(dict(stage="cost",pass_id=index)),encoding="utf-8")
        pd.concat([before[before.dataset!="secom"],after],ignore_index=True).to_csv(folder/"times.csv",index=False)
        merged_times.append(folder)
    reports.summarize(quality,merged_times,out/"summary")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="stage",required=True)
    meas=sub.add_parser("measure")
    meas.add_argument("output",type=Path)
    meas.add_argument("quality",type=Path)
    meas.add_argument("--pass-id",required=True,type=int,choices=(1,2))
    meas.add_argument("--reference",type=Path)
    merge=sub.add_parser("integrate")
    merge.add_argument("output",type=Path)
    merge.add_argument("quality",type=Path)
    merge.add_argument("--original-times",nargs=2,type=Path,required=True)
    merge.add_argument("--corrected",nargs=2,type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=="measure":
            if bool(args.reference)!=(args.pass_id==2):
                parser.error("Second pass requires first pass as reference")
            measure(args.output,args.pass_id,args.quality,args.reference)
        else:
            integrate(args.output,args.quality,args.original_times,args.corrected)
