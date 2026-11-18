"""Compare fixed and automatically calibrated consensus minPI on six frozen deployments."""
import argparse
import json
from itertools import combinations
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from experiments.minpi_backward_confirmation import evaluate, splits
from experiments.minpi_stable_shift import SEEDS, digest, get_shift, serializable
from experiments.minpi_study import ROOT, file_hash, train
from minppi_mofs.consensus_minpi import ConsensusMinPISelector, consensus_top_from_training


RUNS = ROOT / "results/minpi-study"
SHIFT = RUNS / "stable-shift-20261117b"
FIXED = RUNS / "backward-confirmation-20261117a"
AUTO = RUNS / "stable-confirmation-20261117b"
SOURCES = ("gas", "supercon", "har", "mhealth", "diabetes", "isolet")
SHIFT_SOURCES = ("gas", "supercon", "har")
METHODS = ("auto", "consensus50", "full")


def benchmarks(source):
    if source in SHIFT_SOURCES:
        data = get_shift(source)
        for seed in SEEDS:
            yield seed, data
    else:
        yield from splits(source)


def prior_results():
    shift = pd.read_csv(SHIFT/"trials.csv")
    names = {"stable":"previous_auto", "environment50":"previous_forward", "local50":"previous_local",
             "f50":"F50", "pfi50":"PFI50", "calibrated_f":"calibrated_F", "full":"previous_full"}
    shift = shift[shift.method.isin(names)].copy()
    shift["method"] = shift.method.map(names)
    fixed = pd.read_csv(FIXED/"cases.csv")
    fixed = fixed[(~fixed.source.isin(SHIFT_SOURCES)) | (fixed.method=="backward")].copy()
    fixed["method"] = fixed.method.map({"full":"previous_full", "backward":"previous_backward",
                       "local":"previous_local", "forward":"previous_forward", "F":"F50", "PFI":"PFI50"})
    automatic = pd.read_csv(AUTO/"cases.csv").assign(method="previous_auto")
    return pd.concat([x[["source","seed","method","width","score"]] for x in (shift, fixed, automatic)])


def run(output, verify):
    if not verify:
        output.mkdir(parents=True,exist_ok=False)
    paths = [Path(__file__), *(ROOT/"src/minppi_mofs"/f for f in (
        "consensus_minpi.py","stable_minpi.py","environment_minpi.py")), ROOT/"experiments/minpi_study.py",
        ROOT/"experiments/minpi_backward_confirmation.py", ROOT/"experiments/minpi_stable_shift.py"]
    protocol = dict(sources=SOURCES,seeds=SEEDS, methods=METHODS, n_paths=3, tolerance=.02,threads=1,
                    fraction=.5, automatic_fractions=(.1,.5,.75), sklearn=sklearn.__version__,
                    split_protocol_hashes={p.name:file_hash(p/"protocol.json") for p in (SHIFT,FIXED,AUTO)},
                    source_sha256={str(f.relative_to(ROOT)):file_hash(f) for f in paths},
                    timing="One fresh execution per seed/split, rotating order, charge selection, slicing, ridge refit and prediction. Other methods' scores imported from frozen same-split runs; timings are not pooled across runs.",
                    units="Historical Gas forward batches, official HAR test people; material composition blocked. MHEALTH worst wrist/ankle BA per held person; ISOLET paired sample split; diabetes fixed folds.",
                    sensitivity="For Gas/materials/HAR/MHEALTH the three selector seeds use identical data. For diabetes/ISOLET they pair with different outer splits. Reproduce each exact configuration in a separate process.")
    manifest=output/"protocol.json"
    if verify:
        assert json.loads(manifest.read_text(encoding="utf-8"))==json.loads(json.dumps(protocol))
    else:
        manifest.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows, units_rows, jaccard=[],[],[]
    for source in SOURCES:
        support_banks = {name:{} for name in METHODS if name!="full"}
        for seed,(X,y,groups,xt,truth,units,task) in benchmarks(source):
            if groups is not None:
                assert not np.intersect1d(groups,units).size
            saved = {"truth":truth,"units":units}
            methods=np.roll(METHODS,SEEDS.index(seed))
            for method in methods:
                start=perf_counter_ns()
                if method=="auto":
                    estimator=ConsensusMinPISelector(task=task,random_state=seed)
                    estimator.fit(X,y,groups=groups,ordered_groups=source=="gas",environment_groups=True)
                    cols,info=estimator.get_support(indices=True),estimator.diagnostics_
                elif method=="consensus50":
                    cols,info=consensus_top_from_training(X,y,ceil(X.shape[1]/2),task=task,random_state=seed,
                            groups=groups,ordered_groups=source=="gas",environment_groups=True)
                else:
                    cols,info=np.arange(X.shape[1]),{}
                selected=perf_counter_ns()
                A,B=(X,xt) if len(cols)==X.shape[1] else (X[:,cols],xt[:,cols])
                pred,_=train("ridge",A,y,B,task,seed)
                done=perf_counter_ns()
                assert np.array_equal(np.unique(cols),cols)
                if method=="consensus50":
                    assert len(cols)==ceil(X.shape[1]/2)
                if method=="auto":
                    discovery=np.asarray(info["discovery_rows"])
                    validation=[np.asarray(s) for s in info["validation_rows"]]
                    for group in validation:
                        assert not np.intersect1d(discovery,group).size
                        if groups is not None:
                            assert not np.intersect1d(groups[discovery],groups[group]).size
                if method!="full":
                    support_banks[method][seed]=cols
                    path=output/f"{source}-{seed}-{method}-decisions.json"
                    dec=json.dumps(info,sort_keys=True,default=serializable)
                    if verify:
                        assert dec==path.read_text(encoding="utf-8")
                    else:
                        path.write_text(dec,encoding="utf-8")
                saved[method+"-support"],saved[method+"-pred"]=cols,pred
                score,per_unit=evaluate(source,truth,pred,units,task)
                units_rows.extend(dict(source=source,seed=seed,method=method,unit=k,score=v)
                                  for k,v in per_unit.items())
                rows.append(dict(source=source,seed=seed,method=method,chosen_rule=info.get("rule",method),
                                 width=len(cols),p=X.shape[1],score=score,select_ms=(selected-start)/1e6,
                                 total_ms=(done-start)/1e6,support_sha256=digest(cols),prediction_sha256=digest(pred)))
            if source in SHIFT_SOURCES:
                archive=SHIFT/f"{source}-predictions.npz"
                pred_key=f"full-{seed}-pred"
            else:
                archive=FIXED/f"{source}-{seed}.npz"
                pred_key="full-pred"
            with np.load(archive) as oracle:
                np.testing.assert_array_equal(oracle[pred_key],saved["full-pred"])
                np.testing.assert_array_equal(oracle["truth"],truth)
            archive=output/f"{source}-{seed}.npz"
            if verify:
                with np.load(archive) as old:
                    for key,value in saved.items():
                        np.testing.assert_array_equal(old[key],value)
            else:
                np.savez_compressed(archive,**saved)
            print(f"{source} seed={seed} width={len(saved['auto-support'])}/{X.shape[1]}",flush=True)
        for method, supports in support_banks.items():
            for a,b in combinations(supports,2):
                overlap=len(np.intersect1d(supports[a],supports[b]))/len(np.union1d(supports[a],supports[b]))
                jaccard.append(dict(source=source,method=method,left=a,right=b,jaccard=overlap))
    frame=pd.DataFrame(rows)
    if verify:
        frame.drop(columns=["select_ms","total_ms"]).to_csv(output/"fresh-process-audit.csv",index=False)
        return
    frame.to_csv(output/"cases.csv",index=False)
    pd.DataFrame(jaccard).to_csv(output/"seed-overlap.csv",index=False)
    pd.DataFrame(units_rows).to_csv(output/"unit_scores.csv",index=False)
    fields=["source","seed","method","width","score"]
    combined=pd.concat([frame[fields],prior_results()],ignore_index=True)
    assert not combined.duplicated(["source","seed","method"]).any()
    combined.to_csv(output/"paired-methods.csv",index=False)
    paired=frame.merge(frame[frame.method=="full"][["source","seed","score"]].rename(columns={"score":"full_score"}),
                       on=["source","seed"])
    paired["delta_full"]=paired.score-paired.full_score
    paired["passes"]=paired.delta_full>=-.02-1e-12
    paired["compressed"]=paired.width<paired.p
    paired.to_csv(output/"paired.csv",index=False)
    summary=paired.groupby(["source","method"],as_index=False).agg(score=("score","mean"),
            worst_gap=("delta_full","min"),passes=("passes","sum"),compressed=("compressed","sum"),
            total_ms=("total_ms","median"),select_ms=("select_ms","median"))
    summary.to_csv(output/"summary.csv",index=False)
    print(summary.round(4).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify",action="store_true")
    opts=parser.parse_args()
    with threadpool_limits(limits=1):
        run(opts.output,opts.verify)
