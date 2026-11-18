"""Apply an already frozen automatic width rule to the three additional confirmation cohorts."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_backward_confirmation import evaluate, splits
from experiments.minpi_stable_shift import digest, serializable
from experiments.minpi_study import ROOT, file_hash, train
from minppi_mofs.stable_minpi import StableMinPISelector


def run(output, baseline, verify):
    if not verify:
        output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/"experiments/minpi_backward_confirmation.py",
             ROOT/"src/minppi_mofs/stable_minpi.py", ROOT/"src/minppi_mofs/environment_minpi.py"]
    protocol=dict(sources=["mhealth","diabetes","isolet"], seeds=[1129,2027,3469], tolerance=.02,
                  parent_protocol_hash=file_hash(baseline/"protocol.json"),
                  source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths})
    if verify:
        assert json.loads((output/"protocol.json").read_text(encoding="utf-8")) == protocol
    else:
        (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    for source in protocol["sources"]:
        for seed, (X,y,groups,xt,truth,units,task) in splits(source):
            start=perf_counter_ns()
            estimator=StableMinPISelector(task=task,random_state=seed)
            estimator.fit(X,y,groups=groups)
            cols=estimator.get_support(indices=True)
            selected=perf_counter_ns()
            A,B=(X,xt) if len(cols)==X.shape[1] else (X[:,cols],xt[:,cols])
            pred,_=train("ridge",A,y,B,task,seed)
            done=perf_counter_ns()
            full_start=perf_counter_ns()
            full_pred,_=train("ridge",X,y,xt,task,seed)
            full_end=perf_counter_ns()
            with np.load(baseline/f"{source}-{seed}.npz") as old:
                np.testing.assert_array_equal(units,old["test_units"])
                np.testing.assert_array_equal(truth,old["truth"])
                np.testing.assert_array_equal(full_pred,old["full-pred"])
                score_full,_=evaluate(source,truth,old["full-pred"],units,task)
            score,_=evaluate(source,truth,pred,units,task)
            rows.append(dict(source=source,seed=seed,rule=estimator.diagnostics_["rule"],
                             width=len(cols),p=X.shape[1],score=score,full=score_full,delta=score-score_full,
                             pass_tolerance=score>=score_full-.02,support_sha256=digest(cols),prediction_sha256=digest(pred),
                             select_ms=(selected-start)/1e6,total_ms=(done-start)/1e6,
                             full_ms=(full_end-full_start)/1e6))
            filepath=output/f"{source}-{seed}.npz"
            decision_path=output/f"{source}-{seed}-decision.json"
            decision=json.dumps(estimator.diagnostics_,sort_keys=True,default=serializable)
            if verify:
                with np.load(filepath) as old:
                    np.testing.assert_array_equal(old["cols"],cols)
                    np.testing.assert_array_equal(old["pred"],pred)
                assert decision==decision_path.read_text(encoding="utf-8")
            else:
                np.savez_compressed(filepath,cols=cols,pred=pred)
                decision_path.write_text(decision,encoding="utf-8")
    frame=pd.DataFrame(rows)
    frame.to_csv(output/("fresh-process-audit.csv" if verify else "cases.csv"),index=False)
    print(frame.drop(columns=["support_sha256","prediction_sha256"]).round(4).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.baseline,args.verify)
