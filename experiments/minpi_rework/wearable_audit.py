"""Check native sklearn truncation at every device state and time the two group selectors."""
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import wearable_data as data, wearable_run as run
from experiments.minpi_rework.predictive_selection import prepare_paths,select_width
from experiments.minpi_rework.wide_veto_verify import native_after_pruning


def verify():
    clocks,audits = [],[]
    rng = np.random.default_rng(351)
    pilot = rng.normal(size=(100,6))
    target = np.arange(100)%3
    forest = run.rf.fit(pilot[:40],target[:40],"classification",13,n_estimators=4)
    pilot_paths,null,_ = prepare_paths(forest,pilot[40:70],target[40:70],pilot[70:],target[70:],
                                       "classification",rank_backend="interval",unit_ids=np.repeat(range(3),2))
    select_width(pilot_paths,null,target[70:],"classification",1.)
    for name,(_,_,n,_) in data.SOURCES.items():
        lock = json.loads((data.OUT/f"lock-{name}.json").read_bytes())
        for pos in range(n):
            folder = run.RUN/f"{name}-{pos}"
            assert data.digest(folder/"selection.json")==lock["receipts"][folder.name]
            receipt = json.loads((folder/"selection.json").read_bytes())
            assert data.digest(folder/"forecasts.npz")==receipt["forecast_sha256"]
            x,y,unit,idx,_ = data.fold(name,pos)
            xf,yf = x[idx["fit"]],y[idx["fit"]]
            xv,yv = x[idx["vote"]],y[idx["vote"]]
            xa,ya = x[idx["width"]],y[idx["width"]]
            xt = x[idx["test"]]
            with np.load(folder/"forecasts.npz",allow_pickle=False) as z:
                first = run.rf.fit(xf,yf,"classification",42)
                # Compare archived probabilities on actual held rows at every mask.
                for mask in range(receipt["full"]+1):
                    pred,reachable = native_after_pruning(first,xt,run.columns(unit,mask))
                    err = float(np.max(abs(pred-z[f"frozen_{mask}"])))
                    assert err<1e-10
                    audits.append(dict(dataset=name,fold=pos,mask=mask,reachable_numeric=reachable,max_error=err))
                schedules = ["mdi","portfolio"]
                if pos%2:
                    schedules.reverse()
                for repeat in range(3):
                    for method in schedules:
                        start = perf_counter()
                        model = run.rf.fit(xf,yf,"classification",42)
                        paths,null,_ = prepare_paths(model,xv,yv,xa,ya,"classification",
                            include_min=method=="portfolio",rank_backend="interval",unit_ids=unit)
                        devs,_ = select_width(paths,null,ya,"classification",1.)
                        mask = sum(1<<int(j) for j in devs)
                        keep = run.columns(unit,mask)
                        forecast = model.predict(xt[:,keep],keep)
                        elapsed = perf_counter()-start
                        assert mask==receipt["arms"][method]
                        np.testing.assert_allclose(forecast,z[f"frozen_{mask}"],atol=1e-13,rtol=0.)
                        clocks.append(dict(dataset=name,fold=pos,rep=repeat,method=method,seconds=elapsed))
                    schedules.reverse()
            pd.DataFrame(clocks).to_csv(data.OUT/"selector-runtime.csv",index=False)
            pd.DataFrame(audits).to_csv(data.OUT/"native-checks.csv",index=False)
        print(name,n,"completed",flush=True)


if __name__=="__main__":
    from experiments.minpi_rework.wearable_assets import cli_args,inputs
    args = cli_args()
    with threadpool_limits(1),inputs(args.prepared,args.archive,args.output):
        verify()
