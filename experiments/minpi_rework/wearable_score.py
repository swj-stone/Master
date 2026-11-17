"""Score only after all registered participant folds have sealed forecasts."""
import json
from hashlib import sha256

import numpy as np
import pandas as pd
from experiments.minpi_rework import wearable_data as data
from experiments.minpi_rework import wearable_run as run


def lock_all():
    receipts = {}
    for name,(_,_,n,_) in data.SOURCES.items():
        lock = json.loads((data.OUT/f"lock-{name}.json").read_bytes())
        assert set(lock["receipts"])=={f"{name}-{i}" for i in range(n)}
        for key,stamp in lock["receipts"].items():
            folder = run.RUN/key
            file = folder/"selection.json"
            assert data.digest(file)==stamp
            rec = json.loads(file.read_bytes())
            assert data.digest(folder/"forecasts.npz")==rec["forecast_sha256"]
            receipts[key]=stamp
    return receipts


def score():
    receipts = lock_all()
    rows,clock,candidates = [],[],[]
    for key in receipts:
        folder = run.RUN/key
        rec = json.loads((folder/"selection.json").read_bytes())
        name,fold = rec["dataset"],rec["fold"]
        x,truth,subject,unit = data.read(name)
        ix = np.flatnonzero(np.isin(subject,rec["people"]["test"]))
        assert sha256(ix.astype("<i8").tobytes()).hexdigest()==rec["rows_sha256"]
        p = len(data.SOURCES[name][1])
        full = (1<<p)-1
        use = truth[ix]>=0
        y = truth[ix][use]
        for kind in ("frozen","refit"):
            for mask,info in rec[kind].items():
                candidates.append(dict(dataset=name,fold=fold,receiver=kind,**{k:v for k,v in info.items() if k!="risk"}))
        with np.load(folder/"forecasts.npz",allow_pickle=False) as archive:
            ref = {s:run.brier(y,archive[f"frozen_{s}"][use]) for s in range(full+1)}
            full_auc,full_acc = run.metrics(y,archive[f"frozen_{full}"][use])
            for arm,mask in rec["arms"].items():
                trained = arm.startswith("refit")
                book = rec["refit" if trained else "frozen"][str(mask)]
                scores = {s:archive[f"refit_{mask}_{s}" if trained else f"frozen_{s}"][use]
                          for s in run.submasks(mask)}
                auc,acc = run.metrics(y,scores[mask])
                risk = run.curve({s:run.brier(y,v) for s,v in scores.items()},mask,ref,ref[0],p)
                rows.append(dict(dataset=name,fold=fold,subject=rec["people"]["test"][0],arm=arm,
                                 k=mask.bit_count(),devices="|".join(lab for i,lab in enumerate(data.SOURCES[name][1]) if mask&(1<<i)),
                                 test_n=len(y),test_n_features=len(ix),n_observed_classes=len(np.unique(y)),
                                 auc=auc,accuracy=acc,delta_auc=auc-full_auc,delta_acc=acc-full_acc,
                                 worst_risk=max(risk),eligible_accept=book["eligible"],
                                 eligible_held=bool(auc>=full_auc-.01 and acc>=full_acc-.01 and max(risk)<=.02+1e-12),
                                 clean_held=bool(auc>=full_auc-.01 and acc>=full_acc-.01)))
        clock.append(dict(dataset=name,fold=fold,**{k:v for k,v in rec.items() if k.endswith("_s") or k=="n_additional_fits"}))
    frame = pd.DataFrame(rows)
    frame.to_csv(data.OUT/"held.csv",index=False)
    pd.DataFrame(candidates).to_csv(data.OUT/"acceptance-subsets.csv",index=False)
    pd.DataFrame(clock).to_csv(data.OUT/"clocks.csv",index=False)
    grouped = frame.copy()
    grouped["retired"] = [k<len(data.SOURCES[name][1]) for k,name in zip(grouped.k,grouped.dataset)]
    out = grouped.groupby(["dataset","arm"]).agg(
        n=("fold","size"),k=("k","mean"),retired=("retired","sum"),auc=("auc","mean"),accuracy=("accuracy","mean"),
        delta_auc=("delta_auc","mean"),delta_acc=("delta_acc","mean"),max_risk=("worst_risk","max"),
        eligible_accept=("eligible_accept","sum"),clean_held=("clean_held","sum"),eligible_held=("eligible_held","sum"))
    out.to_csv(data.OUT/"summary.csv")
    print(out.to_string(float_format=lambda z:f"{z:.4f}"))


if __name__=="__main__":
    score()
