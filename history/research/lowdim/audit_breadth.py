"""Independently check the F cutoff, uniform draws and full-prediction archives."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.feature_game_eval.audit_controls import effect_size
from research.lowdim.extra_study import SIZES, cases
from research.lowdim.method import fast_support
from research.lowdim.survey_inputs import study_cases


def peer_change(X,y,task,seed,archive,metadata):
    cols,info=fast_support(X,y,task=task,seed=seed)
    empty=np.sort(np.argsort(-info["tier_scores"][0],kind="stable")[:len(cols)])
    with np.load(archive) as saved:
        np.testing.assert_array_equal(cols,saved["new_cols"])
        f_cols=saved["F_cols"]
    return dict(**metadata,k=len(cols),p=X.shape[1],empty_replacements=len(np.setdiff1d(cols,empty)),
                F_replacements=len(np.setdiff1d(cols,f_cols)))


def run(base,out,survey):
    rows,peer=[],[]
    for cohort,count in SIZES.items():
        folders=[base/(cohort+str(i)) for i in (1,2)]
        states=[json.loads((p/"complete.json").read_text()) for p in folders]
        assert states[0]["sha256"]==states[1]["sha256"] and all(s["n"]==count for s in states)
        for d,archive,full_key in cases(cohort):
            X=d.X
            if not np.isfinite(X).all():
                valid=np.isfinite(X)
                means=np.where(valid,X,0.).sum(axis=0)/np.maximum(1,valid.sum(axis=0))
                X=np.where(valid,X,means)
            effects=np.concatenate([effect_size(X[:,start:start+256],d.y,d.task) for start in range(0,X.shape[1],256)])
            with np.load(folders[0]/f"{d.name}-{d.fold}-{d.seed}.npz") as f:
                cols=f["F_cols"]
                remaining=np.setdiff1d(np.arange(len(effects)),cols)
                gap=max(0.,effects[remaining].max()-effects[cols].min()) if len(remaining) else 0.
                assert gap<1e-9,(cohort,d.name,gap)
                np.testing.assert_array_equal(f["random_cols"],np.sort(np.random.default_rng(d.seed).permutation(len(effects))[:len(cols)]))
                if archive:
                    with np.load(archive) as old:
                        np.testing.assert_array_equal(f["full_pred"],old[full_key])
            rows.append(dict(cohort=cohort,dataset=d.name,fold=d.fold,seed=d.seed,F_boundary_violation=gap,
                             F_exact_set=np.array_equal(cols,np.sort(np.argsort(-effects,kind="stable")[:len(cols)])),
                             archive_full=archive is not None))
            if X.shape[1]<=32:
                key=dict(cohort=cohort,dataset=d.name,fold=d.fold,seed=d.seed)
                peer.append(peer_change(X,d.y,d.task,d.seed,folders[0]/f"{d.name}-{d.fold}-{d.seed}.npz",key))
    if survey is not None:
        for c in study_cases():
            if c.X.shape[1]<=32:
                key=dict(cohort="archive_supplement",dataset=c.dataset,fold=c.fold,seed=c.seed)
                peer.append(peer_change(c.X,c.y,c.task,c.seed,survey/f"{c.dataset}-{c.fold}-{c.seed}.npz",key))
    pd.DataFrame(peer).to_csv(out/"peer-support.csv",index=False)
    assert len(rows)==sum(SIZES.values())
    pd.DataFrame(rows).to_csv(out/"f-audit.csv",index=False)
    print("F cutoffs and uniform draws verified",len(rows),"max F violation",max(r["F_boundary_violation"] for r in rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("base","out"):
        p.add_argument(arg,type=Path)
    p.add_argument("--survey",type=Path)
    args=p.parse_args()
    with threadpool_limits(1):
        run(args.base,args.out,args.survey)

