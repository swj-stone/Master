"""Summarize min/mean contributions using paired predictions from the same held-out rows."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.test.compact_quality_bounds import _load
from experiments.test.piv3_lite_uncertainty import _csv, _paired_scores, _quantile, _relative_loss
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, _sha
from experiments.test.run_three_part import DEFAULT, DATASETS, FAMILIES
from experiments.test.run_compact_minpi import read

REPS=399

def run(root):
    detail=[]
    joint=[]
    for i,d in enumerate(DATASETS):
        meta=read(DEFAULT_ROOT/d/"metadata.json")
        file=DEFAULT_ROOT/d/"test_y.npy"
        if _sha(file)!=meta["sha256"][file.name]:
            raise ValueError("Held-out labels changed")
        y=np.load(file,allow_pickle=False)
        for f in FAMILIES:
            arms=["full",f,"min_"+f,"mean_"+f]
            pred={}
            expected={}
            for a in arms:
                data=_load(root,d,a,meta,y)
                if data is not None: pred[a],expected[a]=data
            if len(pred)!=len(arms):
                joint.append(dict(dataset=d,backend=f,status="incomplete"))
                continue
            with threadpool_limits(limits=1):
                point,samples=_paired_scores(d,y,pred,REPS,np.random.default_rng([42,i]))
            for a in arms:
                if any(abs(v-expected[a][k])>1e-12 for k,v in point[a].items()):
                    raise ValueError(f"Cannot reproduce {d}/{a} test metric")
            for a in arms[2:]:
                measures=[]
                pts=[]
                for b in ("full",f):
                    for k,v in point[a].items():
                        vals=np.array([_relative_loss(x,z,k) for x,z in zip(samples[a][k],samples[b][k],strict=True)])
                        loss=_relative_loss(v,point[b][k],k)
                        detail.append(dict(dataset=d,backend=f,comparison=f"{a} vs {b}",metric=k,
                                           relative_loss=loss,upper95_relative_loss=_quantile(vals)))
                        pts.append(loss)
                        measures.append(vals)
                upper=_quantile(np.max(measures,axis=0))
                joint.append(dict(dataset=d,backend=f,method=a,status="completed",point_worst_loss=max(pts),
                                  upper95_worst_loss=upper,point_within_5pct=(max(pts)<=.05+1e-12),
                                  upper_within_5pct=(upper<=.05+1e-12)))
            a,b=arms[2:]
            for k in point[a]:
                vals=np.array([_relative_loss(x,z,k) for x,z in zip(samples[a][k],samples[b][k],strict=True)])
                detail.append(dict(dataset=d,backend=f,comparison="min vs mean",metric=k,
                                   relative_loss=_relative_loss(point[a][k],point[b][k],k),
                                   upper95_relative_loss=_quantile(vals)))
    _csv(root/"quality-bounds.csv",joint)
    _csv(root/"quality-by-metric.csv",detail)
    return joint

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,nargs="?",default=DEFAULT)
    out=run(parser.parse_args().run)
    print(f"{len(out)} dataset/backend/mode contrasts; 399 paired, stratified test-row draws when classification.")

