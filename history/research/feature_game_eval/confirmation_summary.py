"""Pair independent clock runs and test the frozen redundant-proxy criterion."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.feature_game_eval.confirmation import ARMS, REPEATS

KEY = ["mechanism","task","rows","p","replicate"]


def summarize(first, second, out):
    if (first/"protocol.json").read_bytes()!=(second/"protocol.json").read_bytes():
        raise ValueError("Different confirmation protocols")
    tables=[]
    frames=[]
    for folder in [first,second]:
        frames.append(pd.read_csv(folder/"cases.csv").set_index(KEY+["method"]).sort_index())
        clock = pd.read_csv(folder/"trials.csv").groupby(KEY+["method"])
        assert (clock.size()==REPEATS).all() and (clock.position.nunique()==REPEATS).all()
        tables.append(clock.total_ms.median().unstack("method"))
    pd.testing.assert_frame_equal(*frames)
    q=frames[0].unstack("method")
    assert len(q)==64 and set(q.score.columns)==set(ARMS)
    result = q.score.add_suffix("_score").copy()
    result["width"] = q.width.minpi
    for name in ARMS:
        result[name+"_ms"] = np.maximum(*[t[name] for t in tables])
        result[name+"_relative_full"] = np.maximum(*[t[name]/t.full for t in tables])
        result["minpi_relative_"+name] = np.maximum(*[t.minpi/t[name] for t in tables])
        result[name+"_coverage"] = q.latent_groups[name]
    result=result.reset_index()
    result["joint_full"]=(result.width<result.p)&(result.minpi_score>=result.full_score-.02-1e-12)&(result.minpi_relative_full<1)
    result["strict_test"]=result.joint_full&(result.minpi_score>result.F_score+1e-12)&(result.minpi_score>result.random_score+1e-12)
    out.mkdir(parents=True,exist_ok=False)
    result.to_csv(out/"paired.csv",index=False)
    records=[]
    for key,g in result.groupby(["mechanism","task"]):
        for method in ARMS:
            gap=g.minpi_score-g[method+"_score"]
            ratio=g["minpi_relative_"+method]
            records.append(dict(mechanism=key[0],task=key[1],comparator=method,
                configurations=len(g), average_score=g[method+"_score"].mean(), average_minpi=g.minpi_score.mean(),
                minpi_wins=(gap>1e-12).sum(),ties=(gap.abs()<=1e-12).sum(),
                minpi_faster=(ratio<1).sum(),minpi_pareto=((gap>=-1e-12)&(ratio<1)).sum(),
                joint_vs_Full=((g.width<g.p)&(g[method+"_score"]>=g.full_score-.02-1e-12)
                               &(g[method+"_relative_full"]<1)&(method!="full")).sum(),
                median_cost_ratio=ratio.median(), mean_covered_groups=g[method+"_coverage"].mean()))
    pd.DataFrame(records).to_csv(out/"method-summary.csv",index=False)
    shape=result.groupby(["mechanism","task","p"]).agg(cases=("width","size"),width_median=("width","median"),
        joint=("joint_full","sum"),strict=("strict_test","sum")).reset_index()
    shape.to_csv(out/"shape-summary.csv",index=False)
    decision={}
    for task,g in shape.query("mechanism=='proxy'").groupby("task"):
        decision[task]=dict(strict=int(g.strict.sum()), strict_by_width={int(p):int(c) for p,c in zip(g.p,g.strict)},
                            passed=bool((g.strict>=5).all() and g.strict.sum()>=12))
    (out/"decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    print(json.dumps(decision,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("first","second","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    summarize(args.first,args.second,args.output)
