"""Sum recorded shared stages for each drift-study algorithm path."""
import json
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/context_review/drift-locked"


def main():
    times=json.loads((OUT/"frozen.json").read_bytes())["components_seconds"]
    cols=json.loads((OUT/"supports.json").read_bytes())["supports"]
    stages={
        "full":[],
        "compact_gain":["anchor"],
        "vote":["anchor","vote"],
        "random_family":["anchor","random_family"],
        "vote_guard":["anchor","vote","guard_prep","guard_compact_gain","guard_vote"],
        "maximin":["anchor","vote","maximin"],
        "weak_gain":["anchor","vote"],
        "maximin_guard":["anchor","vote","maximin","guard_prep","guard_compact_gain","guard_vote","guard_maximin"],
        "weak_gain_guard":["anchor","vote","guard_prep","guard_compact_gain","guard_vote","guard_weak_gain"],
        "gain_at_vote_width":["anchor"],
        "rfe_at_vote_width":["rfe_at_vote_width"],
        "rfe_at_challenger_width":["rfe_at_challenger_width"],
        "anova_at_10pct":["anova_at_10pct"],
        "anova_at_25pct":["anova_at_25pct"]
    }
    assert stages.keys()==cols.keys()
    timing=[]
    for arm,costs in stages.items():
        selection=sum(times[s] for s in costs)
        final=times["final_prep"]+times["final_"+arm]
        timing.append(dict(arm=arm,k=len(cols[arm]),channels=len({i//8 for i in cols[arm]}),
                           component_sum_s=selection+final,selection_s=selection,refit_predict_s=final,
                           selection_stages=",".join(costs)))
    pd.DataFrame(timing).to_csv(OUT/"costs.csv",index=False)
    scores=pd.read_csv(OUT/"by-batch.csv")
    recall=scores.pivot(index="arm",columns="batch",values="recall")
    assert len(recall)==len(stages)
    print(recall.to_string(float_format=lambda x:f"{x:.5f}"))


if __name__=="__main__":
    main()
