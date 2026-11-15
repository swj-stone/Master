"""Matched per-task differences on independently reproduced refit-guard folds."""
from pathlib import Path

import pandas as pd

from rounds.pmlb_baselines import ROOT, read

RESULTS = ROOT/"results/context_review"
HERE = Path(__file__).resolve().parent
rows = []
for source in ("refit-guard","challenger"):
    for path in sorted((RESULTS/source).glob("*/fold-*/completed.json")):
        rows.extend(read(path)["rows"])
df = pd.DataFrame(rows)
assert not df.duplicated(["dataset","fold","arm"]).any()
df["retained"] = df.k / df.width
fields = ["auc","accuracy","retained","exceeds"]
df[fields] = df[fields].astype(float)
wide = df.set_index(["dataset","fold","arm"])[fields].unstack("arm")
summaries = []
for a,b in (("maximin","weak_gain"),("maximin_guard","weak_gain_guard"),("maximin_guard","vote_guard"),
            ("vote_guard","half_guard"),("vote_guard","same_width_guard"),("vote","same_width")):
    first,second = (wide.xs(arm,axis=1,level="arm") for arm in (a,b))
    assert len(first)==len(second)==41 and not (first.isna() | second.isna()).any().any()
    d = first - second
    by_task = d.groupby("dataset").mean()
    by_task.insert(0,"folds",d.groupby("dataset").size())
    by_task.insert(0,"contrast",a+" minus "+b)
    summaries.append(by_task.reset_index())
out = pd.concat(summaries,ignore_index=True)
out.to_csv(HERE/"refit-review-paired-means.csv",index=False)
print(out.groupby("contrast")[fields].mean().to_string())
