"""Complete both refit-guard studies, recording folds with insufficient ranking classes.

The frozen worker sources retain the original implementation hashes. Their
initial sequential driver stopped at the first unavailable split. This entry
checks eligibility before dispatch and continues through the named cohort.
"""
import argparse
import csv
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.context_review import challenger, refit_guard
from experiments.context_study.benchmark import store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest


def eligibility(name, fold):
    path = PRIMARY/name/f"fold-{fold}"/"selection.json"
    rec = json.loads(path.read_bytes())
    rank = np.asarray(rec["rank"])
    y = pd.read_csv(DATA/name/f"{name}.tsv.gz",sep="\t",usecols=["target"]).target.to_numpy()
    counts = dict(zip(*np.unique(y[rank],return_counts=True),strict=True))
    if min(counts.values())<2:
        return False, "A ranking class occurs only once"
    left,right = train_test_split(rank,test_size=.5,stratify=y[rank],random_state=39823+fold)
    if set(y[left])!=set(counts) or set(y[right])!=set(counts):
        return False, "A proposed half lacks an outer-training class"
    return True,""


def run_stage(module, eligible):
    out = module.OUT
    out.mkdir(exist_ok=True,parents=True)
    script = ROOT/"experiments/context_review"/(module.__name__.rsplit(".",1)[-1]+".py")
    if module is refit_guard:
        paths = [module.DESIGN,script,PRIMARY/"protocol-lock.json",ROOT/"experiments/context_review/refit_target.py"]
    else:
        paths = [module.DESIGN,script,refit_guard.OUT/"protocol.json",ROOT/"src/minppi_mofs/maximin_challenger.py"]
    lock = {p.name:digest(p) for p in paths}
    protocol = out/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_bytes())!=lock:
        raise ValueError("Changed original study lock")
    store(protocol,lock)
    rows, costs = [], []
    for entry in eligible:
        name, fold = entry["dataset"],entry["fold"]
        folder = out/name/f"fold-{fold}"
        if not entry["eligible"]:
            if (folder/"completed.json").exists():
                raise ValueError(f"Published completed result for ineligible fold {folder}")
            continue
        if not (folder/"completed.json").exists():
            subprocess.run([sys.executable,"-m",module.__name__,"--worker",name,str(fold)],
                           check=True,timeout=3600)
        rec = json.loads((folder/"completed.json").read_bytes())
        assert rec["protocol"]==digest(protocol) and rec["predictions"]==digest(folder/"predictions.npz")
        assert rec["selection"]==digest(folder/"selection.json")
        rows.extend(rec["rows"])
        if module is challenger:
            nominee=json.loads((folder/"proposals.json").read_bytes())
            costs.append(dict(dataset=name,fold=fold,masks=nominee["masks"],eager_masks=nominee["eager_masks"],
                              visits=nominee["counts"]["visits"],eager_visits=nominee["eager"]["visits"]))
    df = pd.DataFrame(rows)
    df["retained"] = df.k/df.width
    df.to_csv(out/"per-fold.csv",index=False)
    cols=["auc","accuracy","retained","exceeds"]+(["reduced","different"] if costs else ["fallback"])
    by_task=df.groupby(["dataset","arm"])[cols].mean()
    by_task.to_csv(out/"by-dataset.csv")
    by_task.groupby("arm").mean().to_csv(out/"summary.csv")
    full={name for name in config()["datasets"] if all(e["eligible"] for e in eligible if e["dataset"]==name)}
    by_task.loc[by_task.index.get_level_values("dataset").isin(full)].groupby("arm").mean().to_csv(out/"complete-tasks.csv")
    if costs:
        pd.DataFrame(costs).to_csv(out/"costs.csv",index=False)
    print(f"{module.__name__}: {len(df)} arm-fold records, {len(full)} fully covered tasks",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["guard","challenger","both"],default="both")
    args=parser.parse_args()
    eligible=[]
    for name in config()["datasets"]:
        for fold in range(3):
            ok, why=eligibility(name,fold)
            eligible.append(dict(dataset=name,fold=fold,eligible=ok,reason=why))
    stages={"guard":(refit_guard,), "challenger":(challenger,), "both":(refit_guard,challenger)}
    for module in stages[args.stage]:
        module.OUT.mkdir(parents=True,exist_ok=True)
        with (module.OUT/"eligibility.csv").open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,list(eligible[0]))
            writer.writeheader()
            writer.writerows(eligible)
        run_stage(module,eligible)


if __name__=="__main__":
    main()
