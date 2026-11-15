"""Re-score fixed transport selections on feature vectors absent from development."""
from collections import Counter
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd


audit = runpy.run_path(str(Path(__file__).with_name("transport-forensic.py")))
scores, passes, root = (audit[k] for k in ("scores","passes","RUN"))


def row_keys(x):
    return [tuple(None if np.isnan(v) else float(v) for v in row) for row in x]


def compare():
    records, shifts = [], []
    total = Counter()
    for sealed in sorted((root/"low-fidelity").glob("*/fold-*/sealed.json")):
        rec = json.loads(sealed.read_bytes())
        task, fold = rec["dataset"],rec["fold"]
        parent = root/"transport-uci"/task/f"fold-{fold}"
        primary = json.loads((parent/"selection.json").read_bytes())
        s = rec["supports"]
        uwidth = 1 - len(s[-1])/len(s[0])
        with np.load(parent/"inputs.npz") as source, np.load(parent/"targets.npz") as truth:
            dev = set(row_keys(source["x"]))
            keep = np.array([r not in dev for r in row_keys(source["x_outer"])])
            y = truth["y"]
            guard_y = source["y"][source["guard"]]
        assert len(np.unique(y[keep])) == len(np.unique(y)) and keep.sum()>1
        total[task]+=len(y)-sum(keep)
        with np.load(sealed.parent/"predictions.npz") as proba:
            q = np.array([scores(y,proba[f"o{i}"]) for i in range(len(s))])
            qu = np.array([scores(y[keep],proba[f"o{i}"][keep]) for i in range(len(s))])
            guard = passes(np.array([scores(guard_y,proba[f"g{i}"]) for i in range(len(s))]))
        o, ou = passes(q),passes(qu)
        nominees = {m:s.index(choice["nominee"]) for m,choice in primary["choices"].items()}
        nominees["small"] = rec["low_fidelity"]
        for method,i in nominees.items():
            final = i if guard[i] else 0
            if o[final] != ou[final]:
                shifts.append((task,fold,method,bool(o[final]),bool(ou[final])))
            records.append(dict(method=method,task=task,fold=fold,
                                u=uwidth*bool(final and ou[final]),breach=float(not ou[final]),
                                outer=uwidth*any(ou[1:]),joint=uwidth*any((ou&guard)[1:])))
    print("Removed raw-feature duplicates relative to development by task",dict(total))
    print("Changed measured decisions (old, restricted)",shifts)
    means=pd.DataFrame(records).groupby(["method","task"])[["u","breach","outer","joint"]].mean()
    print(means.groupby("method").mean().to_string())


if __name__=="__main__":
    compare()
