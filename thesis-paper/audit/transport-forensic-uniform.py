"""Exact uniform-nomination expectation on each recorded UCI slate, conditional on outcomes."""
from fractions import Fraction
from pathlib import Path
import runpy

import numpy as np
import pandas as pd


base = runpy.run_path(str(Path(__file__).with_name("transport-forensic.py")))
RUN, EVIDENCE, scores, passes = (base[k] for k in ("RUN","EVIDENCE","scores","passes"))


def main():
    rows=[]
    for file in sorted((RUN/"low-fidelity").glob("*/fold-*/sealed.json")):
        rec=base["read"](file)
        task, fold = rec["dataset"],rec["fold"]
        src=RUN/"transport-uci"/task/f"fold-{fold}"
        with np.load(src/"inputs.npz") as data, np.load(src/"targets.npz") as test:
            yg=data["y"][data["guard"]]
            yo=test["y"]
        with np.load(file.parent/"predictions.npz") as pred:
            supports=rec["supports"]
            g=passes(np.array([scores(yg,pred[f"g{i}"]) for i in range(len(supports))]))
            o=passes(np.array([scores(yo,pred[f"o{i}"]) for i in range(len(supports))]))
            k=len(supports[0])
            d=Fraction(k-(k+1)//2,k)
            m=len(supports)-1
            u=d*Fraction(int(np.sum((g&o)[1:])),m) if m else Fraction(0)
            breach=Fraction(int(np.sum((g&~o)[1:])),m) if m else Fraction(0)
            accepted=Fraction(int(g[1:].sum()),m) if m else Fraction(0)
            rows.append(dict(task=task,fold=fold,u=u,breach=breach,accepted=accepted))
    for metric in ("u","breach","accepted"):
        value=sum(r[metric] for r in rows)/30
        print("uniform",metric,value,float(value))
    # All ten tasks have exactly three folds, so the raw mean is the equal-task mean.
    uf=pd.DataFrame(rows)
    uf["u"]=uf.u.astype(float)
    uniform=uf.groupby("task").u.mean()
    actual=pd.read_csv(EVIDENCE/"low_fidelity/folds.csv").groupby(["method","dataset"]).u.mean()
    for method in ["point","slack","vote","small","gain"]:
        x=(actual.loc[method]-uniform).to_numpy()
        idx=np.random.default_rng(33119).integers(len(x),size=(30000,len(x)))
        print(method,"minus uniform",x.mean(),np.percentile(x[idx].mean(axis=1),[2.5,97.5]))


if __name__=="__main__":
    main()
