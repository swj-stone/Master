"""Evaluate every feasible nomination and the exact uniform lottery on the two external panels."""
import argparse
import numpy as np
import pandas as pd
from experiments.context_review import budgeted_guard as bg
from experiments.context_review.ballot_refinement import settings
from experiments.context_review.refit_opportunity import outcome

ROOT=bg.ROOT
EVIDENCE=ROOT/"thesis-paper/evidence"


def records(panel):
    src,refined=settings(panel)
    name="low-fidelity" if panel=="first" else "wide-slate"
    complete=ROOT/"results/context_review"/name
    manifest=EVIDENCE/("transport/low-fidelity-lock.json" if panel=="first" else "wide_slate/prediction-lock.json")
    seals=bg.read(manifest)
    factors=bg.read(EVIDENCE/"margin_factorial"/f"{panel}-lock.json")
    cells={s["key"]:s for s in factors["folds"]}
    refinement=bg.read(EVIDENCE/"refinement"/f"{panel}-lock.json")["selections"]
    assert set(cells)==set(seals)==set(refinement)
    for key,sha in seals.items():
        f=complete/key
        assert bg.digest(f/"sealed.json")==sha and bg.read(f/"sealed.json")["predictions"]==bg.digest(f/"predictions.npz")
        assert cells[key]["selection_hash"]==bg.digest(src/key/"selection.json")
        assert bg.digest(refined/key/"sealed.json")==refinement[key]
    rows=[]
    for key in sorted(seals):
        f=complete/key
        r=bg.read(f/"sealed.json")
        pool=r["supports"]
        base=bg.read(src/key/"selection.json")
        g=np.array([outcome(m,r["guard_metrics"][0],.03) for m in r["guard_metrics"]])
        nominees={arm:pool.index(c["nominee"]) for arm,c in base["choices"].items()}
        nominees["small"]=r["low_fidelity"]
        nominees["refined"]=pool.index(bg.read(refined/key/"sealed.json")["nominee"])
        nominees.update({arm:pool.index(c) for arm,c in cells[key]["winners"].items()})
        assert nominees["slack"]==nominees["family4"] and nominees["point"]==nominees["empty1"]
        with np.load(f/"predictions.npz") as arrays,np.load(src/key/"targets.npz") as truth:
            assert bg.digest(src/key/"layout.json")==r["layout"]
            assert bg.digest(src/key/"targets.npz")==bg.read(src/key/"layout.json")["targets_sha256"]
            y=truth["y"]
            scores=[bg.metrics(y,arrays[f"o{i}"],r["rare"]) for i in range(len(pool))]
            with np.load(refined/key/"prediction.npz") as check:
                i=nominees["refined"]
                np.testing.assert_array_equal(check["p"],arrays[f"o{i if g[i] else 0}"])
        for tau in (.01,.02,.03,.04,.05):
            o=np.array([outcome(m,scores[0],tau) for m in scores])
            final=np.where(g,np.arange(len(pool)),0)
            saved=np.array([1-len(c)/len(pool[0]) for c in pool])
            vals=pd.DataFrame(dict(u=saved[final]*o[final],breach=(~o[final]).astype(int),
                     auc=[scores[i]["auc"] for i in final],accuracy=[scores[i]["accuracy"] for i in final],
                     saved=saved[final],accept=(final!=0).astype(int)))
            # At a singleton anchor the common no-action policy is the sole option.
            lottery=vals.iloc[1:].mean() if len(vals)>1 else vals.iloc[0]
            evaluations={arm:vals.iloc[i] for arm,i in nominees.items()}
            evaluations["uniform"]=lottery
            # An oracle ceiling identifies maximum attainable utility. Several
            # nominees can attain it with different risks and predictive scores.
            evaluations["hindsight"] = pd.Series({"u": vals.u.max()})
            for arm,v in evaluations.items():
                rows.append(dict(dataset=r["dataset"],fold=r["fold"],method=arm,tolerance=tau,candidates=len(pool)-1,
                          outer_ceiling=max(saved*o),joint_ceiling=max(vals.u),**v.to_dict()))
    return pd.DataFrame(rows)


def report(panel):
    df=records(panel)
    dest=EVIDENCE/"chance"
    dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/(panel+"-folds.csv"),index=False)
    fields=["u","breach","auc","accuracy","saved","accept","outer_ceiling","joint_ceiling"]
    tasks=df.groupby(["tolerance","method","dataset"])[fields].mean()
    means=tasks.groupby(["tolerance","method"]).mean()
    means.to_csv(dest/(panel+"-means.csv"))
    tasks.to_csv(dest/(panel+"-tasks.csv"))
    rows=[]
    # The interaction compares the change from four panels at all contexts to
    # the change from four panels at the empty context.
    primary=tasks.loc[.03]
    comparisons={arm+"_minus_uniform":primary.loc[arm]-primary.loc["uniform"]
                 for arm in ("point","slack","vote","gain","small","refined","family1","empty4")}
    for arm,ref in (("empty4","empty1"),("family1","empty1"),("family4","family1"),("family4","empty4")):
        comparisons[arm+"_minus_"+ref]=primary.loc[arm]-primary.loc[ref]
    comparisons["panel_by_family"]=(primary.loc["family4"]-primary.loc["family1"]
                                    -primary.loc["empty4"]+primary.loc["empty1"])
    for name,delta in comparisons.items():
        n=len(delta)
        indices=np.random.default_rng(33119).integers(n,size=(30000,n))
        for field in ("u","breach","auc","accuracy"):
            values=delta[field].to_numpy()
            lo,hi=np.percentile(values[indices].mean(axis=1),[2.5,97.5])
            rows.append(dict(comparison=name,metric=field,mean=values.mean(),lo95=lo,hi95=hi,
                             positive=(values>1e-14).sum(),negative=(values< -1e-14).sum(),n_tasks=n))
    pd.DataFrame(rows).to_csv(dest/(panel+"-contrasts.csv"),index=False)
    print(means.loc[.03].round(4).to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("first","wide"))
    report(parser.parse_args().panel)
