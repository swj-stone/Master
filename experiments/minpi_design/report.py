"""Check development result completeness and compare matched searches by source."""
import numpy as np
import pandas as pd

from experiments.minpi_compression.benchmark import eligible
from experiments.minpi_design.study import OUT


def paired(frame,left,right):
    result = []
    for name, group in frame.groupby("dataset",sort=True):
        arm = group[(group["rank"]==left[0]) & (group.search==left[1])].iloc[0]
        base = group[(group["rank"]==right[0]) & (group.search==right[1])].iloc[0]
        result.append(dict(dataset=name,candidate="/".join(left),control="/".join(right),
                           k=arm.k,control_k=base.k,fits=arm.fits,control_fits=base.fits,
                           sec=arm.elapsed_s,control_sec=base.elapsed_s,ratio=arm.elapsed_s/base.elapsed_s,
                           auc=arm.auc,control_auc=base.auc,accuracy=arm.accuracy,
                           control_accuracy=base.accuracy,r2=arm.r2,control_r2=base.r2,
                           mae=arm.mae,control_mae=base.mae,
                           score_pass=eligible(arm,base,arm.task),
                           joint=(arm.k<=base.k and arm.elapsed_s<base.elapsed_s
                                  and eligible(arm,base,arm.task))))
    return result


def main():
    native = pd.read_csv(OUT/"native-six-pilot.csv")
    paths = pd.read_csv(OUT/"path-six-pilot.csv")
    if len(native)!=6*7 or len(paths)!=6*9:
        raise ValueError("Incomplete native downstream results")
    results=[]
    for candidate in [("min","direct"),("min","bisect")]:
        for control in [("gain","direct"),("gain","bisect"),("magnitude","bisect"),("pooled","bisect")]:
            results += paired(paths,candidate,control)
    pairs=pd.DataFrame(results)
    pairs.to_csv(OUT/"path-paired.csv",index=False)
    archive=OUT.parents[2]/"docs/thesis/archives/piv1/pfi-paired.csv"
    old=pd.read_csv(archive).rename(columns={"gain_retained":"k_base","gain_wall_s":"seconds_base",
                                             "gain_test_primary":"primary_base"})
    old=old[old.seed==42][["dataset","k_base","seconds_base","primary_base"]]
    native_min=paths[(paths["rank"]=="min")&(paths.search=="bisect")]
    scoped=native_min.merge(old,on="dataset",validate="one_to_one")
    assert len(scoped)==6
    scoped[["dataset","k","elapsed_s","auc","r2","k_base","seconds_base","primary_base"]].to_csv(
        OUT/"archived-reference.csv",index=False)
    print(pairs.groupby(["candidate","control"]).agg(joint=("joint","sum"),
            median_time_ratio=("ratio","median")).round(3).to_string())
    heads=pd.concat([pd.read_csv(OUT/(n+"-pilot.csv")) for n in ("prepared","seven")])
    assert len(heads) == 13*2*2*9
    for basis in ("linear","spline"):
        game=heads[(heads.basis==basis)&(heads["rank"]=="min")&(heads["head"]=="robust")]
        print(basis, "total exact / all candidates",game.exact.sum(),game.possible.sum())
    for col in ("k","auc","accuracy","r2","mae"):
        assert np.isfinite(paths[col].dropna()).all()


if __name__=="__main__":
    main()
