"""Paired timing contrasts, with a complete four-source cluster bootstrap distribution."""
from itertools import product

import numpy as np
import pandas as pd
from experiments.minpi_rework.predictive_polynomial_timing import OUT, wide


def summarize():
    rows = pd.read_csv(OUT/"repeats.csv")
    groups = ["dataset","seed","method"]
    assert rows.groupby(groups).size().eq(3).all()
    assert rows.groupby(["dataset","seed"]).path_sha256.nunique().eq(1).all()
    assert len(rows.groupby(["dataset","seed"])) == 20
    med = rows.groupby(groups).median(numeric_only=True)
    rows = []
    for (name,seed),sub in med.groupby(level=[0,1]):
        sub = sub.droplevel([0,1])
        other = sub.drop(index="interval")
        best = other.rank_s.idxmin()
        exhaustive = other.drop(index="coalesced_early").rank_s.idxmin()
        arm,baseline,exh = sub.loc["interval"],sub.loc[best],sub.loc[exhaustive]
        rows.append(dict(dataset=name,seed=seed,p=int(arm.p),baseline=best,exhaustive=exhaustive,
                         interval_ms=1000*arm.rank_s,baseline_ms=1000*baseline.rank_s,exhaustive_ms=1000*exh.rank_s,
                         early_ms=1000*sub.at["coalesced_early","rank_s"],blas_ms=1000*sub.at["coalesced_blas","rank_s"],
                         straight_ms=1000*sub.at["coalesced_straight","rank_s"],common_s=arm.common_s,
                         rank_ratio=arm.rank_s/baseline.rank_s,exhaustive_ratio=arm.rank_s/exh.rank_s,
                         total_ratio=(arm.common_s+arm.rank_s)/(arm.common_s+baseline.rank_s),
                         checks_ratio=arm.checks/baseline.checks,interval_checks=int(arm.checks),
                         interval_bounds=int(arm.intervals)))
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT/"paired.csv",index=False)
    ratios = ["rank_ratio","exhaustive_ratio","total_ratio","checks_ratio"]
    means = np.log(paired[ratios]).groupby(paired.dataset).mean()
    draws = np.array([means.to_numpy()[list(ix)].mean(axis=0) for ix in product(range(4), repeat=4)])
    bounds = np.exp(np.quantile(draws,[.025,.975],axis=0))
    overall = pd.DataFrame(dict(measure=ratios,ratio=np.exp(means.mean()).to_numpy(),l95=bounds[0],u95=bounds[1]))
    overall.to_csv(OUT/"contrasts.csv",index=False)
    by_source = paired.groupby("dataset")[["p","interval_ms","baseline_ms","exhaustive_ms","early_ms","blas_ms",
                                            "straight_ms","common_s","interval_checks","interval_bounds"]].median()
    by_source = by_source.join(np.exp(means))
    by_source.to_csv(OUT/"sources.csv")
    registered = pd.read_csv(wide.OUT/"paired-jobs.csv")
    assert len(registered)==20
    print(by_source.to_string(float_format=lambda n: f"{n:.4f}"))
    print(overall.to_string(index=False,float_format=lambda n:f"{n:.5f}"))


if __name__ == "__main__":
    summarize()
