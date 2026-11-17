"""Source-equal summaries of matched analytic/Monte Carlo comparisons."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.minpi_rework.availability_trial import DEST


def summary(frame):
    records=[]
    keys=["dataset","seed","draw"]
    exact=frame[frame.method=="exact"].set_index(keys).sort_index()
    for m in frame.method.unique():
        if m=="exact":
            continue
        other=frame[frame.method==m].set_index(keys).sort_index()
        if not exact.index.equals(other.index):
            raise ValueError(f"Unpaired jobs for {m}")
        statistics={
            "selector_time_ratio": np.log(exact.selector_s/other.selector_s),
            "search_time_ratio": np.log(exact.search_s/other.search_s),
            "training_inclusive_ratio": np.log(exact.pipeline_s/other.pipeline_s),
            "additional_columns": other.k-exact.k, "identical_support":other.exact_support,
            "identical_path":other.exact_path, "quota_violation":other.true_violation,
            "jaccard":other.jaccard, "fixed_curve_error":other.fixed_support_error,
            "adaptive_curve_error":other.max_curve_error}
        if "outer_delta_worst" in other:
            statistics["outer_delta_worst"]=other.outer_delta_worst
        for metric,values in statistics.items():
            per_source=values.groupby(level="dataset").mean()
            rng=np.random.default_rng(171126)
            draws=rng.choice(per_source.to_numpy(),size=(20000,len(per_source)),replace=True).mean(axis=1)
            est=np.r_[per_source.mean(),np.quantile(draws,[.025,.975])]
            if metric.endswith("_ratio"):
                est=np.exp(est)
            records.append(dict(method=m,metric=metric,sources=len(per_source),jobs=len(values),
                                estimate=est[0],lower=est[1],upper=est[2]))
    return pd.DataFrame(records)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("csv",nargs="?",type=Path,default=DEST/"matched-mc-seed42.csv")
    input_path=parser.parse_args().csv
    out=summary(pd.read_csv(input_path))
    out.to_csv(input_path.with_stem(input_path.stem+"-contrasts"),index=False)
    print(out[out.metric=="selector_time_ratio"].to_string(index=False))
