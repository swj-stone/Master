"""Summarize exact solver time and compiled cost sensitivity by whole sources."""
import argparse
import numpy as np
import pandas as pd
from experiments.minpi_rework import quota_compress_tree_confirmation as conf


def main(path):
    f=pd.read_csv(path)
    component=f.groupby(["panel","dataset","seed","method"]).seconds.median().unstack()
    raw=pd.read_csv(conf.DEST/"scores.csv")
    base=raw[raw.method=="directions"].set_index(["panel","dataset","seed"])
    shared=base.pipeline_s-base.search_s
    cost=pd.read_csv(conf.DEST/"native-costs.csv").groupby(["panel","dataset","seed"]).median(numeric_only=True)
    search=component.assign(compiled_gain=cost.compiled_s)
    pipeline=component.add(shared,axis=0).assign(compiled_gain=cost.matched_gain_pipeline_s,native_full=cost.native_full_pipeline_s)
    summary=[]
    for scope,values in (("search",search),("pipeline",pipeline)):
        for other in values:
            for panel in ("core","wide","all"):
                ratio=values.four/values[other]
                if panel!="all":
                    ratio=ratio.xs(panel,level="panel")
                log=np.log(ratio).groupby("dataset").mean().to_numpy()
                rng=np.random.default_rng(5002)
                samples=rng.choice(log,(20000,len(log)),replace=True).mean(axis=1)
                centre,lo,hi=np.exp([log.mean(),*np.quantile(samples,[.025,.975])])
                summary.append(dict(scope=scope,reference=other,panel=panel,value=centre,low=lo,high=hi))
    result=pd.DataFrame(summary)
    result.to_csv(path.with_name(path.stem+"-contrasts.csv"),index=False)
    print(result[result.panel=="all"].to_string(index=False))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--partial",action="store_true")
    main(conf.DEST/("compiled-exact-partial.csv" if cli.parse_args().partial else "compiled-exact.csv"))
