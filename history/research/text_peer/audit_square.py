"""Check fresh processes, reconstruct paired totals and signed differences on bydate."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from research.text_peer.bench import original,HERE


def main():
    root=original.ROOT/"results/minpi-study"
    paths=[root/("text-square-bydate-20261118"+i) for i in ("a","b")]
    quality=root/"text-square-bydate-quality-20261118"
    raw=pd.concat([pd.read_csv(path/"cases.csv") for path in paths])
    assert len(raw)==220 and raw.groupby(["dataset","method","run"]).size().eq(5).all()
    counts=0
    for task in ("bydate_computers","bydate_sports"):
        with np.load(quality/(task+".npz")) as orig,np.load(paths[0]/(task+".npz")) as a,np.load(paths[1]/(task+".npz")) as b:
            assert set(a.files)==set(b.files)==set(orig.files)
            for field in a:
                np.testing.assert_array_equal(a[field],b[field]);np.testing.assert_array_equal(a[field],orig[field]);counts+=1
            for method in raw.method.unique():
                part=raw[(raw.method==method)&(raw.dataset==task)]
                assert (abs(part.ba-balanced_accuracy_score(a["gold"],a[method+"_pred"]))<1e-13).all()
    byrun=raw.groupby(["dataset","method","run"]).agg(total_ms=("total_ms","median"),ba=("ba","first"),width=("width","first"),p=("p","first")).reset_index()
    out=[]
    for (dataset,run),g in byrun.groupby(["dataset","run"]):
        g=g.set_index("method").copy();candidate=g.loc["square"]
        g["candidate_delta"]=candidate.ba-g.ba;g["candidate_ratio"]=candidate.total_ms/g.total_ms
        g["candidate_own_joint"]=(candidate.width<candidate.p and candidate.ba>=g.loc["full","ba"]-.02 and candidate.total_ms<g.loc["full","total_ms"])
        out.append(g.reset_index())
    pairs=pd.concat(out)
    summary=pairs.groupby(["dataset","method"]).agg(width=("width","first"),ba=("ba","first"),max_median_ms=("total_ms","max"),delta=("candidate_delta","first"),max_paired_ratio=("candidate_ratio","max"),candidate_joint=("candidate_own_joint","all"))
    summary.to_csv(HERE/"square-bydate-summary.csv")
    print(counts,"identical arrays on two processes and the non-timed quality run")
    print(summary.round(4).to_string())
if __name__=="__main__": main()
