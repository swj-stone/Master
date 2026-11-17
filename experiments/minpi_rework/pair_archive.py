"""Export development rows after verifying the fixed-width exchange replay."""
import gzip
import numpy as np
import pandas as pd

from experiments.minpi_compression import benchmark as b

RUNS={
    "pair-uci-spline":54,"pair-six-spline":6,"pair-robust-uci":54,"pair-robust-six":5,
    "pair-outage-uci":90,"pair-outage-six":30,"pair-frozen-label":90,"pair-frozen-fidelity":90,
    "pair-swap-uci":90,"pair-swap-six":30,"pair-adaptive-uci":90,"pair-adaptive-six":30,
}


def export():
    root=b.ROOT/"results/minpi_rework"
    frames=[]
    for name,count in RUNS.items():
        rows=pd.read_csv(root/name/"scores.csv")
        assert len(rows[["dataset","seed"]].drop_duplicates())==count,name
        keys=["dataset","seed","method"]+(["choice"] if "choice" in rows else [])
        assert not rows.duplicated(keys).any()
        if name.startswith("pair-swap-"):
            rerun=pd.read_csv(root/name.replace("swap-", "swap-replay-")/"scores.csv")
            a,c=rows.set_index(keys).sort_index(),rerun.set_index(keys).sort_index()
            pd.testing.assert_index_equal(a.index,c.index)
            for field in ("p","k","auc","accuracy","r2","scaled_mae","intact","worst"):
                np.testing.assert_allclose(a[field],c[field],atol=1e-10,rtol=1e-10,equal_nan=True)
            a["selected"]=c["selected"]
            rows=a.reset_index()
        frames.append(rows.assign(study=name))
    result=pd.concat(frames,ignore_index=True)
    path=b.ROOT/"experiments/minpi_rework/pair_evidence.csv.gz"
    with path.open("wb") as raw,gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as f:
        f.write(result.to_csv(index=False,lineterminator="\n").encode("utf8"))
    print(f"{len(result)} rows, {len(RUNS)} studies, 120 matched model-exchange split replays")


if __name__=="__main__":
    export()
