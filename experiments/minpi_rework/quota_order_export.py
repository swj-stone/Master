"""Archive the matched development comparisons without reading outer labels."""
import pandas as pd
from experiments.minpi_compression import benchmark as b


def run():
    frames=[]
    for trial,expected in (("quota-order-three",84),("quota-order-width-specific",56),("quota-order-fraud-clocks",42)):
        frame=pd.read_csv(b.ROOT/"results/minpi_rework"/trial/"scores.csv")
        assert len(frame)==expected
        assert (frame.lb<=frame.k).all() and (frame.k==frame.initial_upper).all()
        assert (frame.certified==(frame.k==frame.lb)).all()
        assert (frame.relative_error<=.20+1e-10).all()
        frames.append(frame.assign(trial=trial))
    combined=pd.concat(frames,ignore_index=True)
    assert combined.groupby("dataset").support.nunique().max()==1
    combined.to_csv(b.ROOT/"experiments/minpi_rework/quota_order_evidence.csv",index=False)
    print(f"Archived {len(combined)} matched solver outcomes across {combined.dataset.nunique()} sources")


if __name__=="__main__": run()
