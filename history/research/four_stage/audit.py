"""Recalculate every crossed-panel score; verify archived supports and training-only splits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,r2_score
from threadpoolctl import threadpool_limits
from research.four_stage.benchmark import DATASETS, SEEDS, MODELS, ARMS, ROOT, digest, load, select_features


def verify(folder):
    raw=pd.read_csv(folder/"outcomes.csv").set_index(["source","seed","model","arm"])
    state=json.loads((folder/"protocol.json").read_text())
    assert len(raw)==1080 and raw.index.is_unique
    for name,sha in state["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    count=0
    for source in DATASETS:
        X,y,task,groups=load(source)
        metric=balanced_accuracy_score if task=="classification" else r2_score
        for seed in SEEDS:
            with np.load(folder/f"{source}-{seed}.npz") as data:
                assert data["input_X_sha256"].item()==digest(X) and data["input_y_sha256"].item()==digest(y)
                tr,te=data["train"],data["test"]
                assert np.array_equal(np.sort(np.r_[tr,te]), np.arange(len(X)))
                if groups is not None:
                    assert not np.intersect1d(groups[tr],groups[te]).size
                np.testing.assert_array_equal(y[te], data["truth"])
                for arm in ["minpi_auto","minpi80"]:
                    budget=(4*X.shape[1]+4)//5 if arm=="minpi80" else None
                    selected,info=select_features(X[tr],y[tr],task=task,random_state=seed,count=budget)
                    assert not np.intersect1d(info["fit_rows"],info["vote_rows"]).size
                    np.testing.assert_array_equal(selected,data[arm])
                for model in MODELS:
                    for arm in ARMS:
                        chosen=data[arm]
                        assert np.array_equal(chosen,np.unique(chosen)) and min(chosen)>=0 and max(chosen)<X.shape[1]
                        row=raw.loc[(source,seed,model,arm)]
                        assert row.k==len(chosen) and 0<=row.select_ms<=row.total_ms
                        score=metric(y[te],data[model+"_"+arm+"_pred"])
                        np.testing.assert_allclose(score,row.score,rtol=0,atol=1e-12)
                        count+=1
    print(f"{count} scores checked; 60 candidate supports reproduced; 30 splits and source hashes verified.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    with threadpool_limits(1):verify(parser.parse_args().folder)
