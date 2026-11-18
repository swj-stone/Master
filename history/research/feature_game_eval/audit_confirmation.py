"""Recompute the 64 scores, latent coverage and F ranking independently from arrays."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score
from threadpoolctl import threadpool_limits
from research.feature_game_eval.confirmation import ARMS,SHAPES,panel
from research.feature_game_eval.audit_controls import effect_size


def audit(folder):
    saved=pd.read_csv(folder/"cases.csv").set_index(["mechanism","task","rows","p","replicate","method"])
    maximum=0.
    for mech in ("proxy","diffuse"):
        for task in ("classification","regression"):
            for shape,(n,p) in enumerate(SHAPES):
                for rep in range(1,9):
                    data=panel(mech,task,shape,rep)
                    eta=effect_size(data.X,data.y,task)
                    with np.load(folder/f"{mech}-{task}-{shape}-{rep}.npz") as z:
                        np.testing.assert_array_equal(z["truth"],data.truth)
                        np.testing.assert_array_equal(z["group_ids"],data.group_ids)
                        cols=z["F_cols"]
                        other=np.setdiff1d(np.arange(p),cols)
                        if len(other):
                            maximum=max(maximum,float(eta[other].max()-eta[cols].min()))
                        for arm in ARMS:
                            expected=saved.loc[(mech,task,n,p,rep,arm)]
                            metric=balanced_accuracy_score if task=="classification" else r2_score
                            assert abs(metric(data.truth,z[arm+"_pred"])-expected.score)<1e-12
                            if mech=="proxy":
                                ids=np.unique(data.group_ids[z[arm+"_cols"]])
                                assert (ids>=0).sum()==int(expected.latent_groups)
    assert len(saved)==64*7 and maximum<1e-11
    print(f"Verified all 448 scores, 224 group coverage counts, 64 F supports; maximum effect-size inversion {maximum}.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass_one",type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        audit(args.pass_one)
