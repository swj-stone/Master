"""Locate the vowel accuracy losses in the locked five-seed predictions."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.anchored_vote import audit
from experiments.minpi_replication import direct_gain as native
from experiments.minpi_replication.second_panel import data


def describe(y,pred):
    return roc_auc_score(y,pred,multi_class="ovr"),accuracy_score(y,pred.argmax(axis=1))


def main():
    pairs=[]
    speakers=[]
    for s in data.SEEDS:
        x,y,task,idx=data.split("vowel",s)
        tr=idx["fit"]
        va=np.sort(np.r_[idx["vote"],idx["width"]])
        actual=data.truth("vowel",s)
        groups=data.group_ids("vowel")[idx["test"]]
        arms={}
        for method in ("min","exhaustive"):
            folder=audit.RUN/"prospective-r0"/f"vowel-{s}"/method
            row=json.loads((folder/"receipt.json").read_bytes())
            if audit.digest((folder/"forecast.npz").read_bytes())!=row["forecast"]:
                raise ValueError("Vowel forecast changed")
            rec=row["record"]
            cols=rec["selected"]
            fitted=native.fit_native(x[np.ix_(tr,cols)],y[tr],task,s,"cpu")
            validation=native.predict_native(fitted,x[np.ix_(va,cols)],task)
            inner_auc,inner_acc=describe(y[va],validation)
            np.testing.assert_allclose(inner_auc,rec["utility"],atol=1e-10,rtol=0)
            with np.load(folder/"forecast.npz") as ar:
                np.testing.assert_array_equal(ar["indices"],idx["test"])
                pred=ar["primary"]
            outer_auc,outer_acc=describe(actual,pred)
            arms[method]=(pred.argmax(axis=1)==actual)
            pairs.append(dict(seed=s,method=method,k=len(cols),anchor_k=len(rec.get("anchor",[])) or None,
                              validation_auc=inner_auc,validation_accuracy=inner_acc,
                              test_auc=outer_auc,test_accuracy=outer_acc))
        for group in np.unique(groups):
            loc=groups==group
            a,b=arms["min"][loc],arms["exhaustive"][loc]
            speakers.append(dict(seed=s,speaker=int(group),utterances=int(loc.sum()),
                                 lost_correct=int((~a&b).sum()),gained_correct=int((a&~b).sum()),
                                 accuracy_difference=float(a.mean()-b.mean())))
    pd.DataFrame(pairs).to_csv(audit.OUT/"vowel-selected.csv",index=False)
    pd.DataFrame(speakers).to_csv(audit.OUT/"vowel-speakers.csv",index=False)
    print(pd.DataFrame(pairs).round(5).to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
