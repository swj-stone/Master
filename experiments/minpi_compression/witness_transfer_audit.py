"""Independently verify observed negative edges and guards on the new supplied groups."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import witness_transfer as study
from experiments.minpi_compression.witness_guard_audit import svd_predict
from minppi_mofs.dictator_minpi import rank_bounds


def audit():
    b=study.b
    counts,guard_checks=[],[]
    receipts=json.loads((study.EVIDENCE/"lock.json").read_bytes())
    assert len(receipts)==30
    for key,digest in receipts.items():
        receipt=study.OUT/key/"selection.json"
        assert b.digest(receipt)==digest
        state=json.loads(receipt.read_bytes())
        name,seed=state["dataset"],state["seed"]
        x,y,task,idx=study.data.split(name,seed)
        groups=study.data.load(name)[3]
        input_ids=np.unique(x,axis=0,return_inverse=True)[1]
        for group in (groups,input_ids):
            parts=[set(group[ix]) for ix in idx.values()]
            assert sum(map(len,parts))==len(set.union(*parts))
        if name=="estate":
            for left,right in (("fit","vote"),("vote","width"),("width","test")):
                assert np.max(groups[idx[left]])<np.min(groups[idx[right]])
        records={r["method"]:r for r in state["rows"] if not r["repeat"]}
        w=records["witness"]
        fit,vote=idx["fit"],idx["vote"]
        prep=b.transform(x,fit)
        rank=rank_bounds(prep.transform(x[fit]),y[fit],prep.transform(x[vote]),y[vote],
                         task=task,seed=seed,closure=True,**b.DESIGN["engine"])
        neg=set(np.flatnonzero(rank.minimum==-1))
        assert set(w["negative"])==neg.intersection(w["prefix"])
        if w["h"]:
            truth=b.targets(y[vote],task,len(np.unique(y[fit])))
            def loss(cols):
                return np.square(truth-svd_predict(x,y,task,fit,vote,cols)).sum()/len(vote)
            base_loss=loss([]) or 1.
            quota=.95/(1.+loss(np.arange(x.shape[1]))/base_loss)
            for i in w["negative"]:
                left=list(rank.witness[i])
                assert i not in left and 1./(1.+loss(left)/base_loss)>=quota>1./(1.+loss(sorted(left+[i]))/base_loss)
        counts.append(dict(dataset=name,seed=seed,all_witnessed=len(neg),inside=w["h"]))
        if not w["optional_refits"]:
            continue
        train=np.sort(np.r_[fit,vote])
        width=idx["width"]
        def validate(cols):
            return b.metrics(y[width],svd_predict(x,y,task,train,width,cols),task,max(y[train].std(),1e-12))
        full=validate(np.arange(x.shape[1]))
        for method in ("witness","tail","uniform"):
            rec=records[method]
            retained=sorted(set(w["prefix"])-set(rec["removed"]))
            check=validate(retained)
            accepted=b.eligible(check,full,task)
            assert accepted==rec["accepted"] and len(rec["removed"])==w["h"]
            assert rec["columns"]==(retained if accepted else sorted(w["prefix"]))
            for metric,v in check.items():
                np.testing.assert_allclose(v,rec["optional"][metric],atol=1e-10)
            guard_checks.append(dict(dataset=name,seed=seed,method=method,accepted=accepted,
                                     same_as_witness=set(w["removed"])==set(rec["removed"]),**check))
    pd.DataFrame(counts).to_csv(study.EVIDENCE/"negative-counts.csv",index=False)
    pd.DataFrame(guard_checks).to_csv(study.EVIDENCE/"guard-audit.csv",index=False)
    print(f"Verified {sum(r['inside'] for r in counts)} within-prefix negative indices, "
          f"{len(guard_checks)} optional fits, no overlapping feature vectors or held-out groups")


if __name__=="__main__":
    with threadpool_limits(limits=1):
        audit()
