"""Norm-scale Gain-prefix correction and outcome-blind audit of signed forecasts."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import quota_compress_tree as tree
from experiments.minpi_rework import quota_compress_tree_confirmation as conf
from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from minppi_mofs.modeling import make_model


def gain_prefix(f,x,order,rho=.2):
    """Return the shortest prefix passing the same norm-scale guard as exact greedy."""
    ranks=np.argsort(order)+1
    inc=tree._prefix(np.asarray(x,dtype=np.float32),*f.buffers(),ranks,f.c)
    residual=inc.sum(axis=1,keepdims=True)-np.cumsum(inc,axis=1)
    norm=np.linalg.norm(residual,axis=0)
    limit=rho*norm[0]+1e-12*max(1.,norm[0])
    return np.sort(order[:int(np.flatnonzero(norm<=limit)[0])])


def small_stump_case():
    left=np.array([[1,-1,-1]],dtype=np.int32)
    right=np.array([[2,-1,-1]],dtype=np.int32)
    f=tree.Forest(left,right,np.zeros_like(left),np.zeros((1,3),np.float32),
        np.ones((1,3),bool),np.array([[0.,-1e-8,1e-8]]),np.zeros(1,int),
        np.array([[-1,0,0]],np.int32),[(0,)],1,1,1,np.zeros(1))
    x=np.array([[-1.],[1.]])
    cols=np.arange(1)
    assert len(f.gain_prefix(x,cols,.2)[0])==0
    np.testing.assert_array_equal(gain_prefix(f,x,cols),cols)
    np.testing.assert_array_equal(f.cache(x).search("exact_greedy",cols)[0],cols)


def audit():
    small_stump_case()
    lock=json.loads((conf.DEST/"lock.json").read_bytes())
    assert conf.signatures()==lock["source"]
    rows=[]
    for panel,source,name,seed in conf.jobs():
        key=f"{name}-{seed}"
        sub=conf.OUT/key
        assert b.digest(sub/"selection.json")==lock["receipts"][key]
        rec=json.loads((sub/"selection.json").read_bytes())
        assert b.digest(sub/"forecasts.npz")==rec["forecast"]
        x,y,task,idx=source.split(name,seed)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        val=x[idx["width"]]
        prob=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model=make_model(task=prob,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        f=tree.extract(model,val[:1])
        g=f.cache(val)
        gain=model.get_booster().get_score(importance_type="gain")
        order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
        corrected=gain_prefix(f,val,order)
        original=rec["records"]["gain_prefix"]["support"]
        changed=tuple(corrected)!=tuple(original)
        for m,item in rec["records"].items():
            keep=item["support"]
            omitted=np.setdiff1d(np.arange(g.p),keep)
            lost=np.asarray(g.incidence[:,omitted].sum(axis=1)).ravel()>0
            res=np.asarray(g.a[:,lost].sum(axis=1)).ravel()
            actual=np.linalg.norm(res)
            scale=g.norm
            assert actual<=.2*scale+1e-12*max(1.,scale)
            assert abs(actual/max(1e-100,scale)-item["relative_error"])<1e-8
            rows.append(dict(panel=panel,dataset=name,seed=seed,method=m,teacher_norm=scale,
                actual_norm=actual,strict_slack=.2*scale-actual,corrected_prefix_changed=changed))
        print(key,"audited",flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(conf.DEST/"fidelity-audit.csv",index=False)
    assert len(result)==585 and not result.corrected_prefix_changed.any()
    print("near-constant regression verified; 585 physically feasible supports; 65 identical corrected prefixes;"
          f" minimum strict slack {result.strict_slack.min():.9g}",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        audit()
