"""Independently reconstruct the three original challenge matrices, baselines and stored predictions."""
import hashlib
import io
import json
from pathlib import Path
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from threadpoolctl import threadpool_limits


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/"results/context_study/nips2003"


def read(p):
    return json.loads(p.read_bytes())


def hash_file(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def official(z, dataset, split, width):
    def read_member(sub):
        found=[name for name in z.namelist() if Path(name).name==f"{dataset}_{split}.{sub}"]
        assert len(found)==1
        data=z.read(found[0])
        assert hashlib.sha256(data).hexdigest()==read(RUN/dataset/"members.json")[f"{split}.{sub}"]
        return data
    raw=read_member("data")
    labels=(np.loadtxt(io.BytesIO(read_member("labels")))>0).astype(int)
    if dataset=="arcene":
        vals=np.fromstring(raw.decode("ascii"),sep=" ").reshape(len(labels),width)
        return csr_matrix(vals),labels
    # Parse the sparse challenge format via scikit-learn's independent LIBSVM parser.
    if dataset=="dorothea":
        lines=(b"0 "+b" ".join(tok+b":1" for tok in row.split()) for row in raw.splitlines())
    else:
        lines=(b"0 "+row for row in raw.splitlines())
    x,_=load_svmlight_file(io.BytesIO(b"\n".join(lines)),n_features=width,zero_based=False)
    assert x.shape==(len(labels),width)
    return x,labels


def logistic(seed,l1=False,c=1):
    return LogisticRegression(random_state=seed,C=c,dual=not l1,solver="liblinear",max_iter=2000,
                              l1_ratio=int(l1),tol=1e-5)


def dense_score(y, model, x):
    proba=model.predict_proba(x)
    return np.array([roc_auc_score(y,proba[:,1]),np.mean(y==proba.argmax(1))])


def audit(name,width,seed,x,y,v,target,table):
    folder=RUN/name/str(seed)
    frozen=read(folder/"frozen.json")
    train,rank=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    np.testing.assert_array_equal(train,frozen["train"])
    np.testing.assert_array_equal(rank,frozen["rank"])
    scale=MaxAbsScaler().fit(x[train])
    tr,rr=scale.transform(x[train]),scale.transform(x[rank])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        warnings.simplefilter("ignore",RuntimeWarning)
        f=f_classif(tr,y[train])[0]
    screen=sorted(np.flatnonzero(~np.isnan(f)),key=lambda j:(-f[j],j))[:256]
    local=logistic(seed).fit(tr[:,screen],y[train])
    ranked=sorted(range(len(screen)),key=lambda j:(-abs(local.coef_[0,j]),j))
    xx=rr[:,screen].toarray()
    bg=np.median(tr[:,screen].toarray(),axis=0)
    mask=np.broadcast_to(bg,xx.shape).copy()
    reference=dense_score(y[rank],local,xx)
    for k,j in enumerate(ranked,1):
        mask[:,j]=xx[:,j]
        if np.all(dense_score(y[rank],local,mask)+1e-12>=.92*reference):
            break
    k_votes=len(frozen["supports"]["separating"])
    fallback=False
    options=[]
    for c in (.01,.1,1,10):
        select=logistic(seed,True,c).fit(tr[:,screen],y[train])
        idx=np.flatnonzero(select.coef_[0]!=0)
        if len(idx):
            clf=logistic(seed).fit(tr[:,screen][:,idx],y[train])
            options.append((roc_auc_score(y[rank],clf.decision_function(rr[:,screen][:,idx])),
                            -len(idx),-c,[screen[j] for j in idx]))
    if options:
        l1=max(options)[3]
    else:
        l1=[screen[ranked[0]]]
        fallback=True
    known={"raw_full":np.arange(width),"screen_full":screen,"anchor":[screen[j] for j in ranked[:k]],
           "prefix_at_separating_k":[screen[j] for j in ranked[:k_votes]],"l1":l1}
    auditrows=[]
    assert frozen["supports"]["raw_full"]=={"all_columns":width}
    with np.load(folder/"predictions.npz") as scores:
        for arm,support in frozen["supports"].items():
            if arm in known:
                cols=known[arm]
                if arm!="raw_full":
                    assert sorted(cols)==support,(name,seed,arm,k,support)
            else:
                # Frozen votes are screen-position sorted before the final refit.
                cols=[j for j in screen if j in set(support)]
                assert len(cols)==len(support)
            final_scaler=MaxAbsScaler().fit(x[:,cols])
            final_model=logistic(seed).fit(final_scaler.transform(x[:,cols]),y)
            p=final_model.predict_proba(final_scaler.transform(v[:,cols]))[:,1]
            np.testing.assert_allclose(p,scores[arm],atol=1e-9,rtol=0,err_msg=f"{name}/{seed}/{arm}")
            auc=roc_auc_score(target,p)
            acc=np.mean(target==(p>.5))
            ber=1-balanced_accuracy_score(target,p>.5)
            row=table.loc[(name,seed,arm)]
            np.testing.assert_allclose([ber,acc,auc],[row.ber,row.accuracy,row.auc],atol=1e-10,rtol=0)
            assert len(cols)==row.k and row.anchor==k
            auditrows.append(dict(dataset=name,seed=seed,arm=arm,k=len(cols),ber=ber,auc=auc,accuracy=acc,
                   independent_support=arm in known,l1_fallback=fallback if arm=="l1" else None,
                   max_prediction_error=float(np.max(abs(p-scores[arm])))))
    return auditrows


def main():
    plan=read(ROOT/"configs/context_study/nips2003.json")
    lock=read(RUN/"protocol.json")
    assert hash_file(ROOT/"configs/context_study/nips2003.json")==lock["protocol"]
    assert hash_file(ROOT/"experiments/context_study/nips_challenge.py")==lock["runner"]
    for mod in ("context_design","context_vote","piv3_compact_witness"):
        assert hash_file(ROOT/f"src/minppi_mofs/{mod}.py")==lock[mod]
    assert hash_file(ROOT/"experiments/test/minpi_convergence_metrics.py")==lock["scoring"]
    table=pd.read_csv(RUN/"scores.csv").set_index(["dataset","seed","arm"],verify_integrity=True)
    audits=[]
    for name,(_,width) in plan["datasets"].items():
        path=ROOT/f"data/raw/nips2003/{name}.zip"
        assert hash_file(path)==lock["sources"][name]
        with ZipFile(path) as z:
            x,y=official(z,name,"train",width)
            v,target=official(z,name,"valid",width)
        for seed in plan["seeds"]:
            audits.extend(audit(name,width,seed,x,y,v,target,table))
        print(f"Verified all {name} runs",flush=True)
    assert len(audits)==len(table)==135
    pd.DataFrame(audits).to_csv(HERE/"nips-baselines.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
