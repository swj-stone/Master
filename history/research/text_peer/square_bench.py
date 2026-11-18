"""Second mechanism: historical and exposed-source rechecks plus a bydate holdout."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile
from time import perf_counter_ns
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets._twenty_newsgroups import strip_newsgroup_header,strip_newsgroup_footer,strip_newsgroup_quoting
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from research.text_peer.bench import CACHE,HERE,original,select_arm
from research.text_peer.square import choose
from research.text_peer.bns import bns

METHODS=("square","no_absence","zero","all","old_vote","F_match","pooled_match","random_match","df_match","BNS","full")


def new_source():
    archive=original.ROOT/"data/raw/text-peer/20news-bydate.tar.gz"
    if original.file_hash(archive)!="8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610":
        raise ValueError("20news-bydate archive mismatch")
    datasets={"sports":("rec.sport.baseball","rec.sport.hockey"),
              "computers":("comp.sys.ibm.pc.hardware","comp.sys.mac.hardware")}
    tokenize=CountVectorizer().build_analyzer();raw={k:[] for k in datasets}
    with tarfile.open(archive,mode="r|gz") as tar:
        for member in tar:
            fields=member.name.strip("./").split("/")
            if not member.isfile() or len(fields)!=3:continue
            split,category,record=fields
            for key,pair in datasets.items():
                if category not in pair:continue
                txt=tar.extractfile(member).read().decode("latin1")
                txt=strip_newsgroup_quoting(strip_newsgroup_footer(strip_newsgroup_header(txt)))
                terms=tuple(sorted(set(tokenize(txt))))
                if terms:raw[key].append((member.name,split," ".join(terms),int(category==pair[1])))
    for key,records in raw.items():
        rows=[(split,text,label) for _,split,text,label in sorted(records)]
        counts=Counter(text for _,text,_ in rows)
        train=[(t,y) for part,t,y in rows if part.endswith("-train") and counts[t]==1]
        held=[(t,y) for part,t,y in rows if part.endswith("-test") and counts[t]==1]
        assert train and held
        cv=CountVectorizer(binary=True,dtype=float)
        X=cv.fit_transform([text for text,_ in train]);Xt=cv.transform([text for text,_ in held])
        X.sort_indices();Xt.sort_indices()
        freq=np.bincount(X.indices,minlength=X.shape[1]);mask=(freq>0)&(freq<len(train))
        X=X[:,mask];Xt=Xt[:,mask]
        known=set(original.csr_row_keys(X));usable=np.array([row not in known for row in original.csr_row_keys(Xt)])
        yield X,np.array([y for _,y in train]),Xt[usable],np.array([y for _,y in held])[usable],dict(dataset="bydate_"+key,fold=1,seed=1129,key="bydate_"+key)


def prior_cases():
    for m in json.loads((CACHE/"panels.json").read_text(encoding="utf8")):
        key=m["key"]; X=sparse.load_npz(CACHE/(key+"-X.npz"));Xt=sparse.load_npz(CACHE/(key+"-Xt.npz"))
        with np.load(CACHE/(key+"-labels.npz")) as z:y,gold=z["y"],z["gold"]
        yield X,y,Xt,gold,m


def arm(method,X,y,seed,width):
    if method=="square":return choose(X,y)
    if method=="no_absence":return choose(X,y,count=width,absence=False)
    if method in ("zero","all"):return choose(X,y,count=width,rule=method)
    if method=="BNS":return bns(X,y,width)
    if method in METHODS:return select_arm(method,X,y,seed,width)
    raise ValueError(method)


def run(directory,cohort,repeats,pass_id):
    directory.mkdir(parents=True,exist_ok=False);rows=[]
    for X,y,Xt,gold,meta in (prior_cases() if cohort=="recheck" else new_source()):
        key=meta["key"];seed=meta["seed"];width=None
        arrays={"gold":gold};scores={}
        for method in METHODS:
            cols=arm(method,X,y,seed,width)
            if method=="square":width=len(cols)
            pred=original.fit_predict(X,y,Xt,cols,"classification",seed,False)
            arrays[method+"_cols"]=cols;arrays[method+"_pred"]=pred
            scores[method]=balanced_accuracy_score(gold,pred)
        order=np.random.default_rng([43157,seed,pass_id]).permutation(METHODS)
        for r in range(max(1,repeats)):
            for pos,method in enumerate(np.roll(order,r)):
                select_ms=total_ms=None;cols=arrays[method+"_cols"]
                if repeats:
                    t=perf_counter_ns();cols=arm(method,X,y,seed,width);chosen=perf_counter_ns()
                    pred=original.fit_predict(X,y,Xt,cols,"classification",seed,False);end=perf_counter_ns()
                    select_ms=(chosen-t)/1e6;total_ms=(end-t)/1e6
                    np.testing.assert_array_equal(pred,arrays[method+"_pred"]);np.testing.assert_array_equal(cols,arrays[method+"_cols"])
                rows.append(dict(dataset=meta["dataset"],fold=meta["fold"],method=method,width=len(cols),p=X.shape[1],ba=scores[method],n=len(y),n_test=len(gold),run=pass_id,repeat=r,position=pos,select_ms=select_ms,total_ms=total_ms))
        np.savez_compressed(directory/(key+".npz"),**arrays)
    pd.DataFrame(rows).to_csv(directory/"cases.csv",index=False)
    sources=[HERE/"square.py",Path(__file__),HERE/"bns.py",HERE/"square-protocol.md",HERE/"bench.py"]
    (directory/"protocol.json").write_text(json.dumps(dict(cohort=cohort,pass_id=pass_id,repeats=repeats,source_sha256={s.name:original.file_hash(s) for s in sources}),indent=2),encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--cohort",choices=("recheck","bydate"),required=True)
    parser.add_argument("--repeats",type=int,default=0);parser.add_argument("--pass-id",type=int,default=1)
    args=parser.parse_args()
    with threadpool_limits(1):run(args.directory,args.cohort,args.repeats,args.pass_id)
