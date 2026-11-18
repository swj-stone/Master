"""Train-only document folds and frozen baseline comparison with fresh selection."""
import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments import document_vote_benchmark as original
from experiments import fast_document_vote_benchmark as fast
from experiments import mail_vote_transfer, refined_document_external, reliability_mail_external
from research.text_peer.sparse_game import select

HERE = Path(__file__).resolve().parent
CACHE = original.ROOT / "tmp/text-peer-panels-20261118b"
ARMS = ("candidate","old_vote","F_match","pooled_match","random_match","df_match","zero","all",
        "F_50","pooled_50","random_50","full")


def cornell():
    root=original.ROOT/"data/raw/text-peer"
    for name, filename, checksum in (
        ("cornell_polarity","review_polarity.tar.gz","fc0dccc2671af5db3c5d8f81f77a1ebfec953ecdd422334062df61ede36b2179"),
        ("cornell_subjectivity","rotten_imdb.tar.gz","985e501fbe891f050e2134be6d4cdb79e955435e240ea8f238b3fdb92ed839aa")):
        path=root/filename
        if original.file_hash(path)!=checksum: raise ValueError("Corpus digest changed")
        docs,labels=[],[]
        with tarfile.open(path) as tar:
            for member in sorted(tar.getmembers(),key=lambda m:m.name):
                if name=="cornell_polarity" and member.name.startswith(("txt_sentoken/neg/","txt_sentoken/pos/")) and member.isfile():
                    docs.append(tar.extractfile(member).read().decode("latin1"));labels.append(int("/pos/" in member.name))
                elif name=="cornell_subjectivity" and member.name in ("quote.tok.gt9.5000","plot.tok.gt9.5000"):
                    lines=[line.decode("latin1") for line in tar.extractfile(member).read().splitlines()]
                    docs.extend(lines); labels.extend([int(member.name.startswith("quote"))]*len(lines))
        assert len(docs)==(2000 if name=="cornell_polarity" else 10000)
        tokenize=CountVectorizer().build_analyzer()
        keys=[tuple(sorted(set(tokenize(doc)))) for doc in docs]
        counts=Counter(keys)
        ids=np.array([i for i,key in enumerate(keys) if key and counts[key]==1])
        docs=np.asarray(docs)[ids]; y=np.array(labels)[ids]
        folds=StratifiedKFold(3,shuffle=True,random_state=39191)
        for fold,(a,b) in enumerate(folds.split(docs,y),1):
            vectorizer=CountVectorizer(binary=True,dtype=float)
            X=vectorizer.fit_transform(docs[a]);Xt=vectorizer.transform(docs[b])
            X.sort_indices();Xt.sort_indices()
            freq=np.bincount(X.indices,minlength=X.shape[1]); mask=(freq>0)&(freq<len(a))
            X=X[:,mask];Xt=Xt[:,mask];known=set(original.csr_row_keys(X))
            valid=np.array([k not in known for k in original.csr_row_keys(Xt)])
            yield X,y[a],Xt[valid],y[b[valid]],dict(dataset=name,fold=fold,seed=original.SEEDS[fold-1],
                train_ids_sha256=original.digest(ids[a]),test_ids_sha256=original.digest(ids[b[valid]]),
                vocabulary_sha256=original.digest(vectorizer.get_feature_names_out()[mask].astype("U")),source_sha256=checksum)


def prepare():
    CACHE.mkdir(exist_ok=False)
    X,y,Xt,gold,meta=mail_vote_transfer.prepare()
    mail=[(X,y,Xt,gold,dict(meta,seed=1129))]
    panels=[]
    for iterator in (original.panels(), refined_document_external.panels(),reliability_mail_external.panels(),mail,cornell()):
        for X,y,Xt,gold,meta in iterator:
            key=f"{meta['dataset']}-{meta['fold']}"
            sparse.save_npz(CACHE/(key+"-X.npz"),X)
            sparse.save_npz(CACHE/(key+"-Xt.npz"),Xt)
            np.savez(CACHE/(key+"-labels.npz"),y=y,gold=gold)
            panels.append(dict(meta,key=key,n_train=len(y),n_test=len(gold),p=X.shape[1]))
    assert len(panels)==30
    (CACHE/"panels.json").write_text(json.dumps(panels,indent=2),encoding="utf8")


def select_arm(method,X,y,seed,width):
    if method in ("candidate","zero","all"):
        return select(X,y,count=None if method=="candidate" else width,
                      rule="both" if method=="candidate" else method,utility="log",scale="sqrt")
    if method=="old_vote":
        return fast.select("fast_vote",X,y,seed,width)
    if method in ARMS:
        return fast.select(method,X,y,seed,width)
    raise ValueError(method)


def score_case(meta,repeats,run_id):
    key=meta["key"];seed=meta["seed"]
    X=sparse.load_npz(CACHE/(key+"-X.npz"));Xt=sparse.load_npz(CACHE/(key+"-Xt.npz"))
    with np.load(CACHE/(key+"-labels.npz")) as labels: y,gold=labels['y'],labels['gold']
    reference={};arrays={"gold":gold};quality={};rows=[]
    rng=np.random.default_rng([93216,seed,run_id])
    width=None
    for method in ARMS:
        cols=select_arm(method,X,y,seed,width)
        if method=="candidate": width=len(cols)
        pred=original.fit_predict(X,y,Xt,cols,"classification",seed,False)
        reference[method]=(cols,pred)
        arrays[method+"_cols"]=cols;arrays[method+"_pred"]=pred
        quality[method]=balanced_accuracy_score(gold,pred)
    schedule=rng.permutation(ARMS)
    for r in range(repeats):
        for position,method in enumerate(np.roll(schedule,r)):
            start=perf_counter_ns();cols=select_arm(method,X,y,seed,width);selected=perf_counter_ns()
            pred=original.fit_predict(X,y,Xt,cols,"classification",seed,False);end=perf_counter_ns()
            np.testing.assert_array_equal(cols,reference[method][0]);np.testing.assert_array_equal(pred,reference[method][1])
            rows.append(dict(dataset=meta["dataset"],fold=meta["fold"],method=method,width=len(cols),p=X.shape[1],
                ba=quality[method],run=run_id,repeat=r,position=position,select_ms=(selected-start)/1e6,total_ms=(end-start)/1e6))
    return rows,arrays


def run(directory,run_id,repeats):
    directory.mkdir(parents=True,exist_ok=False)
    panels=json.loads((CACHE/"panels.json").read_text(encoding="utf8"));rows=[]
    (directory/"panels.json").write_text(json.dumps(panels,indent=2),encoding="utf8")
    files=[Path(__file__), HERE/"vote.py", HERE/"sparse_game.py", HERE/"protocol.md",
        Path(fast.__file__),Path(original.__file__),original.ROOT/"experiments/minpi_constant_transfer.py",
        original.ROOT/"experiments/compare_predictive_minpi.py",original.ROOT/"configs/minpi-study.json",
        original.ROOT/"src/minppi_mofs/fast_document_vote_minpi.py", fast.library_path()]
    (directory/"protocol.json").write_text(json.dumps(dict(arms=ARMS,rounds=repeats,run=run_id,
        frozen_source={str(p.relative_to(original.ROOT)):original.file_hash(p) for p in files}),indent=2),encoding="utf8")
    for meta in panels:
        timed,arrays=score_case(meta,repeats,run_id)
        np.savez_compressed(directory/(meta["key"]+".npz"),**arrays)
        rows.extend(timed)
        print(meta["key"],len(rows),flush=True)
    pd.DataFrame(rows).to_csv(directory/"trials.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","run"))
    parser.add_argument("--directory",type=Path);parser.add_argument("--run-id",type=int,choices=(1,2))
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.action=="prepare":prepare()
        else:run(args.directory,args.run_id,args.repeats)
