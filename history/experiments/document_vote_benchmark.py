"""Compare cross-fitted sparse presence votes against supervised and frequency-based text filters."""
import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.minpi_constant_transfer import fit_predict
from experiments.stable_minpi_stress import ROOT, digest, file_hash, write_json
from minppi_mofs.document_vote_minpi import document_vote_top_from_training


ASSETS = {
    "sms": "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
    "youtube": "bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a",
    "sentences": "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343",
}
ARMS = ("document_vote", "F_match", "chi_match", "random_match", "df_match", "F_50",
        "chi_50", "random_50", "df2", "full")
CASE = ["dataset","fold","seed"]
SPLIT_SEED = 39191
SEEDS = (1129,2027,3469)


def corpora():
    for source, sha in ASSETS.items():
        path = ROOT/"data/raw"/f"text-{source}.zip"
        assert file_hash(path) == sha
        with ZipFile(path) as z:
            if source == "sms":
                lines = z.read("SMSSpamCollection").decode("utf8").splitlines()
                entries = [s.split("\t", 1) for s in lines]
                assert all(label in ("ham", "spam") for label, _ in entries)
                yield "sms", [txt for _, txt in entries], np.array([int(k=="spam") for k,_ in entries]), None
            elif source == "youtube":
                parts = []
                for name in sorted(n for n in z.namelist() if n.startswith("Youtube") and n.endswith(".csv")):
                    with z.open(name) as stream:
                        parts.append(pd.read_csv(stream).assign(video=Path(name).stem))
                assert len(parts) == 5
                data = pd.concat(parts, ignore_index=True)
                assert not data.CONTENT.isna().any() and set(data.CLASS.unique()) == {0,1}
                yield "youtube", data.CONTENT.tolist(), data.CLASS.to_numpy(dtype=int), data.video.to_numpy()
            else:
                for domain in ("amazon_cells","imdb","yelp"):
                    name = "sentiment labelled sentences/"+domain+"_labelled.txt"
                    # U+0085 occurs inside two IMDb sentences; publisher records are LF-delimited.
                    lines = [line.decode("utf8") for line in z.read(name).splitlines()]
                    assert len(lines) == 1000
                    entries = [s.rsplit("\t",1) for s in lines]
                    yield domain, [txt for txt,_ in entries], np.array([int(k) for _,k in entries]), None


def csr_row_keys(X):
    return [X.indices[start:stop].tobytes() for start,stop in zip(X.indptr[:-1], X.indptr[1:])]


def panels():
    analyze = CountVectorizer().build_analyzer()
    for name, documents, y, groups in corpora():
        keys = [tuple(sorted(set(analyze(doc)))) for doc in documents]
        counts = Counter(keys)
        ids = np.array([i for i,k in enumerate(keys) if counts[k] == 1], dtype=np.intp)
        docs = np.asarray(documents)[ids]
        splitter = (StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(docs, y[ids]) if groups is None
                    else LeaveOneGroupOut().split(docs, y[ids], groups[ids]))
        for fold, (train, test) in enumerate(splitter, 1):
            start = perf_counter_ns()
            vectorizer = CountVectorizer(binary=True, dtype=np.float64)
            X = vectorizer.fit_transform(docs[train])
            Xt = vectorizer.transform(docs[test])
            X.sort_indices()
            Xt.sort_indices()
            counts = np.bincount(X.indices,minlength=X.shape[1])
            keep = (counts > 0) & (counts < X.shape[0])
            X, Xt = X[:,keep], Xt[:,keep]
            X.sort_indices()
            Xt.sort_indices()
            train_keys = set(csr_row_keys(X))
            new = np.array([key not in train_keys for key in csr_row_keys(Xt)])
            Xt = Xt[new]
            common_ms = (perf_counter_ns()-start)/1e6
            assert new.any() and np.all((freq := np.bincount(X.indices, minlength=X.shape[1])) > 0)
            assert max(freq) < X.shape[0] and set(y[ids[test[new]]]) == {0,1}
            info = dict(dataset=name,fold=fold,seed=SEEDS[(fold-1)%3],n_train=len(train),n_test=int(new.sum()),
                        p=X.shape[1], source_records=len(documents), drop_duplicate_bags=len(documents)-len(ids),
                        train_vocabulary=len(keep), drop_constant=int((~keep).sum()), drop_cross_test=int((~new).sum()),
                        task="sentiment" if name in ("amazon_cells","imdb","yelp") else "spam",
                        video="" if groups is None else str(groups[ids[test[0]]]), common_prepare_ms=common_ms,
                        train_ids_sha256=digest(ids[train]), test_ids_sha256=digest(ids[test[new]]),
                        vocabulary_sha256=digest(np.array(vectorizer.get_feature_names_out()[keep],dtype="U")),
                        train_rows_sha256=digest(X.indptr), train_cols_sha256=digest(X.indices),
                        test_rows_sha256=digest(Xt.indptr), test_cols_sha256=digest(Xt.indices))
            yield X, y[ids[train]], Xt, y[ids[test[new]]], info


def select(method, X, y, seed, width):
    if method == "document_vote":
        return document_vote_top_from_training(X, y, random_state=seed)[0]
    if method == "full":
        return np.arange(X.shape[1])
    if method in ("df2","df_match"):
        freq = np.bincount(X.indices, minlength=X.shape[1])
        if method == "df2":
            return np.flatnonzero(freq >= 2)
        return np.sort(np.argsort(-freq, kind="stable")[:width])
    rule, budget = method.split("_")
    count = width if budget == "match" else (X.shape[1]+1)//2
    if rule == "chi":
        score = np.nan_to_num(chi2(X,y)[0],nan=-np.inf)
        return np.sort(np.argsort(-score, kind="stable")[:count])
    return choose(rule.lower(), X, y, "classification", seed, count)


def run_case(out, X, y, Xt, truth, info, verify):
    key = {c:info[c] for c in CASE}
    seed = info["seed"]
    arrays, audit, trials, draws, scores = {}, [], [], [], {}
    width = None
    for method in ARMS:
        cols = select(method, X, y, seed, width)
        check_columns(cols, X.shape[1])
        if method == "document_vote":
            width = len(cols)
        if method.endswith("_match"):
            assert len(cols) == width
        pred = fit_predict(X,y,Xt,cols,"classification",seed,False)
        scores[method] = balanced_accuracy_score(truth,pred)
        arrays[method+"_cols"], arrays[method+"_pred"] = cols, pred
        audit.append(dict(**key,method=method,width=len(cols),score=scores[method],
                          support_sha256=digest(cols),prediction_sha256=digest(pred)))
    trace = out/f"{info['dataset']}-{info['fold']}-{seed}.npz"
    if verify:
        with np.load(trace) as prev:
            assert set(arrays) == set(prev)
            for k,v in arrays.items():
                np.testing.assert_array_equal(v, prev[k], err_msg=k)
    else:
        order = np.random.default_rng([473803, info["fold"], seed]).permutation(ARMS)
        for r in range(len(ARMS)):
            for position, method in enumerate(np.roll(order,r)):
                t0=perf_counter_ns()
                cols = select(method, X, y, seed, width)
                t1=perf_counter_ns()
                pred = fit_predict(X,y,Xt,cols,"classification",seed,False)
                t2=perf_counter_ns()
                np.testing.assert_array_equal(cols,arrays[method+"_cols"])
                np.testing.assert_array_equal(pred,arrays[method+"_pred"])
                trials.append(dict(**key,method=method,repeat=r,position=position,width=len(cols),p=X.shape[1],
                                   score=scores[method],select_ms=(t1-t0)/1e6,refit_ms=(t2-t1)/1e6,
                                   total_ms=(t2-t0)/1e6))
        np.savez_compressed(trace,**arrays)
    for draw in range(20):
        rng = np.random.default_rng([108391,info["fold"],seed,draw])
        cols=np.sort(rng.permutation(X.shape[1])[:width])
        pred=fit_predict(X,y,Xt,cols,"classification",seed,False)
        score=balanced_accuracy_score(truth,pred)
        draws.append(dict(**key,draw=draw,width=width,score=score,win=scores["document_vote"]>score+1e-12,
                          support_sha256=digest(cols),prediction_sha256=digest(pred)))
    return trials,audit,draws


def summarize(out, records, panels_frame, draws):
    cases=summarize_timings(out, records)
    panels_frame.to_csv(out/"panels.csv",index=False)
    draws.to_csv(out/"random-draws.csv",index=False)
    vote=cases[cases.method=="document_vote"].set_index(CASE)
    meta=panels_frame.set_index(CASE)
    vote["task"]=meta.task
    vote["ratio_with_vocabulary"]=(vote.total_ms+meta.common_prepare_ms)/(vote.full_total_ms+meta.common_prepare_ms)
    rows=[]
    for method, group in cases.groupby("method"):
        control=group.set_index(CASE)
        comp=vote[["task","joint"]].copy()
        comp["baseline"]=method
        comp["score_delta"]=vote.score-control.score
        comp["width_ratio"]=vote.width/control.width
        comp["time_ratio"]=vote.total_ms/control.total_ms
        safe=vote.joint & (comp.score_delta>=-.02) & (comp.time_ratio<1)
        comp["joint_vs_baseline"]=safe & (comp.width_ratio<=1)
        comp["joint_vs_baseline_10pct"]=safe & (comp.width_ratio<=1.1)
        rows.append(comp)
    paired=pd.concat(rows).reset_index()
    paired.to_csv(out/"paired-baselines.csv",index=False)
    vote=vote.join(draws.groupby(CASE).agg(random_wins=("win","sum"),random_median=("score","median")))
    for field in ("joint_vs_baseline","joint_vs_baseline_10pct"):
        flags=paired.pivot(index=CASE,columns="baseline",values=field)
        vote["F_and_random50_"+field]=flags.F_50 & flags.random_50
        vote["F_chi_random50_"+field]=flags.F_50 & flags.random_50 & flags.chi_50
    vote["strong_joint"]=vote.joint & (vote.random_wins>=15)
    vote.to_csv(out/"vote-comparisons.csv")


def run(out, verify):
    files=[Path(__file__),ROOT/"docs/minpi-study/document-vote-protocol.md",ROOT/"src/minppi_mofs/document_vote_minpi.py",
           ROOT/"src/minppi_mofs/signal_vote_minpi.py",ROOT/"experiments/minpi_constant_transfer.py",
           ROOT/"experiments/compare_predictive_minpi.py",ROOT/"experiments/fast_minpi_five_source.py",
           ROOT/"experiments/minpi_study.py",ROOT/"configs/minpi-study.json"]
    state=dict(sources=ASSETS, code={p.relative_to(ROOT).as_posix():file_hash(p) for p in files},
               methods=list(ARMS),split_seed=SPLIT_SEED,selector_seeds=list(SEEDS),repeats=len(ARMS),
               sklearn=sklearn.__version__,numpy=np.__version__,python=platform.python_version())
    if verify:
        assert state==json.loads((out/"protocol.json").read_text())
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/"protocol.json",state)
    trials,audit,draws,metadata=[],[],[],[]
    for X,y,Xt,truth,info in panels():
        r,a,d=run_case(out,X,y,Xt,truth,info,verify)
        trials.extend(r)
        audit.extend(a)
        draws.extend(d)
        metadata.append(info)
        print("Verified" if verify else "Measured",info["dataset"],info["fold"],flush=True)
    assert len(metadata)==17 and len(draws)==340
    frame=pd.DataFrame(metadata)
    random=pd.DataFrame(draws)
    if verify:
        assert len(audit)==170
        fields=frame.columns.drop("common_prepare_ms")
        pd.testing.assert_frame_equal(frame[fields],pd.read_csv(out/"panels.csv",keep_default_na=False)[fields])
        pd.testing.assert_frame_equal(random,pd.read_csv(out/"random-draws.csv"),rtol=0,atol=1e-12)
        pd.DataFrame(audit).to_csv(out/"fresh-process-audit.csv",index=False)
    else:
        assert len(trials)==1700
        summarize(out,trials,frame,random)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.output,args.verify)
