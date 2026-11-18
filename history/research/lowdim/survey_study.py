"""Time the archived survey, advertisement and three formal simulation datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, metrics, train
from experiments.stable_minpi_stress import digest
from minppi_mofs.stable_minpi import _impute_validation
from research.conditional_dense.original_tasks.paper_splits import check_f, f_support
from research.conditional_dense.vote_width.method import vote_support
from research.lowdim.formal_study import ARMS, HERE, KEYS, hash_source_files, run_arm
from research.lowdim.method import fast_support
from research.lowdim.survey_inputs import fraud_cases, study_cases

SOURCES = {"surveys": (study_cases, 33), "fraud": (fraud_cases, 1)}


def signatures():
    files = [Path(__file__), HERE/"survey_inputs.py", HERE/"survey-protocol.md", HERE/"fraud-protocol.md",
             ROOT/"configs/credit_card_fraud.yaml", ROOT/"experiments/minpi_constant_transfer.py",
             ROOT/"experiments/signal_vote_confirmation.py", ROOT/"experiments/stable_minpi_stress.py",
             ROOT/"src/minppi_mofs/stable_minpi.py"]
    return {**hash_source_files(), **{path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in files}}


def deploy(arm, c, width):
    if not c.impute:
        return run_arm(arm, c.X, c.y, c.Xt, c.task, c.seed, width)
    begin = perf_counter_ns()
    if arm in ("new", "previous", "F"):
        clean, _ = _impute_validation(c.X, c.X[:0])
        if arm == "new":
            cols = fast_support(clean, c.y, task=c.task, seed=c.seed)[0]
        elif arm == "previous":
            cols = vote_support(clean, c.y, seed=c.seed)[0]
        else:
            cols = f_support(clean, c.y, c.task, width)
    elif arm == "full":
        cols = np.arange(c.X.shape[1])
    elif arm == "random":
        cols = np.sort(np.random.default_rng(c.seed).permutation(c.X.shape[1])[:width])
    else:
        raise ValueError(arm)
    selected = perf_counter_ns()
    X, Xt = (c.X, c.Xt) if len(cols) == c.X.shape[1] else (c.X[:, cols], c.Xt[:, cols])
    X, Xt = _impute_validation(X, Xt)
    pred, decision = train("ridge", X, c.y, Xt, c.task, c.seed)
    stop = perf_counter_ns()
    return cols, pred, decision, (selected-begin)/1e6, (stop-begin)/1e6


def run(out, previous, cohort):
    process = 2 if previous else 1
    source = signatures()
    loader, count = SOURCES[cohort]
    if previous:
        assert json.loads((previous/"protocol.json").read_text())["sha256"] == source
    out.mkdir(parents=True, exist_ok=False)
    fields = KEYS + ["source", "family", "task", "metric", "p", "n_train", "n_test", "method",
                     "k", "repeat", "position", "score", "select_ms", "total_ms"]
    n = 0
    with (out/"times.csv").open("x", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for c in loader():
            p = c.X.shape[1]
            base = dict(dataset=c.dataset, source=c.source, family=c.family, fold=c.fold, seed=c.seed,
                        task="binary" if c.task=="classification" else c.task, metric=c.metric, p=p,
                        n_train=len(c.y), n_test=len(c.truth))
            if c.task=="classification":
                assert np.unique(c.y).size==2
            hashes = [digest(a) for a in (c.X, c.y, c.Xt, c.truth)]
            arrays = dict(**c.traces, truth=c.truth, hashes=hashes)
            scores = {}
            width = None
            for arm in ARMS:
                cols, pred, decision, _, _ = deploy(arm, c, width)
                if arm == "new":
                    width = len(cols)
                    assert 1 <= width <= p
                elif arm == "previous" and p>32:
                    np.testing.assert_array_equal(cols, arrays["new_cols"])
                elif arm == "full":
                    np.testing.assert_array_equal(cols, np.arange(p))
                    np.testing.assert_array_equal(pred, c.expected_full)
                    if "archived_decision" in arrays:
                        np.testing.assert_allclose(decision, arrays["archived_decision"], rtol=2e-5, atol=2e-5)
                elif arm == "F":
                    assert len(cols) == width
                    X = _impute_validation(c.X,c.X[:0])[0] if c.impute else c.X
                    check_f(X,c.y,c.task,cols)
                elif arm == "random":
                    expected = np.sort(np.random.default_rng(c.seed).permutation(p)[:width])
                    np.testing.assert_array_equal(cols, expected)
                np.testing.assert_array_equal(np.unique(cols), cols)
                scores[arm] = metrics(c.truth,pred,decision)[c.metric]
                arrays[arm+"_cols"], arrays[arm+"_pred"] = cols,pred
                if decision is not None:
                    arrays[arm+"_decision"] = decision
            name = f"{c.dataset}-{c.fold}-{c.seed}.npz"
            if previous:
                with np.load(previous/name) as ref:
                    assert set(ref.files)==set(arrays)
                    for field, val in arrays.items():
                        np.testing.assert_array_equal(val,ref[field])
            np.savez_compressed(out/name,**arrays)
            order = np.random.default_rng([123983,process,c.fold,c.seed]).permutation(ARMS)
            for repeat in range(len(ARMS)):
                for position, arm in enumerate(np.roll(order,repeat)):
                    cols,pred,decision,sel,total = deploy(arm,c,width)
                    np.testing.assert_array_equal(cols, arrays[arm+"_cols"])
                    np.testing.assert_array_equal(pred, arrays[arm+"_pred"])
                    if decision is not None:
                        np.testing.assert_array_equal(decision, arrays[arm+"_decision"])
                    writer.writerow(dict(**base,method=arm,k=len(cols),repeat=repeat,position=position,
                                         score=scores[arm],select_ms=sel,total_ms=total))
            fp.flush()
            n += 1
            if n%5==0 or n==count:
                print(f"Completed {n}/{count}: {c.dataset}",flush=True)
    assert n==count and signatures()==source
    (out/"protocol.json").write_text(json.dumps(dict(process=process,sha256=source,cases=n,cohort=cohort,threads=1),indent=2))


def summarize(first,second,out):
    medians = []
    keys = KEYS+["method"]
    for process,folder in enumerate((first,second),1):
        protocol=json.loads((folder/"protocol.json").read_text())
        current=signatures()
        runner_key=Path(__file__).relative_to(ROOT).as_posix()
        if current[runner_key]!=protocol["sha256"][runner_key]:
            archived=(folder/"frozen-runner.py").read_bytes()
            current[runner_key]=hashlib.sha256(archived).hexdigest()
            old_source=archived.decode("utf8").split("def summarize(",1)[0].splitlines()
            active_source=Path(__file__).read_text(encoding="utf8").split("def summarize(",1)[0].splitlines()
            assert old_source==active_source, "Timed or input-preparation code differs"
        assert protocol["sha256"]==current and protocol["process"]==process
        cohort=protocol["cohort"]
        count=SOURCES[cohort][1]
        trials = pd.read_csv(folder/"times.csv",keep_default_na=False)
        assert len(trials)==count*len(ARMS)**2 and not trials.duplicated(keys+["repeat"]).any()
        assert trials.groupby(keys).position.nunique().eq(len(ARMS)).all()
        assert not trials.duplicated(KEYS+["repeat","position"]).any()
        assert (trials.total_ms >= trials.select_ms).all() and trials.select_ms.ge(0).all()
        assert trials.groupby(keys).score.nunique().eq(1).all()
        agg = {field:"first" for field in ["source","family","task","metric","p","n_train","n_test","k","score"]}
        medians.append(trials.groupby(keys,as_index=False).agg(dict(**agg,select_ms="median",total_ms="median")).assign(process=process))
    med = pd.concat(medians,ignore_index=True)
    cases = []
    checked=0
    for key,group in med.groupby(KEYS):
        ts = [group[group.process==process].set_index("method") for process in (1,2)]
        data=ts[0].loc["new"]
        case=dict(zip(KEYS,key))
        case.update({field:data[field] for field in ("source","family","p","k","n_train","n_test","metric","task")})
        case["cohort"]="archive_supplement" if cohort=="surveys" else "conference_fraud"
        case["reduced"]=data.k<data.p
        name = "-".join(map(str,key))+".npz"
        with np.load(first/name) as a,np.load(second/name) as b:
            assert set(a.files)==set(b.files)
            for field in a.files:
                np.testing.assert_array_equal(a[field],b[field]); checked+=1
            case["train_sha"],case["target_sha"],case["test_sha"],case["truth_sha"]=a["hashes"]
            for arm in ARMS:
                score=metrics(a["truth"],a[arm+"_pred"],a.get(arm+"_decision"))[data.metric]
                np.testing.assert_allclose([t.at[arm,"score"] for t in ts],score,rtol=0,atol=1e-12)
                case["score_"+arm]=score
                case["k_"+arm]=len(a[arm+"_cols"])
                if cohort=="fraud":
                    case["test_positive"]=int(a["truth"].sum())
                    case["prevalence"]=float(np.mean(a["truth"]))
                    case["average_precision_"+arm]=average_precision_score(a["truth"],a[arm+"_decision"])
                    case["fraud_recall_"+arm]=recall_score(a["truth"],a[arm+"_pred"])
                    case["fraud_detected_"+arm]=int(np.count_nonzero((a["truth"]==1)&(a[arm+"_pred"]==1)))
                for i,t in enumerate(ts,1):
                    case[f"select_ms{i}_{arm}"]=t.at[arm,"select_ms"]
                    case[f"total_ms{i}_{arm}"]=t.at[arm,"total_ms"]
        for arm in ARMS[1:]:
            gap=case["score_new"]-case["score_"+arm]
            ratio=max(t.at["new","total_ms"]/t.at[arm,"total_ms"] for t in ts)
            case["gap_"+arm]=gap
            case["ratio_"+arm]=ratio
            case["faster_"+arm]=ratio<1
            case["better_"+arm]=gap>1e-12
            case["better_faster_"+arm]=ratio<1 and gap>1e-12
        case["cut_faster"] = case["reduced"] and case["faster_full"]
        case["within_002_cut_faster"] = case["cut_faster"] and case["gap_full"]>=-.02-1e-12
        cases.append(case)
    assert len(cases)==count, "Some source names may have been parsed as missing values"
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(cases).to_csv(out/"cases.csv",index=False)
    med.to_csv(out/"process-medians.csv",index=False)
    facts=dict(configurations=len(cases),timed_pipelines=len(cases)*2*len(ARMS)**2,cross_process_arrays=checked)
    (out/"verification.json").write_text(json.dumps(facts,indent=2))
    print(facts)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    cmd=sub.add_parser("run")
    cmd.add_argument("out",type=Path)
    cmd.add_argument("--previous",type=Path)
    cmd.add_argument("--cohort",choices=SOURCES,default="surveys")
    cmd=sub.add_parser("summarize")
    for name in ("first","second","out"):
        cmd.add_argument(name,type=Path)
    a=parser.parse_args()
    with threadpool_limits(1):
        if a.action=="run":
            run(a.out,a.previous,a.cohort)
        else:
            summarize(a.first,a.second,a.out)
