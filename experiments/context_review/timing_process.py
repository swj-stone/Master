"""Interleaved, complete CPU fits with exact post-timing replay of every visited ballot."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.lazy_ballot import LazyAccuracyBallot
from minppi_mofs.piv3_compact_tree import select as gain_anchor
from minppi_mofs.piv3_compact_witness import _sample, _within
from minppi_mofs.veto_schedule import search_with_recent_veto

OUT = ROOT / "results/context_review/timing-process"
DESIGN = ROOT / "configs/context_review/timing_process.json"
ARMS = ["fixed","recent","recent_lazy"]


def bind_logical_cpu():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = ctypes.c_void_p(-1)
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle,ctypes.byref(process),ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        available = [i for i in range(process.value.bit_length()) if process.value & (1<<i)]
        cpu = available[min(2,len(available)-1)]
        if not kernel.SetProcessAffinityMask(handle,ctypes.c_size_t(1<<cpu)):
            raise ctypes.WinError(ctypes.get_last_error())
        return cpu
    if hasattr(os,"sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        cpu = available[min(2,len(available)-1)]
        os.sched_setaffinity(0,{cpu})
        return cpu
    return None


class FullBallot:
    def __init__(self, model, rows, background, anchor, scorer):
        self.model, self.rows, self.background = model, rows, background
        self.anchor, self.scorer = anchor, scorer
        self.buffer = np.empty_like(rows)
        self.scores = {}

    def score(self, visible):
        if visible not in self.scores:
            columns = sorted(visible)
            self.buffer[:] = self.background
            self.buffer[:,columns] = self.rows[:,columns]
            self.scores[visible] = self.scorer(self.model.predict_proba(self.buffer))
        return self.scores[visible]

    def __call__(self, support, extension):
        a, b = self.score(support|extension), self.score(self.anchor|extension)
        return tuple(_within(x,y) for x,y in zip(a,b,strict=True))


def clocks():
    return np.array([perf_counter(),process_time()])


def run_arm(arm, raw, y, train, rank, test, seed):
    timing = {}
    start = last = clocks()
    def mark(stage):
        nonlocal last
        now = clocks()
        timing[stage+"_wall"], timing[stage+"_cpu"] = (now-last).tolist()
        last = now
    dev = np.sort(np.r_[train,rank])
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(raw[train])
    xt, xr = imp.transform(raw[train]), imp.transform(raw[rank])
    task = "binary" if len(np.unique(y[train]))==2 else "multiclass"
    mark("prep")
    model = fit(xt,y[train],seed)
    mark("fit")
    names = [f"c{i}" for i in range(xt.shape[1])]
    _, info = gain_anchor(model,pd.DataFrame(xt,columns=names),y[train],pd.DataFrame(xr,columns=names),
                          y[rank],task=task,utility="ovr_auc",seed=seed,aggregate="gain")
    order = [int(name[1:]) for name in info["order"]]
    k = info["k"]
    mark("anchor")
    family = separating_contexts(order[k:])
    anchor = frozenset(order[:k])
    idx, weights = _sample(y[rank],task,np.random.SeedSequence([seed,659]))
    labels, rows = y[rank][idx], xr[idx]
    bgrows = np.random.default_rng(seed).choice(len(train),min(512,len(train)),replace=False)
    bg = np.median(xt[bgrows],axis=0)
    rare = task=="binary" and min(np.bincount(y[rank]))<=.05*len(rank)
    scorer = make_scorer(labels,weights,task,"ovr_auc",rare)
    if arm=="recent_lazy":
        ballot = LazyAccuracyBallot(model.predict_proba,rows,bg,labels,weights,anchor,scorer)
    else:
        ballot = FullBallot(model,rows,bg,anchor,scorer)
    method = search if arm=="fixed" else search_with_recent_veto
    live, trace = method(order[:k],family,ballot)
    cols = sorted(live)
    mark("delete")
    fi = SimpleImputer(strategy="median",keep_empty_features=True).fit(raw[dev])
    xd,xe = fi.transform(raw[dev])[:,cols],fi.transform(raw[test])[:,cols]
    predictions = fit(xd,y[dev],seed).predict_proba(xe)
    mark("refit")
    timing["total_wall"],timing["total_cpu"] = (last-start).tolist()
    counts = {"full_scores":len(ballot.scores),"proposals":len(trace["trace"])}
    if arm=="recent_lazy":
        counts.update(masks=len(ballot.accuracy),dispatches=ballot.dispatches,quick_veto=ballot.accuracy_vetoes)
    else:
        counts.update(masks=len(ballot.scores),dispatches=len(ballot.scores),quick_veto=0)
    verifier = FullBallot(model,rows,bg,anchor,scorer)
    return order,k,cols,trace,family,verifier,predictions,dict(**timing,**counts)


def run_fold(name, fold):
    cpu = bind_logical_cpu()
    source = PRIMARY/name/f"fold-{fold}"
    path = DATA/name/f"{name}.tsv.gz"
    rec = json.loads((source/"selection.json").read_bytes())
    lock = json.loads((source/"completed.json").read_bytes())
    assert digest(path)==lock["data_sha256"] and digest(source/"selection.json")==lock["selection_sha256"]
    frame = pd.read_csv(path,sep="\t")
    y = frame.pop("target").map({v:i for i,v in enumerate(rec["class_order"])}).to_numpy()
    raw = frame.to_numpy(dtype=np.float32)
    train,rank,test = [np.asarray(rec[p]) for p in ("train","rank","test")]
    with np.load(source/"predictions.npz") as frozen:
        expected = frozen["antichain"]
        np.testing.assert_array_equal(test, frozen["row_id"])
    schedule = ARMS.copy()
    Random(331741+config()["datasets"].index(name)*3+fold).shuffle(schedule)
    records, ballots = [], []
    for repetition in range(3):
        sequence = schedule[repetition:] + schedule[:repetition]
        for position, arm in enumerate(sequence):
            order,k,cols,trace,family,verify,prob,timing = run_arm(arm,raw,y,train,rank,test,20421+fold)
            if order!=rec["order"] or k!=rec["anchor_k"] or cols!=rec["selected"]["antichain"]:
                raise ValueError("Anchor or final selected support differs from the frozen study")
            np.testing.assert_array_equal(prob, expected)
            old = rec["traces"]["antichain"]["trace"]
            actual = trace["trace"]
            assert len(old)==len(actual)
            current = frozenset(order[:k])
            audit = []
            for row, reference in zip(actual,old,strict=True):
                assert all(row[key]==reference[key] for key in ("phase","before","block","accepted"))
                proposal = current.difference(row["block"])
                seen = row.get("evaluated_contexts", family[:len(row["votes"])])
                evaluated = [(tuple(sorted(e)),tuple(v)) for e,v in zip(seen,row["votes"],strict=True)]
                for e,v in evaluated:
                    if v!=verify(proposal,frozenset(e)):
                        raise ValueError(f"Reference votes differ: {name}, {fold}, {arm}, {e}")
                if row["accepted"]:
                    current=proposal
                audit.append((sorted(proposal),evaluated,row["accepted"]))
            path_hash=hashlib.sha256(json.dumps(audit,separators=(",",":")).encode()).hexdigest()
            ballots.append((arm,path_hash))
            records.append(dict(dataset=name,fold=fold,logical_cpu=cpu,repeat=repetition,position=position,
                                arm=arm,selected=cols,vote_hash=path_hash,verified_panels=sum(
                                    len(v)*4 for _,v,_ in audit),**timing))
    assert len({v for a,v in ballots if a=="fixed"})==1
    assert len({v for a,v in ballots if a!="fixed"})==1
    folder=OUT/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    store(folder/"completed.json",dict(protocol=digest(OUT/"protocol.json"),records=records))


def summarize(allrows):
    df = pd.DataFrame(allrows)
    df.to_csv(OUT/"observations.csv",index=False)
    cols = [n for n in df if n.endswith(("_wall","_cpu"))]+["masks","dispatches","full_scores","quick_veto"]
    folds = df.groupby(["dataset","fold","arm"])[cols].median()
    tasks = folds.groupby(["dataset","arm"]).mean()
    tasks.to_csv(OUT/"by-dataset.csv")
    tasks.groupby("arm").mean().to_csv(OUT/"summary.csv")
    for metric in ("total_wall","total_cpu","delete_wall"):
        vals = tasks[metric].unstack("arm")
        for a,b in (("recent","fixed"),("recent_lazy","fixed"),("recent_lazy","recent")):
            ratio=vals[a]/vals[b]
            print(f"{metric} {a}/{b}: median task ratio {ratio.median():.4f}, faster {int((ratio<1).sum())}/14",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            return run_fold(args.worker[0],int(args.worker[1]))
    sources = [DESIGN,Path(__file__),ROOT/"src/minppi_mofs/lazy_ballot.py",
               PRIMARY/"protocol-lock.json",ROOT/"src/minppi_mofs/veto_schedule.py"]
    OUT.mkdir(exist_ok=True,parents=True)
    path = OUT/"protocol.json"
    lock = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    if path.exists() and json.loads(path.read_bytes())!=lock:
        raise ValueError("Protocol changed; use a fresh experiment directory")
    store(path,lock)
    rows = []
    for name in config()["datasets"]:
        for fold in range(3):
            result=OUT/name/f"fold-{fold}"/"completed.json"
            if not result.exists():
                print(f"Starting {name}, fold {fold}, nine independent pipelines",flush=True)
                subprocess.run([sys.executable,"-m","experiments.context_review.timing_process",
                                "--worker",name,str(fold)],check=True,timeout=3600)
            payload=json.loads(result.read_bytes())
            assert payload["protocol"]==digest(path) and len(payload["records"])==9
            rows.extend(payload["records"])
            print(f"Verified {len(rows)}/378 pipelines",flush=True)
    summarize(rows)


if __name__=="__main__":
    main()
