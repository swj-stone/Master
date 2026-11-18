"""Development replay for minPI consensus on the five already-inspected outer splits."""
import argparse
import json
from itertools import combinations
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.backward_minpi_stress import ARCHIVE, SEEDS, summarize
from experiments.stable_minpi_stress import (
    DATASETS, FRACTIONS, PRIMARY_SEED, ROOT, SPLIT_SEED, TOLERANCE, choose, choose_columns,
    digest, file_hash, folds, inputs, manifest, metric, prediction, write_json,
)
from minppi_mofs.consensus_minpi import ConsensusMinPISelector, consensus_top_from_training


OLD = ROOT / "results/minpi-study/backward-stress-20261117a"
METHODS = ("consensus_policy", "consensus_half", "local_half", "full", "F_matched")


def select(method, X, y, task, seed, groups, real_env, width):
    env = dict(groups=groups, ordered_groups=False, environment_groups=real_env)
    if method == "consensus_policy":
        est = ConsensusMinPISelector(task=task,random_state=seed,tolerance=TOLERANCE,fractions=FRACTIONS,n_paths=3)
        est.fit(X,y,**env)
        return est.get_support(indices=True), est.diagnostics_
    if method == "consensus_half":
        return consensus_top_from_training(X,y,ceil(X.shape[1]/2),task=task,random_state=seed,n_paths=3,**env)
    if method == "F_matched":
        return (np.arange(X.shape[1]) if width==X.shape[1] else choose("f",X,y,task,seed,width)), {}
    return choose_columns(method,X,y,task,seed,groups,width)


def check_paths(info, groups, n):
    basis=np.arange(n)
    if "discovery_rows" in info:
        basis=np.asarray(info["discovery_rows"],dtype=int)
        held=np.asarray([row for split in info["validation_rows"] for row in split],dtype=int)
        assert not np.intersect1d(basis,held).size
        if groups is not None:
            assert not np.intersect1d(groups[basis],groups[held]).size
    paths=info.get("consensus_path",info).get("paths",[])
    if paths:
        assert len(paths)==3
        for path in paths:
            for split in path["folds"]:
                fi,va=basis[split["fit_rows"]],basis[split["held_rows"]]
                assert not np.intersect1d(fi,va).size
                if groups is not None:
                    assert not np.intersect1d(groups[fi],groups[va]).size
    return len(paths)


def run_fold(output,name,fold,X,y,Xt,gold,task,groups,oracle):
    rows, arrays, decisions = [], {}, {}
    for seed in SEEDS:
        width = None
        order = np.random.default_rng([SPLIT_SEED,fold,seed]).permutation(METHODS[1:])
        for rep in range(3 if seed==PRIMARY_SEED else 1):
            for position,method in enumerate(np.roll(["consensus_policy",*order], 2 * rep)):
                start = perf_counter_ns()
                cols,info = select(method,X,y,task,seed,groups,name=="crimedata",width)
                selected = perf_counter_ns()
                pred = prediction(X,y,Xt,cols,task,seed)
                end = perf_counter_ns()
                if method=="consensus_policy":
                    width=len(cols)
                if method.endswith("_half"):
                    assert len(cols)==ceil(X.shape[1]/2)
                assert np.array_equal(cols,np.unique(cols)) and len(cols)>0
                key=f"{method}_s{seed}"
                for suffix,val in (("support",cols),("pred",pred)):
                    label=f"{key}_{suffix}"
                    if rep:
                        np.testing.assert_array_equal(val,arrays[label])
                    else:
                        arrays[label]=val
                    if method in ("local_half","full"):
                        np.testing.assert_array_equal(val,oracle[label])
                if rep==0 and method.startswith("consensus"):
                    check_paths(info,groups,len(y))
                    decisions[key]=info
                rows.append(dict(dataset=name,fold=fold,seed=seed,method=method,repeat=rep,position=position,
                                 rule=info.get("rule",method),width=len(cols),p=X.shape[1],score=metric(gold,pred,task),
                                 select_ms=(selected-start)/1e6,refit_ms=(end-selected)/1e6,total_ms=(end-start)/1e6))
    np.savez_compressed(output/f"{name}-{fold}.npz",**arrays)
    write_json(output/f"{name}-{fold}-decisions.json",decisions)
    return rows


def compare_seeds(output):
    rows=[]
    for name in DATASETS:
        for fold in (1,2,3):
            with np.load(ARCHIVE/f"{name}-{fold}-split.npz") as split:
                gold=split["gold"]
            task="regression" if name in ("crimedata","abalone") else "classification"
            for parent,methods in ((OLD,("local_half","environment_half","stable","backward_half","backward_policy")),
                                   (output,("consensus_half","consensus_policy"))):
                with np.load(parent/f"{name}-{fold}.npz") as arrays:
                    for method in methods:
                        for a,b in combinations(SEEDS,2):
                            ca,cb=(arrays[f"{method}_s{i}_support"] for i in (a,b))
                            sa,sb=(metric(gold,arrays[f"{method}_s{i}_pred"],task) for i in (a,b))
                            rows.append(dict(dataset=name,fold=fold,method=method,a=a,b=b,width_a=len(ca),
                                             width_b=len(cb),score_a=sa,score_b=sb,score_change=abs(sa-sb),
                                             width_change=abs(len(ca)-len(cb)),
                                             jaccard=len(np.intersect1d(ca,cb))/len(np.union1d(ca,cb))))
    frame=pd.DataFrame(rows)
    frame.to_csv(output/"seed-overlap.csv",index=False)
    frame.groupby("method").agg(pairs=("fold","size"),median_jaccard=("jaccard","median"),
                                min_jaccard=("jaccard","min"),mean_jaccard=("jaccard","mean"),
                                max_width_change=("width_change","max"),median_width_change=("width_change","median"),
                                max_score_change=("score_change","max"),mean_score_change=("score_change","mean")
                                ).to_csv(output/"seed-summary.csv")


def run(output,verify):
    hashes=manifest()["sources"]
    files=[Path(__file__),ROOT/"experiments/backward_minpi_stress.py",ROOT/"src/minppi_mofs/consensus_minpi.py"]
    hashes.update({str(p.relative_to(ROOT)):file_hash(p) for p in files})
    archive_protocol=json.loads((ARCHIVE/"protocol.json").read_text(encoding="utf-8"))
    if verify:
        assert json.loads((output/"protocol.json").read_text(encoding="utf-8"))["sources"]==hashes
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"protocol.json",dict(sources=hashes,archive_digest=file_hash(ARCHIVE/"protocol.json"),
                   development=True,outer_sources=DATASETS,seeds=SEEDS,n_paths=3,tolerance=TOLERANCE,
                   compressed_fractions=FRACTIONS,semantic_environments=["crimedata"],only_duplicate_groups=["spambase"],
                   protocol="Known five-source replay; all three primary repeats independently regenerate rankings. Two other seeds each run once. F receives only the training-selected consensus width. Intrinsic groups are kept intact for both selector paths and outer testing."))
    rows,audit=[],[]
    for name in DATASETS:
        X,y,task,groups,source,duplicates=inputs(name)
        assert archive_protocol["input_hashes"][name]==dict(source=source,X=digest(X),y=digest(y),
                                                             groups=None if groups is None else digest(groups))
        for fold,(tr,te) in enumerate(folds(X,y,task,groups),1):
            with np.load(ARCHIVE/f"{name}-{fold}-split.npz") as split:
                np.testing.assert_array_equal(tr,split["train"])
                np.testing.assert_array_equal(te,split["test"])
                np.testing.assert_array_equal(y[te],split["gold"])
            assert not np.intersect1d(duplicates[tr],duplicates[te]).size
            g=None if groups is None else groups[tr]
            if verify:
                with np.load(output/f"{name}-{fold}.npz") as arrays:
                    for method in METHODS[:2]:
                        seed=PRIMARY_SEED
                        cols,info=select(method,X[tr],y[tr],task,seed,g,name=="crimedata",None)
                        pred=prediction(X[tr],y[tr],X[te],cols,task,seed)
                        np.testing.assert_array_equal(cols,arrays[f"{method}_s{seed}_support"])
                        np.testing.assert_array_equal(pred,arrays[f"{method}_s{seed}_pred"])
                        layers=check_paths(info,g,len(tr))
                        audit.append(dict(dataset=name,fold=fold,method=method,score=metric(y[te],pred,task),path_count=layers,
                                          width=len(cols),rule=info.get("rule",method),support_sha256=digest(cols),
                                          prediction_sha256=digest(pred)))
            else:
                with np.load(OLD/f"{name}-{fold}.npz") as oracle:
                    rows.extend(run_fold(output,name,fold,X[tr],y[tr],X[te],y[te],task,g,oracle))
            print(f"{'Verified' if verify else 'Finished'} {name} {fold}",flush=True)
    if verify:
        pd.DataFrame(audit).to_csv(output/"fresh-process-audit.csv",index=False)
    else:
        summarize(output,rows)
        compare_seeds(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.verify)
