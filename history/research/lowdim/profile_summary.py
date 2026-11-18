"""Summarize the native-load cost and separately measured warm selection costs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.lowdim.build_native import library_path


def summarize(folders, output):
    output.mkdir(parents=True,exist_ok=True)
    loads, samples=[],[]
    for process, folder in enumerate(folders,1):
        state=json.loads((folder/"startup.json").read_text())
        assert state["thread_limit"]==1
        loads.append(dict(process=process,folder=folder.name,native_load_ms=state["native_load_ms"]))
        rows=pd.read_csv(folder/"selection.csv")
        assert len(rows)==6*31*2 and (rows.select_ms>0).all()
        assert not rows.duplicated(["task","n","p","method","repeat"]).any()
        samples.append(rows.assign(process=process))
    assert len(loads)==5
    load_times=pd.DataFrame(loads)
    load_times.to_csv(output/"cold-native.csv",index=False)
    keys=["task","n","p","method"]
    proc=pd.concat(samples).groupby(keys+["process"]).agg(ms=("select_ms","median"),k=("k","first"))
    proc.reset_index().to_csv(output/"warm-selection-processes.csv",index=False)
    ms=proc.ms.unstack("method")
    ms["ratio"]=ms.new/ms.previous
    stats=ms.groupby(["task","n","p"]).agg(new_ms=("new","median"),previous_ms=("previous","median"),
            ratio=("ratio","median"),min_ratio=("ratio","min"),max_ratio=("ratio","max"))
    stats.to_csv(output/"warm-selection.csv")
    checksum=hashlib.sha256(library_path().read_bytes()).hexdigest()
    facts=dict(processes=len(load_times),warm_observations=sum(map(len,samples)),
               native_load_median_ms=float(np.median(load_times.native_load_ms)),
               native_load_min_ms=load_times.native_load_ms.min(),
               native_load_max_ms=load_times.native_load_ms.max(),library_sha256=checksum)
    (output/"startup-summary.json").write_text(json.dumps(facts,indent=2),encoding="utf-8")
    print(json.dumps(facts,indent=2))
    print(stats.to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("folders",type=Path,nargs=5)
    args=parser.parse_args()
    summarize(args.folders,args.output)
