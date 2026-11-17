"""Replay sealed tree deletions; this benchmark never calls an outer scoring API."""
import ctypes
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as tree, quota_compress_tree_exact as bins
from experiments.minpi_rework import quota_compress_tree_confirmation as confirm, quota_compress_tree_run as dev
from minppi_mofs.modeling import make_model
from minppi_mofs.stratified_tree import greedy_path


DEST = b.ROOT / "thesis-paper/evidence/minpi_rework/stratified-tree"


def workloads():
    for panel,source,name,seed in confirm.jobs():
        x,y,task,idx=source.split(name,seed)
        yield panel,name,seed,x,y,task,idx,confirm.OUT/f"{name}-{seed}","directions"
    jobs=legacy.jobs()
    for name in legacy.CONFIG["datasets"]:
        for seed in (42,123,456,789,1024):
            directory,row=jobs[name,seed]
            x,y,_,idx=legacy.read_split(directory,row,seed)
            yield "development",name,seed,x,y,row["task"],idx,dev.OUT/f"{name}-{seed}","exact_greedy"


def main():
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6)):
            raise ctypes.WinError(ctypes.get_last_error())
    DEST.mkdir(parents=True,exist_ok=True)
    rows=[]
    start=perf_counter()
    for i,(panel,name,seed,x,y,task,idx,folder,arm) in enumerate(workloads()):
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model=make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        f=tree.extract(model,x[idx["width"][:1]])
        g=f.cache(x[idx["width"]])
        gain=model.get_booster().get_score(importance_type="gain")
        order=np.array(sorted(range(g.p),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
        locked=json.loads((folder/"selection.json").read_bytes())["records"][arm]
        ref_cols,ref=g.search("exact_greedy",order)
        np.testing.assert_array_equal(ref_cols,locked["support"])
        rng=np.random.default_rng(seed)
        if i==0:
            for mode in ("cache","last","all"):
                greedy_path(g.a,g.incidence,order,mode=mode)
            bins.search(g,order)
        for repeat in range(3):
            for mode in rng.permutation(["cache","last","all","four_bins"]):
                tick=perf_counter()
                if mode=="four_bins":
                    cols,d=bins.search(g,order)
                    diag=dict(initial=g.p,refreshed=d["recomputed"],contenders=d["all_candidates"])
                else:
                    cols,d=greedy_path(g.a,g.incidence,order,mode=mode)
                    diag={key:val for key,val in d.items() if key!="deletion_order"}
                seconds=perf_counter()-tick
                if ref["deletion_order"]!=d["deletion_order"]:
                    raise AssertionError((name,seed,mode,ref["deletion_order"],d["deletion_order"]))
                np.testing.assert_array_equal(cols,ref_cols)
                rows.append(dict(panel=panel,dataset=name,seed=seed,repeat=repeat,method=mode,p=g.p,k=len(cols),
                                 search_s=seconds,pipeline_s=locked["pipeline_s"]-locked["search_s"]+seconds,**diag))
        pd.DataFrame(rows).to_csv(DEST/"timings.csv",index=False)
        elapsed=perf_counter()-start
        print(f"[{i+1}/95] {name}/{seed} {len(ref_cols)}/{g.p} retained, elapsed {elapsed:.1f}s ETA {elapsed*(94-i)/(i+1):.1f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
