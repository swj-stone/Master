"""Time cardinality geometry on the 65 sealed transfer paths, without predictions."""
import ctypes
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain
from experiments.minpi_rework import quota_compress_tree as tree, quota_compress_tree_confirmation as conf
from experiments.minpi_rework import span_geometry
from experiments.minpi_rework.stratified_tree_benchmark import DEST
from minppi_mofs.modeling import make_model


def main():
    if os.name=="nt":
        dll = ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<6)):
            raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    methods = ["last","all","all_sizes","three_span","order_span","all_residual","order_span_residual"]
    for i,(panel,source,name,seed) in enumerate(conf.jobs()):
        x,y,task,idx = source.split(name,seed)
        pool = np.sort(np.r_[idx["fit"],idx["vote"]])
        if task=="classification":
            task = "binary" if len(np.unique(y[pool]))==2 else "multiclass"
        teacher = make_model(task=task,seed=seed,backend=legacy_gain.CONFIG["backend"],
                             model_params=legacy_gain.CONFIG["model_params"])
        teacher.fit(x[pool],y[pool])
        cal = x[idx["width"]]
        f = tree.extract(teacher,cal[:1])
        g = f.cache(cal)
        gains = teacher.get_booster().get_score(importance_type="gain")
        order = np.array(sorted(range(g.p),key=lambda j:(gains.get(f"f{j}",0.),j),reverse=True))
        lock = json.loads((conf.OUT/f"{name}-{seed}"/"selection.json").read_bytes())
        oracle = lock["path"]
        rng = np.random.default_rng(seed)
        for repeat in range(-1 if i==0 else 0,3):
            for method in rng.permutation(methods):
                tick = perf_counter()
                cols,diag = span_geometry.search(g,order,method)
                elapsed = perf_counter()-tick
                assert diag.pop("deletion_order") == oracle
                np.testing.assert_array_equal(cols,lock["records"]["directions"]["support"])
                if repeat>=0:
                    rows.append(dict(panel=panel,dataset=name,seed=seed,repeat=repeat,method=method,
                                     search_s=elapsed,p=g.p,k=len(cols),**diag))
        if (i+1)%5==0:
            print(f"[{i+1}/65] {name}, all methods matched locked order",flush=True)
    pd.DataFrame(rows).to_csv(DEST/"span-timings.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
