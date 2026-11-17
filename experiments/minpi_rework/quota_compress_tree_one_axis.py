"""Matched exact-search ablation of one anchor versus its four orthogonal pieces."""
import json
import ctypes
import os
from time import perf_counter
from unittest.mock import patch
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as trees,quota_compress_tree_exact as exact
from experiments.minpi_rework import quota_compress_tree_run as historical
from minppi_mofs.modeling import make_model


@njit(cache=True)
def one_axis(delta,ap,ai,av,ip,ii):
    den=np.zeros((len(delta),1))
    for j in range(len(delta)):
        for v in delta[j]:
            den[j,0]+=v*v
    contributions=np.zeros((len(ii),1))
    for g in range(len(ip)-1):
        for pos in range(ip[g],ip[g+1]):
            j=ii[pos]
            total=0.
            for z in range(ap[g],ap[g+1]):
                total+=av[z]*delta[j,ai[z]]
            contributions[pos,0]=total
    return den,contributions


def run():
    jobs=legacy.jobs()
    data=[]
    original=exact._projections
    for seed in legacy.CONFIG["seeds"]:
        for name in legacy.CONFIG["datasets"]:
            d,row=jobs[name,seed]
            x,y,_,idx=legacy.read_split(d,row,seed)
            pool=np.sort(np.r_[idx["fit"],idx["vote"]])
            val=x[idx["width"]]
            model=make_model(task=row["task"],seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
            model.fit(x[pool],y[pool])
            f=trees.extract(model,val[:1])
            gain=model.get_booster().get_score(importance_type="gain")
            order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
            g=f.cache(val)
            cols,ref=g.search("exact_greedy",order)
            reference=json.loads((historical.OUT/f"{name}-{seed}"/"selection.json").read_bytes())
            np.testing.assert_array_equal(cols,reference["records"]["exact_greedy"]["support"])
            rng=np.random.default_rng(seed)
            methods=("zero","one","four")
            for rep in range(-1,3):
                for name_method in rng.permutation(methods):
                    with patch.object(exact,"_projections",one_axis if name_method=="one" else original):
                        t=perf_counter()
                        keep,info=exact.search(g,order,name_method!="zero")
                        elapsed=perf_counter()-t
                    np.testing.assert_array_equal(keep,cols)
                    assert info.pop("deletion_order")==ref["deletion_order"]
                    if rep>=0:
                        data.append(dict(dataset=name,seed=seed,method=name_method,repeat=rep,k=len(cols),
                            seconds=elapsed,**info))
            print(name,seed,"all exact paths equal",flush=True)
    pd.DataFrame(data).to_csv(historical.DEST/"exact-one-axis.csv",index=False)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        run()
