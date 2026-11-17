"""Compare same-winner greedy solvers; all trials share the identical path-increment game."""
import argparse
import json
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as trees
from experiments.minpi_rework import quota_compress_tree_exact as exact
from experiments.minpi_rework.quota_compress_tree_run import OUT, DEST
from minppi_mofs.modeling import make_model


def run_game(g,order,seed,repeats=3):
    oracle=None
    records=[]
    arms=("dense","cache_only","directions","lazy_cache","lazy_directions")
    rng=np.random.default_rng(seed)
    for repeat in range(repeats):
        for method in rng.permutation(arms):
            t=perf_counter()
            if method=="dense":
                cols,diag=g.search("exact_greedy",order)
            else:
                cols,diag=exact.search(g,order,"directions" in method,lazy=method.startswith("lazy"))
            seconds=perf_counter()-t
            path=diag.pop("deletion_order")
            if oracle is None:
                oracle=path
            elif oracle!=path:
                raise AssertionError((method,oracle,path))
            records.append(dict(method=method,repeat=repeat,seconds=seconds,k=len(cols),**diag))
    return records,cols


def warm():
    rng=np.random.default_rng(9001)
    for d in (3,7,9):
        x=rng.normal(size=(60,d))
        x[:,-1]=x[:,0]
        x[:,-2]=0
        y=x[:,0]>0
        model=XGBClassifier(n_estimators=12,max_depth=3,n_jobs=1).fit(x,y)
        game=trees.extract(model,x[:1]).cache(x)
        run_game(game,rng.permutation(d),42,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[42])
    parser.add_argument("--datasets",nargs="+",default=legacy.CONFIG["datasets"])
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    all_results=[]
    jobs=legacy.jobs()
    with threadpool_limits(1):
        warm()
        for seed in args.seeds:
            for name in args.datasets:
                directory,row=jobs[name,seed]
                x,y,_,idx=legacy.read_split(directory,row,seed)
                pool=np.sort(np.r_[idx["fit"],idx["vote"]])
                val=x[idx["width"]]
                t=perf_counter()
                model=make_model(task=row["task"],seed=seed,backend=legacy.CONFIG["backend"],
                    model_params=legacy.CONFIG["model_params"])
                model.fit(x[pool],y[pool])
                f=trees.extract(model,val[:1])
                gain=model.get_booster().get_score(importance_type="gain")
                order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
                game=f.cache(val)
                prep=perf_counter()-t
                rec,cols=run_game(game,order,seed,args.repeats)
                original=json.loads((OUT/f"{name}-{seed}"/"selection.json").read_bytes())["records"]["exact_greedy"]
                np.testing.assert_array_equal(cols,original["support"])
                for rr in rec:
                    rr.update(dataset=name,seed=seed,train_parse_cache_s=prep,
                        original_prediction_s=original["pred_s"],pipeline_s=prep+rr["seconds"]+original["pred_s"])
                all_results.extend(rec)
                sub=pd.DataFrame(rec).groupby("method")[["seconds","recomputed"]].median()
                print(name,seed,sub.to_string(),flush=True)
    pd.DataFrame(all_results).to_csv(DEST/("exact-lazy-"+"-".join(map(str,args.seeds))+".csv"),index=False)


if __name__=="__main__":
    main()
